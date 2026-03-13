"""
agent.py
--------
AI Industrial Maintenance Expert for the semiconductor-grade chiller system.

This agent:
  1. Loads detected anomalies from the most recent anomalies_summary JSON.
  2. Reads the last 20 rows of the most recent sensor CSV for current system state.
  3. Constructs a physics-aware diagnostic prompt.
  4. Calls Claude Opus 4.6 (with adaptive thinking + streaming) to produce a
     structured Root Cause Analysis (RCA).
  5. Writes the report to data/maintenance_report.md and prints a summary.

Usage
-----
  python src/agent.py          # run from project root
  python -m src.agent          # alternative invocation
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
_SRC_DIR      = Path(__file__).parent
_PROJECT_ROOT = _SRC_DIR.parent
DATA_DIR      = _PROJECT_ROOT / "data"

# Number of trailing rows used to summarise the current system state
RECENT_ROWS = 20


# ---------------------------------------------------------------------------
# Context loading
# ---------------------------------------------------------------------------

def load_anomaly_summary() -> dict:
    """
    Load the most recent anomaly summary JSON from DATA_DIR.

    Prefers a file named ``anomalies_summary.json`` (fixed name); falls back
    to the most recently modified timestamped file if that does not exist.

    Returns
    -------
    dict
        Parsed JSON content of the anomaly summary.

    Raises
    ------
    FileNotFoundError
        If no anomaly summary file can be located.
    """
    fixed = DATA_DIR / "anomalies_summary.json"
    if fixed.exists():
        path = fixed
    else:
        candidates = sorted(
            DATA_DIR.glob("anomalies_summary*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(
                f"No anomaly summary JSON found in {DATA_DIR}. "
                "Run src/anomaly.py first to generate one."
            )
        path = candidates[0]

    log.info("Loading anomaly summary from: %s", path)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_recent_sensor_state(n_rows: int = RECENT_ROWS) -> pd.DataFrame:
    """
    Return the last *n_rows* rows of the most recently generated sensor CSV.

    Parameters
    ----------
    n_rows : int
        Number of trailing rows to return (default: 20).

    Returns
    -------
    pd.DataFrame
        Subset of the sensor data sorted ascending by ``time``.

    Raises
    ------
    FileNotFoundError
        If no sensor CSV is found in DATA_DIR.
    """
    candidates = sorted(
        DATA_DIR.glob("synthetic_sensor_data*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No synthetic_sensor_data CSV found in {DATA_DIR}. "
            "Run src/main.py to generate one."
        )
    path = candidates[0]
    log.info("Loading sensor data from: %s", path)
    df = pd.read_csv(path).sort_values("time")
    return df.tail(n_rows).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _format_sensor_snapshot(df: pd.DataFrame) -> str:
    """Render the recent sensor rows as a concise tabular string for the prompt."""
    cols = ["time", "water_temp", "power_consumption", "flow_rate",
            "motor_vibration", "control_signal"]
    available = [c for c in cols if c in df.columns]
    return df[available].to_string(index=False, float_format=lambda x: f"{x:.4f}")


def _summarise_anomalies(summary: dict) -> str:
    """
    Produce a compact prose summary of the anomaly detection results
    suitable for inclusion in the LLM prompt.
    """
    meta   = summary.get("summary", {})
    base   = summary.get("baseline_statistics", {})
    events = summary.get("anomalies", [])

    # Aggregate stats
    total    = meta.get("total_anomalies", len(events))
    by_method = meta.get("anomalies_by_method", {})
    by_sensor = meta.get("anomalies_by_sensor", {})

    # Find the onset of the first non-startup anomaly (skip t < 10 s)
    meaningful = [a for a in events if a.get("time", 0) >= 10.0]
    onset_time = meaningful[0]["time"] if meaningful else None

    # Highest-severity events (top 3)
    top_events = sorted(events, key=lambda a: a.get("severity_score", 0), reverse=True)[:3]

    lines = [
        f"Total anomaly events detected: {total}",
        f"  • Z-score anomalies   : {by_method.get('z_score', 0)}",
        f"  • Efficiency anomalies: {by_method.get('efficiency_logic', 0)}",
        "",
        "Anomalies by sensor channel:",
    ]
    for sensor, count in by_sensor.items():
        lines.append(f"  • {sensor}: {count} events")

    if onset_time is not None:
        lines += ["", f"Earliest meaningful anomaly onset: t = {onset_time:.1f} s"]

    lines += [
        "",
        "Baseline statistics (Normal Phase, t ≤ 300 s):",
        f"  • power_consumption  mean={base['mean'].get('power_consumption', 'N/A'):.4f} kW, "
        f"std={base['std'].get('power_consumption', 'N/A'):.4f}",
        f"  • water_temp         mean={base['mean'].get('water_temp', 'N/A'):.4f} °C, "
        f"std={base['std'].get('water_temp', 'N/A'):.4f}",
        f"  • health_metric      mean={base['mean'].get('health_metric', 'N/A'):.4f}, "
        f"std={base['std'].get('health_metric', 'N/A'):.4f}",
        "",
        "Top-severity anomaly events:",
    ]
    for ev in top_events:
        lines.append(
            f"  [t={ev['time']:.1f}s, severity={ev.get('severity_score', '?'):.3f}] "
            f"{ev.get('description', '(no description)')}"
        )

    return "\n".join(lines)


def build_diagnostic_prompt(anomaly_summary: dict, recent_df: pd.DataFrame) -> tuple[str, str]:
    """
    Construct the system and user messages for the LLM diagnostic call.

    Returns
    -------
    tuple[str, str]
        ``(system_prompt, user_message)``
    """
    system_prompt = (
        "You are a Senior Semiconductor Equipment Engineer with deep expertise in "
        "cryogenic cooling systems, refrigeration thermodynamics, and predictive "
        "maintenance analytics. Your responsibilities include performing Root Cause "
        "Analysis (RCA) on equipment faults detected by automated sensor systems. "
        "You produce structured, evidence-based maintenance reports that field "
        "technicians can act on immediately. Your tone is professional, objective, "
        "and data-driven. Avoid speculation without supporting data."
    )

    anomaly_text  = _summarise_anomalies(anomaly_summary)
    snapshot_text = _format_sensor_snapshot(recent_df)

    # Latest readings for quick reference
    last = recent_df.iloc[-1]
    current_temp    = last.get("water_temp",        float("nan"))
    current_power   = last.get("power_consumption", float("nan"))
    current_ctrl    = last.get("control_signal",    float("nan"))

    user_message = f"""
## Equipment Profile

**System:** High-precision semiconductor process chiller
**Function:** Maintains coolant temperature at a 20.0 °C setpoint for wafer processing.
**PID controller:** Active — continuously adjusting compressor output to hold setpoint.

---

## Statistical Anomaly Report (from automated detection engine)

{anomaly_text}

---

## Current System State — Last {RECENT_ROWS} Sensor Readings

```
{snapshot_text}
```

**Latest reading summary:**
  - Water temperature   : {current_temp:.4f} °C  (setpoint: 20.0 °C)
  - Power consumption   : {current_power:.4f} kW
  - PID control signal  : {current_ctrl:.4f}

---

## Physics Reminder

In a refrigerant-based chiller loop:

1. **Power spike without temperature deviation** — When the PID controller maintains
   the temperature setpoint but power consumption rises disproportionately relative
   to the commanded control signal (health metric = power / control_signal increases),
   this almost always indicates **efficiency loss rather than a thermal load increase**.
   Common root causes: refrigerant leakage (reduced mass flow → compressor works harder
   to achieve the same heat transfer), partial valve obstruction, or condenser fouling.

2. **Stable temperature during fault** — A PID-controlled system can mask early-stage
   faults by increasing compressor demand. The fault is only detectable through
   energy-efficiency metrics, not temperature alone.

3. **Flow anomalies** — Transient flow-rate deviations (z-score spikes) alongside
   elevated power may indicate refrigerant loss events causing momentary pressure drops.

---

## Task

Based on all evidence above, produce a professional Root Cause Analysis (RCA) report
structured **exactly** as follows (use these exact section headers):

**[DIAGNOSIS]**
State the single most probable root cause of the observed fault. Be specific about
the fault type, affected subsystem, and estimated severity.

**[EVIDENCE]**
Provide bullet points that directly link each piece of sensor/statistical evidence
to the stated diagnosis. Reference specific timestamps, z-scores, or metric values
from the data above.

**[RECOMMENDATION]**
List specific, prioritised next steps for the field technician. Include immediate
inspection actions, any parts/tools required, and a recommended timeline.
""".strip()

    return system_prompt, user_message


# ---------------------------------------------------------------------------
# LLM Diagnostic Call
# ---------------------------------------------------------------------------

OLLAMA_MODEL = "llama3.2"


def ensure_ollama_ready() -> None:
    """
    Ensure Ollama is installed, the server is running, and the model is pulled.

    Steps
    -----
    1. Install Ollama via Homebrew if the ``ollama`` CLI is not found.
    2. Start ``ollama serve`` in the background if the server is not reachable.
    3. Pull OLLAMA_MODEL if it has not been downloaded yet.
    4. Install the ``ollama`` Python package if missing.
    """
    # ── 1. Install Ollama CLI ────────────────────────────────────────────────
    if not shutil.which("ollama"):
        log.info("Ollama CLI not found — installing via Homebrew…")
        if not shutil.which("brew"):
            sys.exit(
                "ERROR: Homebrew is required to auto-install Ollama.\n"
                "Install Homebrew from https://brew.sh, then re-run,\n"
                "or install Ollama manually from https://ollama.com."
            )
        subprocess.run(["brew", "install", "ollama"], check=True)
        log.info("Ollama installed successfully.")

    # ── 2. Start the server if not already running ───────────────────────────
    probe = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
    )
    if probe.returncode != 0:
        log.info("Ollama server not running — starting in background…")
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Give it a moment to start accepting connections
        time.sleep(3)
        log.info("Ollama server started.")

    # ── 3. Pull model if not present ─────────────────────────────────────────
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if OLLAMA_MODEL not in result.stdout:
        log.info("Model '%s' not found — pulling now (this may take a few minutes)…", OLLAMA_MODEL)
        subprocess.run(["ollama", "pull", OLLAMA_MODEL], check=True)
        log.info("Model '%s' ready.", OLLAMA_MODEL)

    # ── 4. Ensure Python ollama package is installed ──────────────────────────
    try:
        import ollama as _  # noqa: F401
    except ImportError:
        log.info("Installing ollama Python package…")
        subprocess.run([sys.executable, "-m", "pip", "install", "ollama"], check=True)
        log.info("ollama package installed.")


def run_diagnostic(system_prompt: str, user_message: str) -> str:
    """
    Call a local Ollama model using the native ollama library with streaming.

    Calls ensure_ollama_ready() first to auto-install and start Ollama if needed.

    Parameters
    ----------
    system_prompt : str
        The expert-role system instruction.
    user_message : str
        The full diagnostic context and task.

    Returns
    -------
    str
        The raw text of the RCA report.
    """
    ensure_ollama_ready()

    import ollama  # imported here so auto-install above takes effect first

    log.info("Calling Ollama (%s) for Root Cause Analysis (streaming)…", OLLAMA_MODEL)

    print("\n" + "=" * 70)
    print(f"  AI INDUSTRIAL MAINTENANCE EXPERT — RCA in progress ({OLLAMA_MODEL})…")
    print("=" * 70 + "\n")

    rca_text = ""
    for chunk in ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        stream=True,
    ):
        text = chunk["message"]["content"]
        print(text, end="", flush=True)
        rca_text += text

    print("\n")

    log.info("RCA complete — %d characters generated", len(rca_text))

    return rca_text


# ---------------------------------------------------------------------------
# Report output
# ---------------------------------------------------------------------------

def save_report(rca_text: str, anomaly_summary: dict, recent_df: pd.DataFrame) -> Path:
    """
    Write the complete maintenance report to ``data/maintenance_report.md``.

    The report includes a metadata header, the anomaly overview, the current
    system snapshot, and the full AI-generated RCA.

    Parameters
    ----------
    rca_text : str
        Claude's RCA output.
    anomaly_summary : dict
        Parsed anomaly summary for the header section.
    recent_df : pd.DataFrame
        Recent sensor data for the snapshot section.

    Returns
    -------
    Path
        Path to the written report file.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    report_path = DATA_DIR / "maintenance_report.md"

    meta      = anomaly_summary.get("summary", {})
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    last      = recent_df.iloc[-1]

    report = f"""# Semiconductor Chiller — AI Maintenance Report

**Generated:** {timestamp}
**System:** High-precision semiconductor process chiller
**Analysis engine:** Claude Opus 4.6 (adaptive thinking)

---

## Anomaly Detection Summary

| Metric                      | Value                                   |
|-----------------------------|-----------------------------------------|
| Total anomalies detected    | {meta.get('total_anomalies', 'N/A')}   |
| Z-score anomalies           | {meta.get('anomalies_by_method', {}).get('z_score', 0)} |
| Efficiency anomalies        | {meta.get('anomalies_by_method', {}).get('efficiency_logic', 0)} |
| Baseline duration           | {meta.get('baseline_duration_s', 300)} s |
| Z-score threshold           | {meta.get('z_score_threshold', 3.0)} σ |
| Efficiency Z threshold      | {meta.get('efficiency_z_threshold', 3.0)} σ |

**Anomalies by sensor channel:**

{chr(10).join(f"- **{sensor}**: {count} events" for sensor, count in meta.get("anomalies_by_sensor", {}).items())}

---

## Current System State (last {RECENT_ROWS} readings)

```
{_format_sensor_snapshot(recent_df)}
```

**Latest values:**
- Water temperature  : {last.get('water_temp', float('nan')):.4f} °C (setpoint: 20.0 °C)
- Power consumption  : {last.get('power_consumption', float('nan')):.4f} kW
- PID control signal : {last.get('control_signal', float('nan')):.4f}

---

## Root Cause Analysis

{rca_text}

---

*This report was generated automatically by the AI Industrial Maintenance Expert agent.
All findings should be validated by a qualified field engineer before action is taken.*
"""

    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report)

    log.info("Maintenance report saved → %s", report_path)
    return report_path


def print_summary(rca_text: str, report_path: Path, anomaly_summary: dict) -> None:
    """Print a concise terminal summary of the agent's findings."""
    meta  = anomaly_summary.get("summary", {})
    total = meta.get("total_anomalies", "N/A")

    # Extract just the [DIAGNOSIS] section for the terminal summary
    diagnosis_line = "(see full report)"
    for line in rca_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("**[DIAGNOSIS]**") or stripped == "[DIAGNOSIS]":
            # Grab the next non-empty line as the one-liner
            continue
        if "[DIAGNOSIS]" in stripped and len(stripped) > 15:
            diagnosis_line = stripped.replace("**[DIAGNOSIS]**", "").strip(" *:")
            break

    # Fallback: first substantive paragraph after the header
    if diagnosis_line == "(see full report)":
        capture = False
        for line in rca_text.splitlines():
            if "[DIAGNOSIS]" in line:
                capture = True
                continue
            if capture and line.strip():
                diagnosis_line = line.strip()
                break

    print("\n" + "=" * 70)
    print("  AGENT SUMMARY")
    print("=" * 70)
    print(f"  Anomalies analysed : {total}")
    print(f"  Primary diagnosis  : {diagnosis_line[:120]}")
    print(f"  Full report        : {report_path}")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=== AI Industrial Maintenance Expert — Starting ===")

    # 1. Load context
    anomaly_summary = load_anomaly_summary()
    recent_df       = load_recent_sensor_state()

    log.info(
        "Context loaded — %d total anomalies, %d recent sensor rows",
        anomaly_summary.get("summary", {}).get("total_anomalies", 0),
        len(recent_df),
    )

    # 2. Build prompt
    system_prompt, user_message = build_diagnostic_prompt(anomaly_summary, recent_df)

    # 3. Run LLM diagnostic
    rca_text = run_diagnostic(system_prompt, user_message)

    # 4. Save report and print summary
    report_path = save_report(rca_text, anomaly_summary, recent_df)
    print_summary(rca_text, report_path, anomaly_summary)

    log.info("=== Agent run complete ===")


if __name__ == "__main__":
    main()
