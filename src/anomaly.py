"""
anomaly.py
----------
Fault detection engine for the semiconductor-grade chiller system.

Two complementary detection strategies are employed:

1. **Z-Score Detection** — statistical deviation from the normal-phase
   baseline.  Any sensor reading more than ``Z_SCORE_THRESHOLD`` standard
   deviations from the baseline mean is flagged as an anomaly.

2. **Efficiency Anomaly Detection** — logic-based rule targeting refrigerant
   or coolant leakage scenarios.  A leakage fault forces the PID controller
   to demand more compressor work (higher control signal) to maintain the
   temperature setpoint, causing power consumption to exceed the expected
   value for a given control signal while water temperature remains stable.
   This pattern is invisible to pure temperature-based monitoring.

Pipeline
--------
  load_data()           → raw DataFrame from the most recent CSV
  compute_baseline()    → mean / std from the Normal Phase (t ≤ 300 s)
  detect_anomalies()    → scan full dataset, emit anomaly records
  save_anomaly_summary()→ persist anomaly records to JSON
  plot_anomalies()      → annotated matplotlib figure → PNG

Usage
-----
  python -m src.anomaly          # run from project root
  from src.anomaly import detect_anomalies  # import from other modules
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # headless backend — safe for servers / CI pipelines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
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
# Project layout constants
# ---------------------------------------------------------------------------
_SRC_DIR      = Path(__file__).parent
_PROJECT_ROOT = _SRC_DIR.parent
DATA_DIR      = _PROJECT_ROOT / "data"
OUTPUTS_DIR   = _PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------------------
# Detection hyper-parameters
# ---------------------------------------------------------------------------

# Number of seconds at the start of the dataset treated as normal operation.
# The simulator injects the first fault at t = 300 s.
BASELINE_DURATION_S: float = 300.0

# Z-score threshold above which a data point is considered anomalous.
Z_SCORE_THRESHOLD: float = 3.0

# Efficiency anomaly thresholds
# -- Water temperature must remain within this band (°C) of the baseline mean
#    for an observation to qualify as a "stable temperature" data point.
TEMP_STABILITY_BAND: float = 0.5   # °C

# -- Health metric (power / control_signal) must exceed the baseline mean by
#    this many baseline standard deviations to trigger an efficiency anomaly.
EFFICIENCY_Z_THRESHOLD: float = 3.0

# Sensors to monitor with z-score detection
Z_SCORE_SENSORS: list[str] = ["power_consumption", "flow_rate", "motor_vibration"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(csv_path: Path | None = None) -> pd.DataFrame:
    """
    Load sensor data from a CSV file.

    If *csv_path* is not provided the function automatically selects the
    most recently modified CSV file in the ``data/`` directory, so that
    re-generated datasets are picked up without changing any configuration.

    Parameters
    ----------
    csv_path : Path, optional
        Explicit path to a sensor CSV.  Falls back to auto-discovery.

    Returns
    -------
    pd.DataFrame
        Raw sensor data, sorted ascending by the ``time`` column.

    Raises
    ------
    FileNotFoundError
        If no CSV file can be located in ``data/``.
    """
    if csv_path is None:
        candidates = sorted(DATA_DIR.glob("synthetic_sensor_data*.csv"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError(
                f"No synthetic_sensor_data CSV found in {DATA_DIR}. "
                "Run src/main.py to generate one."
            )
        csv_path = candidates[0]

    log.info("Loading sensor data from: %s", csv_path)
    df = pd.read_csv(csv_path)

    # Defensive sort — ensures chronological ordering regardless of CSV order
    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)

    log.info("Loaded %d rows spanning t=%.1f s → t=%.1f s",
             len(df), df["time"].iloc[0], df["time"].iloc[-1])
    return df


# ---------------------------------------------------------------------------
# Baseline computation
# ---------------------------------------------------------------------------

def compute_baseline(df: pd.DataFrame) -> dict[str, Any]:
    """
    Derive statistical baselines from the Normal Phase of the dataset.

    The Normal Phase is defined as all rows where ``time ≤ BASELINE_DURATION_S``.
    Mean and standard deviation are calculated for each monitored sensor plus
    the derived Health Metric (power consumption normalised by control signal).

    Parameters
    ----------
    df : pd.DataFrame
        Full sensor dataset.

    Returns
    -------
    dict
        Keys:
        ``mean``  — dict[sensor_name → float]
        ``std``   — dict[sensor_name → float]
        ``n_samples`` — number of rows used for baseline estimation
    """
    normal_mask = df["time"] <= BASELINE_DURATION_S
    baseline_df = df.loc[normal_mask].copy()

    if baseline_df.empty:
        raise ValueError(
            f"No data points found in the first {BASELINE_DURATION_S} s. "
            "Verify that the CSV contains a Normal Phase."
        )

    log.info("Computing baseline from %d samples (t ≤ %.0f s).",
             len(baseline_df), BASELINE_DURATION_S)

    # --- Health Metric: power efficiency per unit control signal ------------
    # Guard against zero control signal to prevent division-by-zero errors.
    # Rows where the compressor is fully off are excluded from the health
    # metric baseline, as power/signal is undefined (and physically irrelevant)
    # when the compressor is commanded off.
    valid_ctrl = baseline_df["control_signal"] > 0.0
    if valid_ctrl.sum() == 0:
        raise ValueError(
            "All baseline control_signal values are zero — cannot compute "
            "Health Metric.  Check that the Normal Phase has PID activity."
        )

    baseline_df.loc[valid_ctrl, "health_metric"] = (
        baseline_df.loc[valid_ctrl, "power_consumption"]
        / baseline_df.loc[valid_ctrl, "control_signal"]
    )

    sensors = Z_SCORE_SENSORS + ["water_temp", "health_metric"]
    means: dict[str, float] = {}
    stds:  dict[str, float]  = {}

    for sensor in sensors:
        col_data = baseline_df[sensor].dropna()
        means[sensor] = float(col_data.mean())
        # Use a minimum std floor to avoid division-by-zero in z-score
        # computation for nearly-constant sensors during the normal phase.
        raw_std = float(col_data.std(ddof=1))
        stds[sensor] = max(raw_std, 1e-6)
        log.debug("  %-22s  mean=%8.4f  std=%8.4f", sensor, means[sensor], stds[sensor])

    return {
        "mean":      means,
        "std":       stds,
        "n_samples": int(valid_ctrl.sum()),
    }


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

def detect_anomalies(
    csv_path: Path | None = None,
    *,
    save_json: bool = True,
    save_plot: bool = True,
) -> list[dict[str, Any]]:
    """
    Scan the full sensor dataset and return a list of detected anomaly events.

    Each anomaly record contains:
    - ``time``          — timestamp (s) of the anomalous reading
    - ``sensor_type``   — name of the affected sensor / metric
    - ``severity_score``— magnitude of the deviation (z-score or derived score)
    - ``description``   — human-readable fault description

    Parameters
    ----------
    csv_path : Path, optional
        Explicit path to the sensor CSV.  Auto-discovers if omitted.
    save_json : bool
        If True, persist the anomaly list to ``data/anomalies_summary.json``.
    save_plot : bool
        If True, generate the annotated diagnostic plot.

    Returns
    -------
    tuple[list[dict], Path | None, Path | None]
        ``(anomalies, json_path, plot_path)`` — anomaly records sorted by
        ``time``, path to the saved JSON file, and path to the saved plot
        (each is ``None`` if the corresponding *save_** flag is False).
    """
    df       = load_data(csv_path)
    baseline = compute_baseline(df)
    means    = baseline["mean"]
    stds     = baseline["std"]

    anomalies: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Pre-compute the Health Metric for the full dataset.
    # Rows with zero control signal get NaN so they are safely skipped.
    # ------------------------------------------------------------------
    df["health_metric"] = np.where(
        df["control_signal"] > 0.0,
        df["power_consumption"] / df["control_signal"],
        np.nan,
    )

    # ------------------------------------------------------------------
    # Strategy 1 — Z-Score Detection
    # ------------------------------------------------------------------
    # For each monitored sensor, compute the z-score of every data point
    # relative to the Normal Phase statistics.  Points that exceed the
    # threshold are recorded as anomalies with a severity equal to the
    # absolute z-score, giving engineers a relative magnitude measure.
    # ------------------------------------------------------------------
    log.info("Running Z-score anomaly detection (threshold=%.1f σ).",
             Z_SCORE_THRESHOLD)

    for sensor in Z_SCORE_SENSORS:
        z_scores = (df[sensor] - means[sensor]) / stds[sensor]
        flagged  = df[np.abs(z_scores) > Z_SCORE_THRESHOLD]

        for _, row in flagged.iterrows():
            z = float(z_scores[row.name])
            direction = "above" if z > 0 else "below"
            anomalies.append({
                "time":           float(row["time"]),
                "sensor_type":    sensor,
                "severity_score": round(abs(z), 3),
                "description": (
                    f"{sensor} is {abs(z):.2f}σ {direction} the normal baseline "
                    f"(observed={row[sensor]:.4f}, "
                    f"baseline_mean={means[sensor]:.4f})."
                ),
                "detection_method": "z_score",
            })

    log.info("  → %d z-score anomaly events detected.", len(anomalies))

    # ------------------------------------------------------------------
    # Strategy 2 — Efficiency Anomaly Detection (Logic-based)
    # ------------------------------------------------------------------
    # An efficiency anomaly occurs when:
    #   (a) water_temp remains within the normal operating band — the PID
    #       controller is still maintaining the setpoint, masking the fault
    #       from simple temperature alarms.
    #   (b) The Health Metric (power / control_signal) significantly exceeds
    #       its baseline value — the compressor is consuming disproportionately
    #       more energy per unit of commanded cooling, which is the hallmark
    #       signature of refrigerant leakage causing efficiency degradation.
    #
    # This rule catches the leakage fault that the PID controller partially
    # compensates for, making it invisible to temperature-only monitoring.
    # ------------------------------------------------------------------
    log.info("Running efficiency anomaly detection.")

    temp_stable = (
        (df["water_temp"] - means["water_temp"]).abs() <= TEMP_STABILITY_BAND
    )
    hm_z_scores  = (df["health_metric"] - means["health_metric"]) / stds["health_metric"]
    hm_elevated  = hm_z_scores > EFFICIENCY_Z_THRESHOLD
    efficiency_mask = temp_stable & hm_elevated & df["health_metric"].notna()

    eff_anomaly_count = 0
    for _, row in df[efficiency_mask].iterrows():
        hm_z = float(hm_z_scores[row.name])
        # Severity scaled 0–10 based on how many σ above threshold
        severity = round(min(10.0, (hm_z / EFFICIENCY_Z_THRESHOLD) * 5.0), 3)
        anomalies.append({
            "time":           float(row["time"]),
            "sensor_type":    "health_metric (power/control_signal)",
            "severity_score": severity,
            "description": (
                f"Efficiency anomaly at t={row['time']:.1f}s: water_temp is stable "
                f"({row['water_temp']:.3f}°C ≈ setpoint) but power consumption is "
                f"disproportionately high relative to control signal "
                f"(health_metric={row['health_metric']:.3f}, "
                f"{hm_z:.2f}σ above baseline). "
                "Consistent with coolant/refrigerant leakage degrading cooling "
                "efficiency while PID compensates by increasing compressor demand."
            ),
            "detection_method": "efficiency_logic",
        })
        eff_anomaly_count += 1

    log.info("  → %d efficiency anomaly events detected.", eff_anomaly_count)

    # Sort chronologically for readability
    anomalies.sort(key=lambda a: a["time"])
    log.info("Total anomalies detected: %d", len(anomalies))

    # ------------------------------------------------------------------
    # Persist outputs
    # ------------------------------------------------------------------
    json_path = save_anomaly_summary(anomalies, baseline) if save_json else None
    plot_path = plot_anomalies(df, anomalies, baseline)   if save_plot else None

    return anomalies, json_path, plot_path


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def save_anomaly_summary(
    anomalies: list[dict[str, Any]],
    baseline:  dict[str, Any],
) -> Path:
    """
    Persist detected anomalies and baseline statistics to a JSON file.

    Parameters
    ----------
    anomalies : list[dict]
        Output of ``detect_anomalies()``.
    baseline : dict
        Baseline statistics from ``compute_baseline()``.

    Returns
    -------
    Path
        Absolute path to the written JSON file.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    timestamp   = datetime.now().strftime("%m%d%Y_%H%M")
    output_path = DATA_DIR / f"anomalies_summary_{timestamp}.json"

    # Summarise anomaly counts by detection method and sensor type
    method_counts: dict[str, int] = {}
    sensor_counts: dict[str, int] = {}
    for a in anomalies:
        method_counts[a["detection_method"]] = (
            method_counts.get(a["detection_method"], 0) + 1
        )
        sensor_counts[a["sensor_type"]] = (
            sensor_counts.get(a["sensor_type"], 0) + 1
        )

    payload = {
        "summary": {
            "total_anomalies":        len(anomalies),
            "anomalies_by_method":    method_counts,
            "anomalies_by_sensor":    sensor_counts,
            "baseline_duration_s":    BASELINE_DURATION_S,
            "z_score_threshold":      Z_SCORE_THRESHOLD,
            "efficiency_z_threshold": EFFICIENCY_Z_THRESHOLD,
        },
        "baseline_statistics": {
            "mean": baseline["mean"],
            "std":  baseline["std"],
            "n_samples_used": baseline["n_samples"],
        },
        "anomalies": anomalies,
    }

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    log.info("Anomaly summary saved → %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_anomalies(
    df:        pd.DataFrame,
    anomalies: list[dict[str, Any]],
    baseline:  dict[str, Any],
) -> Path:
    """
    Generate an annotated diagnostic plot highlighting anomaly regions.

    The figure contains four subplots:
      1. Power Consumption — with z-score and efficiency anomaly markers
      2. Water Temperature — showing thermal stability during the leakage phase
      3. Health Metric (power / control_signal) — efficiency ratio over time
      4. Motor Vibration & Flow Rate — secondary fault indicators

    Anomaly regions are shaded with translucent overlays, and individual
    flagged points are marked with coloured scatter symbols.

    Parameters
    ----------
    df : pd.DataFrame
        Full sensor dataset including the pre-computed ``health_metric`` column.
    anomalies : list[dict]
        Detected anomaly records from ``detect_anomalies()``.
    baseline : dict
        Baseline statistics for reference line rendering.

    Returns
    -------
    Path
        Absolute path to the saved PNG file.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUTS_DIR / "anomaly_detection_plot.png"

    # Separate anomalies by detection method for distinct visual treatment
    zscore_times = {
        sensor: [] for sensor in Z_SCORE_SENSORS
    }
    efficiency_times: list[float] = []

    for a in anomalies:
        if a["detection_method"] == "z_score" and a["sensor_type"] in zscore_times:
            zscore_times[a["sensor_type"]].append(a["time"])
        elif a["detection_method"] == "efficiency_logic":
            efficiency_times.append(a["time"])

    # Build contiguous anomaly time windows for region shading
    def _anomaly_spans(times: list[float], gap_s: float = 5.0) -> list[tuple[float, float]]:
        """
        Merge nearby anomaly timestamps into contiguous shaded spans.

        Consecutive timestamps separated by less than *gap_s* seconds are
        merged into a single span to produce cleaner shading on the plot.
        """
        if not times:
            return []
        sorted_t = sorted(times)
        spans: list[tuple[float, float]] = []
        start = end = sorted_t[0]
        for t in sorted_t[1:]:
            if t - end <= gap_s:
                end = t
            else:
                spans.append((start, end))
                start = end = t
        spans.append((start, end))
        return spans

    efficiency_spans = _anomaly_spans(efficiency_times)
    power_zscore_spans = _anomaly_spans(zscore_times.get("power_consumption", []))

    # ------------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    fig.suptitle(
        "Chiller System — Anomaly Detection Report\n"
        f"Z-Score Threshold: {Z_SCORE_THRESHOLD}σ  |  "
        f"Efficiency Z-Threshold: {EFFICIENCY_Z_THRESHOLD}σ  |  "
        f"Baseline: first {int(BASELINE_DURATION_S)} s",
        fontsize=13, fontweight="bold", y=0.98,
    )

    time = df["time"].values

    # ---- Colour palette (colourblind-friendly) -------------------------
    C_NORMAL    = "#2196F3"   # blue   — normal operation line
    C_BASELINE  = "#4CAF50"   # green  — baseline reference level
    C_ZSCORE    = "#F44336"   # red    — z-score anomaly markers
    C_EFFIC     = "#FF9800"   # orange — efficiency anomaly shading
    C_SHADE_BG  = "#FFEB3B"   # yellow — normal-phase background band

    def _shade_normal_phase(ax: plt.Axes) -> None:
        """Shade the Normal Phase window to orient the viewer."""
        ax.axvspan(0, BASELINE_DURATION_S, alpha=0.07,
                   color=C_SHADE_BG, label="Normal Phase")

    def _shade_anomaly_spans(ax: plt.Axes, spans: list[tuple[float, float]],
                              color: str, label: str, alpha: float = 0.25) -> None:
        """Draw translucent shading over detected anomaly windows."""
        for i, (t0, t1) in enumerate(spans):
            ax.axvspan(t0, t1, alpha=alpha, color=color,
                       label=label if i == 0 else "_nolegend_")

    # ---- Subplot 1: Power Consumption ----------------------------------
    ax1 = axes[0]
    ax1.plot(time, df["power_consumption"], color=C_NORMAL,
             linewidth=0.8, alpha=0.85, label="Power Consumption (kW)")
    ax1.axhline(baseline["mean"]["power_consumption"], color=C_BASELINE,
                linestyle="--", linewidth=1.2, label="Baseline Mean")
    ax1.axhline(
        baseline["mean"]["power_consumption"] + Z_SCORE_THRESHOLD * baseline["std"]["power_consumption"],
        color=C_ZSCORE, linestyle=":", linewidth=1.0, alpha=0.7, label=f"+{Z_SCORE_THRESHOLD}σ Threshold",
    )

    # Shade efficiency anomaly windows on the power subplot
    _shade_normal_phase(ax1)
    _shade_anomaly_spans(ax1, efficiency_spans, C_EFFIC, "Efficiency Anomaly Region")
    _shade_anomaly_spans(ax1, power_zscore_spans, C_ZSCORE, "Z-Score Anomaly Region", alpha=0.15)

    # Mark individual z-score flagged power points
    if zscore_times["power_consumption"]:
        pwr_flag_df = df[df["time"].isin(zscore_times["power_consumption"])]
        ax1.scatter(pwr_flag_df["time"], pwr_flag_df["power_consumption"],
                    color=C_ZSCORE, s=18, zorder=5, label="Flagged Point (Z-Score)")

    ax1.set_ylabel("Power (kW)", fontsize=10)
    ax1.set_title("Power Consumption", fontsize=11, loc="left")
    ax1.legend(loc="upper left", fontsize=7.5, ncol=3)
    ax1.grid(True, alpha=0.3)

    # ---- Subplot 2: Water Temperature ----------------------------------
    ax2 = axes[1]
    ax2.plot(time, df["water_temp"], color=C_NORMAL,
             linewidth=0.8, alpha=0.85, label="Water Temperature (°C)")
    ax2.axhline(baseline["mean"]["water_temp"], color=C_BASELINE,
                linestyle="--", linewidth=1.2, label="Baseline Mean")
    ax2.axhline(
        baseline["mean"]["water_temp"] + TEMP_STABILITY_BAND,
        color=C_EFFIC, linestyle=":", linewidth=1.0, alpha=0.8,
        label=f"Stability Band (±{TEMP_STABILITY_BAND}°C)",
    )
    ax2.axhline(
        baseline["mean"]["water_temp"] - TEMP_STABILITY_BAND,
        color=C_EFFIC, linestyle=":", linewidth=1.0, alpha=0.8,
        label="_nolegend_",
    )
    _shade_normal_phase(ax2)
    # Highlight where temp remains stable despite fault — the "hidden fault" zone
    _shade_anomaly_spans(ax2, efficiency_spans, C_EFFIC,
                         "Stable Temp / High Power (Hidden Fault)")

    ax2.set_ylabel("Temperature (°C)", fontsize=10)
    ax2.set_title("Water Temperature", fontsize=11, loc="left")
    ax2.legend(loc="upper left", fontsize=7.5, ncol=2)
    ax2.grid(True, alpha=0.3)

    # ---- Subplot 3: Health Metric (Power Efficiency) --------------------
    ax3 = axes[2]
    hm_values = df["health_metric"].values
    ax3.plot(time, hm_values, color="#9C27B0",
             linewidth=0.8, alpha=0.85, label="Health Metric (kW / control)")
    ax3.axhline(baseline["mean"]["health_metric"], color=C_BASELINE,
                linestyle="--", linewidth=1.2, label="Baseline Mean")
    hm_upper = (baseline["mean"]["health_metric"]
                + EFFICIENCY_Z_THRESHOLD * baseline["std"]["health_metric"])
    ax3.axhline(hm_upper, color=C_EFFIC, linestyle=":",
                linewidth=1.2, label=f"+{EFFICIENCY_Z_THRESHOLD}σ Efficiency Threshold")
    _shade_normal_phase(ax3)
    _shade_anomaly_spans(ax3, efficiency_spans, C_EFFIC, "Efficiency Anomaly Region")

    # Mark individual efficiency anomaly points
    if efficiency_times:
        eff_flag_df = df[df["time"].isin(efficiency_times)]
        ax3.scatter(eff_flag_df["time"], eff_flag_df["health_metric"],
                    color=C_EFFIC, s=18, zorder=5, label="Efficiency Anomaly Point")

    ax3.set_ylabel("kW / (control signal)", fontsize=10)
    ax3.set_title("Health Metric — Power Efficiency Ratio", fontsize=11, loc="left")
    ax3.legend(loc="upper left", fontsize=7.5, ncol=2)
    ax3.grid(True, alpha=0.3)

    # ---- Subplot 4: Motor Vibration & Flow Rate (twin axis) ------------
    ax4  = axes[3]
    ax4b = ax4.twinx()

    vib_times = zscore_times.get("motor_vibration", [])
    flow_times = zscore_times.get("flow_rate", [])

    ax4.plot(time, df["motor_vibration"], color="#795548",
             linewidth=0.8, alpha=0.85, label="Motor Vibration (mm/s)")
    ax4.axhline(baseline["mean"]["motor_vibration"], color="#795548",
                linestyle="--", linewidth=1.0, alpha=0.6, label="Vibration Baseline")
    if vib_times:
        vib_flag_df = df[df["time"].isin(vib_times)]
        ax4.scatter(vib_flag_df["time"], vib_flag_df["motor_vibration"],
                    color=C_ZSCORE, s=18, zorder=5, label="Vibration Anomaly")

    ax4b.plot(time, df["flow_rate"], color="#00BCD4",
              linewidth=0.8, alpha=0.7, label="Flow Rate (L/min)")
    ax4b.axhline(baseline["mean"]["flow_rate"], color="#00BCD4",
                 linestyle="--", linewidth=1.0, alpha=0.6, label="Flow Baseline")
    if flow_times:
        flow_flag_df = df[df["time"].isin(flow_times)]
        ax4b.scatter(flow_flag_df["time"], flow_flag_df["flow_rate"],
                     color="#F44336", s=18, zorder=5, marker="^", label="Flow Anomaly")

    _shade_normal_phase(ax4)
    _shade_anomaly_spans(ax4, efficiency_spans, C_EFFIC, "Efficiency Anomaly Region")

    ax4.set_ylabel("Vibration (mm/s)", fontsize=10)
    ax4b.set_ylabel("Flow Rate (L/min)", fontsize=10)
    ax4.set_xlabel("Time (s)", fontsize=10)
    ax4.set_title("Motor Vibration & Flow Rate", fontsize=11, loc="left")

    # Merge legends from both y-axes
    lines_left,  labels_left  = ax4.get_legend_handles_labels()
    lines_right, labels_right = ax4b.get_legend_handles_labels()
    ax4.legend(lines_left + lines_right, labels_left + labels_right,
               loc="upper left", fontsize=7.5, ncol=3)
    ax4.grid(True, alpha=0.3)

    # ---- Vertical fault-injection marker on all axes -------------------
    for ax in axes:
        ax.axvline(BASELINE_DURATION_S, color="#607D8B", linestyle="-.",
                   linewidth=1.2, alpha=0.8, label="_nolegend_")
        ax.text(BASELINE_DURATION_S + 5, ax.get_ylim()[0],
                "Fault\nInjected", fontsize=7, color="#607D8B", va="bottom")

    # ---- Global legend patch -------------------------------------------
    legend_patches = [
        mpatches.Patch(color=C_SHADE_BG,  alpha=0.4, label="Normal Phase"),
        mpatches.Patch(color=C_EFFIC,     alpha=0.35, label="Efficiency Anomaly Region"),
        mpatches.Patch(color=C_ZSCORE,    alpha=0.25, label="Z-Score Anomaly Region"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, 0.005), framealpha=0.9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    log.info("Anomaly detection plot saved → %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    csv_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    results, json_path, plot_path = detect_anomalies(
        csv_path=csv_arg, save_json=True, save_plot=True
    )

    print(f"\nDetection complete — {len(results)} anomaly events found.")
    print(f"  JSON summary : {json_path}")
    print(f"  Plot         : {plot_path}")
