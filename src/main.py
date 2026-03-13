"""
main.py
-------
Primary entry point for the chiller simulation loop.

Integrates ChillerSimulator and PIDController to generate a synthetic
sensor dataset for AI-based fault detection and diagnostics research.

Simulation timeline
-------------------
  0 –  300 s : Normal operation — system reaches thermal steady state.
300 –  700 s : Leakage fault injected (severity=0.7) — PID compensates,
               power consumption rises as cooling efficiency degrades.
700 – 1000 s : Continued operation under fault — observe long-run behaviour.

Output
------
  data/synthetic_sensor_data_<MMDDYYYY_HHMM>.csv
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Resolve project root so this script can be run from any working directory
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).parent
_PROJECT_ROOT = _SRC_DIR.parent
sys.path.insert(0, str(_SRC_DIR))

from simulator import ChillerSimulator  # noqa: E402
from controller import PIDController    # noqa: E402

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
TOTAL_STEPS   = 1000
DT            = 1.0       # seconds per step
SETPOINT      = 20.0      # °C
FAULT_START   = 300       # step at which fault is injected
FAULT_END     = 700       # (fault persists; this is just logged for clarity)

OUTPUT_DIR = _PROJECT_ROOT / "data"


def run_simulation() -> list[dict]:
    """
    Execute the full 1000-step simulation and return collected records.

    Returns
    -------
    list[dict]
        One dictionary per time step containing all sensor readings plus
        fault severities and the PID control signal.
    """
    sim = ChillerSimulator(dt=DT, seed=42)
    pid = PIDController(setpoint=SETPOINT, kp=0.4, ki=0.08, kd=0.02)

    fault_injected = False
    records: list[dict] = []

    log.info("Starting simulation: %d steps × %.1f s/step", TOTAL_STEPS, DT)

    for step in range(TOTAL_STEPS):
        # ------------------------------------------------------------------ #
        # Event scheduling
        # ------------------------------------------------------------------ #
        if step == FAULT_START:
            log.info(
                "Step %d — injecting leakage fault (severity=0.7, ramp_rate=0.005)",
                step,
            )
            sim.inject_fault("leakage", severity=0.7, ramp_rate=0.005)
            fault_injected = True

        if step == FAULT_END and fault_injected:
            log.info(
                "Step %d — fault injection phase complete; "
                "leakage remains at achieved severity.",
                step,
            )

        # ------------------------------------------------------------------ #
        # Control → simulate → log
        # ------------------------------------------------------------------ #
        state = sim.get_current_state()
        control_signal = pid.update(state["water_temp"], dt=DT)
        state = sim.update(control_signal)

        # Flatten faults into top-level columns for clean CSV output
        record = {
            "time":               state["time"],
            "water_temp":         state["water_temp"],
            "flow_rate":          state["flow_rate"],
            "motor_vibration":    state["motor_vibration"],
            "power_consumption":  state["power_consumption"],
            "control_signal":     round(control_signal, 6),
            "fault_leakage":      state["faults"]["leakage"],
            "fault_clogging":     state["faults"]["clogging"],
            "fault_bearing_wear": state["faults"]["bearing_wear"],
        }
        records.append(record)


    log.info("Simulation complete — %d records collected.", len(records))
    return records


def save_csv(records: list[dict]) -> None:
    """Persist *records* to a timestamped CSV file under ``data/``."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%m%d%Y_%H%M")
    output_file = OUTPUT_DIR / f"synthetic_sensor_data_{timestamp}.csv"
    df = pd.DataFrame(records)
    df.to_csv(output_file, index=False)
    log.info("Dataset saved → %s  (%d rows × %d cols)", output_file, *df.shape)


def main() -> None:
    log.info("=== Chiller Simulation — AI Dataset Generator ===")
    records = run_simulation()
    save_csv(records)
    log.info("Done.")


if __name__ == "__main__":
    main()
