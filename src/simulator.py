"""
simulator.py
------------
Physics-based simulator for a high-precision Chiller System used in
semiconductor manufacturing (e.g., immersion lithography / etching).

The simulator implements a first-order lag thermal model with realistic
sensor noise and a fault injection system for AI agent diagnostics.
"""

import random
import math
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Physical / default constants
# ---------------------------------------------------------------------------

TEMP_SETPOINT       = 20.0   # °C   — target chilled water temperature
FLOW_NOMINAL        = 50.0   # L/min — nominal coolant flow rate
VIBRATION_NOMINAL   = 0.3    # mm/s  — healthy bearing vibration level
MAX_COOLING_POWER   = 15.0   # kW    — rated cooling capacity of the chiller
THERMAL_CAPACITY    = 5.0    # kJ/°C — lumped thermal mass of the chilled loop
DT                  = 1.0    # s     — simulation time step

# Gaussian noise amplitudes (1-sigma) for each sensor
NOISE_TEMP      = 0.05   # °C
NOISE_FLOW      = 0.2    # L/min
NOISE_VIBRATION = 0.02   # mm/s
NOISE_POWER     = 0.05   # kW


@dataclass
class FaultState:
    """Holds the progressive severity of each injected fault (0.0 - 1.0)."""

    leakage:      float = 0.0   # coolant / refrigerant leakage severity
    clogging:     float = 0.0   # filter / heat-exchanger clogging severity
    bearing_wear: float = 0.0   # pump bearing wear severity


class ChillerSimulator:
    """
    High-fidelity simulator for a semiconductor-grade chiller system.

    Dynamics are governed by a first-order lag thermal model:

        T_next = T_curr + (Q_load - Q_cooling) * (dt / C_thermal)

    where
        Q_load    — heat injected by the production tool (kW)
        Q_cooling — actual cooling power delivered (kW)
        C_thermal — lumped thermal capacity of the chilled loop (kJ/°C)

    Parameters
    ----------
    dt : float
        Simulation time step in seconds (default: 1.0 s).
    seed : int, optional
        Random seed for reproducible noise sequences.
    """

    def __init__(self, dt: float = DT, seed: Optional[int] = None) -> None:
        self._dt     = dt
        self._rng    = random.Random(seed)
        self._time   = 0.0          # elapsed simulation time (s)

        # --- State variables ---
        self._water_temp        = TEMP_SETPOINT
        self._flow_rate         = FLOW_NOMINAL
        self._motor_vibration   = VIBRATION_NOMINAL
        self._power_consumption = 0.0

        # --- Fault state ---
        self._faults = FaultState()

        # --- Last control signal (0.0 – 1.0) ---
        self._control_signal = 0.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, control_signal: float) -> dict:
        """
        Advance the simulation by one time step.

        Parameters
        ----------
        control_signal : float
            Desired cooling power fraction in [0.0, 1.0].
            0.0 = compressor off, 1.0 = full cooling capacity.

        Returns
        -------
        dict
            Current sensor readings after the step (same as
            ``get_current_state()``).
        """
        control_signal = max(0.0, min(1.0, float(control_signal)))
        self._control_signal = control_signal
        self._step(control_signal)
        return self.get_current_state()

    def get_current_state(self) -> dict:
        """
        Return a snapshot of all sensor readings.

        Returns
        -------
        dict with keys:
            time              (s)
            water_temp        (°C)
            flow_rate         (L/min)
            motor_vibration   (mm/s)
            power_consumption (kW)
            faults            (dict of active fault severities)
        """
        return {
            "time":               round(self._time, 2),
            "water_temp":         round(self._water_temp, 4),
            "flow_rate":          round(self._flow_rate, 4),
            "motor_vibration":    round(self._motor_vibration, 4),
            "power_consumption":  round(self._power_consumption, 4),
            "faults": {
                "leakage":      round(self._faults.leakage, 4),
                "clogging":     round(self._faults.clogging, 4),
                "bearing_wear": round(self._faults.bearing_wear, 4),
            },
        }

    def inject_fault(
        self,
        fault_type: str,
        severity: float = 1.0,
        ramp_rate: float = 0.002,
    ) -> None:
        """
        Initiate or intensify a fault scenario.

        The fault severity ramps up gradually each ``update()`` call so
        that the AI agent must detect a progressive anomaly rather than
        an instantaneous step change.

        Parameters
        ----------
        fault_type : str
            One of ``'leakage'``, ``'clogging'``, or ``'bearing_wear'``.
        severity : float
            Target fault severity in [0.0, 1.0].  1.0 = fully developed
            fault.
        ramp_rate : float
            Per-step increment used to ramp toward ``severity``.
            Default 0.002 means ~500 steps to reach full severity.

        Raises
        ------
        ValueError
            If ``fault_type`` is not a recognised fault name.
        """
        valid = {"leakage", "clogging", "bearing_wear"}
        if fault_type not in valid:
            raise ValueError(
                f"Unknown fault type '{fault_type}'. Choose from {valid}."
            )
        severity  = max(0.0, min(1.0, float(severity)))
        ramp_rate = max(0.0, float(ramp_rate))

        setattr(
            self._faults,
            fault_type,
            # Store target severity; actual value is ramped in _step()
            # We use a small sentinel by storing as a negative to signal
            # "ramp toward |value|" — instead, we store the target and
            # ramp inside _apply_fault_ramp().
            getattr(self._faults, fault_type),  # no-op placeholder
        )
        # Store ramp target on the fault state directly
        self._faults.__dict__.setdefault("_targets", {})[fault_type] = (
            severity,
            ramp_rate,
        )

    def reset(self) -> None:
        """Reset simulator to the clean nominal state."""
        self._time              = 0.0
        self._water_temp        = TEMP_SETPOINT
        self._flow_rate         = FLOW_NOMINAL
        self._motor_vibration   = VIBRATION_NOMINAL
        self._power_consumption = 0.0
        self._control_signal    = 0.0
        self._faults            = FaultState()

    # ------------------------------------------------------------------
    # Internal simulation logic
    # ------------------------------------------------------------------

    def _step(self, control_signal: float) -> None:
        """Compute the next simulator state for one time step."""
        self._time += self._dt

        # 1. Ramp fault severities toward their targets
        self._apply_fault_ramp()

        # 2. Compute heat loads and cooling power
        q_load    = self._compute_heat_load()
        q_cooling = self._compute_cooling_power(control_signal)

        # 3. Thermal dynamics — first-order lag
        delta_t = (q_load - q_cooling) * (self._dt / THERMAL_CAPACITY)
        self._water_temp += delta_t

        # 4. Flow rate dynamics (affected by clogging)
        self._flow_rate = self._compute_flow_rate()

        # 5. Motor vibration dynamics (affected by bearing wear)
        self._motor_vibration = self._compute_vibration()

        # 6. Power consumption (reflects actual cooling effort)
        self._power_consumption = self._compute_power(control_signal)

        # 7. Add sensor noise to all readings
        self._apply_sensor_noise()

    def _compute_heat_load(self) -> float:
        """
        Simulate a time-varying heat load from the production tool.

        Returns a value in kW with slow sinusoidal fluctuation plus
        a small stochastic component to mimic real fab conditions.
        """
        # Slow 0.01 Hz oscillation (one cycle ≈ 100 s) ±1 kW around 5 kW
        base_load  = 5.0
        oscillation = 1.0 * math.sin(2.0 * math.pi * 0.01 * self._time)
        noise       = self._rng.gauss(0.0, 0.1)
        return base_load + oscillation + noise

    def _compute_cooling_power(self, control_signal: float) -> float:
        """
        Translate the control signal into actual delivered cooling power.

        Leakage fault degrades cooling efficiency so that even at full
        control signal, less heat is removed.

        Returns a value in kW.
        """
        # Leakage reduces effective cooling efficiency (up to 80% degradation)
        efficiency = 1.0 - 0.8 * self._faults.leakage
        return control_signal * MAX_COOLING_POWER * efficiency

    def _compute_flow_rate(self) -> float:
        """
        Compute coolant flow rate in L/min.

        Clogging progressively restricts flow (up to 60% reduction).
        """
        restriction = 1.0 - 0.6 * self._faults.clogging
        return FLOW_NOMINAL * restriction

    def _compute_vibration(self) -> float:
        """
        Compute pump motor vibration in mm/s.

        Bearing wear introduces escalating vibration spikes.  Healthy
        bearings produce low-level broadband vibration; worn bearings
        develop periodic impulse spikes at the bearing defect frequency.
        """
        # Baseline healthy vibration
        base = VIBRATION_NOMINAL

        # Bearing wear adds periodic spikes and a rising floor
        if self._faults.bearing_wear > 0.0:
            wear = self._faults.bearing_wear
            # Rising noise floor due to wear (up to +2 mm/s at full wear)
            floor_rise = 2.0 * wear
            # Impulse spikes at ~2 Hz defect frequency (spike amplitude ∝ wear²)
            spike = (wear ** 2) * 3.0 * max(
                0.0,
                math.sin(2.0 * math.pi * 2.0 * self._time) ** 8,
            )
            return base + floor_rise + spike

        return base

    def _compute_power(self, control_signal: float) -> float:
        """
        Estimate total electrical power consumption in kW.

        Clogging raises back-pressure, forcing the pump to draw more
        power even as actual cooling flow drops.  Leakage causes the
        compressor to run harder (higher control signal) for the same
        net cooling effect.
        """
        # Compressor power — proportional to control signal
        compressor_power = control_signal * MAX_COOLING_POWER * 0.35

        # Pump power — rises with clogging backpressure (up to 2× nominal)
        pump_power_base  = 0.5   # kW at nominal conditions
        backpressure_factor = 1.0 + 1.5 * self._faults.clogging
        pump_power = pump_power_base * backpressure_factor

        return compressor_power + pump_power

    def _apply_sensor_noise(self) -> None:
        """Add independent Gaussian noise to each sensor reading."""
        self._water_temp      += self._rng.gauss(0.0, NOISE_TEMP)
        self._flow_rate       += self._rng.gauss(0.0, NOISE_FLOW)
        self._motor_vibration += self._rng.gauss(0.0, NOISE_VIBRATION)
        self._motor_vibration  = max(0.0, self._motor_vibration)
        self._power_consumption += self._rng.gauss(0.0, NOISE_POWER)
        self._power_consumption  = max(0.0, self._power_consumption)

    def _apply_fault_ramp(self) -> None:
        """
        Incrementally ramp each active fault severity toward its target.

        Targets are set by ``inject_fault()`` and stored in
        ``self._faults._targets``.
        """
        targets: dict = getattr(self._faults, "_targets", {})
        for fault_name, (target, rate) in targets.items():
            current = getattr(self._faults, fault_name)
            if current < target:
                new_val = min(current + rate, target)
            elif current > target:
                new_val = max(current - rate, target)
            else:
                continue
            setattr(self._faults, fault_name, new_val)


# ---------------------------------------------------------------------------
# Quick smoke-test (run with: python simulator.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sim = ChillerSimulator(seed=42)

    print("=== Normal operation (20 steps) ===")
    for i in range(20):
        state = sim.update(control_signal=0.4)
        print(
            f"t={state['time']:5.1f}s  "
            f"T={state['water_temp']:6.3f}°C  "
            f"Q={state['flow_rate']:5.2f} L/min  "
            f"V={state['motor_vibration']:.3f} mm/s  "
            f"P={state['power_consumption']:.3f} kW"
        )

    print("\n=== Injecting bearing_wear fault ===")
    sim.inject_fault("bearing_wear", severity=0.8, ramp_rate=0.05)
    for i in range(20):
        state = sim.update(control_signal=0.4)
        print(
            f"t={state['time']:5.1f}s  "
            f"V={state['motor_vibration']:.3f} mm/s  "
            f"wear={state['faults']['bearing_wear']:.3f}"
        )
