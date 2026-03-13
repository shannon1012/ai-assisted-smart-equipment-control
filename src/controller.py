"""
PID Controller for chiller temperature regulation.

Controls a chiller's cooling output to maintain a target temperature setpoint
by combining proportional, integral, and derivative feedback terms.
"""


class PIDController:
    """
    Standard PID controller for chiller temperature management.

    The control signal represents cooling power as a fraction [0.0, 1.0]:
      - 0.0 = chiller fully off
      - 1.0 = chiller at maximum cooling capacity

    Tuning guide:
      - kp: Higher values give faster response but risk oscillation.
      - ki: Eliminates steady-state error (e.g., when chiller efficiency drops).
      - kd: Dampens overshoot by reacting to how fast the error is changing.
    """

    def __init__(
        self,
        setpoint: float = 20.0,
        kp: float = 0.5,
        ki: float = 0.1,
        kd: float = 0.05,
    ) -> None:
        """
        Initialise the PID controller.

        Args:
            setpoint: Target temperature in °C (default 20.0).
            kp: Proportional gain — scales the immediate reaction to current error.
            ki: Integral gain — scales the reaction to accumulated past error.
            kd: Derivative gain — scales the reaction to the rate of error change.
        """
        self.setpoint = setpoint
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self._integral: float = 0.0
        self._prev_error: float = 0.0

    def update(self, current_temp: float, dt: float) -> float:
        """
        Compute the next control signal given the current temperature reading.

        The three PID terms work together:
          - Proportional: Provides an output proportional to the current error.
            Larger errors produce a stronger cooling response immediately.
          - Integral: Accumulates error over time and applies a correction that
            grows until the steady-state error is eliminated. This is especially
            important when chiller efficiency degrades and a persistent offset
            would otherwise remain uncorrected.
          - Derivative: Measures how quickly the error is changing and applies a
            braking force to prevent the temperature from overshooting the
            setpoint. Effectively predicts near-future error and acts early.

        Anti-windup: If the raw output would exceed the [0.0, 1.0] range the
        integral accumulator is not updated for that timestep, preventing it
        from winding up while the output is saturated.

        Args:
            current_temp: Current measured temperature in °C.
            dt: Time elapsed since the last call in seconds. Must be > 0.

        Returns:
            control_signal: Cooling power demand in the range [0.0, 1.0].
        """
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")

        error = current_temp - self.setpoint  # positive → too hot → need more cooling

        # --- Proportional term ---
        p_term = self.kp * error

        # --- Derivative term (based on error change, not measurement) ---
        d_term = self.kd * (error - self._prev_error) / dt

        # Compute raw output without integral to check for saturation
        raw_output_no_integral = p_term + self._integral + d_term

        # --- Anti-windup: only accumulate integral when output is not saturated ---
        unsaturated = 0.0 <= raw_output_no_integral <= 1.0
        if unsaturated:
            self._integral += self.ki * error * dt

        # --- Full control signal ---
        control_signal = p_term + self._integral + d_term

        # Clamp to valid actuator range
        control_signal = max(0.0, min(1.0, control_signal))

        self._prev_error = error
        return control_signal

    def reset(self) -> None:
        """Reset accumulated state (integral and previous error)."""
        self._integral = 0.0
        self._prev_error = 0.0


if __name__ == "__main__":
    # ------------------------------------------------------------------ #
    #  Quick smoke tests — run with:  python src/controller.py            #
    # ------------------------------------------------------------------ #
    import math

    PASS = "PASS"
    FAIL = "FAIL"

    def check(name: str, condition: bool) -> None:
        print(f"  [{PASS if condition else FAIL}] {name}")

    print("\n=== PIDController tests ===\n")

    # --- 1. Output clamped to [0, 1] when far above setpoint ---
    pid = PIDController(setpoint=20.0)
    sig = pid.update(current_temp=40.0, dt=1.0)
    check("output clamped to 1.0 when very hot", sig == 1.0)

    # --- 2. Output clamped to 0 when far below setpoint ---
    pid.reset()
    sig = pid.update(current_temp=0.0, dt=1.0)
    check("output clamped to 0.0 when very cold", sig == 0.0)

    # --- 3. Zero error → small/zero output (only D term since integral=0) ---
    pid.reset()
    sig = pid.update(current_temp=20.0, dt=1.0)
    check("zero error produces zero output", sig == 0.0)

    # --- 4. Integral grows over repeated above-setpoint readings ---
    pid.reset()
    for _ in range(10):
        pid.update(current_temp=21.0, dt=1.0)
    check("integral accumulates with sustained error", pid._integral > 0.0)

    # --- 5. Anti-windup: integral should NOT grow while output is saturated ---
    pid = PIDController(setpoint=20.0, kp=2.0, ki=1.0, kd=0.0)
    pid.reset()
    for _ in range(20):
        pid.update(current_temp=40.0, dt=1.0)
    integral_after_saturation = pid._integral
    pid.update(current_temp=40.0, dt=1.0)
    check(
        "anti-windup: integral stops growing at saturation",
        math.isclose(pid._integral, integral_after_saturation, rel_tol=1e-6),
    )

    # --- 6. reset() clears state ---
    pid.reset()
    check("reset clears integral", pid._integral == 0.0)
    check("reset clears prev_error", pid._prev_error == 0.0)

    # --- 7. Invalid dt raises ValueError ---
    pid = PIDController()
    try:
        pid.update(current_temp=20.0, dt=0.0)
        check("dt=0 raises ValueError", False)
    except ValueError:
        check("dt=0 raises ValueError", True)

    # --- 8. Convergence: simulate simple first-order system toward setpoint ---
    # Plant model: temp changes by -(control_signal * 2) + 0.5 per second
    pid = PIDController(setpoint=20.0, kp=0.5, ki=0.1, kd=0.05)
    temp = 25.0
    dt = 1.0
    for _ in range(200):
        sig = pid.update(current_temp=temp, dt=dt)
        temp += (-sig * 2.0 + 0.5) * dt
    check(f"converges to setpoint (final temp={temp:.2f}°C)", abs(temp - 20.0) < 0.5)

    print()
