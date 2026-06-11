"""Generic PID with integrator clamp, output clamp, deadband.

Anti-windup via back-calculation: when the output saturates, the integrator
is rewound by the over-saturation so it doesn't keep growing.

`derivative` is d(error)/dt — caller computes it (no sign assumption about
error convention inside the class). Passing it externally enables
D-on-measurement (avoids derivative kick on setpoint changes); omit it and
the class differences the error internally.

Example: with `error = measurement - target` convention,
  d(error)/dt = +d(measurement)/dt, so pass `derivative=v_measurement`.
With the textbook `error = target - measurement` convention,
  d(error)/dt = -d(measurement)/dt, so pass `derivative=-v_measurement`.
"""


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


class PID:
    def __init__(self, kp, ki, kd, *, i_clamp=None, output_clamp=None,
                 deadband=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.i_clamp = i_clamp
        self.output_clamp = output_clamp
        self.deadband = deadband
        self.integral = 0.0
        self.prev_error = None

    def reset(self):
        self.integral = 0.0
        self.prev_error = None

    def update(self, error, dt, *, derivative=None):
        """Return PID output. `derivative` is d(error)/dt (see module docstring)."""
        if abs(error) < self.deadband:
            error = 0.0

        p = self.kp * error

        if dt > 0.0:
            self.integral += error * dt
            if self.i_clamp is not None:
                self.integral = _clamp(self.integral,
                                       -self.i_clamp, self.i_clamp)
        i = self.ki * self.integral

        if derivative is not None:
            d_input = derivative
        elif self.prev_error is not None and dt > 1e-9:
            d_input = (error - self.prev_error) / dt
        else:
            d_input = 0.0
        self.prev_error = error
        d = self.kd * d_input

        out = p + i + d
        if self.output_clamp is not None:
            sat = _clamp(out, -self.output_clamp, self.output_clamp)
            if sat != out and abs(self.ki) > 1e-9:
                # Back-calculate so the integrator doesn't keep growing
                # against the saturation.
                self.integral -= (out - sat) / self.ki
                if self.i_clamp is not None:
                    self.integral = _clamp(self.integral,
                                           -self.i_clamp, self.i_clamp)
            out = sat
        return out
