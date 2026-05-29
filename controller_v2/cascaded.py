"""Cascaded hover controller (FC in Angle mode → inner attitude loop in firmware).

  Outer position PID (lat → roll deg)
  Outer position PID (fwd → pitch deg)            ─┐
                                                   ├─► CRSF channels → FC
  Altitude (velocity-loop PD, asymmetric throttle)  │
  Yaw     (P on heading error w/ deadband)        ─┘

Position errors are computed in the CURRENT body frame. Targets are stored
in the arm-time body frame and rotated into the current body frame by
(yaw_now - yaw_at_arm), so a small yaw drift doesn't get mistaken for an
XY translation. With yaw held at arm-time (the goal), this rotation is
near-identity — it's there to keep the controller well-behaved during
transient yaw error.

The FC's Angle mode interprets CRSF roll/pitch us as target angles; we
convert our desired-angle (deg) outputs to us via config.STICK_US_PER_DEG.
"""
import math

from . import config
from .pid import PID, _clamp


def _wrap_pi(rad):
    while rad > math.pi:
        rad -= 2.0 * math.pi
    while rad < -math.pi:
        rad += 2.0 * math.pi
    return rad


def deg_to_us(deg):
    """Body-frame angle (deg, signed) → CRSF stick offset (us, signed)."""
    return deg * config.STICK_US_PER_DEG


class CascadedHoverController:
    """Four control loops in the layout described in the module docstring."""

    def __init__(self):
        ki_x = config.KI_X_DEG_PER_M_S
        ki_z = config.KI_Z_DEG_PER_M_S
        # No output_clamp here — the single output limit lives downstream in us
        # (MAX_ROLL_US / MAX_PITCH_US). i_clamp still bounds the integrator state
        # independently so |ki·integral| ≤ MAX_*_INT_DEG (anti-windup). Note: with
        # KI>0, output saturation now happens at the us clamp, which the PID's
        # back-calc can't see — i_clamp is the windup guard in that case.
        self.pid_x = PID(
            kp=config.KP_X_DEG_PER_M,
            ki=ki_x,
            kd=config.KD_X_DEG_PER_MPS,
            i_clamp=(config.MAX_X_INT_DEG / ki_x) if ki_x > 1e-9 else None,
        )
        self.pid_z = PID(
            kp=config.KP_Z_DEG_PER_M,
            ki=ki_z,
            kd=config.KD_Z_DEG_PER_MPS,
            i_clamp=(config.MAX_Z_INT_DEG / ki_z) if ki_z > 1e-9 else None,
        )

        # Targets in arm-time body frame. set_targets() updates these on FLY entry.
        self.target_fwd_m = config.TARGET_FWD_FALLBACK_M
        self.target_lat_m = config.TARGET_LAT_M
        self.target_up_m = config.TARGET_UP_M
        self.yaw_at_arm_deg = 0.0

    def reset(self):
        self.pid_x.reset()
        self.pid_z.reset()

    def set_targets(self, target_fwd_m, target_lat_m, target_up_m,
                    yaw_at_arm_deg):
        self.target_fwd_m = target_fwd_m
        self.target_lat_m = target_lat_m
        self.target_up_m = target_up_m
        self.yaw_at_arm_deg = yaw_at_arm_deg
        self.reset()

    def _rotate_targets_to_body(self, yaw_deg):
        """Rotate (target_fwd, target_lat) from arm-time frame into the current
        body frame. Identity when yaw hasn't drifted; corrects the transient
        when yaw is briefly off-target."""
        dyaw = _wrap_pi(math.radians(yaw_deg - self.yaw_at_arm_deg))
        c, s = math.cos(dyaw), math.sin(dyaw)
        tgt_fwd_b = c * self.target_fwd_m + s * self.target_lat_m
        tgt_lat_b = -s * self.target_fwd_m + c * self.target_lat_m
        return tgt_fwd_b, tgt_lat_b

    def _altitude_us(self, est_up, v_up):
        """Velocity-loop PD with asymmetric throttle clamp — same control law
        and gains as the existing tag_hover_controller.

        Tag-frame up: drone climbing → tag's up DECREASES → v_up < 0.
        """
        e_pos = est_up - self.target_up_m
        v_des = _clamp(-config.KP_UP * e_pos,
                       -config.VMAX_UP_MPS, config.VMAX_UP_MPS)
        e_v = v_up - v_des           # >0 → climbing too slow / need more thrust
        thr_corr = config.KV_UP_US_PER_MPS * e_v
        thr_corr = _clamp(thr_corr,
                          -config.THR_DESC_TRIM_US, config.THR_CLIMB_TRIM_US)
        thr_us = _clamp(config.HOVER_THROTTLE_US + thr_corr,
                        config.IDLE_THR_US, config.MAX_THROTTLE_US)
        return thr_us, e_pos, v_des, e_v

    def _yaw_us(self, est_fwd, est_lat):
        """P-on-tag-bearing with deadband — drift-free yaw hold.

        The Air75 has no magnetometer; its CRSF yaw is free-running gyro
        heading that drifts without bound, so holding against a captured
        yaw_at_arm always winds off and pins the stick. Instead we yaw to
        keep the TAG centered: bearing = atan2(lat, fwd) is the tag's
        horizontal angle off the nose, an absolute reference that never
        drifts.

        bearing > 0 (tag to the RIGHT) → yaw RIGHT to recenter → us > 1500.
        Guard the degenerate fwd≈0 case (tag directly overhead/behind).
        """
        if est_fwd < 0.05:
            return config.NEUTRAL_US, 0.0
        e_rad = math.atan2(est_lat, est_fwd)
        if abs(e_rad) < config.YAW_DEADBAND_RAD:
            return config.NEUTRAL_US, e_rad
        u = _clamp(config.KP_YAW_US_PER_RAD * e_rad,
                   -config.MAX_YAW_US, config.MAX_YAW_US)
        return round(config.NEUTRAL_US + config.SIGN_YAW * u), e_rad

    def step(self, state, dt):
        """One control step.

        state: dict — est_fwd, est_lat, est_up, v_fwd, v_lat, v_up, yaw_deg
        dt:    loop period (s)

        Returns a dict of channel values and the intermediate quantities the
        main loop needs for logging.
        """
        # 1) Outer position loops — body-frame errors and PID → desired angles.
        tgt_fwd_b, tgt_lat_b = self._rotate_targets_to_body(state["yaw_deg"])
        e_fwd_b = state["est_fwd"] - tgt_fwd_b
        e_lat_b = state["est_lat"] - tgt_lat_b
        v_fwd_b = state["v_fwd"]
        v_lat_b = state["v_lat"]

        desired_roll_deg = self.pid_x.update(
            e_lat_b, dt, derivative=v_lat_b)
        desired_pitch_deg = self.pid_z.update(
            e_fwd_b, dt, derivative=v_fwd_b)

        # 2) Deg → us with stick sign, then final us clamp (defense in depth).
        roll_us_offset = _clamp(desired_roll_deg * config.STICK_US_PER_DEG,
                                -config.MAX_ROLL_US, config.MAX_ROLL_US)
        pitch_us_offset = _clamp(desired_pitch_deg * config.STICK_US_PER_DEG,
                                 -config.MAX_PITCH_US, config.MAX_PITCH_US)
        roll_us = round(config.NEUTRAL_US + config.SIGN_ROLL * roll_us_offset)
        pitch_us = round(config.NEUTRAL_US + config.SIGN_PITCH * pitch_us_offset)

        # 3) Altitude (velocity-loop PD).
        throttle_us, e_up, v_des_up, e_v_up = self._altitude_us(
            state["est_up"], state["v_up"])

        # 4) Yaw (P on tag bearing — drift-free, no compass needed).
        yaw_us, e_yaw_rad = self._yaw_us(state["est_fwd"], state["est_lat"])

        return {
            "roll_us": int(roll_us),
            "pitch_us": int(pitch_us),
            "yaw_us": int(yaw_us),
            "throttle_us": int(throttle_us),
            "desired_roll_deg": desired_roll_deg,
            "desired_pitch_deg": desired_pitch_deg,
            "e_fwd_b": e_fwd_b,
            "e_lat_b": e_lat_b,
            "e_up": e_up,
            "v_des_up": v_des_up,
            "e_v_up": e_v_up,
            "e_yaw_rad": e_yaw_rad,
            "tgt_fwd_b": tgt_fwd_b,
            "tgt_lat_b": tgt_lat_b,
        }
