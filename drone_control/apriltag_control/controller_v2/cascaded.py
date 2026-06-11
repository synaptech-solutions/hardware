"""Cascaded hover controller (FC in Angle mode → inner attitude loop in firmware).

  Outer position PID (lat → roll deg)
  Outer position PID (fwd → pitch deg)            ─┐
                                                   ├─► CRSF channels → FC
  Altitude (PI velocity loop, learns hover throttle) │
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

from . import config            # __init__ puts drone_control on sys.path
from common.pid import PID, _clamp


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

        # Altitude PI velocity loop. hover_us IS the integrator state, carried as
        # an ABSOLUTE throttle (us): it integrates velocity error directly into a
        # hover-throttle estimate, clamped to the HOVER_BAND. No fixed hover
        # constant lives in the control law — the loop discovers hover in flight.
        # Starts each flight at the band floor (re-floored in reset()).
        self.hover_us = float(config.HOVER_BAND_LO_US)

        # Targets in arm-time body frame. set_targets() updates these on FLY entry.
        self.target_fwd_m = config.TARGET_FWD_FALLBACK_M
        self.target_lat_m = config.TARGET_LAT_M
        self.target_up_m = config.TARGET_UP_M
        self.yaw_at_arm_deg = 0.0

    def reset(self, keep_alt_trim=False):
        """Zero the integrators. keep_alt_trim=True preserves the learned hover
        throttle (hover_us) — used on a brief tag loss, where the X/Z position
        integrals are stale and must clear, but the learned hover is a slow
        battery-droop estimate that's still valid and whose loss would lurch
        throttle on recovery. set_targets() (a fresh flight) does a full reset,
        re-flooring hover_us to the band bottom so it re-learns from scratch."""
        self.pid_x.reset()
        self.pid_z.reset()
        if not keep_alt_trim:
            self.hover_us = float(config.HOVER_BAND_LO_US)

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

    def _altitude_us(self, est_up, v_up, dt):
        """PI velocity loop: position error → capped target velocity → velocity
        error → P (transient) + I. The integrator state (self.hover_us) IS the
        hover throttle, in us: it accumulates velocity error directly into an
        absolute throttle, clamped to the HOVER_BAND. There is NO hardcoded hover
        constant — the loop finds hover in flight and tracks it as the pack sags.
        At the fixed point (v_up=0, e_pos=0 → e_v=0) hover_us stops moving, parked
        at the true hover throttle. PI-on-velocity also zeroes steady-state
        position error, which the old PD could not.

        SIGN CONVENTION (foot-gun): est_up/v_up track the TAG's position in the
        body frame, NOT the drone's. The tag is the fixed reference. So the
        intuitions invert from "drone position":
          - drone CLIMBS  → tag appears LOWER  → est_up DECREASES → v_up < 0
          - drone BELOW target → tag appears HIGH → est_up > target → e_pos > 0
        These two inversions cancel: e_pos>0 (below) → v_des<0 (want to climb,
        climb is -v_up) → correct. Don't "fix" a sign here by reasoning about
        the drone's motion directly — reason about the tag's apparent motion.
        """
        # e_pos > 0 means the drone is BELOW target (tag rides high in frame).
        e_pos = est_up - self.target_up_m
        v_des = _clamp(-config.KP_UP * e_pos,
                       -config.VMAX_UP_MPS, config.VMAX_UP_MPS)
        e_v = v_up - v_des           # >0 → climbing too slow / need more thrust

        # P: transient velocity correction, asymmetric authority clamp.
        thr_p = _clamp(config.KV_UP_US_PER_MPS * e_v,
                       -config.THR_DESC_TRIM_US, config.THR_CLIMB_TRIM_US)

        # I: integrate velocity error straight into the hover-throttle estimate.
        # The HOVER_BAND clamp on hover_us IS the anti-windup limit — it's a narrow
        # physical window, so the integrator can't run away on a bad estimate, and
        # because the band ⊂ [IDLE, MAX] the final output clamp never has to fight
        # it (no back-calculation needed).
        if dt > 0.0:
            self.hover_us = _clamp(
                self.hover_us + config.KI_UP_US_PER_M * e_v * dt,
                config.HOVER_BAND_LO_US, config.HOVER_BAND_HI_US)

        thr_us = _clamp(self.hover_us + thr_p,
                        config.IDLE_THR_US, config.MAX_THROTTLE_US)
        return thr_us, e_pos, v_des, e_v, self.hover_us

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

        # 3) Altitude (PI velocity loop — learns hover throttle).
        throttle_us, e_up, v_des_up, e_v_up, hover_us = self._altitude_us(
            state["est_up"], state["v_up"], dt)

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
            "hover_us": hover_us,
            "e_yaw_rad": e_yaw_rad,
            "tgt_fwd_b": tgt_fwd_b,
            "tgt_lat_b": tgt_lat_b,
        }
