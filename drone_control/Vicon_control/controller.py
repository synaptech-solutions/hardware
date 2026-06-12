"""ViconHoverController — closed-loop hover on Vicon world-frame pose.

  Position (world→body) → desired pitch/roll angle (deg)   ─┐
  Altitude (PI velocity loop; integral learns hover throttle)├─► CRSF us → FC (Angle mode)
  Yaw      (P on absolute heading error, drift-free Vicon)  ─┘

The FC is in Angle mode, so it interprets roll/pitch us as TARGET ANGLES; we
convert our desired angles to us via config.STICK_US_PER_DEG and the FC's attitude
PID closes the inner loop.

Reuses common.pid.PID. The altitude loop is the same self-learning-hover design
proven in apriltag_control (the integrator IS the hover throttle), but with clean
DRONE-frame altitude signs — z is the drone's own altitude (no tag inversion).

SIGN CONVENTIONS (verify in DRY_RUN with props off — this is the gate that catches
the kind of inversion that bit the apriltag controller):
  - World frame is Z-up; yaw is CCW-positive about +Z (from the Vicon quaternion).
  - Body forward is +X at yaw=0. err_* below is (target - drone) in the body frame.
  - err_fwd > 0 → target ahead → pitch forward (nose down, us>1500) → fly forward.
  - err_lat > 0 → target to the right → roll right (us>1500).
  - yaw err > 0 (drone CCW of target) → yaw right (us>1500) to recenter.
If a hand-displacement test shows a correction pushing the WRONG way, flip the
matching config.SIGN_* (ROLL for lateral, PITCH for forward, YAW for heading).
"""
import math

from . import config
from common import channels
from common.pid import PID, _clamp


def _wrap_pi(rad):
    while rad > math.pi:
        rad -= 2.0 * math.pi
    while rad < -math.pi:
        rad += 2.0 * math.pi
    return rad


class ViconHoverController:
    def __init__(self):
        # Independent per-axis gains (fwd = pitch/world-Y, lat = roll/world-X): the
        # two axes saw different disturbances in flight, so they're tuned apart.
        kf = config.KI_FWD_DEG_PER_M_S
        kf_clamp = (config.MAX_FWD_INT_DEG / kf) if kf > 1e-9 else None
        kl = config.KI_LAT_DEG_PER_M_S
        kl_clamp = (config.MAX_LAT_INT_DEG / kl) if kl > 1e-9 else None
        # D-on-measurement: caller passes derivative = d(error)/dt, so no output
        # clamp inside the PID — the tilt clamp is applied downstream.
        self.pid_fwd = PID(kp=config.KP_FWD_DEG_PER_M, ki=kf,
                           kd=config.KD_FWD_DEG_PER_MPS, i_clamp=kf_clamp)
        self.pid_lat = PID(kp=config.KP_LAT_DEG_PER_M, ki=kl,
                           kd=config.KD_LAT_DEG_PER_MPS, i_clamp=kl_clamp)
        # hover_us IS the altitude integrator state, carried as absolute throttle.
        # Seeds near true hover (HOVER_START_US) so takeoff is quick.
        self.hover_us = float(config.HOVER_START_US)
        self.x_tgt = 0.0
        self.y_tgt = 0.0
        self.z_tgt = 0.0
        self.yaw_tgt = 0.0

    def set_target(self, x, y, z, yaw):
        """Capture the hover setpoint (call once at takeoff) and reset state."""
        self.x_tgt, self.y_tgt, self.z_tgt, self.yaw_tgt = x, y, z, yaw
        self.reset()

    def set_setpoint(self, x, y, z, yaw):
        """Update the target WITHOUT resetting state — for following a moving
        reference (waypoints). Unlike set_target this preserves hover_us (the
        learned hover throttle) and the position integrators, so the altitude
        trim and steady-state bias carry across legs. The reference moves smoothly
        (a crawling carrot), so there's no setpoint step to kick the loop."""
        self.x_tgt, self.y_tgt, self.z_tgt, self.yaw_tgt = x, y, z, yaw

    def reset(self, keep_alt_trim=False):
        """Zero the position integrators. keep_alt_trim preserves the learned
        hover throttle (a slow battery estimate, still valid); a fresh target
        re-seeds it to HOVER_START_US so takeoff lifts quickly."""
        self.pid_fwd.reset()
        self.pid_lat.reset()
        if not keep_alt_trim:
            self.hover_us = float(config.HOVER_START_US)

    # --- frame transforms (world → body), using the drone's current yaw ---
    def _body_errors(self, x, y, yaw):
        """(target - drone) projected on the body forward / right axes."""
        ex, ey = self.x_tgt - x, self.y_tgt - y          # world error (target-drone)
        c, s = math.cos(yaw), math.sin(yaw)
        err_fwd = c * ex + s * ey                         # along body forward (+X@yaw0)
        err_lat = s * ex - c * ey                         # along body right
        return err_fwd, err_lat

    def _body_vel(self, vx, vy, yaw):
        """World velocity → body forward / right velocity."""
        c, s = math.cos(yaw), math.sin(yaw)
        v_fwd = c * vx + s * vy
        v_lat = s * vx - c * vy
        return v_fwd, v_lat

    def _altitude_us(self, z, vz, dt, descent_rate=None):
        """PI velocity loop. position error → capped target velocity (or a fixed
        descent rate when landing) → velocity error → P (transient) + I. The
        integrator (hover_us) accumulates velocity error into an absolute throttle
        clamped to the HOVER_BAND, so it discovers + tracks true hover. Clean
        drone-frame signs: alt_err>0 (below target) → climb → +vz."""
        alt_err = None
        if descent_rate is None:
            alt_err = self.z_tgt - z                      # >0 → below target
            v_des = _clamp(config.KP_UP * alt_err,
                           -config.VMAX_UP_MPS, config.VMAX_UP_MPS)
        else:
            v_des = -abs(descent_rate)                     # commanded descent
        e_v = v_des - vz                                   # >0 → need more lift
        thr_p = _clamp(config.KV_UP_US_PER_MPS * e_v,
                       -config.THR_DESC_TRIM_US, config.THR_CLIMB_TRIM_US)
        if dt > 0.0:
            self.hover_us = _clamp(
                self.hover_us + config.KI_UP_US_PER_M * e_v * dt,
                config.HOVER_BAND_LO_US, config.HOVER_BAND_HI_US)
        thr_us = _clamp(self.hover_us + thr_p,
                        channels.IDLE_THR_US, config.MAX_THROTTLE_US)
        return thr_us, alt_err, v_des, e_v

    def _yaw_us(self, yaw):
        """P on absolute heading error with deadband. Vicon yaw is drift-free, so
        we hold the captured heading directly."""
        e = _wrap_pi(yaw - self.yaw_tgt)                   # >0 → drone CCW of target
        if abs(e) < config.YAW_DEADBAND_RAD:
            return channels.NEUTRAL_US, e
        u = _clamp(config.KP_YAW_US_PER_RAD * e,
                   -config.MAX_YAW_US, config.MAX_YAW_US)
        return int(round(channels.NEUTRAL_US + config.SIGN_YAW * u)), e

    def _angle_to_us(self, deg, sign):
        deg = _clamp(deg, -config.MAX_TILT_DEG, config.MAX_TILT_DEG)
        return int(round(channels.NEUTRAL_US + sign * deg * config.STICK_US_PER_DEG))

    def step(self, pose, dt, descent_rate=None, level_only=False):
        """One control step.

        pose: dict from ViconPoseSource.get_pose() (x,y,z,yaw,vx,vy,vz).
        descent_rate: if set (m/s), command that descent instead of holding z
                      (used for landing). Horizontal + yaw hold continue.
        level_only: takeoff/ground phase — command LEVEL (no roll/pitch tilt) and
                    FREEZE the horizontal integrators (don't accumulate), so the
                    drone lifts straight up instead of scooting. Altitude + yaw
                    still run. Set this until the drone is airborne.

        Returns channel us + the intermediate quantities for logging.
        """
        x, y, z, yaw = pose["x"], pose["y"], pose["z"], pose["yaw"]
        if level_only:
            # Lift straight up: no horizontal correction, integrators untouched.
            err_fwd = err_lat = v_fwd = v_lat = 0.0
            desired_pitch_deg = desired_roll_deg = 0.0
        else:
            err_fwd, err_lat = self._body_errors(x, y, yaw)
            v_fwd, v_lat = self._body_vel(pose["vx"], pose["vy"], yaw)
            # d(err)/dt = d(target - pos)/dt = -v_body  (D-on-measurement, no kick).
            desired_pitch_deg = self.pid_fwd.update(err_fwd, dt, derivative=-v_fwd)
            desired_roll_deg = self.pid_lat.update(err_lat, dt, derivative=-v_lat)

        pitch_us = self._angle_to_us(desired_pitch_deg, config.SIGN_PITCH)
        roll_us = self._angle_to_us(desired_roll_deg, config.SIGN_ROLL)

        thr_us, alt_err, v_des_up, e_v_up = self._altitude_us(
            z, pose["vz"], dt, descent_rate=descent_rate)
        yaw_us, e_yaw = self._yaw_us(yaw)

        return {
            "roll_us": int(roll_us),
            "pitch_us": int(pitch_us),
            "yaw_us": int(yaw_us),
            "throttle_us": int(thr_us),
            "desired_roll_deg": desired_roll_deg,
            "desired_pitch_deg": desired_pitch_deg,
            "err_fwd": err_fwd, "err_lat": err_lat,
            "v_fwd": v_fwd, "v_lat": v_lat,
            "alt_err": alt_err, "v_des_up": v_des_up, "e_v_up": e_v_up,
            "hover_us": self.hover_us, "e_yaw_rad": e_yaw,
        }
