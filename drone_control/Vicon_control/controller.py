"""ViconHoverController — closed-loop hover on Vicon world-frame pose.

  Position (world→body) → desired pitch/roll angle (deg)   ─┐
  Altitude (PI velocity loop; integral learns hover throttle)├─► CRSF us → FC (Angle mode)
  Yaw      (heading PID + rate FF, drift-free Vicon yaw)    ─┘

The FC is in Angle mode, so it interprets roll/pitch us as TARGET ANGLES; we
convert our desired angles to us via config.STICK_US_PER_DEG and the FC's attitude
PID closes the inner loop.

ACRO option (config.ACRO_MODE, default OFF — see config.py): the FC is put in ACRO
and the roll/pitch us become RATE commands. The FC's 8 kHz inner RATE PID still runs
untouched; we close the ATTITUDE loop ourselves with a small angle→rate P stage
(_angle_to_rate_us) fed by Vicon body roll/pitch. Everything else (position, altitude,
yaw, throttle) is identical — yaw is a rate command in both modes. ANGLE stays the
default + fallback; with ACRO_MODE off this file behaves exactly as before.

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
        # Heading PID (us out) — the yaw-rate FF is added outside it in _yaw_us.
        # The class deadband zeroes the error fed to P+I (no twitch at rest) while
        # the D term (externally supplied derivative) still damps.
        ky = config.KI_YAW_US_PER_RAD_S
        ky_clamp = (config.MAX_YAW_I_US / ky) if ky > 1e-9 else None
        self.pid_yaw = PID(kp=config.KP_YAW_US_PER_RAD, ki=ky,
                           kd=config.KD_YAW_US_PER_RAD_PER_S, i_clamp=ky_clamp,
                           deadband=config.YAW_DEADBAND_RAD)
        # hover_us IS the altitude integrator state, carried as absolute throttle.
        # Seeds near true hover (HOVER_START_US) so takeoff is quick.
        self.hover_us = float(config.HOVER_START_US)
        self.x_tgt = 0.0
        self.y_tgt = 0.0
        self.z_tgt = 0.0
        self.yaw_tgt = 0.0
        # Setpoint velocity (world frame) — the carrot's own motion, for the D-term
        # velocity feedforward. Zero for a static hover.
        self.vx_tgt = 0.0
        self.vy_tgt = 0.0
        # Setpoint acceleration (world frame) + the LPF'd body-frame FF tilt state.
        self.ax_tgt = 0.0
        self.ay_tgt = 0.0
        self.aff_fwd_f = 0.0
        self.aff_lat_f = 0.0
        # Previous commanded heading, for the yaw-rate feedforward (None = no rate
        # yet — first tick after a reset contributes zero FF).
        self.prev_yaw_tgt = None
        # Measured-yaw-rate state for the heading D term: previous Vicon yaw and
        # the low-pass-filtered rate (differencing 50 Hz yaw is noisy raw).
        self.prev_yaw_meas = None
        self.yaw_rate_f = 0.0

    def set_target(self, x, y, z, yaw):
        """Capture the hover setpoint (call once at takeoff) and reset state."""
        self.x_tgt, self.y_tgt, self.z_tgt, self.yaw_tgt = x, y, z, yaw
        self.vx_tgt = self.vy_tgt = 0.0
        self.ax_tgt = self.ay_tgt = 0.0
        self.reset()

    def set_setpoint(self, x, y, z, yaw, vx=0.0, vy=0.0, ax=0.0, ay=0.0):
        """Update the target WITHOUT resetting state — for following a moving
        reference (waypoints). Unlike set_target this preserves hover_us (the
        learned hover throttle) and the position integrators, so the altitude
        trim and steady-state bias carry across legs. The reference moves smoothly
        (a crawling carrot), so there's no setpoint step to kick the loop.

        (vx, vy): the reference's own world-frame velocity (the mission knows it
        exactly — it moves the carrot). Feeds the D term the TRUE d(error)/dt =
        v_target - v_drone, so cruising with the carrot generates no braking tilt;
        without it the drone must trail by KD*v/KP (~1 m at 0.8 m/s) just to
        cancel the phantom brake. Omitting them (static target) keeps the old
        D-on-measurement behavior exactly.

        (ax, ay): the reference's world-frame ACCELERATION (ramp accel on
        straights, v²/r centripetal on arcs). Fed forward as the tilt that
        acceleration physically requires — atan(a/g) — so curves and speed
        changes cost no standing error either (flight 20260612_160149's −4 cm
        radius / 8° lag was the feedback manufacturing the inward lean)."""
        self.x_tgt, self.y_tgt, self.z_tgt, self.yaw_tgt = x, y, z, yaw
        self.vx_tgt, self.vy_tgt = vx, vy
        self.ax_tgt, self.ay_tgt = ax, ay

    def reset(self, keep_alt_trim=False):
        """Zero the position integrators. keep_alt_trim preserves the learned
        hover throttle (a slow battery estimate, still valid); a fresh target
        re-seeds it to HOVER_START_US so takeoff lifts quickly."""
        self.pid_fwd.reset()
        self.pid_lat.reset()
        self.pid_yaw.reset()
        self.aff_fwd_f = self.aff_lat_f = 0.0
        self.prev_yaw_tgt = None
        self.prev_yaw_meas = None
        self.yaw_rate_f = 0.0
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

    def _altitude_us(self, z, vz, dt, descent_rate=None, tilt_comp=1.0):
        """PI velocity loop. position error → capped target velocity (or a fixed
        descent rate when landing) → velocity error → P (transient) + I. The
        integrator (hover_us) accumulates velocity error into an absolute throttle
        clamped to the HOVER_BAND, so it discovers + tracks true hover. Clean
        drone-frame signs: alt_err>0 (below target) → climb → +vz.

        tilt_comp: 1/cos(commanded tilt) — scales the hover term so the VERTICAL
        thrust component stays constant while tilted (feedforward; the reactive
        loop alone let altitude dip −22 cm on the 2 m/s laps). Applied OUTSIDE
        the integrator: hover_us keeps learning LEVEL hover."""
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
        # Thrust ∝ throttle ABOVE idle, so compensate that span, not the absolute
        # us (scaling 1352us outright would overcorrect ~4x): +42us at 27° tilt.
        hover_comp = (channels.IDLE_THR_US
                      + (self.hover_us - channels.IDLE_THR_US) * tilt_comp)
        thr_us = _clamp(hover_comp + thr_p,
                        channels.IDLE_THR_US, config.MAX_THROTTLE_US)
        return thr_us, alt_err, v_des, e_v

    def _yaw_us(self, yaw, dt, tgt_rate):
        """Heading PID + RATE FEEDFORWARD on the commanded heading.

        FF: the mission slews its heading setpoint (never steps), so differencing
        yaw_tgt across ticks gives a clean rate; the FF converts it to us via the
        FC's LINEAR yaw curve (config.YAW_LINEAR_MAX_DPS — see the paired
        Betaflight change) so holding a moving heading costs zero standing error.

        PID around it: P on the (deadbanded) error; D on the MEASURED yaw rate
        (differenced Vicon yaw, low-passed) minus the target rate — i.e. true
        d(err)/dt with no setpoint kick — damps the transport+FC lag; small
        clamped I trims residuals (e.g. FF scale error while the heading ramps).

        Sign: e > 0 / target rotating CW(-) → yaw RIGHT (us>1500 with
        SIGN_YAW=+1), so FF enters with the OPPOSITE sign of the target rate.

        tgt_rate: rad/s (CCW+) of the commanded heading, computed in step()
        (shared with the accel-FF lead rotation)."""
        e = _wrap_pi(yaw - self.yaw_tgt)                   # >0 → drone CCW of target
        # Measured yaw rate for the D term (diff + 1-pole LPF; spike-guarded).
        if dt > 1e-6 and self.prev_yaw_meas is not None:
            raw = _clamp(_wrap_pi(yaw - self.prev_yaw_meas) / dt,
                         -2.0 * math.pi, 2.0 * math.pi)
            alpha = min(dt / config.YAW_RATE_LPF_S, 1.0)
            self.yaw_rate_f += (raw - self.yaw_rate_f) * alpha
        self.prev_yaw_meas = yaw
        # d(err)/dt = d(yaw - yaw_tgt)/dt = yaw_rate - tgt_rate (both CCW+).
        u_pid = self.pid_yaw.update(e, dt,
                                    derivative=self.yaw_rate_f - tgt_rate)
        u_ff = -config.YAW_FF_US_PER_DPS * math.degrees(tgt_rate)
        u = _clamp(u_pid + u_ff, -config.MAX_YAW_US, config.MAX_YAW_US)
        if u == 0.0:
            return channels.NEUTRAL_US, e
        return int(round(channels.NEUTRAL_US + config.SIGN_YAW * u)), e

    def _angle_to_us(self, deg, sign):
        deg = _clamp(deg, -config.MAX_TILT_DEG, config.MAX_TILT_DEG)
        return int(round(channels.NEUTRAL_US + sign * deg * config.STICK_US_PER_DEG))

    def _angle_to_rate_us(self, desired_deg, measured_deg, sign):
        """ACRO output: the outer angle→rate P loop — exactly Betaflight Angle mode's
        leveling P, run here on the laptop instead of on the FC. Both angles are in
        the controller convention (deg; pitch>0 nose-down, roll>0 right). The rate
        error is clamped to the FC's linear ACTUAL-rate ceiling and converted to us
        through that curve (ACRO_RATE_US_PER_DPS). The FC's 8 kHz inner rate PID still
        closes the fast loop — we only swap the angle setpoint for a rate setpoint.
        The restoring SIGN is the SAME SIGN_* as Angle mode (the rate-stick and
        angle-stick directions are identical), so this needs no new sign constant —
        but DRY_RUN-verify it anyway (tilt → us moves to oppose the tilt)."""
        rate_dps = _clamp(config.KP_ANGLE_RATE * (desired_deg - measured_deg),
                          -config.ACRO_MAX_RATE_DPS, config.ACRO_MAX_RATE_DPS)
        return int(round(channels.NEUTRAL_US
                         + sign * rate_dps * config.ACRO_RATE_US_PER_DPS))

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
        # Rate of the commanded heading (rad/s, CCW+): drives the yaw-rate FF and
        # the accel-FF lead rotation. The mission slews yaw_tgt (never steps), so
        # the diff is clean; spike-guarded anyway.
        tgt_yaw_rate = 0.0
        if dt > 1e-6 and self.prev_yaw_tgt is not None:
            tgt_yaw_rate = _clamp(_wrap_pi(self.yaw_tgt - self.prev_yaw_tgt) / dt,
                                  -math.pi, math.pi)
        self.prev_yaw_tgt = self.yaw_tgt
        if level_only:
            # Lift straight up: no horizontal correction, integrators untouched.
            err_fwd = err_lat = v_fwd = v_lat = tv_fwd = tv_lat = 0.0
            desired_pitch_deg = desired_roll_deg = 0.0
            self.aff_fwd_f = self.aff_lat_f = 0.0
        else:
            err_fwd, err_lat = self._body_errors(x, y, yaw)
            v_fwd, v_lat = self._body_vel(pose["vx"], pose["vy"], yaw)
            # True d(err)/dt = d(target - pos)/dt = v_target - v_drone (body frame).
            # The v_target part is the velocity FEEDFORWARD: keeping pace with a
            # moving carrot gives derivative ≈ 0 → no braking tilt → no KD*v/KP
            # standing lag. For a parked target (v_tgt = 0) this is exactly the
            # old D-on-measurement: still no kick on setpoint position steps.
            tv_fwd, tv_lat = self._body_vel(self.vx_tgt, self.vy_tgt, yaw)
            # ACCELERATION FEEDFORWARD: the tilt the reference's acceleration
            # physically requires (atan(a/g), horizontal-thrust kinematics),
            # LPF'd to swallow one-tick spikes. Same body rotation as velocity;
            # added OUTSIDE the PIDs so the integrators never have to learn it.
            ta_fwd, ta_lat = self._body_vel(self.ax_tgt, self.ay_tgt, yaw)
            alpha = min(dt / config.ACCEL_FF_LPF_S, 1.0) if dt > 0.0 else 0.0
            self.aff_fwd_f += (ta_fwd - self.aff_fwd_f) * alpha
            self.aff_lat_f += (ta_lat - self.aff_lat_f) * alpha
            # DRAG FEEDFORWARD: cruising costs ~K_DRAG·v of forward tilt. Ships
            # WITH the accel FF: without this, the loop manufactures that tilt
            # from a standing error — phase lag pre-accel-FF, or (worse) a speed
            # deficit that shrinks curves once the accel FF removes the radial
            # burden. Uses the REFERENCE velocity (already body-frame as tv_*),
            # so it's exactly zero in hover/landing.
            aff_pitch_deg = (math.degrees(math.atan2(self.aff_fwd_f, 9.81))
                             + config.K_DRAG_DEG_PER_MPS * tv_fwd)
            aff_roll_deg = (math.degrees(math.atan2(self.aff_lat_f, 9.81))
                            + config.K_DRAG_DEG_PER_MPS * tv_lat)
            desired_pitch_deg = aff_pitch_deg + self.pid_fwd.update(
                err_fwd, dt, derivative=tv_fwd - v_fwd)
            desired_roll_deg = aff_roll_deg + self.pid_lat.update(
                err_lat, dt, derivative=tv_lat - v_lat)

        # Measured body attitude (controller convention) from Vicon — drives the
        # ACRO angle→rate loop and is logged to cross-check the FC attitude telemetry.
        # Verified, heading-invariant map: vicon_source.drone_roll_pitch.
        meas_roll_deg = math.degrees(pose.get("roll", 0.0))
        meas_pitch_deg = math.degrees(pose.get("pitch", 0.0))
        if config.ACRO_MODE:
            # FC in ACRO: WE close the attitude loop (Vicon roll/pitch → rate cmd).
            # level_only feeds desired=0, so the same call actively LEVELS on takeoff.
            pitch_us = self._angle_to_rate_us(desired_pitch_deg, meas_pitch_deg,
                                              config.SIGN_PITCH)
            roll_us = self._angle_to_rate_us(desired_roll_deg, meas_roll_deg,
                                             config.SIGN_ROLL)
        else:
            # FC in Angle mode: send the desired angle as a setpoint (unchanged).
            pitch_us = self._angle_to_us(desired_pitch_deg, config.SIGN_PITCH)
            roll_us = self._angle_to_us(desired_roll_deg, config.SIGN_ROLL)

        # Tilt compensation from the COMMANDED (clamped) angles — what the FC is
        # being asked to fly right now; instant, unlike waiting for the dip.
        pc = math.radians(_clamp(desired_pitch_deg,
                                 -config.MAX_TILT_DEG, config.MAX_TILT_DEG))
        rc = math.radians(_clamp(desired_roll_deg,
                                 -config.MAX_TILT_DEG, config.MAX_TILT_DEG))
        tilt_comp = min(1.0 / max(math.cos(pc) * math.cos(rc), 1e-3),
                        config.TILT_COMP_MAX)

        thr_us, alt_err, v_des_up, e_v_up = self._altitude_us(
            z, pose["vz"], dt, descent_rate=descent_rate, tilt_comp=tilt_comp)
        yaw_us, e_yaw = self._yaw_us(yaw, dt, tgt_yaw_rate)

        return {
            "roll_us": int(roll_us),
            "pitch_us": int(pitch_us),
            "yaw_us": int(yaw_us),
            "throttle_us": int(thr_us),
            "desired_roll_deg": desired_roll_deg,
            "desired_pitch_deg": desired_pitch_deg,
            "meas_roll_deg": meas_roll_deg, "meas_pitch_deg": meas_pitch_deg,
            "err_fwd": err_fwd, "err_lat": err_lat,
            "v_fwd": v_fwd, "v_lat": v_lat,
            "tv_fwd": tv_fwd, "tv_lat": tv_lat,
            "aff_fwd": self.aff_fwd_f, "aff_lat": self.aff_lat_f,
            "alt_err": alt_err, "v_des_up": v_des_up, "e_v_up": e_v_up,
            "tilt_comp": tilt_comp,
            "hover_us": self.hover_us, "e_yaw_rad": e_yaw,
        }
