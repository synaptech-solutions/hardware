"""Tunable parameters for the cascaded hover controller.

Architecture (FC in Angle mode → inner attitude loop lives in firmware):
  PID_X  : lat-position error (m) → desired roll angle (deg)
  PID_Z  : fwd-position error (m) → desired pitch angle (deg)
  Altitude (velocity-loop PD, asymmetric throttle trim) → throttle (us)
  Yaw    : P-on-heading error with deadband → yaw (us)

Desired angles are converted to CRSF us via the FC's Angle-mode stick scaling
(STICK_US_PER_DEG below). The FC's attitude PID then drives the drone to
that angle.

Edit values here. Re-run in DRY_RUN first to verify directions and magnitudes.
"""
import math


# ============ run mode ============
DRY_RUN = False                # True = never arm; print sticks only

# ============ target / reference ============
TARGET_TAG_ID = 0
TAG_SIZE_M = 0.078
TARGET_LAT_M = 0.0
TARGET_UP_M = 0.0              # 0 = level with tag (tag height)
TARGET_FWD_FALLBACK_M = 0.50  # only used if no vision lock at arm (shouldn't happen)
# FLIGHT: hold whatever fwd/lat the drone reads at launch (launch where it sits,
# don't fly to a fixed setpoint). Both True → target = vision pose at arm.
LOCK_LAT_AT_ARM = True
LOCK_FWD_AT_ARM = True

# ============ vision input ============
# WIDTH/HEIGHT must match camera_setup/camera_calibration.npz (K is in pixels).
DEVICE_INDEX = 4
WIDTH, HEIGHT = 1280, 720
TAG_FAMILY = "tag25h9"
MIN_TAG_DECISION_MARGIN = 18.0
MAX_TRACK_RANGE_M = 2.5
MAX_TRACK_LAT_M = 1.5
MAX_TRACK_UP_M = 1.8
VISION_ROI_RECOVERY_S = 1.0
VISION_ROI_HALF_SIZE_PX = 260
CAMERA_TILT_DEG = 15.0

# ============ CRSF channels (Air75 Betaflight dump) ============
CH_ROLL, CH_PITCH, CH_THR, CH_YAW = 0, 1, 2, 3
CH_ARM, CH_MODE, CH_PREARM = 5, 6, 9
ARM_HIGH_US, ARM_LOW_US = 1500, 1000
PREARM_HIGH_US, PREARM_LOW_US = 1800, 1000
MODE_ANGLE_US = 1500
NEUTRAL_US = 1500
IDLE_THR_US = 1000

# ============ stick signs (verify in DRY_RUN before flying) ============
# Ground truth: setup/test_stick_directions.py sends 1700us on ROLL → "ROLL RIGHT"
# and on PITCH → "PITCH FORWARD" (both verified in Betaflight Configurator).
# So on this FC: ch>1500 ⇒ roll right / pitch forward / yaw right.
# Controller convention: desired_*_deg > 0 ⇒ want roll-right / nose-down (fwd).
# Therefore SIGN_PITCH must be +1 (not -1 — that was the inverted axis that
# made the 2026-05-29 v2 flight drift backward away from the tag).
SIGN_PITCH = +1
SIGN_ROLL = +1
SIGN_YAW = +1
# IMU R/P signs for the camera-leveling rotation in cam_to_world. Air75 FC
# reports +pitch = nose DOWN (non-standard), so SIGN_IMU_PITCH flips it.
SIGN_IMU_ROLL = +1
SIGN_IMU_PITCH = -1

# ============ FC angle-mode stick → angle scaling ============
# Measured 2026-05-29 with a TX in Betaflight: full-deflection endpoints are
# roll right=2012us / left=989us (center 1500, half-range 511.5us). The FC
# reaches angle_limit (60°) at that ±511.5us, so the true scaling is
# 511.5/60 = 8.525 us/deg — slightly above the 8.333 textbook default. Pitch,
# yaw, thrust measured the same endpoints. Centered at 1500 as expected.
FC_ANGLE_LIMIT_DEG = 60.0
STICK_FULL_DEFLECTION_US = 511.5                  # measured half-range, not 500
STICK_US_PER_DEG = STICK_FULL_DEFLECTION_US / FC_ANGLE_LIMIT_DEG   # 8.525

# ============ throttle ============
# HOVER_THROTTLE_US: the no-correction baseline. 2026-05-28 flight stabilised
# at T≈1340 on a 4.10V pack, but the 2026-05-29 v2 flight started at 3.70V
# (sagging to 3.60V) — at that voltage 1340 is below hover and the drone never
# leaves the ground. Bumped to 1380 as a starting point for ~3.7V packs.
# Refine after each flight by watching v_up at steady throttle.
HOVER_THROTTLE_US = 1340
MAX_THROTTLE_US = 1700
# Throttle authority relaxed: the controller now has wide room to climb/descend
# (was ±60/±80 around hover, which capped the band at 1300–1440us and starved
# altitude authority). Widened so altitude isn't the bottleneck. MAX_THROTTLE_US
# is still the hard ceiling; IDLE_THR_US the floor.
THR_CLIMB_TRIM_US = 100        # max +correction (climbing) — halved: fresh pack
                               # climbed too fast at full +200 authority
THR_DESC_TRIM_US = 200         # max -correction (descending)

# ============ outer position PIDs ============
# Output: desired body roll/pitch angle (deg), converted to us downstream and
# bounded by a SINGLE output clamp (MAX_ROLL_US / MAX_PITCH_US, below). Tune
# order: KP → KD → KI. KP is sized so output magnitude matches the yaw loop —
# a 0.3 m error → ~12° → ~100us, comparable to yaw's ~100us for an off-center
# tag. MAX_*_INT_DEG bounds only the integrator state (anti-windup), a separate
# signal from the output.
#
# X (lateral): err_lat (m) → desired roll angle (deg).
# +err_lat (tag drifted right of where it should be → drone drifted LEFT)
# → +desired_roll → drone rolls right → back to target.
KP_X_DEG_PER_M = 40.0
KI_X_DEG_PER_M_S = 0.0         # leave at 0; enable once flight is stable
KD_X_DEG_PER_MPS = 20.0
MAX_X_INT_DEG = 15.0           # |Ki·integral| clamp (anti-windup)

# Z (forward): err_fwd (m) → desired pitch angle (deg).
# +err_fwd (tag too far ahead → drone too far back) → +desired_pitch
# → nose down → drone flies forward → back to target.
KP_Z_DEG_PER_M = 40.0
KI_Z_DEG_PER_M_S = 0.0
KD_Z_DEG_PER_MPS = 20.0
MAX_Z_INT_DEG = 15.0

# THE single roll/pitch output clamp (in us, applied after deg→us). Set to the
# measured full-deflection half-range (±511us), so the only effective limit is
# the FC's own 60° angle_limit — the controller can command full stick.
MAX_ROLL_US = 511
MAX_PITCH_US = 511

# ============ altitude controller (carried over from existing) ============
# Architecture: position error → target velocity (capped) → velocity error
# → throttle correction (asymmetric clamp). Same gains as tag_hover_controller.py.
VMAX_UP_MPS = 0.30
KP_UP = 0.7                    # 1/s (position → target velocity); halved from
                               # 1.4 — fresh pack climbed too fast
KV_UP_US_PER_MPS = 40.0        # throttle us per (m/s) of velocity error

# ============ yaw controller (tag-bearing hold) ============
# The Air75 has no magnetometer, so CRSF yaw is free-running gyro heading that
# drifts without bound — holding it against a captured yaw_at_arm always winds
# off and pins the stick (the 2026-05-29 "yaw stuck at 1350" bug). Instead we
# hold the TAG centered: error = atan2(lat, fwd), the tag's bearing off the
# nose, which is absolute and drift-free.
# bearing > 0 (tag to the RIGHT) → yaw RIGHT (us > 1500) to recenter.
# KP_YAW: us per rad of bearing error. A 17° (0.30 rad) off-center tag → 90us;
# the clamp caps it at ±150us. Centered at 1500; inside YAW_DEADBAND = 1500.
KP_YAW_US_PER_RAD = 300.0
MAX_YAW_US = 150
YAW_DEADBAND_RAD = math.radians(2.5)

# ============ pose filter ============
ALPHA_POS = 0.40
ALPHA_VEL = 0.25
VISION_FRESH_S = 0.20
ESTIMATOR_MAX_DT_S = 0.05
VEL_DECAY_TAU_S = 0.25
RESEED_AFTER_LOSS_S = 0.10
GROUND_PIN_BAND_US = 60

# ============ tag-loss behavior ============
STATE_MAX_AGE_S = 0.35
LAND_DESCENT_US_PER_S = 130.0
KILL_AFTER_LOST_S = 3.5

# ============ arming ============
ARM_CONFIRM_S = 8.0
ARM_LOCK_MIN_S = 0.45

# ============ battery cutoff (1S whoop) ============
LOW_BATT_CUTOFF = True
BATT_PRESENT_V = 2.5
MIN_CELL_V = 3.3
CELLS = 1

# ============ loop rates ============
TX_HZ = 50.0
DISPLAY_HZ = 30.0
LOG_HZ = 12.0
LOG_EVERY_LOOP = False
