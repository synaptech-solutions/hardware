"""Tunable parameters for the Vicon hover controller.

Architecture (FC in Angle mode → inner attitude loop in firmware):
  Position (world→body)  : body fwd/lat error (m) → desired pitch/roll angle (deg)
  Altitude (PI velocity)  : the integrator IS the learned hover throttle → us
  Yaw                     : P on ABSOLUTE heading error (Vicon yaw, drift-free) → us

Desired angles → CRSF us via the FC Angle-mode stick scaling (STICK_US_PER_DEG).

START CONSERVATIVE. Vicon pose is clean + low-latency, so these can be tightened,
but the bring-up order is: verify DIRECTIONS in DRY_RUN (props off) → low hover →
raise CLIMB_M → tune gains. The history to respect (see apriltag_control + memory):
the SIGN_PITCH inversion drove the drone away from target — verify signs first.
"""
import math

from common import channels   # __init__ puts drone_control on sys.path


# ============ run mode ============
DRY_RUN = False                 # True = compute + print, NEVER open Ranger / arm.
                               # Flip to False only after DRY_RUN dir-checks pass.
RECORD_VIDEO = False            # Vicon flights don't need the drone-feed video (Vicon
                               # is the truth) — skip it to save disk. The other
                               # streams (vicon/commands/telemetry) still record.

# ============ target ============
# Hover holds the takeoff x/y and heading, CLIMB_M above the takeoff altitude.
# FIRST FLIGHTS: set CLIMB_M = 0.3 and confirm a stable low hover before 1.0 m.
CLIMB_M = 1.0

# ============ FC angle-mode stick → angle scaling (from the Air75 measurement) ===
# Measured 2026-05-29: full deflection ≈ ±511.5us reaches angle_limit (60°), so
# 511.5/60 = 8.525 us/deg. (TODO: confirm via Air75 CLI `get angle_limit`.)
FC_ANGLE_LIMIT_DEG = 60.0
STICK_FULL_DEFLECTION_US = 511.5
STICK_US_PER_DEG = STICK_FULL_DEFLECTION_US / FC_ANGLE_LIMIT_DEG   # 8.525

# ============ stick signs (Air75, verified via test_stick_directions.py) ========
# us > 1500 ⇒ roll RIGHT / pitch FORWARD (nose down) / yaw RIGHT (CW).
# These are the knobs to flip if DRY_RUN shows a corrective direction inverted.
SIGN_ROLL = +1
SIGN_PITCH = +1
SIGN_YAW = +1

# ============ Vicon↔drone frame alignment (THE 2026-06-11 flyaway fix) ============
# The Vicon rigid body's local axes are a clean 90° off from the drone's roll/pitch
# axes: Vicon's body X = the drone's RIGHT, Vicon's body Y = the drone's NOSE. So
# Vicon yaw=0 means the nose points world +Y, while the controller's convention is
# nose = +X at heading 0. We add this constant to the Vicon yaw so the controller
# decomposes position error in the drone's TRUE body frame (true_heading = vicon_yaw
# + 90). Uncorrected this is a pure 90° rotation -> the position loop orbits ->
# the 2026-06-11 widening-spiral flyaway. MEASURED BY GROUND CALIBRATION 2026-06-11
# (vicon_read.py): nose=+Y -> vicon yaw 0; nose=+X -> vicon yaw -90; both give +90.
# (Supersedes the noisy +50 estimate from the crashed-flight regression.) Does NOT
# affect manual flying (the pilot never used Vicon coords to drive the sticks).
VICON_YAW_OFFSET_DEG = 90.0

# ============ horizontal position loop (world→body PID → desired angle deg) =====
# err is (target - drone) projected on the body axes (see controller._body_errors):
#   err_fwd > 0 → target ahead of nose → fly forward (nose down, us>1500)
#   err_lat > 0 → target to the right  → roll right (us>1500)
# Output desired angle (deg) = KP*err + KD*d(err)/dt, clamped by MAX_TILT_DEG.
# Conservative start (apriltag used 40/20 in tag frame; Vicon is cleaner so we can
# raise once stable). A 1 m error → 8°; 0.3 m → 2.4°.
# Tuning after flight 20260611_135158: stable hover but ~1m steady-state offset
# (P-only settled where KP*err balanced a ~8°-worth bias). Enabled integral to
# zero the offset + modest KP bump for a firmer return (your "harder corrections").
KP_POS_DEG_PER_M = 11.0        # was 8 — firmer proportional return (1m err -> 11°)
KD_POS_DEG_PER_MPS = 14.0      # damping unchanged (it wasn't oscillating)
KI_POS_DEG_PER_M_S = 2.5       # was 0 — THE fix for the 1m steady-state offset;
                               # accumulates standing error, adds trim until err->0
MAX_POS_INT_DEG = 12.0         # was 8 — room for the integrator to hold the ~8° trim
MAX_TILT_DEG = 18.0            # output clamp on commanded roll/pitch angle

# ============ altitude loop (PI velocity loop, self-learning hover) =============
# The integrator state (hover_us) IS the hover throttle, in us: it SEEDS at
# HOVER_START_US (near true hover, for a quick takeoff) and integrates velocity
# error up/down, clamped to the band. No hover constant is hardcoded in the law —
# the loop finds it and tracks pack sag. (This fixed the 2026-05-31 pure-PD
# slow-sink.) Measured hover ≈ 1351-1405us; seeding at 1350 lifts in <1s instead
# of the ~2.5s slow ramp from the 1300 floor.
HOVER_BAND_LO_US = 1300        # integrator floor (clamp; lets it trim down if climbing fast)
HOVER_BAND_HI_US = 1450        # integrator ceiling (clamp; measured hover ~1351-1405)
HOVER_START_US = 1350          # takeoff SEED for the integrator (near true hover → fast lift)
MAX_THROTTLE_US = 1700         # hard rail; band ⊂ [IDLE, MAX] so it's never hit
VMAX_UP_MPS = 0.50             # climb/descend speed cap (was 0.30 — faster takeoff)
KP_UP = 0.7                    # 1/s: altitude error → target vertical velocity
KV_UP_US_PER_MPS = 40.0        # P: throttle us per (m/s) of velocity error
KI_UP_US_PER_M = 80.0          # I: hover-throttle us per (m) of accumulated v-error
THR_CLIMB_TRIM_US = 100        # max +P correction (climbing) — asymmetric:
THR_DESC_TRIM_US = 200         # max -P correction (gravity aids descent)
LAND_SPEED_MPS = 0.25          # commanded descent rate when landing (record-off)
LAND_DONE_M = 0.10             # treat as landed when within this of takeoff alt
# Clean takeoff: until the drone is this far above its takeoff altitude, hold LEVEL
# (no roll/pitch) and freeze the horizontal integrators so it lifts STRAIGHT UP
# instead of scooting on the ground; engage horizontal hold once above it.
TAKEOFF_AIRBORNE_M = 0.15

# ============ yaw loop (absolute heading hold — Vicon yaw is drift-free) ========
# err = wrap(yaw - yaw_target); err > 0 (drone yawed CCW/left of target) → yaw
# RIGHT (us>1500) to recenter, with SIGN_YAW=+1. No magnetometer hack needed.
KP_YAW_US_PER_RAD = 200.0
MAX_YAW_US = 120
YAW_DEADBAND_RAD = math.radians(2.0)

# ============ safety ============
VICON_STALE_S = 0.15           # pose older than this in flight → land then disarm
VICON_KILL_S = 0.60            # pose lost this long → immediate disarm
LOW_BATT_CUTOFF = True         # disarm on a sagging 1S pack
BATT_PRESENT_V = 2.5           # below this = no/!valid pack reading, ignore
MIN_CELL_V = 3.3
CELLS = 1
MAX_FLIGHT_S = 60.0            # hard cap on a single FLYING session (then land)

# ============ loop rate (shared) ============
TX_HZ = channels.TX_HZ         # 50 Hz, same as the data logger
