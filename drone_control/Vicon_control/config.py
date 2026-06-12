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
# CONFIRMED via Betaflight CLI (2026-06-11): angle_limit = 60, so full ±511us stick
# = ±60° → 8.525 us/deg. (My earlier "13°" guess from a flight regression was wrong —
# the FC config is authoritative.) NOTE the measured FACT: in flight 20260611_142905
# the drone tilted ~4.6x LESS than commanded near center (commanded ~5° → ~1° actual).
# With angle_limit=60 confirmed, that softness is RC expo and/or the response lag near
# center — NOT a low angle limit. We don't need this conversion exact: the integral
# auto-winds-up to whatever us actually moves the drone. We compensate the soft
# near-center response purely by RAISING THE GAINS below (more aggressive corrections).
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
# Gains are in commanded-degrees at 8.525 us/deg (angle_limit=60). The numbers look
# large because the FC's near-center response is soft (~4.6x, expo/lag) — so we send
# more us to get real authority. Behavior is set by the us output (gain × 8.525);
# the integral auto-trims steady-state regardless of the exact conversion. vs the
# last flight (KP=11/KD=14/KI=2.5) this is ~1.7x more P + damping authority, with a
# gentler integral (KI=2.5 wound up and slow-oscillated). WATCH for fast oscillation
# now — if it shakes quickly the response lag is the limit; lower KP.
#
# PER-AXIS GAINS — the two horizontal PIDs are independent (controller.pid_fwd =
# pitch/world-Y, controller.pid_lat = roll/world-X), so each gets its own gains.
#
# Diagnosis from flight 20260611_154818 (KP=19/KD=25/KI=1.0, shared): BOTH axes
# slow-oscillate ±0.2–0.3 m about the origin with a ~25–35 s period (x a bit wider
# than y, but the SAME problem — y is NOT actually tight). Two facts pin the cause:
#   (1) it's CENTERED (mean ≈ 0) → the integral IS trimming the average; this is a
#       swing AROUND target, not a standing offset.
#   (2) the period is ~30 s → far too slow for a proportional/damping/lag oscillation
#       (that cycles in seconds) → the INTEGRAL is driving the slow limit cycle.
# Cross-flight trend confirms KI is the driver: KI=0 → ~1 m offset, no swing; KI=1.0
# → centered, ±0.25 m swing; KI=2.5 → ±0.3 m worse swing. So the lever to REDUCE the
# swing is to LOWER KI (1.0→0.5), NOT raise it — while stiffening P (so the gentler
# integral still centers it: a stiffer P shrinks the offset the integral must chase)
# and adding damping. Applied to BOTH axes (both oscillate); LAT gets a touch more
# P+D since x swings wider. Gains are commanded-deg at 8.525 us/deg (large because the
# FC near-center response is soft). UNFLOWN — if either axis now shakes FAST, the
# ~300 ms response lag is the ceiling → drop that axis' KP. Gain-tuning has a limit
# here (soft + laggy plant); if the swing persists, the real fixes are reducing the
# command→tilt lag and/or linearizing the FC rates curve.

# !!! RETUNE FOR A LINEARIZED FC RATE CURVE (after flight 20260611_164212) !!!
# That flight MEASURED the real problem (FC attitude telemetry @97.5Hz AND the Vicon
# quaternion AGREE): the drone tilts only ~1.5° when the controller commands ~7.6° —
# a ~5x soft near-center response. Root cause (verified): BF 4.5 Angle mode follows
# the ACTUAL-rates curve, and the stock curve (Center Sensitivity 70°/s vs Max Rate
# 670°/s) is ~10:1 progressive = soft center. NO controller gain fixes a plant that
# ignores ~80% of the command — that's why KP 11→19→26 barely helped.
# THE FIX is FC-side: linearize roll+pitch rates (set roll_srate=pitch_srate=7 so
# Center Sensitivity = Max Rate → straight-line curve → commanded angle = actual).
# ONCE LINEAR, the drone tilts ~5x MORE for the same command, so these gains are cut
# ~5x from the values above (KP 26-28 → 6) to a CONSERVATIVE baseline that ~preserves
# the previous (stable) loop gain but now in HONEST degrees. Start here, confirm the
# linear response is stable + that actual≈commanded tilt, THEN raise to tighten the
# hover (now that the drone actually responds, raising KP will work). Both axes start
# equal — the old per-axis split chased an asymmetry that may have been the curve;
# re-split only if x still lags y once linear. APPLY THE FC CHANGE AND THESE TOGETHER
# — linear curve with the OLD gains would be ~5x too hot. First linear flight = a
# careful test: low CLIMB, finger on the disarm; if it shakes FAST, the ~0.3-0.5s
# response lag is the ceiling → lower KP.

# After flight 20260611_171558 (FIRST LINEAR flight): the rate-curve fix WORKED —
# actual/commanded tilt = 0.99 (was 0.20), stable, gentle (max 3°), centered, alt
# rock-solid. Swing was still ±0.2 m because the gains were cut 5x for safety, so net
# authority ≈ before. Now that the plant is HONEST + has lots of margin (no fast
# shake), raising KP finally tightens the hover for real. STEP 1: 2x (KP 6→12, KD
# 8→16, ratio kept). If still loose + stable → raise again; if it shakes FAST → the
# ~0.3-0.5 s cmd→tilt lag is the ceiling, back off. Both axes still equal (response is
# now ~symmetric: roll slope 0.91 / pitch 0.74).

# --- forward axis (pitch / world-Y) ---
KP_FWD_DEG_PER_M = 15.0        # 2x (was 6) — honest deg/m, tighten the hold
KD_FWD_DEG_PER_MPS = 20.0      # 2x (was 8) — damping, ratio kept ~1.33
KI_FWD_DEG_PER_M_S = 0.4       # gentle auto-trim (unchanged)
MAX_FWD_INT_DEG = 10.0         # integral contribution cap

# --- lateral axis (roll / world-X) ---
KP_LAT_DEG_PER_M = 15.0        # 2x (was 6)
KD_LAT_DEG_PER_MPS = 20.0      # 2x (was 8)
KI_LAT_DEG_PER_M_S = 0.4       # gentle auto-trim
MAX_LAT_INT_DEG = 10.0         # integral contribution cap

MAX_TILT_DEG = 15.0            # output clamp — real tilt; 15° is plenty (it used ~3°)

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
KI_UP_US_PER_M = 85.0          # I: hover-throttle us per (m) of accumulated v-error
THR_CLIMB_TRIM_US = 100        # max +P correction (climbing) — asymmetric:
THR_DESC_TRIM_US = 200         # max -P correction (gravity aids descent)
LAND_SPEED_MPS = 0.25          # commanded descent rate when landing (SPACEBAR / low batt)
LAND_CUT_M = 0.30              # descend to this height above launch, then CUT throttle +
                               # disarm + close the session + EXIT the program. Cutting
                               # at 0.3 m (not all the way down) avoids ground-effect
                               # wobble; the Air75 is light enough to drop the last 0.3 m.
# Clean takeoff: until the drone is this far above its takeoff altitude, hold LEVEL
# (no roll/pitch) and freeze the horizontal integrators so it lifts STRAIGHT UP
# instead of scooting on the ground; engage horizontal hold once above it.
TAKEOFF_AIRBORNE_M = 0.3

# ============ yaw loop (absolute heading hold — Vicon yaw is drift-free) ========
# err = wrap(yaw - yaw_target); err > 0 (drone yawed CCW/left of target) → yaw
# RIGHT (us>1500) to recenter, with SIGN_YAW=+1. No magnetometer hack needed.
KP_YAW_US_PER_RAD = 200.0
MAX_YAW_US = 120
YAW_DEADBAND_RAD = math.radians(2.0)

# ============ safety ============
# The flight ends ONLY on: SPACEBAR (laptop), low battery, manual disarm (TX12),
# or Vicon loss — each descends/cuts and EXITS the program (no time cap, no
# auto-relaunch). The TX12 arm switch is always the instant kill.
VICON_STALE_S = 0.15           # pose older than this in flight → blind gentle sink
VICON_KILL_S = 0.60            # pose lost this long → cut + exit
LOW_BATT_CUTOFF = True         # auto-LAND on a sagging 1S pack (then cut + exit)
BATT_PRESENT_V = 2.5           # below this = no/!valid pack reading, ignore
MIN_CELL_V = 3.3
CELLS = 1

# ============ waypoint mission (square_flight.py only) ==========================
# square_flight.py flies a course built in the LAUNCH BODY FRAME (forward/right
# relative to the nose at takeoff) and converted to FIXED world waypoints ONCE, at
# launch, using the captured launch yaw (mission.build_square_mission). The default
# course: take off + hover, then forward → right → back → left by LEG_M with DWELL_S
# holds at each vertex, returning over the origin, then land. Heading is HELD at the
# launch yaw throughout (the legs are strafes, not turns). vicon_hover.py ignores all
# of this (it flies a static HoldMission); only square_flight.py reads these.
LEG_M = 2.0                    # square side length (forward/right/back/left distance)
CRUISE_SPEED_MPS = 0.80        # moving-setpoint ("carrot") speed between waypoints —
                               # the horizontal analog of VMAX_UP_MPS. (Note:
                               # D-on-measurement adds ~KD*cruise of opposing tilt, so
                               # the drone trails the carrot ~KD*v/KP m: ~0.3 m measured
                               # at 0.4 m/s, so ~0.6 m expected here at 0.8 m/s — hence
                               # LEASH_M is raised to stay above it. The arrival gate
                               # waits for the DRONE, not the carrot, so the trailing is
                               # benign; at speed the circle just flies a bit smaller +
                               # more phase-lagged. Tighten with velocity feedforward.)
DWELL_S = 5.0                  # hold time at each square vertex
INITIAL_HOVER_S = 3.0          # settle time at the takeoff hover before leg 1
ARRIVE_TOL_M = 0.25            # carrot AT the WP and drone within this (horiz + vert)
                               # → start the hold
ARRIVE_TIMEOUT_S = 12.0        # backstop (counted only while airborne): proceed to the
                               # hold after this long in a leg even if never within tol,
                               # so a drone that never quite settles can't hang the run
LEASH_M = 1.2                  # the carrot never gets more than this far ahead of the
                               # drone — bounds the position error (and thus tilt/speed)
                               # if the drone falls behind; 0.0 disables the leash. Keep
                               # it ~2x the steady-state trailing lag (≈KD*v/KP) so it
                               # only catches a real stall, not normal cruise: 0.6 worked
                               # at 0.4 m/s (lag ~0.3), so 1.2 here at 0.8 m/s (lag ~0.6).
                               # Too low and the leash chops the motion (stutters/stops).

# ============ circle mission (circle_flight.py only) ===========================
# circle_flight.py: take off + hover, fly FORWARD CIRCLE_RADIUS_M to reach a circle
# CENTERED ON THE LAUNCH ORIGIN (so the forward point lands exactly on it), trace
# one full circle, return to the origin, settle, land. The forward distance and the
# radius are the SAME value by construction (the origin is the centre). Heading is
# held at the launch yaw the whole time (the circle is flown by translating). Reuses
# CRUISE_SPEED_MPS / LEASH_M / ARRIVE_TOL_M / ARRIVE_TIMEOUT_S / INITIAL_HOVER_S.
CIRCLE_RADIUS_M = 1.0          # circle radius AND the forward approach distance
CIRCLE_CW = True               # True = clockwise viewed from above (the carrot goes
                               # forward-point → right → back → left → forward-point);
                               # False = counter-clockwise
SETTLE_S = 2.0                 # hold at the circle entry (clean start) and again at
                               # the origin on return, before landing

# ============ loop rate (shared) ============
TX_HZ = channels.TX_HZ         # 50 Hz, same as the data logger
