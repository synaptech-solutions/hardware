"""Tunable parameters for the Vicon hover controller.

Architecture (FC in Angle mode → inner attitude loop in firmware):
  Position (world→body)  : body fwd/lat error (m) → desired pitch/roll angle (deg)
  Altitude (PI velocity)  : the integrator IS the learned hover throttle → us
  Yaw                     : heading PID + rate FF (ABSOLUTE Vicon yaw, drift-free) → us

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
RECORD_VIDEO = True             # record the drone-feed video alongside the other
                               # streams (vicon/commands/telemetry). Needs the VRX →
                               # Cam Link chain plugged in (device index in channels.py).

# ============ target ============
# Hover holds the takeoff x/y and heading, CLIMB_M above the takeoff altitude.
# FIRST FLIGHTS: set CLIMB_M = 0.3 and confirm a stable low hover before 1.0 m.
# SHARED by hover / circle / square / figure-8 (all fly CLIMB_M above launch). Set
# to 1.0 for the figure-8's spec'd flat 1 m height (z=1); the circle last flew 0.8.
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

# ============ ACRO mode (ADDITIVE — ANGLE is the default + the fallback) ========
# With ACRO_MODE True the FC is put in ACRO (MODE_CH → MODE_ACRO_US): Betaflight
# stops self-leveling and reads the roll/pitch sticks as RATE setpoints, still
# closing its own 8 kHz inner rate PID. We then close the ATTITUDE (leveling) loop
# ourselves on Vicon roll/pitch — the controller's angle→rate P stage replaces
# Angle mode's outer P. Yaw + throttle are UNCHANGED (yaw is a rate command in BOTH
# modes; throttle is direct). With ACRO_MODE False all of this is dead code and the
# flight is the proven Angle-mode behavior, byte-for-byte.
#
# Rate curve (Air75 `dump all`, rateprofile 0, ACTIVE): rates_type=ACTUAL, roll/
# pitch srate=7, expo=0 → LINEAR, full stick (±511.5us) = 70°/s. So a commanded body
# rate → us at 511.5/70 = 7.307 us per °/s. (70°/s is low — it was set to linearize
# Angle mode, not for aerobatic ACRO — but it's ample for a hover leveling loop;
# raise srate later for a faster envelope, then cut KP_ANGLE_RATE proportionally.)
#
# THE VICON ATTITUDE MAP is empirically verified (2026-06-16, acro_attitude_check.py,
# 3-pose bench test) and lives in vicon_source.drone_roll_pitch (heading-invariant;
# the 90° mount swaps std roll/pitch vs the drone's axes). Still DRY_RUN the restoring
# direction before arming: tilt nose-down → pitch us must drop BELOW 1500 (commands
# nose-up); roll right → roll us BELOW 1500. That's the gate (defense in depth).
ACRO_MODE = True              # master switch; flip True only after the DRY_RUN check
KP_ANGLE_RATE = 10.0            # °/s of commanded body rate per ° of attitude error
                               # (≈ Angle mode's outer P; 1/KP ≈ 0.2 s leveling time
                               # constant). START LOW: Vicon attitude is ~100 Hz but
                               # transport-delayed and there is NO FC self-level net
                               # under this loop — if it wobbles FAST, LOWER this; if
                               # it's sluggish to level, raise it.
ACRO_MAX_RATE_DPS = 70.0       # clamp on the commanded rate = the FC ACTUAL-rate
                               # ceiling (srate=7 → 70°/s); full stick at the clamp.
ACRO_RATE_US_PER_DPS = STICK_FULL_DEFLECTION_US / ACRO_MAX_RATE_DPS   # 7.307

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

MAX_TILT_DEG = 55.0            # output clamp. The 3 m/s circle (r=1) needs total bank
                               # hypot(centripetal 42.5°, drag 8.2°) = 43.3°; 55° keeps
                               # PID correction headroom while staying under the FC
                               # angle_limit (60°) — NEVER set this ≥60 (the FC clips
                               # there) or near 90 (a quad makes ZERO lift at 90°).

# --- acceleration feedforward (the last rung: position ref → velocity FF → this) ---
# The mission reports the carrot's acceleration (finite thanks to the trapezoid):
# the ramp accel on straights, v²/r centripetal on arcs. The controller converts
# it straight to the tilt that acceleration requires — atan(a/g) — instead of
# letting feedback squeeze it out of error. Flight 20260612_160149 showed the
# cost of its absence: −4.3 cm radius + 8° phase lag were the loop's way of
# generating the ~3.7° inward lean from the D term.
MAX_FF_ACCEL_MPS2 = 12.0       # cap on the relayed accel — its PURPOSE is spike
                               # rejection: the carrot accel is a finite diff of carrot
                               # velocity, so a leash engage / timeout-park (carrot
                               # velocity steps in one tick) finite-differences into a
                               # huge bogus accel; this bounds it (ACCEL_FF_LPF_S smooths
                               # it too). Raised 9→12 (2026-06-15) for the 3 m/s circle:
                               # v²/r = 9 m/s² of centripetal must pass UNCLIPPED (at the
                               # old 9 cap it sat exactly at the limit with no margin);
                               # 12 m/s² = atan(12/9.81)=50.7°, under the 55° MAX_TILT
                               # clamp, so the CLAMP — not this guard — stays the real
                               # ceiling. (figure-8 @2m/s R=0.5 needs only 8.) Raise WITH
                               # MAX_TILT for faster flight / tighter radii in future.
ACCEL_FF_LPF_S = 0.08          # 1-pole LPF on the body-frame accel FF: swallows
                               # one-tick spikes, adds only ~0.08 s to the (already
                               # step-shaped) ramp transitions
# DRAG feedforward — the partner the accel FF NEEDS. Cruising costs ~K_DRAG·v of
# forward tilt against air drag; with no FF for it, the loop must generate that
# tilt from a standing error. Without accel FF it used the tangential/phase-lag
# channel (radius stayed ≈1); with accel FF alone the loop switches to a SPEED
# DEFICIT channel instead — the drone runs slower on a smaller circle (sim:
# r 0.90) — so the two FFs ship together. Tilt += K_DRAG · v_reference (body
# frame). Value measured from flight 20260612_160149: holding 0.8 m/s took
# ~2.2° of forward tilt → 2.75 °/(m/s). Refine from future flight logs.
K_DRAG_DEG_PER_MPS = 2.75

# ============ altitude loop (PI velocity loop, self-learning hover) =============
# The integrator state (hover_us) IS the hover throttle, in us: it SEEDS at
# HOVER_START_US (near true hover, for a quick takeoff) and integrates velocity
# error up/down, clamped to the band. No hover constant is hardcoded in the law —
# the loop finds it and tracks pack sag. (This fixed the 2026-05-31 pure-PD
# slow-sink.) Measured hover ≈ 1351-1405us; seeding at 1350 lifts in <1s instead
# of the ~2.5s slow ramp from the 1300 floor.
HOVER_BAND_LO_US = 1300        # integrator floor (clamp; lets it trim down if climbing fast)
HOVER_BAND_HI_US = 1550        # integrator ceiling (clamp; measured hover ~1351-1405)
HOVER_START_US = 1350          # takeoff SEED for the integrator (near true hover → fast lift)
MAX_THROTTLE_US = 1700         # hard rail; band ⊂ [IDLE, MAX] so it's never hit
VMAX_UP_MPS = 0.50             # climb/descend speed cap (was 0.30 — faster takeoff)
KP_UP = 0.7                    # 1/s: altitude error → target vertical velocity
KV_UP_US_PER_MPS = 40.0        # P: throttle us per (m/s) of velocity error
KI_UP_US_PER_M = 85.0          # I: hover-throttle us per (m) of accumulated v-error
# TILT COMPENSATION feedforward: at tilt θ only cosθ of the thrust points up, so
# holding altitude needs hover/cosθ of throttle. The reactive PI loop is hover-
# soft and only responds AFTER altitude error builds — flight 20260612_165231
# dipped −22 cm (std 8 cm) on the 22-27° laps while throttle sat at ~1352 (NOT
# band-limited). Fix: scale the hover_us term by 1/(cos(pitch_cmd)·cos(roll_cmd))
# using the COMMANDED (clamped) tilt — instant, no lag, zero at hover. The
# integrator still learns LEVEL hover (comp applied outside it), so no transient
# when the tilt returns to zero.
TILT_COMP_MAX = 1.5            # cap on the 1/cos factor (1.414 at the 45° clamp;
                               # guards against pathological cos→0)
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
#
# !!! PAIRED WITH AN FC CHANGE (2026-06-12, for the tangent-facing circle) !!!
# Yaw in Angle mode is a RATE command through the ACTUAL-rates curve, and yaw was
# still on the stock progressive curve (center 70°/s / max 670°/s) — the same
# soft-center plant that broke roll/pitch before 20260611_171558. Within our
# MAX_YAW_US the drone could only do ~16°/s; the circle needs 46°/s sustained.
# FIX (Betaflight CLI): linearize yaw at 300°/s —
#     set yaw_rc_rate = 30 ; set yaw_srate = 30 ; set yaw_expo = 0 ; save
# YAW_LINEAR_MAX_DPS documents that FC setting; the feedforward gain is derived
# from it. FLY THE FC CHANGE AND THIS CONFIG TOGETHER (without the FC change the
# FF is ~4x too weak near center — safe, the dwell gate just waits — but the
# tangent tracking will lag badly).
YAW_LINEAR_MAX_DPS = 300.0     # FC linear yaw rate: full stick (511.5us) = 300°/s
# Yaw-rate FEEDFORWARD: the mission's heading setpoint moves (46°/s around the
# circle, YAW_SLEW_DPS in pre-rotations); P-only would need ~22° of standing error
# to hold that rate. The controller differences the commanded heading and adds
# us = rate / (linear curve slope), so the rate costs zero error — same fix as
# the position-loop velocity FF, applied to yaw.
YAW_FF_US_PER_DPS = STICK_FULL_DEFLECTION_US / YAW_LINEAR_MAX_DPS   # 1.705
# Full heading PID (P+I+D around the FF). Why yaw got away with P-only while
# position needed PID: our yaw output is a RATE command closed by the FC's own
# yaw-rate PID, so heading(u) is a SINGLE integrator — P alone is a stable
# first-order loop (no overshoot in the ideal). Position is a DOUBLE integrator
# (tilt→accel→vel→pos), unstable under pure P, hence KD from day one. The real
# chain has ~0.2-0.4 s of transport+FC lag though, so:
#   D (on MEASURED yaw rate, LPF'd, minus the target rate — no setpoint kick)
#     buys damping margin against that lag;
#   I (small, clamped) trims constant residuals the deadbanded P never fixes —
#     e.g. FF scale error while the heading ramps around the circle (true curve
#     slope ≠ exactly 300°/s ⇒ constant rate deficit ⇒ standing heading error).
KP_YAW_US_PER_RAD = 200.0      # 3.49 us/deg
KI_YAW_US_PER_RAD_S = 30.0     # trims a 5.7° standing offset in ~10 s
MAX_YAW_I_US = 30              # integral contribution cap (windup guard)
KD_YAW_US_PER_RAD_PER_S = 25.0 # damping: 1 rad/s of error rate → 25 us opposing
YAW_RATE_LPF_S = 0.10          # 1-pole LPF on the differenced Vicon yaw rate the D
                               # term uses (50 Hz diff of ~0.2° noise ⇒ ~14°/s rate
                               # noise raw — filter before it reaches KD)
MAX_YAW_US = 450               # 450us = 264°/s authority (450/1.705). The 3 m/s circle
                               # needs 172°/s sustained (FF = 293us), leaving 157us
                               # (~92°/s) for the yaw PID on top — plenty of correction
                               # headroom. The PHYSICAL ceiling is full stick 511.5us =
                               # 300°/s (the FC linear yaw curve); 450 stays under it.
                               # (Was mistakenly 300us = only 176°/s, which left just
                               # 7us of headroom and looked like saturation — that was a
                               # too-low clamp, NOT the drone's limit.)
YAW_DEADBAND_RAD = math.radians(2.0)   # zeroes the error fed to P+I (no twitching at
                               # rest); FF and D always run

# ============ sync-spin maneuver (clock + latency witness) ======================
# Bookends every flight with a deliberate, crisp 360° flat yaw spin (settle → spin
# → settle) BEFORE the program and again BEFORE landing. The spin is the loud, clean
# yaw event the post-flight sync needs: combine.py cross-correlates Vicon yaw-rate
# (laptop clock) against the FC gyro (FC clock) to recover the laptop↔drone uplink
# latency (~42 ms measured 2026-06-17) and, from the start-vs-end spins, the clock
# DRIFT. The master timeline stays the deterministic shared-trigger; this only adds
# a second, gated witness. See [[project_latency_analysis]].
#
# Drive: CLOSED-LOOP — the mission slews the heading setpoint at SPIN_RATE_DPS and
# the controller's existing yaw-rate FF tracks it while still holding x/y/z. Keep
# SPIN_RATE_DPS at/under the FF's ±180°/s clamp (controller.step) so the spin is a
# clean constant rate fed entirely by the FF, not a PID catch-up. DRY_RUN-verify the
# spin direction + that position holds before flying.
SYNC_SPIN_ENABLED = True        # master switch for the bookend spins
SPIN_RATE_DPS = 160.0           # spin yaw rate (°/s). 160 → ~2.25 s/turn; stays under
                                # the ±180°/s yaw-rate FF clamp for a clean constant rate.
SPIN_TURNS = 1.0                # full turns per spin (1.0 = one 360°, returns to start heading)
SPIN_SETTLE_S = 1.0             # hover-settle before AND after each spin (per the flight plan)
SPIN_DIR = +1                   # +1 = CCW (yaw-setpoint increasing); -1 = CW. Sync works
                                # either way — DRY_RUN just confirms it actually rotates.
# The ENTRY spin waits until the climb is (near-)complete — gating on "airborne"
# (0.3 m) alone spun it at 0.6 m mid-climb (flight 20260617_131652). Start once the
# drone is within SPIN_CLIMB_TOL_M of the CLIMB_M target, or after the timeout backstop.
SPIN_CLIMB_TOL_M = 0.15         # consider the climb done within this of CLIMB_M
SPIN_CLIMB_TIMEOUT_S = 8.0      # backstop: spin anyway if it never quite settles to height

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
CRUISE_SPEED_MPS = 0.8         # SQUARE carrot speed (moving-setpoint speed between
                               # waypoints) — the horizontal analog of VMAX_UP_MPS.
                               # The circle has its OWN speed (CIRCLE_SPEED_MPS); this
                               # is square_flight.py only. (History: pure D-on-
                               # measurement once made the drone trail the carrot by
                               # ~KD*v/KP — 0.83-0.98 m / 55° phase lag at 0.8 m/s,
                               # flight 20260612_121316; FIXED by the velocity FF, so
                               # the residual lag is now just the FC response delay.
                               # LEASH_M is the stall backstop on top of that.)
CARROT_ACCEL_MPS2 = 2.0        # carrot speed-ramp accel (trapezoidal profile, SHARED
                               # by all carrot missions): the
                               # carrot speeds 0→cruise over cruise/a s and BRAKES to
                               # arrive at every move-segment end with ZERO speed.
                               # Raised 1→2 (2026-06-15) for 3 m/s: at a=1 the ramp
                               # distance v²/2a = 4.5 m would eat most of the 6.3 m
                               # lap; a=2 → 2.25 m, so the 18.85 m (3-lap) arc holds
                               # full speed for ~14 m. The ramp tilt = a/g = 11.5° is
                               # still gentle vs the 55° clamp.
                               # (ramp distance v²/2a = 0.32 m at 0.8). Replaces the
                               # instant 0↔cruise velocity steps that slammed the
                               # tilt cmd into the 15° clamp at every transition and
                               # overswung the drone to 1.4 m/s (flight 20260612_132718).
                               # Also makes the carrot acceleration finite, so accel
                               # feedforward becomes possible later.
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
# LEASH_M / ARRIVE_TOL_M / ARRIVE_TIMEOUT_S / INITIAL_HOVER_S / CARROT_ACCEL_MPS2;
# has its OWN speed (CIRCLE_SPEED_MPS) — the circle banks/yaws far harder than the
# square's strafes, so they're tuned independently.
CIRCLE_SPEED_MPS = 3.0         # CIRCLE carrot speed. 3.0 is ~the fastest the CURRENT
                               # 1 m circle sustains: bank 43° (g-limited, 17° under
                               # the FC limit) is comfy, but the yaw FF hits 293us at
                               # ω=172°/s against the 300°/s FC linear yaw curve — yaw
                               # authority is the real wall. Go faster only on a LARGER
                               # radius (bank ∝ v²/r, yaw rate ∝ v/r — both ease with r).
CIRCLE_RADIUS_M = 2.0          # circle radius AND the forward approach distance
CIRCLE_LAPS = 3                # consecutive laps of the circle (one continuous arc —
                               # no dwells between laps; entry/exit dwells unchanged)
CIRCLE_CW = False              # True = clockwise viewed from above (the carrot goes
                               # forward-point → right → back → left → forward-point);
                               # False = counter-clockwise. CCW with FACE_TANGENT: the
                               # entry pre-rotation turns LEFT 90° from the launch
                               # heading (tangent at the forward point is -X).
SETTLE_S = 2.0                 # hold at the circle entry (clean start), at the circle
                               # EXIT (the dwell gates on the DRONE arriving, so any
                               # phase lag closes the lap before the carrot heads home
                               # — flight 20260612_121316 lost the last 60° without
                               # this), and at the origin on return before landing
CIRCLE_FACE_TANGENT = True     # True: nose follows the direction of travel around the
                               # circle (pre-rotates to the tangent during the entry
                               # dwell, yaws continuously through the lap at ω =
                               # cruise/radius, rotates back to the launch heading
                               # during the exit dwell). False: old strafing behavior
                               # (heading held at launch yaw for the whole course).
YAW_SLEW_DPS = 200.0           # yaw-setpoint slew rate: pre-rotations in dwells ramp
                               # the heading target at this rate (no 90° step → no
                               # saturated yaw command), and it caps tangent-following.
                               # MUST exceed the circle's yaw rate ω = cruise/radius
                               # (172°/s at 3 m/s, r=1.0) or the heading ref lags.
YAW_ARRIVE_TOL_DEG = 5.0       # a dwell with a heading target waits (same arrive_
                               # timeout backstop) until the drone's heading is within
                               # this of the target before its countdown starts — the
                               # circle can't begin until the nose actually points
                               # down the tangent. 5° is tight for the P-only yaw
                               # loop (deadband is 2°) — needs the FC yaw rates
                               # linearized so small commands actually turn the drone

# ============ figure-8 mission (figure8_flight.py only) ========================
# figure8_flight.py: take off + hover AT the figure-8's crossover (the launch
# origin), trace one full figure-8, settle back at the origin, land — the circle's
# twin (same PathMission / carrot / feedforward / leash / dwell machinery). The
# path is a single smooth LEMNISCATE OF BERNOULLI (mission._lemniscate_seg), long
# axis along WORLD X, half-span FIG8_END_X_M, so the far ends pass through
# (x0±FIG8_END_X_M, y0) and the crossover is the launch origin. Reuses LEASH_M /
# ARRIVE_TOL_M / ARRIVE_TIMEOUT_S / INITIAL_HOVER_S / SETTLE_S / CARROT_ACCEL_MPS2,
# exactly like the circle. Held at CLIMB_M above launch (set CLIMB_M=1.0 for the
# spec'd flat 1 m height).
#
# !!! DYNAMICS — READ BEFORE FLYING !!! This replaces the old two-tangent-circles ∞,
# whose curvature flipped sign (+1/R → -1/R) at the crossover — an INSTANT lateral-
# accel reversal (±8 m/s² at 2 m/s, a 16 m/s² step) the drone couldn't track, which
# is why the loops deviated at the centre. The lemniscate's curvature is CONTINUOUS:
# ZERO at the crossover (the drone flies nearly straight through) and peaking at
# κ = 3/FIG8_END_X_M at the FAR ENDS. So the worst-case bank is now at the ends, not
# the centre: centripetal v²·κ = v²·3/FIG8_END_X_M. At FIG8_END_X_M=1.0 that is
# 3·v² m/s² → keep it under the 9 m/s² accel-FF cap (g·tan45°≈9.8 tilt clamp) with
# PID headroom, i.e. v ≲ 1.5 m/s; the default below is 1.2 m/s (peak ≈ 4.3 m/s² →
# 24° bank, like the circle). The tangent yaw rate also peaks at v·3/FIG8_END_X_M
# (206°/s at 1.2 m/s) > the ~147°/s yaw authority, so FIG8_FACE_TANGENT must stay
# False unless you slow down further. DRY-RUN and preview.py first.
FIG8_END_X_M = 2.0             # centre→end distance along world X; loop radius R is
                               # half this. Far ends pass through (±FIG8_END_X_M, 0).
FIG8_LAPS = 2                  # full figure-8 traversals (each = right loop + left
                               # loop). The whole run flows at cruise; only the first
                               # loop ramps up and only the last brakes to the home dwell.
FIG8_CW = False                # sense of the lemniscate (which loop is traced first,
                               # viewed from above): False = left loop first; True =
                               # right loop first (flips the sign of y). Both are one
                               # continuous smooth ∞ through the crossover.
FIG8_FACE_TANGENT = True      # nose follows the travel direction. KEEP FALSE at the
                               # default speed/size: the yaw rate peaks at v·3/
                               # FIG8_END_X_M (206°/s at 1.2 m/s) — over the ~147°/s yaw
                               # authority. Only enable if FIG8_SPEED_MPS is low enough
                               # that v·3/FIG8_END_X_M < YAW_SLEW_DPS.
FIG8_SPEED_MPS = 2.0           # carrot speed. LOWER than the circle's cruise on purpose:
                               # the lemniscate's peak curvature is 3/FIG8_END_X_M at the
                               # ends, so peak bank ∝ v²; 1.2 m/s keeps it ~4.3 m/s² (24°,
                               # like the circle). Raise toward ~1.5 m/s max (see DYNAMICS).

# ============ loop rate (shared) ============
TX_HZ = channels.TX_HZ         # 50 Hz, same as the data logger
