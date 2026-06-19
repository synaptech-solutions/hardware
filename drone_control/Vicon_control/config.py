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
CLIMB_M = 3.0

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
# Rate curve (Air75, rateprofile 0, ACTIVE): rates_type=ACTUAL, expo=0 → LINEAR.
# Bumped 2026-06-19 from 70°/s to 800°/s full stick for a more aggressive envelope —
# FC setting: roll_rc_rate = roll_srate = pitch_rc_rate = pitch_srate = 80, *_expo = 0
# (ACTUAL: max°/s = srate×10; rc_rate=srate & expo=0 ⇒ linear). So a commanded body
# rate → us at 511.5/800 = 0.639 us per °/s.
#   APPLY THE FC CHANGE AND ACRO_MAX_RATE_DPS BELOW TOGETHER — they MUST match, or the
#   leveling loop sends the wrong us for the curve. KP_ANGLE_RATE does NOT change with
#   the rate ceiling: the law commands a physical °/s (KP·err) and ACRO_RATE_US_PER_DPS
#   converts it through the curve, so the hover levels identically — only the headroom
#   (and the µs-per-°/s resolution) changes. KEEP IT LINEAR (expo 0) or this straight-
#   line conversion breaks. (Re-tuning for aggressive moves is a SEPARATE later step.)
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
ACRO_MAX_RATE_DPS = 800.0      # clamp on the commanded rate = the FC ACTUAL-rate
                               # ceiling. MUST equal the FC's linear full-stick rate
                               # (rc_rate=srate=80 → 800°/s); full stick at the clamp.
ACRO_RATE_US_PER_DPS = STICK_FULL_DEFLECTION_US / ACRO_MAX_RATE_DPS   # 0.639 (511.5/800)

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
# FIX (Betaflight CLI): linearize yaw. Bumped 2026-06-19 from 300 to 700°/s (to match
# the 800°/s roll/pitch envelope) —
#     set yaw_rc_rate = 70 ; set yaw_srate = 70 ; set yaw_expo = 0 ; save
# YAW_LINEAR_MAX_DPS documents that FC setting; the feedforward gain is derived from
# it. FLY THE FC CHANGE AND THIS CONFIG TOGETHER. NOTE: raising the curve makes the
# µs-based yaw PID below 700/300 = 2.33× HOTTER per µs, so those gains were scaled by
# 300/700 to keep the heading-hold behavior (and the °/s authority) IDENTICAL — this
# is a curve-matching change, NOT the aggressive re-tune.
YAW_LINEAR_MAX_DPS = 700.0     # FC linear yaw rate: full stick (511.5us) = 700°/s
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
# Scaled ×300/700 (2026-06-19) when the yaw curve went 300→700°/s, so the heading→
# yaw-rate loop gain is unchanged (the FC now does 2.33× the °/s per µs). Pre-bump
# values were KP 200 / KI 30 / MAX_I 30 / KD 25 at the 300°/s curve.
KP_YAW_US_PER_RAD = 85.7       # was 200 @300°/s → 85.7 @700°/s (same 117°/s per rad)
KI_YAW_US_PER_RAD_S = 12.9     # was 30
MAX_YAW_I_US = 13              # integral contribution cap (windup guard); was 30
KD_YAW_US_PER_RAD_PER_S = 10.7 # damping; was 25
YAW_RATE_LPF_S = 0.10          # 1-pole LPF on the differenced Vicon yaw rate the D
                               # term uses (50 Hz diff of ~0.2° noise ⇒ ~14°/s rate
                               # noise raw — filter before it reaches KD)
MAX_YAW_US = 193               # 193us = 264°/s authority on the 700°/s curve
                               # (193/511.5×700). Scaled from 450 @300°/s — SAME 264°/s
                               # authority, just fewer µs (the physical ceiling is now
                               # full stick 511.5us = 700°/s). Raise toward full stick
                               # during the aggressive re-tune if you want >264°/s yaw.
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

# ============ waypoint mission (waypoint_flight.py) =============================
# waypoint_flight.py flies the course you define in WAYPOINTS below. Points are in
# ABSOLUTE VICON WORLD coordinates — the SAME (x, y, z) you read in Vicon Tracker for
# your gates/obstacles — NOT relative to the drone. They are NOT rotated or offset by
# the launch pose, so a gate at world x=-2 is exactly x=-2 here. Heading is HELD at
# the launch yaw the whole time (the drone strafes between points, nose fixed). The
# setpoint is a crawling "carrot" at CRUISE_SPEED_MPS, so each leg is a smooth
# translation; the hold at a point begins only once the drone has ARRIVED
# (ARRIVE_TOL_M). vicon_hover.py and circle/figure-8 ignore WAYPOINTS.
#
# Each entry: (x_m, y_m, z_m, dwell_s, "label")
#     x_m, y_m   world position (Vicon frame), meters
#     z_m        world height (meters); None → CLIMB_M above the launch altitude
#     dwell_s    hold time once arrived (carrot parks + drone settles)
#     label      shown in the dry-run table + status line
# The carrot starts at the drone's actual launch position and crawls to WP0 first, so
# make WP0 your takeoff/hover point (near where you launch) for a clean straight climb.
# DRY-RUN first — it prints the world point table so you can check it matches your gates.
#
# Example below: an hourglass through gates at world x=±2, y=0, flown at z=1.0 m.
WAYPOINTS = [
    ( 0.0,  0.0, 1.0, 3.0, "takeoff/hover"),   # start/center
    (-2.0,  1.0, 1.0, 3.0, "left-top"),        # ┐ down the left edge → through the
    (-2.0, -1.0, 1.0, 3.0, "left-bottom"),     # ┘   left gate at (-2, 0)
    ( 2.0,  1.0, 1.0, 3.0, "right-top"),        # diagonal across → through center (0,0)
    ( 2.0, -1.0, 1.0, 3.0, "right-bottom"),    #   then down the right edge → right gate (2,0)
    ( 0.0,  0.0, 1.0, 3.0, "home"),            # back to center, settle, then land

    (-2.0,  1.0, 1.0, 3.0, "left-top"),        # ┐ down the left edge → through the
    (-2.0, -1.0, 1.0, 3.0, "left-bottom"),     # ┘   left gate at (-2, 0)
    ( 2.0,  1.0, 1.0, 3.0, "right-top"),        # diagonal across → through center (0,0)
    ( 2.0, -1.0, 1.0, 3.0, "right-bottom"),    #   then down the right edge → right gate (2,0)
    ( 0.0,  0.0, 1.0, 3.0, "home"),            # back to center, settle, then land

    (-2.0,  1.0, 1.0, 3.0, "left-top"),        # ┐ down the left edge → through the
    (-2.0, -1.0, 1.0, 3.0, "left-bottom"),     # ┘   left gate at (-2, 0)
    ( 2.0,  1.0, 1.0, 3.0, "right-top"),        # diagonal across → through center (0,0)
    ( 2.0, -1.0, 1.0, 3.0, "right-bottom"),    #   then down the right edge → right gate (2,0)
    ( 0.0,  0.0, 1.0, 3.0, "home"),            # back to center, settle, then land
]
WAYPOINT_FACE_PATH = True      # True: NOSE FOLLOWS THE PATH — the drone yaws to point
                               # along each leg's direction of travel and pre-rotates to
                               # the next leg during each dwell (slewed at YAW_SLEW_DPS,
                               # gated by YAW_ARRIVE_TOL_DEG, same as the circle). False:
                               # hold the launch heading the whole time (pure strafing).
CRUISE_SPEED_MPS = 3.0         # WAYPOINT carrot speed (moving-setpoint speed between
                               # points) — the horizontal analog of VMAX_UP_MPS.
                               # The circle has its OWN speed (CIRCLE_SPEED_MPS); this
                               # is waypoint_flight.py only. (History: pure D-on-
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
# (Waypoint dwells are per-point in WAYPOINTS above.) INITIAL_HOVER_S is the
# takeoff-hover settle for the circle/figure-8 missions (the waypoint mission uses
# WP0's own dwell instead).
INITIAL_HOVER_S = 3.0          # circle/figure-8 takeoff-hover settle before the first leg
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
CIRCLE_SPEED_MPS = 3.5         # CIRCLE carrot speed. 3.0 is ~the fastest the CURRENT
                               # 1 m circle sustains: bank 43° (g-limited, 17° under
                               # the FC limit) is comfy, but the yaw FF hits 293us at
                               # ω=172°/s against the 300°/s FC linear yaw curve — yaw
                               # authority is the real wall. Go faster only on a LARGER
                               # radius (bank ∝ v²/r, yaw rate ∝ v/r — both ease with r).
CIRCLE_RADIUS_M = 2.5          # circle radius AND the forward approach distance
CIRCLE_LAPS = 5                # consecutive laps of the circle (one continuous arc —
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
FIG8_LAPS = 5                  # full figure-8 traversals (each = right loop + left
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

# ============ helix mission (helix_flight.py only) =============================
# helix_flight.py: the CIRCLE, but the carrot climbs while it laps. Take off + hover
# at CLIMB_M, fly forward HELIX_RADIUS_M to the circle (centred on the launch
# origin), trace HELIX_LAPS turns while rising HELIX_HEIGHT_M total, settle at the
# top, return, land. Bird's-eye it IS the circle (same machinery, same FACE_TANGENT/
# CW behavior); only z ramps along the laps. Reuses SETTLE_S / INITIAL_HOVER_S /
# LEASH_M / ARRIVE_TOL_M / ARRIVE_TIMEOUT_S / CARROT_ACCEL_MPS2 / YAW_SLEW_DPS.
#
# ALTITUDE: the laps span world z from z_base = launch+CLIMB_M to
# z_top = z_base + HELIX_HEIGHT_M  →  with CLIMB_M=1.0 and the default below the
# drone tops out ~1.6 m above launch. CHECK YOUR CEILING and DRY-RUN/preview first.
# The climb rate the drone must hold is HELIX_HEIGHT_M·HELIX_SPEED_MPS /
# (2π·HELIX_RADIUS_M·HELIX_LAPS) — keep it under VMAX_UP_MPS (0.50) or the drone
# lags the rising carrot (default: 0.6·2.0/(2π·1.5·3) ≈ 0.04 m/s, very gentle).
HELIX_RADIUS_M = 2.0           # spiral radius AND the forward approach distance
HELIX_LAPS = 5                 # number of turns climbed (one continuous rising arc)
HELIX_HEIGHT_M = 3.5           # TOTAL climb over the laps (z_base → z_base+this),
                               # linear with arc length. The helix-specific knob.
HELIX_CW = False               # True = clockwise viewed from above; False = CCW
                               # (same convention as CIRCLE_CW)
HELIX_FACE_TANGENT = True      # nose follows travel direction (yaw rate = v/R =
                               # 1.33 rad/s = 76°/s at the defaults, well under the
                               # ~264°/s yaw authority). False = strafe at launch yaw.
HELIX_SPEED_MPS = 2.5          # carrot ground speed along the spiral (its OWN knob,
                               # like CIRCLE_SPEED_MPS). Bank ∝ v²/R, yaw rate ∝ v/R.

# ============ sine-circle mission (sine_circle_flight.py only) =================
# sine_circle_flight.py: the CIRCLE, but the HEIGHT oscillates like a sine wave while
# it laps. Take off + hover at CLIMB_M, fly forward SINE_RADIUS_M to the circle
# (centred on the launch origin), trace SINE_LAPS laps while z rides
# z_mid + SINE_AMP_M·sin(2π·SINE_CYCLES_PER_LAP·lap_fraction), settle, return, land.
# Bird's-eye it IS the circle (same machinery / FACE_TANGENT / CW); only z bobs.
# Reuses SETTLE_S / INITIAL_HOVER_S / LEASH_M / ARRIVE_TOL_M / CARROT_ACCEL_MPS2 / etc.
#
# ALTITUDE: z oscillates about z_mid = launch + CLIMB_M with amplitude SINE_AMP_M →
# spans [z_mid - SINE_AMP_M, z_mid + SINE_AMP_M]. KEEP SINE_AMP_M < CLIMB_M so the
# trough stays above the ground (with CLIMB_M=1.0 the default 0.3 → z in [0.7, 1.3]).
# Peak vertical speed = SINE_AMP_M·SINE_CYCLES_PER_LAP·SINE_SPEED_MPS / SINE_RADIUS_M
# — keep it under VMAX_UP_MPS (0.50) or the drone lags the bob (default 0.3·1·2/2 =
# 0.3 m/s, fine). DRY-RUN + preview.py first.
SINE_RADIUS_M = 2.0            # circle radius AND the forward approach distance
SINE_LAPS = 5                 # laps flown while the height oscillates (you asked for 5)
SINE_AMP_M = 1.0              # height-oscillation amplitude (peak above/below z_mid).
                               # MUST be < CLIMB_M (keeps the trough above ground)
SINE_CYCLES_PER_LAP = 3.0     # sine humps (full up-down periods) per lap. 1 = one
                               # rise+dip per revolution; 2 = two, etc.
SINE_CW = False               # True = clockwise from above; False = CCW (like CIRCLE_CW)
SINE_FACE_TANGENT = True      # nose follows travel direction (like the circle); False
                               # = strafe at the launch yaw
SINE_SPEED_MPS = 2.0          # carrot ground speed along the circle (its OWN knob)

# ============ flip maneuver (flip_flight.py = roll · pitch_flip_flight.py = pitch) ==
# hover → ONE open-loop 360° body flip about the ROLL or PITCH axis (dead-reckoned) →
# the controller catches + re-stabilizes at the hover point → hold stable
# FLIP_STABLE_HOLD_S → land. ACRO-ONLY (the flip is a RATE command; Angle mode caps at
# angle_limit 60°, so it can't rotate past level). OPEN LOOP: for 360/FLIP_RATE_DPS
# seconds the leveling loop is bypassed and a fixed rate stick is sent on the chosen
# axis — the FC's inner rate loop tracks it; then the leveling + position loop cleans
# up the rest. The AXIS is chosen by the SCRIPT (flip_flight.py → roll,
# pitch_flip_flight.py → pitch); both reuse every knob below.
#   FLY WITH ALTITUDE MARGIN (CLIMB_M ≥ ~3): MEASURED on the first good roll
#   (20260619_122711) the drone dropped ~1.9 m (2.98 → 1.05 m), MOSTLY AFTER the flip
#   while the altitude loop arrested the downward velocity built while inverted. That
#   needs both the margin AND the recovery-throttle boost below. The flip only fires
#   once stable at hover height; the leveling sign is already proven by the acro hover,
#   so there's no new hand/DRY_RUN direction check (the flip direction is cosmetic — a
#   full 360° returns to level either way).
FLIP_SEQUENCE = ["roll", "pitch", "roll", "pitch"]       # the flips to do, in order — each "roll" or "pitch". The
                               # drone recovers + re-stabilizes at hover BETWEEN each
                               # (FLIP_BETWEEN_STABLE_S). E.g. ["roll","pitch","roll","pitch"]
                               # = roll, settle, pitch, settle, roll, settle, pitch, land.
                               # Default ["roll"] = the proven single flip; extend once the
                               # inverted-throttle-cut below is confirmed on a single flip.
FLIP_RATE_DPS = 800.0          # commanded body rate during the flip = FULL STICK (=
                               # ACRO_MAX_RATE_DPS): flip for 360/800 = 0.45 s → one turn.
                               # The channel clamps at 2000us (±500), just shy of the
                               # 511.5us full-deflection, so the stick saturates at
                               # ~782°/s → ~0.45 s ≈ 352°; the controller catches the
                               # remainder (the 20260619_122711 roll recovered to level
                               # cleanly at scale 1.0).
FLIP_DURATION_SCALE = 1.0      # flip time = (360/FLIP_RATE_DPS)·this. TRIM after a flight:
                               # >1 if it UNDER-rotates (ends short), <1 if it OVER-rotates
                               # past level. (Dead reckoning ignores the FC rate ramp +
                               # stick clamp, so 360° is approximate by design.)
FLIP_DIR = +1                  # +1 = positive stick (roll RIGHT / pitch FORWARD nose-down),
                               # -1 = the other way. Cosmetic — a full 360° ends level.
FLIP_THROTTLE_BOOST_US = 80    # added to the LEARNED hover throttle while UPRIGHT during
                               # the flip (entry/exit) to offset lift lost while tilted.
# THROTTLE WHILE INVERTED: when the body has rotated past FLIP_INVERTED_TILT_DEG from
# level, its thrust points partly DOWN — so holding throttle there drives the drone
# into the ground (the main cause of the flip sink). Instead CUT throttle to
# FLIP_INVERTED_THROTTLE_US through that window; the FC's rate PID (airmode) keeps the
# flip spinning and the drone coasts ballistically. Dead-reckoned from the flip clock,
# so it's right even through a Vicon dropout. If the flip stalls mid-rotation at idle,
# raise FLIP_INVERTED_THROTTLE_US a little (gives the mix more authority).
FLIP_INVERTED_TILT_DEG = 90.0  # cut throttle once rotated past this from level (90° =
                               # thrust horizontal; >90° = pointing down). Lower = cut
                               # for a wider window (earlier/later); 90 = the inverted half.
FLIP_INVERTED_THROTTLE_US = channels.IDLE_THR_US  # throttle while inverted (idle = no
                               # thrust = "no throttle command", as requested).
# RECOVERY throttle (AFTER the flip): the hover altitude loop is deliberately gentle
# (THR_CLIMB_TRIM_US=100, VMAX_UP_MPS=0.5) and arrests the ~2 m flip sink too slowly.
# During the recover phase the loop uses these HIGHER caps so it punches throttle hard
# to catch the drop, then reverts to the gentle hover tuning. These ONLY bind during
# the big post-flip transient — hover + takeoff are untouched.
FLIP_RECOVER_CLIMB_TRIM_US = 300  # max +us above hover while recovering (vs 100 hover) →
                               # hover≈1356 + 300 = ~1656us, just under MAX_THROTTLE_US.
                               # Raise toward ~340 if it still sinks (keep hover+this < 1700).
FLIP_RECOVER_VMAX_MPS = 2.0    # climb-speed cap while recovering (vs 0.5 hover), so the
                               # climb-back isn't throttled down once the sink is arrested.
FLIP_RECOVER_KV_UP = 80.0      # throttle us per (m/s) of velocity error while recovering
                               # (vs 40 hover) — twice the punch per m/s of sink, so it
                               # actually reaches the higher trim cap at real fall speeds.
FLIP_PREROLL_STABLE_S = 1.5    # must hover stable (within the tols below) this long
                               # before the flip fires
FLIP_BETWEEN_STABLE_S = 2.0    # BETWEEN flips in a sequence: after recovering, hold this
                               # long of steady hover (back at the hover point) before the
                               # next flip fires — "enough time to stabilize and recover".
FLIP_STABLE_HOLD_S = 3.0       # after the LAST flip, hold stable this long before landing
FLIP_STABLE_POS_M = 0.4        # "stable" = within this of the hover point (horiz + vert)
FLIP_STABLE_TILT_DEG = 20.0    # AND roll/pitch within this of level
FLIP_RECOVER_TIMEOUT_S = 8.0   # backstop: land after this long post-flip even if never
                               # fully "stable", so it can't hang in the air

# ============ loop rate (shared) ============
TX_HZ = channels.TX_HZ         # 50 Hz, same as the data logger
