"""Air75 CRSF channel map, µs levels, camera + loop rate — the SINGLE SOURCE OF TRUTH.

These were verified on the Air75 (Betaflight dump + Configurator + bench tests,
2026-06-04/09) and consolidated here so the data-logging pipeline and both
controllers stay in lockstep. NOTE: the old apriltag `controller_v2/config.py`
carried `CH_ARM=5`/`CH_MODE=6` from a *Meteor75* dump — those are WRONG on the
Air75 (arm is AUX3=idx 6; mode is AUX4=idx 7). Use the values here.

A CRSF RC_CHANNELS frame is 16 channels; gimbals are the standard AETR layout on
indices 0-3, AUX1=4, AUX2=5, AUX3=6, AUX4=7.
"""

# --- gimbals (AETR) ---
CH_ROLL, CH_PITCH, CH_THR, CH_YAW = 0, 1, 2, 3

# --- neutral / idle ---
NEUTRAL_US = 1500
IDLE_THR_US = 1000

# Betaflight refuses to arm while throttle > min_check (~1050µs) — the THROTTLE
# arming-disable flag. The throttle stick / commanded throttle must be at the
# bottom to arm. (Betaflight docs, verified 2026-06-04.)
ARM_THROTTLE_MAX_US = 1050

# --- ARM: AUX3 (CRSF array index 6) ---
# Armed 1508µs / disarmed 1000µs (user-verified 2026-06-04). config.CH_ARM=5 was
# the wrong Meteor75 value and the FC ignored it.
ARM_CH = 6
ARM_ARMED_US = 1508
ARM_DISARMED_US = 1000

# --- blackbox / record switch: AUX2 (CRSF array index 5), 3-position ---
#   HIGH 2000µs → START blackbox logging  (also the laptop record trigger)
#   MID  1500µs → nothing
#   LOW  1000µs → ERASE blackbox dataflash
# Each detent MUST send its own distinct value — a binary relay sent LOW (erase)
# for the middle detent and wrongly erased. send-disarm paths must send MID, never
# LOW, or every disarm/exit would erase the log.
AUX2_CH = 5
AUX2_HIGH_US = 2000
AUX2_MID_US = 1500
AUX2_LOW_US = 1000

# --- flight-mode switch: AUX4 (CRSF array index 7), 3-position ---
# Betaflight ranges: ANGLE centered ~1500, HORIZON 1700-2100. So LOW→ACRO (1000,
# no self-level), MID→ANGLE (1500), HIGH→HORIZON (1900, safely inside 1700-2100).
MODE_CH = 7
MODE_ACRO_US = 1000
MODE_ANGLE_US = 1500
MODE_HORIZON_US = 1900

# --- camera (Cam Link 4K) ---
# WIDTH/HEIGHT must match apriltag_control/camera_setup/camera_calibration.npz for
# the AprilTag controller; the data logger + Vicon controller use the camera only
# for raw video capture. The Cam Link can re-enumerate 4<->5.
DEVICE_INDEX = 4
WIDTH, HEIGHT = 1280, 720

# --- loop rate ---
# 100 Hz (was 50, raised 2026-06-12). Within the link budget: the ELRS link runs
# 150 Hz over-the-air (rf_mode 24 in flight telemetry, LQ 99-100), the laptop→
# Ranger CRSF input rate is independent of the OTA rate, and an RC frame is only
# ~0.62 ms at 420k baud (~6% bus). Vicon delivers ~99 Hz, so at 100 Hz the loop
# consumes nearly every pose sample instead of every other one. The flight loops
# pace on absolute deadlines and held 20.00±0.26 ms at 50 Hz with zero overruns —
# verify the first 100 Hz session's commands.csv dt the same way.
TX_HZ = 100.0
