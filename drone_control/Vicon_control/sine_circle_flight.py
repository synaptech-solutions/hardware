#!/usr/bin/env python3
"""Vicon autonomous SINE-CIRCLE: take off → forward → circle with a bobbing (sine)
height → home → land.

The CIRCLE, but the altitude oscillates like a sine wave as it laps. Built on
vicon_hover.py exactly like circle_flight.py / helix_flight.py — SAME arming,
controller, recording, status line and failsafes — it just feeds the shared flight
loop a PathMission whose circular arc carries a SINE z-profile, so the carrot rides
up and down while it goes around. The course (relative to the LAUNCH pose):
  1. climb to config.CLIMB_M and hover config.INITIAL_HOVER_S
  2. fly forward config.SINE_RADIUS_M to the circle (centred on the launch origin),
     settle config.SETTLE_S — and (SINE_FACE_TANGENT) pre-rotate to the first tangent
  3. trace config.SINE_LAPS laps (default 5) of radius config.SINE_RADIUS_M, direction
     config.SINE_CW, at config.SINE_SPEED_MPS, while the height rides
        z = (launch+CLIMB_M) + SINE_AMP_M · sin(2π · SINE_CYCLES_PER_LAP · lap_fraction)
     — i.e. SINE_CYCLES_PER_LAP up-and-down bobs per lap, amplitude SINE_AMP_M
  4. settle config.SETTLE_S at the exit (back at the hover height), return to the
     origin, settle, land → cut → disarm → save → exit
Bird's-eye it is identical to the circle; only z oscillates. All tunables (SINE_RADIUS_M,
SINE_LAPS, SINE_AMP_M, SINE_CYCLES_PER_LAP, SINE_CW, SINE_FACE_TANGENT, SINE_SPEED_MPS,
+ the shared SETTLE_S / CARROT_ACCEL_MPS2 / YAW_SLEW_DPS …) live in config.py.

!!! z spans launch + CLIMB_M ± SINE_AMP_M — keep SINE_AMP_M < CLIMB_M so the trough
stays above the ground, and keep the peak vertical speed under VMAX_UP_MPS (see the
config note). DRY-RUN / preview.py first. !!!

SPACEBAR / disarm / low batt / Vicon loss abort to a controlled landing at any point,
exactly as in vicon_hover. Arm + record triggers are identical.

Dry-run FIRST (prints the segment plan + the z bob range at launch; never arms):
  .venv/bin/python drone_control/Vicon_control/sine_circle_flight.py --dry-run
  .venv/bin/python drone_control/Vicon_control/sine_circle_flight.py            # LIVE
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_DRONE_CONTROL = os.path.dirname(HERE)
if _DRONE_CONTROL not in sys.path:
    sys.path.insert(0, _DRONE_CONTROL)

from Vicon_control import vicon_hover                          # noqa: E402
from Vicon_control.mission import build_sine_circle_mission    # noqa: E402


def main():
    args = vicon_hover.build_parser(description=__doc__).parse_args()
    vicon_hover.run(args, make_mission=build_sine_circle_mission)


if __name__ == "__main__":
    main()
