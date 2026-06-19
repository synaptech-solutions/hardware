#!/usr/bin/env python3
"""Vicon autonomous HELIX course: take off → forward → climbing spiral → home → land.

The CIRCLE, but rising. Built on vicon_hover.py exactly like circle_flight.py — SAME
arming, controller, recording, status line and failsafes — it just feeds the shared
flight loop a PathMission whose circular arc carries a z-ramp, so the carrot spirals
UP while it laps. The course (relative to the LAUNCH pose; you start at the origin):
  1. climb to config.CLIMB_M and hover config.INITIAL_HOVER_S
  2. fly forward config.HELIX_RADIUS_M to the circle (centred on the launch origin),
     settle config.SETTLE_S — and (HELIX_FACE_TANGENT) pre-rotate the nose to the
     spiral's first tangent; the dwell WAITS for the nose to get there
  3. trace config.HELIX_LAPS turns of radius config.HELIX_RADIUS_M, direction
     config.HELIX_CW, at config.HELIX_SPEED_MPS, RISING config.HELIX_HEIGHT_M total
     (linearly with arc length) — nose following the travel direction (or strafing
     at the launch yaw if HELIX_FACE_TANGENT=False)
  4. settle config.SETTLE_S at the top, return to the origin (at the top height),
     settle, land → cut → disarm → save → exit
Bird's-eye it is identical to the circle; only z changes. All tunables (HELIX_RADIUS_M,
HELIX_LAPS, HELIX_HEIGHT_M, HELIX_CW, HELIX_FACE_TANGENT, HELIX_SPEED_MPS, + the shared
SETTLE_S / CARROT_ACCEL_MPS2 / YAW_SLEW_DPS …) live in config.py.

!!! The laps top out at launch + CLIMB_M + HELIX_HEIGHT_M — CHECK YOUR CEILING. The
climb rate must stay under the altitude loop's VMAX_UP_MPS (see the config note). !!!

SPACEBAR / disarm / low batt / Vicon loss abort to a controlled landing at any point,
exactly as in vicon_hover. Arm + record triggers are identical.

Dry-run FIRST (prints the segment plan + the z-ramp at launch; never arms):
  .venv/bin/python drone_control/Vicon_control/helix_flight.py --dry-run
  .venv/bin/python drone_control/Vicon_control/helix_flight.py            # LIVE
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_DRONE_CONTROL = os.path.dirname(HERE)
if _DRONE_CONTROL not in sys.path:
    sys.path.insert(0, _DRONE_CONTROL)

from Vicon_control import vicon_hover                      # noqa: E402
from Vicon_control.mission import build_helix_mission      # noqa: E402


def main():
    args = vicon_hover.build_parser(description=__doc__).parse_args()
    vicon_hover.run(args, make_mission=build_helix_mission)


if __name__ == "__main__":
    main()
