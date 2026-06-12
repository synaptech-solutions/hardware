#!/usr/bin/env python3
"""Vicon autonomous CIRCLE course: take off → forward → full circle → home → land.

Built on vicon_hover.py exactly like square_flight.py — SAME arming, controller,
recording, status line and failsafes — it just feeds the shared flight loop a
PathMission (a carrot that FLOWS along a line+arc+line path) instead of a static
hover. The course (relative to the LAUNCH pose; you start at the origin, nose +Y),
heading HELD at the launch yaw throughout:
  1. climb to config.CLIMB_M and hover config.INITIAL_HOVER_S
  2. fly forward config.CIRCLE_RADIUS_M to the circle (centred on the launch origin),
     settle config.SETTLE_S
  3. trace ONE FULL CIRCLE of radius config.CIRCLE_RADIUS_M, direction config.CIRCLE_CW
     (True = clockwise from above), at config.CRUISE_SPEED_MPS
  4. return to the origin, settle config.SETTLE_S
  5. land → cut → disarm → save → exit
All tunables (CIRCLE_RADIUS_M, CIRCLE_CW, CRUISE_SPEED_MPS, SETTLE_S, …) live in
config.py. The setpoint is a continuous crawling carrot, so the circle is smooth;
the drone trails it ~0.3 m at 0.4 m/s (so it flies a slightly smaller, phase-lagged
circle — fine for now, tighten later with velocity feedforward).

SPACEBAR / disarm / low batt / Vicon loss abort to a controlled landing at any
point, exactly as in vicon_hover. Arm + record triggers are identical.

Dry-run FIRST (prints the segment plan + circle geometry at launch; never arms) —
verify the direction + radius before going live:
  .venv/bin/python drone_control/Vicon_control/circle_flight.py --dry-run
  .venv/bin/python drone_control/Vicon_control/circle_flight.py            # LIVE
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_DRONE_CONTROL = os.path.dirname(HERE)
if _DRONE_CONTROL not in sys.path:
    sys.path.insert(0, _DRONE_CONTROL)

from Vicon_control import vicon_hover                      # noqa: E402
from Vicon_control.mission import build_circle_mission     # noqa: E402


def main():
    args = vicon_hover.build_parser(description=__doc__).parse_args()
    vicon_hover.run(args, make_mission=build_circle_mission)


if __name__ == "__main__":
    main()
