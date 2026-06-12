#!/usr/bin/env python3
"""Vicon autonomous SQUARE course: take off → fly a 2 m square → land.

Built directly on vicon_hover.py — SAME arming, controller, recording, status
line and failsafes — it just feeds the shared flight loop a WaypointMission
instead of a static hover. The course (relative to the LAUNCH pose; you start at
the origin, nose pointing +Y), heading HELD at the launch yaw throughout:
  1. climb to config.CLIMB_M and hover config.INITIAL_HOVER_S
  2. forward config.LEG_M, hold config.DWELL_S
  3. right   config.LEG_M, hold config.DWELL_S
  4. back    config.LEG_M, hold config.DWELL_S
  5. left    config.LEG_M, hold config.DWELL_S   (back over the origin)
  6. land → cut → disarm → save → exit
All tunables (LEG_M, CRUISE_SPEED_MPS, DWELL_S, ARRIVE_TOL_M, …) live in config.py.

The setpoint is a crawling "carrot" between waypoints (CRUISE_SPEED_MPS), not a
step, so each leg is a smooth, gentle translation; each vertex hold begins only
once the drone has actually arrived (ARRIVE_TOL_M). SPACEBAR / disarm / low batt /
Vicon loss abort to a controlled landing at any time, exactly as in vicon_hover.

Arming + record triggers are identical to vicon_hover.py / the data logger:
  - ARM switch (AUX3): master enable; flick to DISARM = instant kill.
  - RECORD switch (AUX2): with the drone armed, flip HIGH to launch + record.

Dry-run FIRST (prints the world waypoint table at launch; never opens Ranger /
never arms) — verify the square geometry + directions before going live:
  .venv/bin/python drone_control/Vicon_control/square_flight.py --dry-run
  .venv/bin/python drone_control/Vicon_control/square_flight.py            # LIVE
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_DRONE_CONTROL = os.path.dirname(HERE)
if _DRONE_CONTROL not in sys.path:
    sys.path.insert(0, _DRONE_CONTROL)

from Vicon_control import vicon_hover                      # noqa: E402
from Vicon_control.mission import build_square_mission     # noqa: E402


def main():
    args = vicon_hover.build_parser(description=__doc__).parse_args()
    vicon_hover.run(args, make_mission=build_square_mission)


if __name__ == "__main__":
    main()
