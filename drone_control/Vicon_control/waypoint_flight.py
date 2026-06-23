#!/usr/bin/env python3
"""Vicon autonomous WAYPOINT course: take off → fly the points in config.WAYPOINTS → land.

Built directly on vicon_hover.py — SAME arming, controller, recording, status line,
sync-spin bookends and failsafes — it just feeds the shared flight loop a
WaypointMission built from config.WAYPOINTS instead of a static hover.

Define the course in config.py → WAYPOINTS: a list of points in ABSOLUTE VICON WORLD
coordinates (the same x/y/z you read in Vicon Tracker for your gates):
  (x_m, y_m, z_m, dwell_s, "label")
Points are used as-is — NOT rotated or offset by the launch pose — so a gate at world
x=-2 is exactly x=-2. z=None → CLIMB_M above launch. With WAYPOINT_FACE_PATH=True the
NOSE FOLLOWS THE PATH (yaws to point along each leg, pre-rotating to the next leg
during each dwell); set it False to strafe with the nose held at the launch yaw. The
carrot starts at the launch position and crawls to WP0 first, so make WP0 your takeoff
point. Tune CRUISE_SPEED_MPS, ARRIVE_TOL_M, ARRIVE_TIMEOUT_S, LEASH_M, WAYPOINT_FACE_PATH
(and per-point dwell_s) in config.py.

The setpoint is a crawling "carrot" between points (CRUISE_SPEED_MPS), not a step, so
each leg is a smooth translation; each hold begins only once the drone has arrived
(ARRIVE_TOL_M). SPACEBAR / disarm / low batt / Vicon loss abort to a controlled
landing at any time, exactly as in vicon_hover.

Arming + record triggers are identical to vicon_hover.py / the data logger:
  - ARM switch (AUX3): master enable; flick to DISARM = instant kill.
  - RECORD switch (AUX2): with the drone armed, flip HIGH to launch + record.

Dry-run FIRST (prints the world waypoint table at launch; never opens Ranger /
never arms) — verify the course geometry + directions before going live:
  .venv/bin/python drone_control/Vicon_control/waypoint_flight.py --dry-run
  .venv/bin/python drone_control/Vicon_control/waypoint_flight.py            # LIVE
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_DRONE_CONTROL = os.path.dirname(HERE)
if _DRONE_CONTROL not in sys.path:
    sys.path.insert(0, _DRONE_CONTROL)

from Vicon_control import vicon_hover                       # noqa: E402
from Vicon_control.mission import build_waypoint_mission    # noqa: E402


def main():
    args = vicon_hover.build_parser(description=__doc__).parse_args()
    vicon_hover.run(args, make_mission=build_waypoint_mission)


if __name__ == "__main__":
    main()
