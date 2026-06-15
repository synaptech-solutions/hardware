#!/usr/bin/env python3
"""Vicon autonomous FIGURE-8 course: take off → full figure-8 → land.

Built on vicon_hover.py exactly like circle_flight.py / square_flight.py — SAME
arming, controller, recording, status line and failsafes — it just feeds the
shared flight loop a PathMission (a carrot that FLOWS along a smooth ∞ path)
instead of a static hover. The course (relative to the LAUNCH pose; you start at
the origin, which IS the figure-8's crossover):
  1. climb to config.CLIMB_M and hover config.INITIAL_HOVER_S at the origin
  2. trace config.FIG8_LAPS full figure-8(s) — two circles tangent at the origin,
     long axis along world X, radius config.FIG8_END_X_M/2, so the ends pass
     through (±config.FIG8_END_X_M, 0). The loops alternate sense (config.FIG8_CW
     sets the first), so the path tangent is continuous at the crossover and the
     carrot flows through at config.FIG8_SPEED_MPS — no stop at the centre. With
     config.FIG8_FACE_TANGENT the nose follows the travel direction (default off:
     strafe at the launch heading — see the DYNAMICS note in config.py)
  3. settle config.SETTLE_S back at the origin, then land
Unlike the circle there is NO forward approach leg: the crossover is the launch
point, so the drone is already on the path at takeoff. All tunables live in
config.py (the FIG8_* block + the shared carrot params).

!!! The figure-8 loops are HALF the circle's radius, so at the same speed the
centripetal load is DOUBLE the circle's and reverses at each crossover — at 2 m/s
that is 39° of bank (vs 22° for the circle) and a 229°/s tangent yaw rate the yaw
loop can't follow. See the DYNAMICS note in config.py; fly ~1.0–1.2 m/s to match
the circle's dynamics. DRY-RUN first and read the commanded bank in the plan. !!!

SPACEBAR / disarm / low batt / Vicon loss abort to a controlled landing at any
point, exactly as in vicon_hover. Arm + record triggers are identical.

Dry-run FIRST (prints the segment plan + figure-8 geometry at launch; never arms):
  .venv/bin/python drone_control/Vicon_control/figure8_flight.py --dry-run
  .venv/bin/python drone_control/Vicon_control/figure8_flight.py            # LIVE
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_DRONE_CONTROL = os.path.dirname(HERE)
if _DRONE_CONTROL not in sys.path:
    sys.path.insert(0, _DRONE_CONTROL)

from Vicon_control import vicon_hover                       # noqa: E402
from Vicon_control.mission import build_figure8_mission     # noqa: E402


def main():
    args = vicon_hover.build_parser(description=__doc__).parse_args()
    vicon_hover.run(args, make_mission=build_figure8_mission)


if __name__ == "__main__":
    main()
