#!/usr/bin/env python3
"""Vicon autonomous FIGURE-8 course: take off → full figure-8 → land.

Built on vicon_hover.py exactly like circle_flight.py / square_flight.py — SAME
arming, controller, recording, status line and failsafes — it just feeds the
shared flight loop a PathMission (a carrot that FLOWS along a smooth ∞ path)
instead of a static hover. The course (relative to the LAUNCH pose; you start at
the origin, which IS the figure-8's crossover):
  1. climb to config.CLIMB_M and hover config.INITIAL_HOVER_S at the origin
  2. trace config.FIG8_LAPS smooth figure-8(s) — a BERNOULLI LEMNISCATE (the classic
     ∞) centred at the origin, peaks at (±config.FIG8_PEAK_M, 0) on world X, crossing
     at the origin. Its curvature is CONTINUOUS (zero at the crossing → straight
     through the centre, greatest at the peak tips), so there's no instantaneous
     bank/yaw reversal at the middle — that was the awkward, untrackable transition
     of the old two-tangent-circles design. One continuous segment at
     config.FIG8_SPEED_MPS (ramps up once, brakes once into the home dwell). With
     config.FIG8_FACE_TANGENT (default ON, like the circle) the nose follows the
     direction of travel; pre-rotates to the start tangent during takeoff.
  3. settle config.SETTLE_S back at the origin, then land
Unlike the circle there is NO forward approach leg: the crossing is the launch
point, so the drone is already on the path at takeoff. All tunables live in
config.py (the FIG8_* block + the shared carrot params).

!!! SPEED is yaw-limited for tangent-facing: the lemniscate's tip turn radius is
≈PEAK/3 (0.67 m at PEAK=2), so the peak yaw rate is v/0.67. FIG8_SPEED_MPS=1.3 →
112°/s yaw, 14° bank — gentle and trackable. NOT the circle's 2 m/s: that lapped the
drone on the old tight figure-8 (flight 20260615_160611). DRY-RUN first. !!!

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
