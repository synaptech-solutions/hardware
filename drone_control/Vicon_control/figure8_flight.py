#!/usr/bin/env python3
"""Vicon autonomous FIGURE-8 course: take off → full figure-8 → land.

Built on vicon_hover.py exactly like circle_flight.py / square_flight.py — SAME
arming, controller, recording, status line and failsafes — it just feeds the
shared flight loop a PathMission (a carrot that FLOWS along a smooth ∞ path)
instead of a static hover. The course (relative to the LAUNCH pose; you start at
the origin, which IS the figure-8's crossover):
  1. climb to config.CLIMB_M and hover config.INITIAL_HOVER_S at the origin
  2. trace config.FIG8_LAPS smooth figure-8(s) — a BERNOULLI LEMNISCATE (the classic
     ∞) centred at the origin, long axis along world X with half-span
     config.FIG8_END_X_M (far ends at (±config.FIG8_END_X_M, 0)), crossing at the
     origin. Its curvature is CONTINUOUS (zero at the crossover → straight through the
     centre, peaking at 3/FIG8_END_X_M at the FAR ENDS), so there's no instantaneous
     bank/yaw reversal at the middle — that was the awkward, untrackable transition
     of the old two-tangent-circles design. One continuous segment at
     config.FIG8_SPEED_MPS (ramps up once, brakes once into the home dwell). With
     config.FIG8_FACE_TANGENT the nose follows the direction of travel; pre-rotates
     to the start tangent during takeoff.
  3. settle config.SETTLE_S back at the origin, then land
Unlike the circle there is NO forward approach leg: the crossing is the launch
point, so the drone is already on the path at takeoff. All tunables live in
config.py (the FIG8_* block + the shared carrot params).

!!! SPEED is yaw-limited for tangent-facing: the lemniscate's peak yaw rate is
v·3/FIG8_END_X_M at the far ends (206°/s at 1.2 m/s, PEAK=2) — over the ~147°/s yaw
authority, so FIG8_FACE_TANGENT must stay False unless you slow down, and peak bank
∝ v². See the config DYNAMICS note; DRY-RUN and preview.py first. !!!

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
