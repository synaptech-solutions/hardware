#!/usr/bin/env python3
"""Vicon autonomous FLIP(S): take off → hover → a configurable SEQUENCE of 360° body
flips (roll and/or pitch), re-stabilizing between each → land.

The sequence is config.FLIP_SEQUENCE, e.g. ["roll"] (one roll), ["pitch"] (one loop),
or ["roll","pitch","roll","pitch"] (alternating). Built on vicon_hover.py exactly like
circle_flight.py — SAME arming, controller, recording, status line and failsafes — it
just feeds the shared loop a FlipManeuver. Per flip:
  1. settle: hover until stable for config.FLIP_PREROLL_STABLE_S (first flip) /
     config.FLIP_BETWEEN_STABLE_S of recovered hover (subsequent flips)
  2. ROLL: full-stick rate command on that flip's axis for 360/FLIP_RATE_DPS s —
     OPEN LOOP, leveling bypassed. Throttle is CUT while inverted (no downward push).
  3. RECOVER: hand back to the leveling + position + altitude loops (with a hard climb
     boost to arrest the sink), back to the hover point.
After the last flip, hold config.FLIP_STABLE_HOLD_S stable → land.

DEAD-RECKONED and deliberately not precise — the controller cleans up the rotation
error. Tune config.FLIP_DURATION_SCALE if a flip ends short/long.

!!! REQUIRES ACRO (config.ACRO_MODE=True): a roll is a RATE command; in Angle mode
the roll stick is a bank ANGLE capped at angle_limit (60°), so it physically cannot
roll past level. !!!
!!! FLY WITH ALTITUDE MARGIN: throttle is held ~hover (open-loop) through the roll,
so the drone DROPS during the inverted portion. config.CLIMB_M should be ≥ ~2.5 m. !!!

The safety-critical sign (the leveling loop's restoring direction) is ALREADY proven
by your working acro hover — the flip reuses it untouched. The roll DIRECTION itself
is cosmetic (a full 360° returns to level either way), so there's nothing new to
verify by hand; the roll only fires once stable AT hover height (it won't trigger on
the ground in a dry-run).

SPACEBAR / disarm / low batt / Vicon loss abort to a controlled landing at any point
(SPACEBAR mid-roll hands straight back to the leveling controller and lands).

Dry-run FIRST (prints the flip plan at launch; never arms):
  .venv/bin/python drone_control/Vicon_control/flip_flight.py --dry-run
  .venv/bin/python drone_control/Vicon_control/flip_flight.py            # LIVE
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_DRONE_CONTROL = os.path.dirname(HERE)
if _DRONE_CONTROL not in sys.path:
    sys.path.insert(0, _DRONE_CONTROL)

from Vicon_control import vicon_hover, config              # noqa: E402
from Vicon_control.mission import FlipManeuver              # noqa: E402


def main():
    if not config.ACRO_MODE:
        sys.exit("flip_flight requires config.ACRO_MODE=True — a 360° roll needs RATE "
                 "commands; Angle mode caps the roll stick at angle_limit (can't roll "
                 "past level). Set ACRO_MODE=True (+ the matching FC ACRO setup) first.")
    args = vicon_hover.build_parser(description=__doc__).parse_args()
    vicon_hover.run(args, make_mission=FlipManeuver)


if __name__ == "__main__":
    main()
