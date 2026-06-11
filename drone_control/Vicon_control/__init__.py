"""Vicon-based autonomous hover controller for the BetaFPV Air75.

Flies on LIVE Vicon pose (100 Hz, absolute position + drift-free yaw) instead of
AprilTags — which deletes the vision-dropout / dead-reckoning / compass-less-yaw
failure modes of apriltag_control/. The TX12 stays in the loop for arm / disarm /
record only (a disarm flick is the hardware kill switch).

Modules:
  config         target climb, gains, hover band, safety limits, loop rate.
  vicon_source   live world-frame pose (x,y,z,yaw,vx,vy,vz) from the ONE Vicon
                 receiver (shared with the recorder).
  controller     ViconHoverController — world→body position PID → angle setpoints,
                 altitude PI that learns hover throttle, absolute yaw hold.
  vicon_hover    main loop: TX12 arm/record gating + state machine + CRSF send +
                 the shared data-collection recording (reuses common/recorders).

Reuses the shared hardware layer (drone_control/common/) and the data pipeline,
so flights record + render exactly like hand-flown ones.
"""
# Put drone_control on sys.path so this package's submodules can `from common
# import ...` (same pattern as apriltag_control/controller_v2).
import os as _os
import sys as _sys
_DRONE_CONTROL = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _DRONE_CONTROL not in _sys.path:
    _sys.path.insert(0, _DRONE_CONTROL)
