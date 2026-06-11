"""Cascaded AprilTag-hover controller (FC handles inner attitude in Angle mode).
Entrypoint: ../tag_hover_v2.py
"""
# The shared hardware layer (CRSF, PID) moved to drone_control/common/. Put
# drone_control on sys.path so this package's submodules can `from common import`.
import os as _os
import sys as _sys
_DRONE_CONTROL = _os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _DRONE_CONTROL not in _sys.path:
    _sys.path.insert(0, _DRONE_CONTROL)
