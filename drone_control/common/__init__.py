"""Shared hardware/IO layer for the drone stack.

Base dependency for the data-logging pipeline (``data_logging/``), the AprilTag
controller (``apriltag_control/``) and the Vicon controller (``Vicon_control/``).
Nothing here depends on a sibling package — consumers add ``drone_control/`` to
``sys.path`` and ``from common import <module>``.

Modules:
  live_telemetry  CRSF/MSP frame builders + parsers, port autodetect.
  ranger          open_ranger() — custom 420000-baud open via TCSETS2 ioctl.
  channels        Air75 CRSF channel map + µs levels + camera + loop rate
                  (the single source of truth, verified on the Air75).
  pid             generic PID with anti-windup.
  tx12            TX12 USB-joystick reader, calibration, switch interpreters.
  recorders       per-session recorders (video / Vicon / commands / telemetry)
                  + session.json writer — the synced data-collection pipeline.
"""
