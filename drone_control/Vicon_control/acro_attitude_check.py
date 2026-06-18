#!/usr/bin/env python3
"""Read-only bench check: print Vicon attitude (roll/pitch/yaw) for the ACRO build.

NEVER opens the Ranger, never arms — pure Vicon read. Used to empirically pin the
Vicon-quaternion → drone-body roll/pitch axis+sign map (the VICON_YAW_OFFSET_DEG=90
frame is rotated 90° from the drone, which SWAPS roll/pitch — see config). Hold the
drone in a known attitude and read which printed value moves and its sign.

Prints, at ~10 Hz:
  - raw Vicon Euler RotX/RotY/RotZ (rad→deg) straight off the wire (NOT yaw-offset)
  - quaternion-derived roll/pitch/yaw in the DRONE body frame (yaw-offset applied),
    i.e. exactly what the controller would consume
  - the same with NO yaw offset, so we can see the 90° swap directly

  .venv/bin/python drone_control/Vicon_control/acro_attitude_check.py
"""
import os
import sys
import math
import time
import struct

HERE = os.path.dirname(os.path.abspath(__file__))
_DRONE_CONTROL = os.path.dirname(HERE)
if _DRONE_CONTROL not in sys.path:
    sys.path.insert(0, _DRONE_CONTROL)

from Vicon_control import config                          # noqa: E402
from Vicon_control.vicon_source import ViconPoseSource    # noqa: E402

CSI = "\033["


def quat_to_rpy(qx, qy, qz, qw):
    """Roll (about body X), pitch (about body Y), yaw (about Z) from a quaternion.
    Standard aerospace ZYX extraction. SIGNS/AXES are what we're here to verify."""
    # roll (x-axis)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    # pitch (y-axis)
    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    # yaw (z-axis)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def main():
    # body_index=1 (b1 = the drone), with the SAME yaw offset the controller uses.
    src = ViconPoseSource(yaw_offset_deg=config.VICON_YAW_OFFSET_DEG)
    print(f"Vicon: probing UDP :{src.port} …")
    if not src.prepare():
        sys.exit(f"No Vicon stream: {src.status}. Start Vicon Tracker streaming here.")
    print(f"Vicon: {src.status}")
    print("Read-only. Ctrl-C to stop.\n")
    print("Columns: rawEuler(RotX/Y/Z)  |  quat RPY no-offset  |  quat RPY +90offset")
    print("(all degrees; the controller would use the +offset roll/pitch)\n")

    # Append every printed reading to a log so the discrete poses can be read back
    # exactly (the live line uses \r and can't be scrolled).
    logf = open(os.path.join(HERE, "acro_attitude_check.log"), "a")
    logf.write(f"\n# session {time.strftime('%Y-%m-%d %H:%M:%S')}  "
               f"(yaw_offset={config.VICON_YAW_OFFSET_DEG})\n")
    logf.flush()
    last = 0.0
    try:
        while True:
            now = time.time()
            pose = src.get_pose(now)
            if pose is None:
                time.sleep(0.02)
                continue
            # raw Euler straight off the parser (bypass the quaternion round-trip)
            raw = src.udp.get_data()[0]
            rotx = roty = rotz = float("nan")
            if raw is not None and len(raw) >= 53:
                # per DataProcessorViCON: 5-byte header, 3-byte per-obj hdr, 24 name,
                # then 6×float64 (TransXYZ mm, RotXYZ rad). First body only.
                try:
                    _x, _y, _z, rotx, roty, rotz = struct.unpack_from(
                        "<dddddd", raw, 5 + 3 + 24)
                except struct.error:
                    pass
            # quaternion RPY without the yaw offset (raw Vicon body frame)
            r0, p0, y0 = quat_to_rpy(*pose_q(src))
            # the controller-facing pose: yaw has the +90 offset baked in already;
            # for roll/pitch we apply the SAME quaternion but show both so the swap
            # is visible.
            if now - last > 0.1:
                d = math.degrees
                line = (f"raw[X{d(rotx):+6.1f} Y{d(roty):+6.1f} Z{d(rotz):+6.1f}]  "
                        f"quat[R{d(r0):+6.1f} P{d(p0):+6.1f} Y{d(y0):+6.1f}]  "
                        f"pose.yaw(+off){d(pose['yaw']):+6.1f}")
                sys.stdout.write(f"\r{CSI}K{line}")
                sys.stdout.flush()
                logf.write(line + "\n")
                logf.flush()
                last = now
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        src.stop()


def pose_q(src):
    """Latest quaternion (qx,qy,qz,qw) for body 1 from the shared receiver."""
    data_raw, _ = src.udp.get_data()
    data, _ = src.dp.process_data(data_raw)
    b = data.get(src.body_index)
    if b is None:
        return 0.0, 0.0, 0.0, 1.0
    return b["qx"], b["qy"], b["qz"], b["qw"]


if __name__ == "__main__":
    main()
