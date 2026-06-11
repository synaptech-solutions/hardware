#!/usr/bin/env python3
"""Live Vicon frame-calibration reader (no Ranger, no arming — pure read).

Samples the drone rigid body (b1) for a few seconds and reports averaged
position, the quaternion, and the Euler roll/pitch/yaw the controller derives
from it. Use it to calibrate the Vicon↔drone frame: place the drone at a known
position/orientation, run this, and read what Vicon transmits.

Usage (repo venv):
  .venv/bin/python drone_control/Vicon_control/vicon_read.py --label "origin nose=+Y"
  .venv/bin/python drone_control/Vicon_control/vicon_read.py --secs 4
"""
import argparse
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                 # drone_control on path
from Vicon_control.vicon_source import ViconPoseSource     # noqa: E402


def quat_to_rpy(qx, qy, qz, qw):
    roll = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (qw * qy - qz * qx))))
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default="", help="what placement this is (for the printout)")
    ap.add_argument("--secs", type=float, default=3.0, help="sampling window (s)")
    args = ap.parse_args()

    src = ViconPoseSource(yaw_offset_deg=0.0)              # RAW yaw (no offset)
    print(f"Probing Vicon :{src.port} …")
    if not src.prepare():
        sys.exit(f"Vicon not streaming ({src.status}). Start Vicon Tracker and retry.")
    print(f"{src.status}\nSampling {args.secs:.0f}s …")

    xs, ys, zs, yaws, qs = [], [], [], [], []
    t0 = time.time()
    while time.time() - t0 < args.secs:
        p = src.get_pose()
        if p is not None:
            b = src.dp.data_list[src.body_index]           # raw quaternion this tick
            xs.append(p["x"]); ys.append(p["y"]); zs.append(p["z"])
            qs.append((b["qx"], b["qy"], b["qz"], b["qw"]))
            yaws.append(p["yaw"])
        time.sleep(0.01)
    src.stop()

    if not xs:
        sys.exit("No Vicon samples — is rigid body b1 present?")
    n = len(xs)

    def ms(v):
        import statistics
        return statistics.fmean(v), (statistics.pstdev(v) if len(v) > 1 else 0.0)
    mx, sx = ms(xs); my, sy = ms(ys); mz, sz = ms(zs)
    # mean yaw via circular mean (avoid wrap); roll/pitch from mean quaternion
    import statistics
    cyaw = math.degrees(math.atan2(statistics.fmean(math.sin(y) for y in yaws),
                                   statistics.fmean(math.cos(y) for y in yaws)))
    mq = [statistics.fmean(q[i] for q in qs) for i in range(4)]
    nq = math.sqrt(sum(c * c for c in mq)) or 1.0
    mq = [c / nq for c in mq]
    roll, pitch, _ = quat_to_rpy(*mq)

    bar = "=" * 60
    print(f"\n{bar}\nPLACEMENT: {args.label or '(unlabeled)'}    ({n} samples)\n{bar}")
    print(f"  position (m):  x = {mx:+.3f} ±{sx:.3f}   y = {my:+.3f} ±{sy:.3f}   z = {mz:+.3f} ±{sz:.3f}")
    print(f"  quaternion:    qx={mq[0]:+.3f} qy={mq[1]:+.3f} qz={mq[2]:+.3f} qw={mq[3]:+.3f}")
    print(f"  Euler (deg):   roll = {roll:+.1f}   pitch = {pitch:+.1f}   YAW = {cyaw:+.1f}")
    print(f"\n  controller-derived heading (raw Vicon yaw) = {cyaw:+.1f}°")
    print(f"  IF the drone's nose currently points world +Y (heading should read 90°),")
    print(f"  then VICON_YAW_OFFSET_DEG = 90 - ({cyaw:+.1f}) = {90 - cyaw:+.1f}°")
    print(bar)


if __name__ == "__main__":
    main()
