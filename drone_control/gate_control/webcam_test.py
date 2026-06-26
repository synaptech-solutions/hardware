#!/usr/bin/env python3
"""Visual GateNet test on a webcam — no drone, Vicon, or Ranger.

Runs the deploy mask pipeline (``MaskSource`` → the size×size mask the policy
actually sees) on a plain webcam so you can hold the gate (or a photo) in front of
the laptop and check GateNet segments it. Shows the frame with the mask overlaid
and the raw mask panel beside it. C03 undistortion is off by default (the webcam
isn't the FPV camera); pass --undistort to enable it.

  .venv/bin/python drone_control/gate_control/webcam_test.py
  .venv/bin/python drone_control/gate_control/webcam_test.py --device 2 --mask-size 64
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate_control.mask_source import MaskSource


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", type=int, default=0, help="webcam /dev/videoN (default 0)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--mask-size", type=int, default=32,
                    help="mask resolution fed to the policy (deploy default 32)")
    ap.add_argument("--conf", type=float, default=0.5, help="overlay/coverage threshold")
    ap.add_argument("--undistort", action="store_true", help="apply C03 rectification")
    ap.add_argument("--weights", default=None, help="GateNet weights dir override")
    args = ap.parse_args()

    src = MaskSource(args.mask_size, device=args.device, width=args.width,
                     height=args.height, fps=args.fps, weights_dir=args.weights,
                     undistort=args.undistort)
    print("loading GateNet + opening camera …")
    src.start()
    print(f"{src.status}  (mask {args.mask_size}x{args.mask_size})  —  Q/ESC to quit")

    win = "GateNet webcam test  [Q]=quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    try:
        while True:
            frame = src.get_latest_frame()
            mask, age = src.get_mask()
            if frame is None:
                if (cv2.waitKey(10) & 0xFF) in (ord("q"), 27):
                    break
                continue

            h, w = frame.shape[:2]
            big = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            view = frame.copy()
            view[big > args.conf] = (0.5 * view[big > args.conf] + (0, 140, 0)).astype(np.uint8)
            panel = cv2.cvtColor((big * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

            hud = (f"{src.infer_ms:4.0f} ms  cover {float(mask.mean()):.2f}  "
                   f"age {age*1000:3.0f} ms")
            cv2.putText(view, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(win, np.hstack([view, np.full((h, 4, 3), 64, np.uint8), panel]))
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
    finally:
        src.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
