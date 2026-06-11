"""Measure the FPV camera's uptilt -> CAMERA_TILT_DEG for tag_hover_controller.py.

Why: the controller rotates the camera view to a gravity-level frame using the
camera's uptilt. When the drone is LEVEL and the tag is at the SAME HEIGHT as the
camera lens, the tag's vertical angle in the camera frame equals that uptilt:
    tilt = atan2(yc, zc)
So you just center the tag and read the number.

Setup:
  1. Put the drone LEVEL on a flat table (or hold it level).
  2. Place AprilTag 0 so its CENTER is at the SAME HEIGHT as the camera lens,
     pointed straight at it. Farther is better (a small height error matters less
     at distance) -- aim for ~1-1.5 m if detection is solid.
  3. Run this. Steer the tag until 'horiz' reads ~0 (tag horizontally centered).
  4. When 'horiz' is ~0 and 'UPTILT' is steady, that UPTILT value is your
     CAMERA_TILT_DEG. The script also prints a running average to copy.

Run:  .venv/bin/python camera_setup/calibrate_tilt.py   (q to quit)
"""
import os
import sys
import math
import time

import cv2
import numpy as np
from pupil_apriltags import Detector

HERE = os.path.dirname(os.path.abspath(__file__))
CALIB_FILE = os.path.join(HERE, "camera_calibration.npz")
DEVICE_INDEX = 4
WIDTH, HEIGHT = 1280, 720
TAG_FAMILY = "tag25h9"
TAG_SIZE_M = 0.078
TARGET_TAG_ID = 0
CENTERED_DEG = 2.0   # |horiz| below this counts as "centered" for averaging

if not os.path.exists(CALIB_FILE):
    sys.exit("no camera_calibration.npz — run calibrate_camera.py first")
c = np.load(CALIB_FILE)
K = c["K"].astype(np.float64)
dist = c["dist"].astype(np.float64)
fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

det = Detector(families=TAG_FAMILY)
cap = cv2.VideoCapture(DEVICE_INDEX, cv2.CAP_V4L2)
if not cap.isOpened():
    sys.exit(f"cannot open /dev/video{DEVICE_INDEX}")
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cv2.namedWindow("tilt", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow("tilt", 1280, 720)
print("Level drone, tag 0 at camera height, centered. Read UPTILT when steady. q to quit.")

avg_sum, avg_n = 0.0, 0
last = 0.0
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        if np.any(dist):
            frame = cv2.undistort(frame, K, dist)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = det.detect(gray, estimate_tag_pose=True,
                          camera_params=(fx, fy, cx, cy), tag_size=TAG_SIZE_M)
        t = next((d for d in tags if d.tag_id == TARGET_TAG_ID), None)
        if t is not None:
            xc, yc, zc = [float(v) for v in t.pose_t.ravel()]
            horiz = math.degrees(math.atan2(xc, zc))
            uptilt = math.degrees(math.atan2(yc, zc))
            centered = abs(horiz) < CENTERED_DEG
            if centered:
                avg_sum += uptilt
                avg_n += 1
            avg = (avg_sum / avg_n) if avg_n else float("nan")
            pts = t.corners.astype(int)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            hud = (f"horiz={horiz:+5.1f}deg {'OK' if centered else 'center me'}   "
                   f"UPTILT={uptilt:+5.1f}deg   avg(centered)={avg:5.1f}   "
                   f"dist={zc*100:4.0f}cm")
            cv2.putText(frame, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
            now = time.time()
            if now - last > 0.25:
                print(hud, flush=True)
                last = now
        else:
            cv2.putText(frame, "tag 0 not visible", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("tilt", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    if avg_n:
        print(f"\nSet CAMERA_TILT_DEG = {avg_sum / avg_n:.1f}  "
              f"(averaged over {avg_n} centered frames)")
