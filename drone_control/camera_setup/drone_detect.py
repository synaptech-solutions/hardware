"""AprilTag detection on the drone video feed via Elgato Cam Link 4K.

Run:    .venv/bin/python camera_setup/drone_detect.py
Quit:   press q in the video window.

Requires nothing else to be holding /dev/video4. If you have the mpv
low-latency viewer running, close it first: pkill -9 mpv
"""
import math
import os
import sys
import time
import cv2
import numpy as np
from pupil_apriltags import Detector

PRINT_HZ = 5.0  # how often to print pose to the console


def rotation_to_euler_xyz_deg(R):
    """Extrinsic XYZ Tait-Bryan Euler angles (degrees) from a 3x3 rotation matrix.
    Returns (rx, ry, rz): rotations around the camera frame's X, Y, Z axes.
    """
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        rx = math.atan2(R[2, 1], R[2, 2])
        ry = math.atan2(-R[2, 0], sy)
        rz = math.atan2(R[1, 0], R[0, 0])
    else:
        rx = math.atan2(-R[1, 2], R[1, 1])
        ry = math.atan2(-R[2, 0], sy)
        rz = 0.0
    return math.degrees(rx), math.degrees(ry), math.degrees(rz)

DEVICE_INDEX = 4  # /dev/video4 is the Cam Link 4K on this machine
WIDTH, HEIGHT = 1280, 720
TAG_FAMILY = "tag25h9"
TAG_SIZE_M = 0.078  # 7.8 cm — outer black-square edge length

HERE = os.path.dirname(os.path.abspath(__file__))
CALIB_FILE = os.path.join(HERE, "camera_calibration.npz")

if os.path.exists(CALIB_FILE):
    _c = np.load(CALIB_FILE)
    K = _c["K"].astype(np.float64)
    DIST = _c["dist"].astype(np.float64)
    FX, FY, CX, CY = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    print(f"using calibration from {CALIB_FILE} (RMS={float(_c['rms']):.3f}px)")
else:
    # Placeholder intrinsics for ~100 deg HFOV (the calibrated effective HFOV of
    # the drone cam -> analog VRX -> HDMI-out -> Cam Link chain). Distance will
    # still be approximate without a real calibration file.
    HFOV_DEG = 100.0
    FX = FY = (WIDTH / 2.0) / math.tan(math.radians(HFOV_DEG) / 2.0)
    CX, CY = WIDTH / 2.0, HEIGHT / 2.0
    K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)
    DIST = np.zeros(5)
    print(f"no {CALIB_FILE} found — using ~{HFOV_DEG:.0f}deg HFOV placeholder")

detector = Detector(families=TAG_FAMILY)

cap = cv2.VideoCapture(DEVICE_INDEX, cv2.CAP_V4L2)
if not cap.isOpened():
    sys.exit(f"could not open /dev/video{DEVICE_INDEX} — is the Cam Link plugged in and free?")

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print(f"capture: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
      f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
      f"@ {cap.get(cv2.CAP_PROP_FPS):.0f} fps, "
      f"family={TAG_FAMILY}")
print(f"pose printed at ~{PRINT_HZ:.0f} Hz")
print("  pos: tag center in camera frame (X right, Y down, Z forward), cm")
print("  rot: tag orientation as XYZ Euler angles around camera axes, deg")
print()

last_print = 0.0
print_interval = 1.0 / PRINT_HZ

WINDOW = "drone apriltag"
cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow(WINDOW, 1280, 720)

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        if np.any(DIST):
            frame = cv2.undistort(frame, K, DIST)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=(FX, FY, CX, CY),
            tag_size=TAG_SIZE_M,
        )
        for d in tags:
            pts = d.corners.astype(int)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

            rvec, _ = cv2.Rodrigues(d.pose_R)
            tvec = d.pose_t.reshape(3, 1)
            cv2.drawFrameAxes(frame, K, DIST, rvec, tvec, TAG_SIZE_M * 0.5, 2)

            dist_m = float(np.linalg.norm(d.pose_t))
            cx, cy = int(d.center[0]), int(d.center[1])
            cv2.putText(frame, f"id={d.tag_id}  {dist_m*100:.1f}cm",
                        (cx - 60, cy - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

        now = time.monotonic()
        if tags and now - last_print >= print_interval:
            for d in tags:
                t = d.pose_t.ravel()
                rx, ry, rz = rotation_to_euler_xyz_deg(d.pose_R)
                print(f"id={d.tag_id:>3}  "
                      f"pos(x={t[0]*100:+7.1f}  y={t[1]*100:+7.1f}  z={t[2]*100:+7.1f}) cm  "
                      f"rot(rx={rx:+7.1f}  ry={ry:+7.1f}  rz={rz:+7.1f}) deg")
            print("-" * 78)
            last_print = now

        cv2.imshow(WINDOW, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
