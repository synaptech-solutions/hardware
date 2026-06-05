"""Interactive camera calibration for the drone feed via Cam Link 4K.

Workflow:
    1. Print checkerboard at 100% scale (make_checkerboard.py or your own).
    2. Set PATTERN and SQUARE_SIZE_M below to match the print.
    3. Tape it FLAT to stiff cardboard. Any curl wrecks accuracy.
    4. Put drone on a table, powered, analog VRX powered + HDMI-out into Cam Link.
    5. Run:  .venv/bin/python camera_setup/calibrate_camera.py
    6. Capture 25+ frames at varied poses (see tips below).
    7. Press C to compute. Script auto-drops worst frames and re-computes.
    8. drone_detect.py auto-loads camera_calibration.npz on next run.

Tips for low RMS (<1.0 px):
    - Mount the print on SOMETHING STIFF. A flexed print is the #1 cause of bad RMS.
    - Move slowly, then hold still for half a second before pressing SPACE.
      Motion blur is the #2 cause.
    - VARY THE POSE between captures. The coverage thumbnail (top-right) shows
      where in the frame you've already captured boards. Fill in gaps,
      especially near the edges and corners.
    - Include strong tilts (30-60 degrees), not just translations.
    - Get the board near each corner of the frame at least 3 times.

Keys:  SPACE = save  |  C = compute & save  |  Q = quit
"""
import os
import sys
import cv2
import numpy as np

DEVICE_INDEX = 4
WIDTH, HEIGHT = 1280, 720
PATTERN = (10, 7)         # inner corners (width, height)
SQUARE_SIZE_M = 0.0168    # 16.8 mm
MIN_FRAMES = 25
DROP_WORST_FRAC = 0.15    # after first compute, drop this fraction of worst frames

# Detect on a downscaled frame for the live preview (60 fps with 1280x720 + 10x7
# is too slow). Full-res is still used for the accurate save-time detection.
PREVIEW_DOWNSCALE = 2
BLUR_THRESH = 60.0        # Laplacian variance below this = too blurry to use

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, "camera_calibration.npz")

# 3D object points: PATTERN[0]*PATTERN[1] corners, z=0, spaced by SQUARE_SIZE_M.
objp = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE_M

cap = cv2.VideoCapture(DEVICE_INDEX, cv2.CAP_V4L2)
if not cap.isOpened():
    sys.exit(f"could not open /dev/video{DEVICE_INDEX}")
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

preview_flags = (cv2.CALIB_CB_ADAPTIVE_THRESH
                 + cv2.CALIB_CB_NORMALIZE_IMAGE
                 + cv2.CALIB_CB_FAST_CHECK)
# CALIB_RATIONAL_MODEL adds 3 extra distortion coefficients (k4,k5,k6).
# Handles wider lenses and non-uniform scaling better than the standard 5-coef model.
calib_flags = cv2.CALIB_RATIONAL_MODEL

objpoints, imgpoints, board_centers = [], [], []
img_shape = (WIDTH, HEIGHT)


def detect_full(gray):
    """Accurate sub-pixel detection on full-resolution gray frame.
    Returns corners (N,1,2) float32 or None.
    """
    sb_flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_ACCURACY
    found, corners = cv2.findChessboardCornersSB(gray, PATTERN, sb_flags)
    return corners if found else None


def sharpness(gray, corners):
    """Laplacian variance over the board's bounding box — low = blurry."""
    xs = corners[:, 0, 0]
    ys = corners[:, 0, 1]
    x1, y1 = int(max(xs.min(), 0)), int(max(ys.min(), 0))
    x2, y2 = int(min(xs.max(), gray.shape[1])), int(min(ys.max(), gray.shape[0]))
    if x2 - x1 < 20 or y2 - y1 < 20:
        return 0.0
    roi = gray[y1:y2, x1:x2]
    return float(cv2.Laplacian(roi, cv2.CV_64F).var())


def compute_calibration(obj_pts, img_pts):
    """Run calibration, return (rms, K, dist, per_frame_errors)."""
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_pts, img_pts, img_shape, None, None, flags=calib_flags
    )
    per_frame = []
    for i in range(len(obj_pts)):
        proj, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], K, dist)
        err = cv2.norm(img_pts[i], proj, cv2.NORM_L2) / len(proj)
        per_frame.append(err)
    return rms, K, dist, per_frame


def coverage_thumb(centers, w=200, h=112):
    """Small thumbnail showing where captured boards landed in the frame."""
    thumb = np.full((h, w, 3), 30, dtype=np.uint8)
    cv2.rectangle(thumb, (0, 0), (w - 1, h - 1), (80, 80, 80), 1)
    for cx, cy in centers:
        x = int(cx * w / WIDTH)
        y = int(cy * h / HEIGHT)
        cv2.circle(thumb, (x, y), 3, (0, 200, 255), -1)
    return thumb


print(f"capture: {WIDTH}x{HEIGHT}, pattern={PATTERN}, square={SQUARE_SIZE_M*1000:.1f}mm")
print(f"distortion model: rational (8 coeffs)  |  need {MIN_FRAMES}+ frames")
print("SPACE = save  |  C = compute  |  Q = quit")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Cheap preview detection on a downscaled frame.
        small = cv2.resize(gray_full, (WIDTH // PREVIEW_DOWNSCALE,
                                       HEIGHT // PREVIEW_DOWNSCALE))
        preview_found, preview_corners = cv2.findChessboardCorners(
            small, PATTERN, preview_flags
        )

        display = frame.copy()
        if preview_found:
            scaled_corners = preview_corners * PREVIEW_DOWNSCALE
            cv2.drawChessboardCorners(display, PATTERN, scaled_corners, preview_found)
            cv2.putText(display, "DETECTED — hold still, press SPACE",
                        (10, HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display, "no checkerboard found",
                        (10, HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)

        cv2.putText(display, f"saved: {len(objpoints)} / {MIN_FRAMES}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        thumb = coverage_thumb(board_centers)
        th, tw = thumb.shape[:2]
        display[10:10 + th, WIDTH - 10 - tw:WIDTH - 10] = thumb
        cv2.putText(display, "coverage", (WIDTH - 10 - tw, 10 + th + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("calibrate", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord(" "):
            if not preview_found:
                print("  no board visible — not saved")
                continue
            # Re-detect on full resolution for sub-pixel accuracy.
            corners = detect_full(gray_full)
            if corners is None:
                print("  full-res detection failed — not saved (try a clearer view)")
                continue
            sharp = sharpness(gray_full, corners)
            if sharp < BLUR_THRESH:
                print(f"  blurry (Laplacian var={sharp:.1f} < {BLUR_THRESH}) — not saved")
                continue
            objpoints.append(objp)
            imgpoints.append(corners)
            cx_b = float(corners[:, 0, 0].mean())
            cy_b = float(corners[:, 0, 1].mean())
            board_centers.append((cx_b, cy_b))
            print(f"  saved frame {len(objpoints)}  (sharpness={sharp:.0f})")

        if key == ord("c"):
            if len(objpoints) < MIN_FRAMES:
                print(f"  need {MIN_FRAMES}+ frames, have {len(objpoints)}")
                continue
            print(f"\ncomputing initial calibration on {len(objpoints)} frames...")
            rms, K, dist, errs = compute_calibration(objpoints, imgpoints)
            print(f"  initial RMS = {rms:.3f} px")
            for i, e in enumerate(errs):
                marker = " <-- worst" if e == max(errs) else ""
                print(f"    frame {i+1:3d}: {e:.3f} px{marker}")

            # Drop the worst DROP_WORST_FRAC frames and recompute.
            n_drop = max(1, int(len(objpoints) * DROP_WORST_FRAC))
            order = np.argsort(errs)
            keep = sorted(order[:-n_drop].tolist())
            kept_obj = [objpoints[i] for i in keep]
            kept_img = [imgpoints[i] for i in keep]
            print(f"\ndropping {n_drop} worst frames, recomputing on {len(kept_obj)}...")
            rms2, K, dist, errs2 = compute_calibration(kept_obj, kept_img)
            print(f"  final RMS = {rms2:.3f} px  (good < 1.0, great < 0.5)")
            print(f"K =\n{K}")
            print(f"distortion = {dist.ravel()}")
            np.savez(OUTPUT, K=K, dist=dist, image_size=img_shape, rms=rms2)
            print(f"\nsaved {OUTPUT}")
            if rms2 > 1.5:
                print("RMS is high. Recommend: flatter print, slower captures, "
                      "more variety near frame corners. Press SPACE to keep adding "
                      "frames, then C again, or Q to quit.")
            else:
                break
finally:
    cap.release()
    cv2.destroyAllWindows()
