"""Live GateNet gate segmentation on the drone video feed (Cam Link 4K).

Opens the analog FPV feed, runs the trained ``checkpoints_flight`` model on each frame
(on the laptop CPU -- ~30 ms/frame, no GPU needed) and shows the predicted gate mask as a
translucent green overlay. Everything is vendored under ``perception/`` so this needs only
the repo venv; it does not touch the GateNet project or its venv.

Keys:  SPACE = start/stop recording the overlay   B = toggle side-by-side mask   Q/ESC = quit

Run with the repo venv (it now carries jax/jaxlib/flax for CPU inference):
  .venv/bin/python live_infer.py
  .venv/bin/python live_infer.py --device 5 --conf 0.4 --side-by-side
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from perception import GatePredictor

# The repo's shared H.264 writer lives as a loose module under pycode_ViCON/.
sys.path.insert(0, str(Path(__file__).parent / "pycode_ViCON"))
from ffmpeg_writer import FfmpegWriter  # noqa: E402

HERE = Path(__file__).parent


def open_camera(device: int, width: int, height: int, fps: int) -> tuple[cv2.VideoCapture, int]:
    """Open /dev/video<device>, falling back to device+1 (Cam Link 4<->5 re-enumeration)."""
    for dev in (device, device + 1):
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
            ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height
            print(f"Camera: /dev/video{dev} {aw}x{ah} (requested {width}x{height})")
            return cap, dev
        cap.release()
    raise SystemExit(f"could not open /dev/video{device} (or {device + 1}) — "
                     "is the Cam Link plugged in and free? (close mpv: pkill -9 mpv)")


def overlay_mask(frame_bgr: np.ndarray, prob: np.ndarray, conf: float) -> np.ndarray:
    """Draw the predicted mask (resized to the frame) as a translucent green fill + contour."""
    h, w = frame_bgr.shape[:2]
    mask = (cv2.resize(prob, (w, h)) > conf).astype(np.uint8)
    out = frame_bgr.copy()
    out[mask > 0] = (0.45 * out[mask > 0] + np.array([0, 165, 0]) * 0.55).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 255, 0), 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", type=int, default=4, help="camera /dev/videoN (default 4)")
    ap.add_argument("--width", type=int, default=720, help="capture width (native NTSC 720x480)")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30, help="C03 is NTSC (~30fps)")
    ap.add_argument("--conf", type=float, default=0.5, help="mask threshold in [0,1]")
    ap.add_argument("--weights", default=str(HERE / "perception" / "weights"),
                    help="dir holding gatenet_flight.msgpack + infer.json")
    ap.add_argument("--side-by-side", action="store_true",
                    help="show the binary mask in a panel beside the overlay")
    ap.add_argument("--out-dir", default=str(HERE / "DataExchange"),
                    help="where SPACE-recorded clips are saved")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("loading model (CPU)…")
    predictor = GatePredictor(args.weights)
    print(f"ready: size={predictor.size}, undistort={predictor.undistort}, pencil={predictor.pencil}")

    cap, _ = open_camera(args.device, args.width, args.height, args.fps)
    cam_fps = cap.get(cv2.CAP_PROP_FPS)
    cam_fps = cam_fps if cam_fps and cam_fps > 1 else float(args.fps)

    win = "gate segmentation  [SPACE]=record  [B]=side-by-side  [Q]=quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    side_by_side = args.side_by_side
    writer: FfmpegWriter | None = None
    rec_path: str | None = None
    rec_start = 0.0
    rec_frames = 0
    ema_ms = 0.0  # smoothed inference time

    print("SPACE = record/stop   B = side-by-side   Q/ESC = quit")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
                continue

            t0 = time.perf_counter()
            proc, prob = predictor(frame)
            infer_ms = (time.perf_counter() - t0) * 1000.0
            ema_ms = infer_ms if ema_ms == 0.0 else 0.9 * ema_ms + 0.1 * infer_ms

            view = overlay_mask(proc, prob, args.conf)
            if side_by_side:
                h = view.shape[0]
                panel = cv2.cvtColor(
                    (cv2.resize(prob, (proc.shape[1], h)) > args.conf).astype(np.uint8) * 255,
                    cv2.COLOR_GRAY2BGR,
                )
                view = np.hstack([view, np.full((h, 4, 3), 64, np.uint8), panel])

            if writer is not None:
                writer.write(view)
                rec_frames += 1

            hud = f"{ema_ms:4.0f} ms ({1000.0 / max(ema_ms, 1e-3):4.1f} fps)  conf>{args.conf:.2f}"
            cv2.putText(view, hud, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if writer is not None:
                el = datetime.datetime.now().timestamp() - rec_start
                cv2.putText(view, f"REC {el:5.1f}s", (12, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow(win, view)

            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("b"):
                side_by_side = not side_by_side
            if k == ord(" "):
                if writer is None:
                    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    rec_path = os.path.join(args.out_dir, f"{stamp}_seg.mp4")
                    h, w = view.shape[:2]
                    writer = FfmpegWriter(rec_path, w, h, cam_fps)
                    rec_start = datetime.datetime.now().timestamp()
                    rec_frames = 0
                    print(f"recording -> {rec_path}")
                else:
                    writer.release()
                    print(f"saved {rec_frames} frames -> {rec_path}")
                    writer = None
    finally:
        if writer is not None:
            writer.release()
            print(f"saved {rec_frames} frames -> {rec_path}")
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
