"""Standalone drone video recorder — just the feed + a record toggle.

Opens the drone's video feed in a window. SPACE starts a recording; SPACE again
stops it and saves the clip to DataExchange/<timestamp>.mp4. You can record as
many clips as you like in one session; Q (or ESC) quits. No ViCON, no blackbox,
no sync — this is purely for grabbing video off the feed.

OpenCV owns the camera (capture + preview window) and pipes each frame to an
ffmpeg subprocess that encodes H.264. This keeps the reliable OpenCV preview but
swaps the bloated 'mp4v' encode for libx264 — which, unlike OpenCV's built-in
H.264, ships with ffmpeg on stock Ubuntu, so collection works on any machine.

Camera: /dev/video4 (auto-tries video5 if the Cam Link re-enumerated). The C03 is
an analog NTSC camera, so we capture at its native 720x480 (4:3) — capturing 1280x720
just upscales SD with no extra detail and bloats files. Pass --width/--height to override.

Run with the cv2 venv (ffmpeg must also be on PATH):
  ../.venv/bin/python simple_video_recorder.py
  ../.venv/bin/python simple_video_recorder.py --device 5 --crf 20 --out-dir /some/where
"""
from __future__ import annotations

import os
import shutil
import argparse
import datetime

import cv2

from ffmpeg_writer import FfmpegWriter

HERE = os.path.dirname(os.path.abspath(__file__))


def open_camera(
    device: int, width: int, height: int, fps: int
) -> tuple[cv2.VideoCapture | None, int | None]:
    """Open /dev/video<device>, falling back to device+1 (Cam Link 4<->5 shift).

    The encoder is sized from the actual frames later, so we only need the cap +
    its index here; the printed size is whatever the device actually negotiated.
    """
    for dev in (device, device + 1):
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)  # C03 is analog NTSC (~30fps); 60 just dupes frames
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
            ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height
            print(f"Camera: /dev/video{dev} {aw}x{ah} (requested {width}x{height})")
            return cap, dev
        cap.release()
    return None, None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", type=int, default=4, help="camera /dev/videoN (default 4)")
    ap.add_argument("--width", type=int, default=720)   # native NTSC (the C03 is SD analog,
    ap.add_argument("--height", type=int, default=480)  # 720x480 4:3 — 1280x720 was upscaled)
    ap.add_argument("--fps", type=int, default=30, help="C03 is NTSC (~30fps); 60 just dupes")
    ap.add_argument("--crf", type=int, default=23,
                    help="H.264 quality, lower=better/bigger (default 23)")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "DataExchange"),
                    help="where clips are saved (default DataExchange/)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found on PATH — install it (e.g. apt install ffmpeg).")

    cap, dev = open_camera(args.device, args.width, args.height, args.fps)
    if cap is None:
        raise SystemExit(f"could not open /dev/video{args.device} (or {args.device + 1}) — "
                         "is the Cam Link plugged in and free?")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 1 else float(args.fps)

    WIN = "drone feed  [SPACE]=record/stop  [Q]=quit"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    writer: FfmpegWriter | None = None
    path: str | None = None
    start_t: float | None = None
    frames = 0
    print("SPACE = start/stop recording   Q/ESC = quit")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
                continue

            if writer is not None:
                writer.write(frame)
                frames += 1

            disp = frame.copy()
            if writer is not None:
                el = datetime.datetime.now().timestamp() - start_t
                cv2.putText(disp, f"REC  {el:5.1f}s  ({frames} frames)", (12, 34),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.putText(disp, "SPACE = record    Q = quit", (12, 34),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
            cv2.imshow(WIN, disp)

            k = cv2.waitKey(1) & 0xFF
            if k == ord(" "):
                if writer is None:                       # start a clip
                    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    path = os.path.join(args.out_dir, stamp + ".mp4")
                    fh, fw = frame.shape[:2]             # encode the real frame size
                    writer = FfmpegWriter(path, fw, fh, fps, crf=args.crf)
                    if not writer.isOpened():
                        print("ERROR: could not start the ffmpeg encoder — not recording.")
                        writer = None
                        continue
                    start_t = datetime.datetime.now().timestamp()
                    frames = 0
                    print(f"REC start -> {path}")
                else:                                    # stop + save the clip
                    writer.release()
                    print(f"Saved {path}  ({frames} frames)")
                    writer = None
            elif k in (ord("q"), 27):
                break
    finally:
        if writer is not None:                           # quit mid-recording -> still save
            writer.release()
            print(f"Saved {path}  ({frames} frames)")
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
