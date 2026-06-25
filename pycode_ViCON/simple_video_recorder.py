"""Standalone drone video recorder — IDENTICAL output to a real flight, no flying.

Grabs the drone's video feed and saves clips that are byte-for-byte the same kind
of file a real flight produces — because it records with the EXACT same VideoRecorder
and channels settings the flight data pipeline uses (vicon_hover.py / joystick_flight.py),
not its own encoder. No Vicon, no blackbox, no commands/telemetry — just the video.

Each clip → <out-dir>/<timestamp>/{video.mkv, video_frames.csv}, the same pair a
flight writes: same camera (channels.DEVICE_INDEX), same capture resolution
(channels.WIDTH×HEIGHT), same uniform downscale (channels.VIDEO_OUT_HEIGHT), same
H.264/libx264 CRF, and the same per-frame real-capture-time sidecar. So a clip from
here drops straight into the same review/extraction tooling as flight footage.

Controls (in the preview window):
  SPACE  start a clip / stop & save     Q or ESC  quit
The preview shows WHILE recording (the recorder owns the camera then, exactly as in
flight); idle shows a prompt. Record as many clips as you like in one session.

Run with the repo venv (ffmpeg + cv2 on it):
  .venv/bin/python pycode_ViCON/simple_video_recorder.py
  .venv/bin/python pycode_ViCON/simple_video_recorder.py --out-dir /some/where
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(HERE)
_DRONE_CONTROL = os.path.join(_REPO, "drone_control")
if _DRONE_CONTROL not in sys.path:
    sys.path.insert(0, _DRONE_CONTROL)

from common import channels                                  # noqa: E402
from common.recorders import (                               # noqa: E402
    VideoRecorder, cv2, CV2_OK, FFMPEG_OK)


def _placeholder(text):
    """A small dark frame with a prompt, so the window + key handling work while
    idle (the recorder only owns the camera — hence a live preview — while recording)."""
    img = np.full((360, 640, 3), 30, np.uint8)
    cv2.putText(img, text, (24, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2)
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "DataExchange"),
                    help="where clips are saved, one <timestamp>/ folder each "
                         "(default pycode_ViCON/DataExchange/)")
    ap.add_argument("--device", type=int, default=channels.DEVICE_INDEX,
                    help=f"camera /dev/videoN (default channels.DEVICE_INDEX={channels.DEVICE_INDEX}, "
                         "same as flights; change if the Cam Link re-enumerated)")
    args = ap.parse_args()

    if not CV2_OK:
        raise SystemExit("OpenCV (cv2) not available — run with the repo .venv.")
    if not FFMPEG_OK:
        raise SystemExit("ffmpeg not found on PATH — install it (e.g. apt install ffmpeg).")
    os.makedirs(args.out_dir, exist_ok=True)

    # The SAME recorder + settings the flight loop builds — this is what makes the
    # output identical (resolution/aspect/downscale/encode/video_frames.csv).
    recorder = VideoRecorder(args.device, channels.WIDTH, channels.HEIGHT,
                             out_height=getattr(channels, "VIDEO_OUT_HEIGHT", None))
    out_h = getattr(channels, "VIDEO_OUT_HEIGHT", None)
    print(f"Recorder: capture {channels.WIDTH}x{channels.HEIGHT} on /dev/video{args.device}"
          f"{f' -> downscale to {out_h}px tall' if out_h else ''}, "
          f"H.264/MKV crf{recorder.crf} + video_frames.csv — identical to a flight clip.")
    print("SPACE = start/stop recording   Q/ESC = quit")

    WIN = "drone feed  [SPACE]=record/stop  [Q]=quit"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    path = None
    try:
        while True:
            if recorder.recording:
                frame = recorder.get_latest_frame()
                if frame is None:
                    disp = _placeholder("starting camera ...")
                else:
                    disp = frame.copy()
                    cv2.putText(disp, f"REC  {recorder.frames} frames", (14, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            else:
                disp = _placeholder("SPACE = record    Q = quit")
            cv2.imshow(WIN, disp)

            k = cv2.waitKey(30) & 0xFF
            if k == ord(" "):
                if not recorder.recording:                    # start a clip
                    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    sdir = os.path.join(args.out_dir, stamp)
                    os.makedirs(sdir, exist_ok=True)
                    path = os.path.join(sdir, "video.mkv")
                    recorder.start(datetime.datetime.now().timestamp(), path)
                    print(f"REC start -> {path}")
                else:                                          # stop + save
                    recorder.stop()
                    print(f"Saved {path}  ({recorder.frames} frames)  | {recorder.status}")
            elif k in (ord("q"), 27):
                break
    finally:
        if recorder.recording:                                # quit mid-clip → still save
            recorder.stop()
            print(f"Saved {path}  ({recorder.frames} frames)")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
