#!/usr/bin/env python3
"""Standalone VIDEO + VICON recorder — flight-format, no flying, no TX12/Ranger.

Captures just the drone's video feed AND the Vicon pose, into the SAME per-session
layout a real flight produces — because it uses the exact flight recorders
(VideoRecorder + ViconRecorder + channels settings), not a separate path. So each
clip drops straight into combine.py + the dashboard like a flight, with no commands,
telemetry, blackbox or control:

  flight_logs/<stamp>/
    video.mkv          H.264/MKV, channels.WIDTH×HEIGHT → VIDEO_OUT_HEIGHT (identical
                       to flight footage)
    video_frames.csv   per-frame real capture times (frame_idx, t_rel, t_wall)
    vicon.mat          Vicon pose @100 Hz, Abs_time stamped t_rel = wall - t0
    session.json       t0 + per-stream offsets + config snapshot (combine-ready)

Both streams start on ONE shared t0 (SPACE), so video and Vicon are aligned exactly
as in a flight. Afterwards: `Vicon_control/combine.py` → flight_synced.csv, then the
dashboard plays the pose + video back together.

Controls (in the preview window): SPACE = start a clip / stop & save · Q/ESC = quit.
The preview shows while recording (the recorder owns the camera then). Vicon must be
streaming on UDP :51001. Record as many clips as you like per session.

Run with the repo venv (cv2 + ffmpeg on it; Vicon Tracker streaming to this laptop):
  .venv/bin/python drone_control/Vicon_control/vicon_video_recorder.py
  .venv/bin/python drone_control/Vicon_control/vicon_video_recorder.py --out-dir /some/where
"""
import argparse
import datetime
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_DRONE_CONTROL = os.path.dirname(HERE)
if _DRONE_CONTROL not in sys.path:
    sys.path.insert(0, _DRONE_CONTROL)

from common import channels                                  # noqa: E402
from common.recorders import (                               # noqa: E402
    VideoRecorder, ViconRecorder, write_session_json, warn_vicon_off,
    cv2, CV2_OK, FFMPEG_OK, VICON_OK, _VICON_ERR)
from Vicon_control import config                             # noqa: E402
from Vicon_control.vicon_source import ViconPoseSource       # noqa: E402

REC_DIR = os.path.join(HERE, "flight_logs")


def _placeholder(text):
    img = np.full((360, 640, 3), 30, np.uint8)
    cv2.putText(img, text, (24, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=REC_DIR,
                    help="where <timestamp>/ session folders land (default flight_logs/)")
    ap.add_argument("--device", type=int, default=channels.DEVICE_INDEX,
                    help=f"camera /dev/videoN (default channels.DEVICE_INDEX={channels.DEVICE_INDEX})")
    args = ap.parse_args()

    if not CV2_OK:
        sys.exit("OpenCV (cv2) not available — run with the repo .venv.")
    if not FFMPEG_OK:
        sys.exit("ffmpeg not found on PATH — install it (e.g. apt install ffmpeg).")
    if not VICON_OK:
        sys.exit(f"Vicon deps missing ({_VICON_ERR}) — run with the repo .venv.")
    os.makedirs(args.out_dir, exist_ok=True)

    # Vicon source owns the ONE UDP receiver; the recorder shares it (one socket).
    source = ViconPoseSource(yaw_offset_deg=config.VICON_YAW_OFFSET_DEG)
    print(f"Vicon:    probing UDP :{source.port} for a stream …")
    if not source.prepare():
        warn_vicon_off(source.status, source.port)
        sys.exit("Vicon is half the point here — start Vicon Tracker (streaming to "
                 "this laptop), then re-run.")
    print(f"Vicon:    {source.status}")

    # The SAME recorders + settings the flight loop builds → flight-identical files.
    recorder = VideoRecorder(args.device, channels.WIDTH, channels.HEIGHT,
                             out_height=getattr(channels, "VIDEO_OUT_HEIGHT", None))
    vicon_rec = ViconRecorder()
    vicon_rec.prepare(external_udp=source.udp)               # share the receiver
    out_h = getattr(channels, "VIDEO_OUT_HEIGHT", None)
    print(f"Recorder: video {channels.WIDTH}x{channels.HEIGHT}"
          f"{f'->{out_h}px tall' if out_h else ''} crf{recorder.crf} + vicon {vicon_rec.status} "
          f"— flight-format → {args.out_dir}/<stamp>/")
    print("SPACE = start/stop recording   Q/ESC = quit")

    session = {"dir": None, "t0": None, "stamp": None}

    def begin():
        t0 = datetime.datetime.now().timestamp()
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        sdir = os.path.join(args.out_dir, stamp)
        os.makedirs(os.path.join(sdir, "blackbox"), exist_ok=True)
        session.update(dir=sdir, t0=t0, stamp=stamp)
        recorder.start(t0, os.path.join(sdir, "video.mkv"))
        vicon_rec.start(t0, os.path.join(sdir, "vicon.mat"))
        print(f"\n● REC {stamp} → {sdir}")

    def end():
        recorder.stop()
        vicon_rec.stop()
        if session["dir"]:
            write_session_json(session, recorder, vicon_rec, None, None, None, None, None,
                               extra={"controller": {"kind": "vicon_video_recorder",
                                                     "note": "standalone video+vicon, no flight"}})
            print(f"■ SAVED {session['stamp']}  video {recorder.frames} frames, "
                  f"vicon {vicon_rec.samples} samples → {session['dir']}\n"
                  f"   combine: Vicon_control/combine.py {session['dir']}")
        session.update(dir=None, t0=None, stamp=None)

    WIN = "drone feed + vicon  [SPACE]=record/stop  [Q]=quit"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    try:
        while True:
            recording = session["dir"] is not None
            if recording:
                frame = recorder.get_latest_frame()
                disp = (frame.copy() if frame is not None
                        else _placeholder("starting camera ..."))
                if frame is not None:
                    elapsed = datetime.datetime.now().timestamp() - session["t0"]
                    mm, ss = divmod(int(elapsed), 60)
                    cv2.putText(disp,
                                f"REC {mm}:{ss:02d}  vid {recorder.frames}  vic {vicon_rec.samples}",
                                (14, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            else:
                disp = _placeholder("SPACE = record    Q = quit")
            cv2.imshow(WIN, disp)

            k = cv2.waitKey(30) & 0xFF
            if k == ord(" "):
                end() if recording else begin()
            elif k in (ord("q"), 27):
                break
    finally:
        if session["dir"]:
            end()
        cv2.destroyAllWindows()
        source.stop()
        print("Stopped.")


if __name__ == "__main__":
    main()
