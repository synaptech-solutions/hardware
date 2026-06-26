#!/usr/bin/env python3
"""Live GateNet gate detection on the drone camera feed, with a SPACE-to-record UI.

The live window ALWAYS shows the camera feed with the gate-mask overlay (detection
runs every frame). Recording is a toggle:
  SPACE  start recording / stop+save the current clip
  q/ESC  quit

When recording, frames are encoded EXACTLY like a flight recording — the same ffmpeg
settings as common.recorders.VideoRecorder (raw BGR piped to libx264 crf18 in an MKV)
plus the paired video_frames.csv of true per-frame capture times. The ONLY difference
from flight: the gate overlay is burned into the encoded frames. Each SPACE-start makes
a new perception/runs/<stamp>/ clip.

  .venv/bin/python -m perception.run_gate_detect
  .venv/bin/python -m perception.run_gate_detect --device 4 --thr 0.4
"""
from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(HERE)
for _p in (_REPO, os.path.join(_REPO, "drone_control")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import channels                                   # noqa: E402
from common.recorders import FFMPEG_BIN, CV2_OK, FFMPEG_OK    # noqa: E402
from perception.predictor import GatePredictor                 # noqa: E402


def overlay_mask(proc_bgr: np.ndarray, prob: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """Gate-probability mask drawn on the (model-input) frame: green wash where
    prob>thr + a red outline. Returns a new BGR frame the size of proc_bgr."""
    h, w = proc_bgr.shape[:2]
    prob_full = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    out = proc_bgr.copy()
    mask = prob_full > thr
    if mask.any():
        green = np.zeros_like(out); green[..., 1] = 255
        out[mask] = cv2.addWeighted(out, 0.45, green, 0.55, 0)[mask]
        cnts, _ = cv2.findContours((mask.astype(np.uint8)) * 255,
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (0, 0, 255), 2)
    return out, mask, float(prob_full.max())


def _fmt_dur(s: float) -> str:
    m, sec = divmod(int(s), 60)
    return f"{m:d}:{sec:02d}"


def draw_hud(frame, *, recording, rec_secs, clips, frames, fps, gate_px, maxp, infer_ms):
    """Top status bar + REC indicator burned onto the displayed (and, while recording,
    encoded) frame so the playback carries the same HUD you saw live."""
    h, w = frame.shape[:2]
    bar_h = 34
    cv2.rectangle(frame, (0, 0), (w, bar_h), (0, 0, 0), -1)
    if recording:
        # red dot + REC time
        cv2.circle(frame, (16, bar_h // 2), 8, (0, 0, 255), -1)
        left = f"REC {_fmt_dur(rec_secs)}  f{frames}@{fps:.0f}fps"
        lcol = (0, 0, 255)
    else:
        left = "READY  [SPACE]=record  [q]=quit"
        lcol = (0, 255, 0)
    cv2.putText(frame, left, (30, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, lcol, 2)
    right = f"gate {gate_px}px  p{maxp:.2f}  {infer_ms:.0f}ms  clips:{clips}"
    (tw, _), _ = cv2.getTextSize(right, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(frame, right, (w - tw - 10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 255), 1)
    return frame


class ClipWriter:
    """ffmpeg encoder for ONE clip — same settings as common.recorders.VideoRecorder
    (raw bgr24 → libx264 crf18, yuv420p, -g 60, MKV) + a paired video_frames.csv of
    real capture times. start()/stop() bracket each SPACE-toggled clip."""

    def __init__(self, w, h, fps, crf=18):
        self.w, self.h, self.fps, self.crf = w, h, fps, crf
        self.proc = None
        self.path = None
        self.frame_times = []
        self.frames = 0
        self.t0 = None

    def start(self, sdir):
        os.makedirs(sdir, exist_ok=True)
        self.path = os.path.join(sdir, "gate_overlay.mkv")
        self._csv = os.path.join(sdir, "video_frames.csv")
        cmd = [
            FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-video_size", f"{self.w}x{self.h}", "-framerate", f"{self.fps:g}", "-i", "-",
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(self.crf),
            "-pix_fmt", "yuv420p", "-g", "60", self.path,
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.frame_times = []
        self.frames = 0
        self.t0 = time.time()
        return self.proc.stdin is not None

    def write(self, frame, cap_t):
        if self.proc is None:
            return
        if (frame.shape[1], frame.shape[0]) != (self.w, self.h):
            frame = cv2.resize(frame, (self.w, self.h))
        try:
            self.proc.stdin.write(frame.tobytes())
        except (BrokenPipeError, ValueError):
            return
        self.frame_times.append((self.frames, cap_t))
        self.frames += 1

    def stop(self):
        if self.proc is None:
            return None
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.wait(timeout=10.0)
        except Exception:
            self.proc.kill()
        # write the paired per-frame capture times (file frame N <-> row N)
        with open(self._csv, "w") as f:
            f.write("frame_idx,t_wall\n")
            for idx, t in self.frame_times:
                f.write(f"{idx},{t:.6f}\n")
        path, n = self.path, self.frames
        self.proc = None
        return path, n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", type=int, default=channels.DEVICE_INDEX)
    ap.add_argument("--width", type=int, default=channels.WIDTH)
    ap.add_argument("--height", type=int, default=channels.HEIGHT)
    ap.add_argument("--thr", type=float, default=0.5, help="gate-mask threshold")
    ap.add_argument("--out", default=os.path.join(HERE, "runs"))
    args = ap.parse_args()

    if not (CV2_OK and FFMPEG_OK):
        sys.exit("cv2/ffmpeg not available — need the repo .venv + ffmpeg on PATH.")

    print("Loading GateNet (CPU jax) …")
    predictor = GatePredictor()
    print(f"GateNet ready: size={predictor.size} undistort={predictor.undistort}")

    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not cap.isOpened():
        sys.exit(f"camera /dev/video{args.device} open FAILED")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    rep = cap.get(cv2.CAP_PROP_FPS)
    fps = rep if rep and rep > 1 else 30.0

    WIN = "GateNet gate detection"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    writer = None
    enc_wh = None
    clips = 0
    print("Live preview. SPACE = start/stop recording, q/ESC = quit.\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            cap_t = time.time()
            t = time.time()
            proc, prob = predictor(frame)            # rectified frame + prob mask
            infer_ms = (time.time() - t) * 1000.0
            overlaid, mask, maxp = overlay_mask(proc, prob, thr=args.thr)
            if enc_wh is None:
                enc_wh = (overlaid.shape[1], overlaid.shape[0])

            recording = writer is not None
            rec_secs = (time.time() - writer.t0) if recording else 0.0
            disp = draw_hud(overlaid.copy(), recording=recording, rec_secs=rec_secs,
                            clips=clips, frames=(writer.frames if recording else 0),
                            fps=fps, gate_px=int(mask.sum()), maxp=maxp, infer_ms=infer_ms)
            # While recording, encode the SAME HUD'd frame the user sees.
            if recording:
                writer.write(disp, cap_t)

            cv2.imshow(WIN, disp)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord(" "):
                if writer is None:                   # START
                    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    sdir = os.path.join(args.out, stamp)
                    writer = ClipWriter(enc_wh[0], enc_wh[1], fps)
                    writer.start(sdir)
                    print(f"\n● REC start → {os.path.join(sdir, 'gate_overlay.mkv')}")
                else:                                # STOP + SAVE
                    path, n = writer.stop()
                    clips += 1
                    print(f"■ saved {path}  ({n} frames)")
                    writer = None
    except KeyboardInterrupt:
        print("\nCtrl-C — stopping.")
    finally:
        if writer is not None:
            path, n = writer.stop()
            print(f"■ saved {path}  ({n} frames)")
        cap.release()
        cv2.destroyAllWindows()
        print("Stopped.")


if __name__ == "__main__":
    main()
