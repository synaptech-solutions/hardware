"""Standalone drone video recorder — just the feed + a record toggle.

Opens the drone's video feed in a window. SPACE starts a recording; SPACE again
stops it and saves the clip to DataExchange/<timestamp>.mp4. You can record as
many clips as you like in one session; Q (or ESC) quits. No ViCON, no blackbox,
no sync — this is purely for grabbing video off the feed.

Camera setup matches the rest of the repo (drone_control/controller_v2/config.py:
/dev/video4, 1280x720 MJPG V4L2; auto-tries video5 if the Cam Link re-enumerated).

Run with the cv2 venv:
  ../.venv/bin/python simple_video_recorder.py
  ../.venv/bin/python simple_video_recorder.py --device 5 --out-dir /some/where
"""
import os
import argparse
import datetime

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))


def open_camera(device, width, height):
    """Open /dev/video<device>, falling back to device+1 (Cam Link 4<->5 shift)."""
    for dev in (device, device + 1):
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, 60)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"Camera: /dev/video{dev} {width}x{height}")
            return cap, dev
        cap.release()
    return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", type=int, default=4, help="camera /dev/videoN (default 4)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "DataExchange"),
                    help="where clips are saved (default DataExchange/)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cap, dev = open_camera(args.device, args.width, args.height)
    if cap is None:
        raise SystemExit(f"could not open /dev/video{args.device} (or {args.device + 1}) — "
                         "is the Cam Link plugged in and free?")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 1 else 30.0

    WIN = "drone feed  [SPACE]=record/stop  [Q]=quit"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    writer = None
    path = None
    start_t = None
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
                    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                                             fps, (args.width, args.height))
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
