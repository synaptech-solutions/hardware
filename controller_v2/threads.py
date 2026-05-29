"""Vision, telemetry, and serial threads + helpers.

Lifted from tag_hover_controller.py (proven). Reuses setup/live_telemetry.py
for CRSF frame building and parsing. All tunables come from controller_v2.config.
"""
import os
import sys
import time
import math
import array
import fcntl
import threading
import contextlib

import cv2
import numpy as np
from pupil_apriltags import Detector

from . import config


# live_telemetry lives in <repo>/setup/live_telemetry.py. Add it to sys.path
# so this module can be imported standalone.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if os.path.join(REPO, "setup") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "setup"))
import serial  # noqa: E402
from live_telemetry import (  # noqa: E402
    build_rc_channels_packed, build_device_ping, autodetect_port, CrsfParser,
    decode_flight_mode, decode_battery, decode_attitude,
    T_FLIGHT_MODE, T_BATTERY, T_ATTITUDE,
)


# Custom-baud open via TCSETS2 / BOTHER. pyserial's normal open path calls
# tcsetattr with B-constants and rejects 420000 on this kernel/pyserial combo
# (termios EINVAL). Switching via the kernel's custom-baud ioctl is the fix.
_TCGETS2, _TCSETS2 = 0x802C542A, 0x402C542B
_BOTHER, _CBAUD = 0o010000, 0o010017


def open_ranger(port, baud=420000):
    ser = serial.Serial(port, baudrate=115200, timeout=0.02)
    buf = array.array('i', [0] * 64)
    fcntl.ioctl(ser.fileno(), _TCGETS2, buf)
    buf[2] = (buf[2] & ~_CBAUD) | _BOTHER
    buf[9] = buf[10] = baud
    fcntl.ioctl(ser.fileno(), _TCSETS2, buf)
    return ser


@contextlib.contextmanager
def suppress_c_stderr():
    """Silence C-library stderr (AprilTag's pose-ambiguity spam) for a call."""
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


def cam_to_world(t_cam, tilt_deg, roll_deg, pitch_deg):
    """Camera-frame point → tag-facing gravity-leveled body frame.
    Returns (fwd, lat, up): +fwd ahead, +lat right, +up up.

    Yaw is intentionally NOT undone — the output XY is in the current
    body-yaw-aligned frame (not a global world frame).
    """
    x, y, z = float(t_cam[0]), float(t_cam[1]), float(t_cam[2])
    # 1) Undo body roll about the optical axis.
    rr = -math.radians(config.SIGN_IMU_ROLL * roll_deg)
    cr, sr = math.cos(rr), math.sin(rr)
    x, y = cr * x + sr * y, -sr * x + cr * y
    # 2) Undo total pitch about the camera right axis (fixed mount + live pitch).
    a = math.radians(tilt_deg + config.SIGN_IMU_PITCH * pitch_deg)
    ca, sa = math.cos(a), math.sin(a)
    y_lvl = ca * y - sa * z
    z_lvl = sa * y + ca * z
    return float(z_lvl), float(x), float(-y_lvl)   # fwd, lat, up


class Shared:
    """Vision output — written by vision thread, read by main."""
    def __init__(self):
        self.lock = threading.Lock()
        self.pos = None
        self.pos_stamp = 0.0
        self.decision_margin = 0.0
        self.ids_seen = ""
        self.detect_label = ""
        self.has_tag = False
        self.frame = None
        self.running = True
        self.vision_status = "vision init"
        self.vision_dbg = ""


class TxState:
    """RC channels + telemetry. Main writes channels; TX thread sends at
    TX_HZ; RX thread decodes telemetry + IMU into these fields."""
    def __init__(self):
        self.lock = threading.Lock()
        # TX and RX both write to the same port (RC frames + ping/MSP).
        # pyserial.write isn't atomic for multi-byte payloads, so concurrent
        # writes corrupt CRSF frames — guard every ser.write with this lock.
        self.ser_write_lock = threading.Lock()
        self.bytes_rx = 0
        self.frames_rx = 0
        self.ch = [config.NEUTRAL_US] * 16
        self.ch[config.CH_THR] = config.IDLE_THR_US
        self.ch[config.CH_ARM] = config.ARM_LOW_US
        self.ch[config.CH_PREARM] = config.PREARM_LOW_US
        self.ch[config.CH_MODE] = config.MODE_ANGLE_US
        self.flight_mode = None
        self.pack_v = None
        self.roll_deg = 0.0
        self.pitch_deg = 0.0
        self.yaw_deg = 0.0
        self.att_stamp = 0.0
        self.stop = False

    def set(self, ch):
        with self.lock:
            self.ch = list(ch)

    def get(self):
        with self.lock:
            return list(self.ch)


def _safe_write(ser, tx, payload):
    """Serial write under the shared write-lock (TX and RX both write)."""
    with tx.ser_write_lock:
        try:
            ser.write(payload)
            return True
        except Exception as e:
            print(f"[serial] write failed: {e}", flush=True)
            return False


def tx_loop(ser, tx):
    period = 1.0 / config.TX_HZ
    nxt = time.monotonic()
    while not tx.stop:
        if not _safe_write(ser, tx, build_rc_channels_packed(tx.get())):
            return
        nxt += period
        s = nxt - time.monotonic()
        if s > 0:
            time.sleep(s)
        else:
            nxt = time.monotonic()


def rx_loop(ser, tx):
    parser = CrsfParser()
    last_ping = 0.0
    while not tx.stop:
        try:
            chunk = ser.read(256)
        except Exception as e:
            print(f"[serial] read failed: {e}", flush=True)
            return
        if chunk:
            tx.bytes_rx += len(chunk)
            for ftype, payload in parser.feed(chunk):
                tx.frames_rx += 1
                if ftype == T_FLIGHT_MODE:
                    tx.flight_mode = decode_flight_mode(payload)
                elif ftype == T_BATTERY:
                    b = decode_battery(payload)
                    if b:
                        tx.pack_v = b["voltage_V"]
                elif ftype == T_ATTITUDE:
                    a = decode_attitude(payload)
                    if a:
                        tx.roll_deg = float(a["roll_deg"])
                        tx.pitch_deg = float(a["pitch_deg"])
                        tx.yaw_deg = float(a["yaw_deg"])
                        tx.att_stamp = time.monotonic()
        now = time.monotonic()
        if now - last_ping > 2.0:
            if not _safe_write(ser, tx, build_device_ping()):
                return
            last_ping = now


def vision_loop(shared, tx, K, dist, cam_params):
    detector = Detector(families=config.TAG_FAMILY, nthreads=4,
                        quad_decimate=1.5, quad_sigma=0.0, refine_edges=0,
                        decode_sharpening=0.25)
    cap = cv2.VideoCapture(config.DEVICE_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("ERROR: cannot open camera; vision thread exiting")
        shared.running = False
        return
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    undist_map1 = undist_map2 = None
    if np.any(dist):
        undist_map1, undist_map2 = cv2.initUndistortRectifyMap(
            K, dist, None, K, (config.WIDTH, config.HEIGHT), cv2.CV_16SC2)

    fx, fy, cx, cy = cam_params
    last_tag_center_px = None
    last_good_stamp = None
    miss = 0
    stat_t0 = time.monotonic()
    n_read = n_hit = 0

    def select_target(detections):
        cands = [d for d in detections if d.tag_id == config.TARGET_TAG_ID]
        if not cands:
            return None, 0
        return (max(cands, key=lambda d: float(getattr(d, "decision_margin", 0.0))),
                len(cands))

    while shared.running:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.003)
            continue
        n_read += 1
        if undist_map1 is not None:
            frame = cv2.remap(frame, undist_map1, undist_map2, cv2.INTER_LINEAR)
        now = time.monotonic()
        if now - stat_t0 >= 0.5:
            elapsed = now - stat_t0
            shared.vision_status = (f"capFPS={n_read/elapsed:.0f} "
                                    f"detFPS={n_hit/elapsed:.0f}")
            stat_t0, n_read, n_hit = now, 0, 0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Pass 1: full-frame.
        with suppress_c_stderr():
            tags = detector.detect(
                gray, estimate_tag_pose=True,
                camera_params=cam_params, tag_size=config.TAG_SIZE_M,
            )
        selected, cand_count = select_target(tags)
        detect_label = "full" if selected is not None else ""
        offset_xy = (0.0, 0.0)

        # Pass 2: CLAHE fallback (every few misses).
        if selected is None and (miss % 3 == 0):
            with suppress_c_stderr():
                tags = detector.detect(
                    clahe.apply(gray), estimate_tag_pose=True,
                    camera_params=cam_params, tag_size=config.TAG_SIZE_M,
                )
            selected, cand_count = select_target(tags)
            if selected is not None:
                detect_label = "clahe"

        # Pass 3: ROI around last good center.
        if (selected is None and last_tag_center_px is not None
                and last_good_stamp is not None
                and (now - last_good_stamp) < config.VISION_ROI_RECOVERY_S):
            cx_px, cy_px = int(last_tag_center_px[0]), int(last_tag_center_px[1])
            r = config.VISION_ROI_HALF_SIZE_PX
            h, w = gray.shape[:2]
            x0, y0 = max(0, cx_px - r), max(0, cy_px - r)
            x1, y1 = min(w, cx_px + r), min(h, cy_px + r)
            if (x1 - x0) >= 80 and (y1 - y0) >= 80:
                roi = gray[y0:y1, x0:x1]
                roi_params = (fx, fy, cx - x0, cy - y0)
                with suppress_c_stderr():
                    tags = detector.detect(
                        roi, estimate_tag_pose=True,
                        camera_params=roi_params, tag_size=config.TAG_SIZE_M,
                    )
                selected, cand_count = select_target(tags)
                if selected is not None:
                    detect_label = "roi"
                    offset_xy = (float(x0), float(y0))

        if selected is not None:
            miss = 0
            n_hit += 1
            dm = float(getattr(selected, "decision_margin", 0.0))
            tvec = selected.pose_t.ravel()
            fwd, lat, up = cam_to_world(tvec, config.CAMERA_TILT_DEG,
                                        tx.roll_deg, tx.pitch_deg)
            plausible = (dm >= config.MIN_TAG_DECISION_MARGIN
                         and 0.10 <= fwd <= config.MAX_TRACK_RANGE_M
                         and abs(lat) <= config.MAX_TRACK_LAT_M
                         and abs(up) <= config.MAX_TRACK_UP_M)
            reject_reason = "ok"
            if not plausible:
                reasons = []
                if dm < config.MIN_TAG_DECISION_MARGIN:
                    reasons.append("low_dm")
                if not (0.10 <= fwd <= config.MAX_TRACK_RANGE_M):
                    reasons.append("fwd_range")
                if abs(lat) > config.MAX_TRACK_LAT_M:
                    reasons.append("lat_range")
                if abs(up) > config.MAX_TRACK_UP_M:
                    reasons.append("up_range")
                reject_reason = "+".join(reasons) if reasons else "rejected"
            vis_dbg = (f"pass={detect_label or 'none'} cands={cand_count} "
                       f"sel_id={int(selected.tag_id)} dm={dm:.1f} "
                       f"plaus={'1' if plausible else '0'} reason={reject_reason} "
                       f"fwd={fwd:+.3f} lat={lat:+.3f} up={up:+.3f}")
            if plausible:
                with shared.lock:
                    shared.pos = (fwd, lat, up)
                    shared.pos_stamp = now
                    shared.decision_margin = dm
                    shared.ids_seen = str(int(selected.tag_id))
                    shared.detect_label = detect_label
                    shared.has_tag = True
                    shared.vision_dbg = vis_dbg
                ox, oy = offset_xy
                poly = (selected.corners + np.array([ox, oy])).astype(int)
                cv2.polylines(frame, [poly], True, (0, 255, 0), 2)
                last_tag_center_px = poly.mean(axis=0)
                last_good_stamp = now
                cv2.putText(frame,
                            f"fwd={fwd*100:5.1f} lat={lat*100:+5.1f} "
                            f"up={up*100:+5.1f}cm dm={dm:.0f} "
                            f"[{detect_label} id={int(selected.tag_id)}]",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0), 2)
            else:
                with shared.lock:
                    shared.has_tag = False
                    shared.vision_dbg = vis_dbg
                cv2.putText(frame, "TAG REJECTED", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        else:
            miss += 1
            with shared.lock:
                shared.has_tag = False
                shared.vision_dbg = (f"pass=none cands=0 sel_id=-1 dm=0.0 "
                                     f"plaus=0 reason=no_target miss={miss}")
            cv2.putText(frame, "NO TARGET", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        with shared.lock:
            shared.frame = frame
    cap.release()
