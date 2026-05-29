"""AprilTag hover controller — take off, climb to tag altitude, hold position.

State estimation
----------------
Position: per-axis low-pass on the tag-frame position from AprilTag PnP.
On vision dropout, position is HELD (never extrapolated) — dead-reckoning
position across dropouts was the snap-back crash failure mode.

Velocity: LPF of the filtered-position delta. Decays toward zero on vision
loss. Tried IMU integration once — motor vibration swung body_z accel from
~6 to ~11 m/s² across a few loop ticks, so the integral was garbage and the
controller couldn't see the real climb. Position-delta velocity is noisier
per-sample but actually tracks reality.

Controller
----------
Per axis (lat, up, fwd) a velocity-loop PD:
  v_target = clamp(-Kp * (pos - target), -v_max, v_max)
  u        = Kv * (v - v_target)
Position error sets a TARGET velocity (capped); velocity error drives the
lean command or throttle. v_max is the main safety — even with a saturated
stick the drone can't accelerate past walking pace.

Vertical adds a slow integral on velocity error to learn hover throttle as
battery sags, and an ASYMMETRIC throttle trim (small upward, larger downward)
so a tag falling out of view can't command a runaway climb.

Pipeline
--------
Cam Link (/dev/video4) → AprilTag corners → solvePnP → tag-frame position
CRSF telem → IMU R/P/Y (level the pose)
                          ↓
                  PoseFilter (hold-on-loss)
                          ↓
          velocity-loop PD → roll / pitch / yaw / throttle
                          ↓
                CRSF RC channels → drone (ANGLE mode)

Run:  .venv/bin/python tag_hover_controller.py
"""
import os
import sys
import time
import math
import array
import fcntl
import threading
import contextlib
from datetime import datetime

import cv2
import numpy as np
from pupil_apriltags import Detector


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


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "setup"))
import serial  # noqa: E402
from live_telemetry import (  # noqa: E402
    build_rc_channels_packed, build_device_ping, autodetect_port, CrsfParser,
    decode_flight_mode, decode_battery, decode_attitude,
    T_FLIGHT_MODE, T_BATTERY, T_ATTITUDE,
)

CALIB_FILE = os.path.join(HERE, "camera_setup", "camera_calibration.npz")
LOG_DIR = os.path.join(HERE, "flight_logs")
LOG_FILE = os.path.join(LOG_DIR, "hover_controller.log")


# Custom baud via TCSETS2 / BOTHER. pyserial's normal open path calls
# tcsetattr with B-constants and rejects 420000 on this kernel/pyserial combo
# (termios EINVAL). Opening at 115200 first and then switching via the
# kernel's custom-baud ioctl is the platform-correct fallback.
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


# =================== TUNABLES (edit before flying) ===========================

DRY_RUN = False           # True = never arm; print sticks only

# Target tags (world frame: tag 0's center is the origin; +X=fwd away from
# camera, +Y=lat right, +Z=up). Multi-tag joint solvePnP for robustness.
TARGET_TAG_IDS = (0, 1)
TAG_WORLD_OFFSETS = {
    0: (0.00, 0.00, 0.00),
    1: (0.52, 0.00, 0.00),     # tag 1 is 52cm right of tag 0
}
TAG_SIZE_M = 0.078

# Hover goal: rise to the tag altitude, stay centered over tag 0 in lat,
# hold the takeoff distance in fwd (locked at arm time).
TARGET_LAT_M = 0.0
TARGET_UP_M = 0.0          # 0 = level with tag
TARGET_FWD_FALLBACK_M = 0.75   # used only if arm-time lock fails

# Camera geometry. Tune CAMERA_TILT_DEG by holding drone level with tag dead
# ahead at same height; the printed 'up' should read ~0.
CAMERA_TILT_DEG = 15.0

# Vision input
DEVICE_INDEX = 4
WIDTH, HEIGHT = 960, 540
TAG_FAMILY = "tag25h9"
MIN_TAG_DECISION_MARGIN = 18.0
MAX_TRACK_RANGE_M = 2.5
MAX_TRACK_LAT_M = 1.5
MAX_TRACK_UP_M = 1.8
VISION_ROI_RECOVERY_S = 1.0
VISION_ROI_HALF_SIZE_PX = 260

# CRSF channels (matches Air75 Betaflight dump)
CH_ROLL, CH_PITCH, CH_THR, CH_YAW, CH_ARM = 0, 1, 2, 3, 5
CH_MODE, CH_PREARM = 6, 9
ARM_HIGH_US, ARM_LOW_US = 1500, 1000
PREARM_HIGH_US, PREARM_LOW_US = 1800, 1000
MODE_ANGLE_US = 1500
NEUTRAL_US = 1500
IDLE_THR_US = 1000

# Stick signs (verify in DRY_RUN before flying)
SIGN_PITCH = +1            # +pitch us → drone pitches forward
SIGN_ROLL = +1             # +roll  us → drone rolls right
SIGN_YAW = +1              # +yaw   us → drone yaws right

# IMU R/P signs for the camera-leveling rotation in cam_to_world. Verify in
# DRY_RUN: hold drone level with the tag dead ahead, then tilt by hand —
# the printed (fwd, lat, up) should stay roughly constant. Flip a sign if
# one of the printed components moves in the wrong direction on tilt.
# This drone's FC reports +pitch = nose DOWN (non-standard), so SIGN_IMU_PITCH
# flips it to the standard +pitch=nose-up convention used by the math.
SIGN_IMU_ROLL = +1
SIGN_IMU_PITCH = -1

# Hover throttle and limits. Use motor_ramp_simple to find the rough hover
# throttle, then refine in flight (the integrator will track battery sag).
HOVER_THROTTLE_US = 1370
MAX_THROTTLE_US = 1550
# Asymmetric throttle authority: small upward trim so a tag falling out of
# view can't command a runaway climb; larger downward trim because the camera
# CAN see the drone fall (tag stays in frame). Net throttle = HOVER + corr.
THR_CLIMB_TRIM_US = 15     # max +correction (PID above HOVER)
THR_DESC_TRIM_US = 80      # max -correction (PID below HOVER)
THR_SLEW_US_PER_S = 600.0  # smooths takeoff and prevents motor jumps

# --- controller gains (velocity-loop PD per axis) ---
# Approach-speed caps. v_max sets the cruising speed when far from target;
# combined with the lean limit, these define the worst-case overshoot if
# vision drops out mid-flight. Keep modest — recovery range is small.
VMAX_LAT_MPS = 0.40        # max lateral approach speed
VMAX_FWD_MPS = 0.40        # max forward/back approach speed
VMAX_UP_MPS = 0.30         # max climb / descend rate (small: a fast climb can
                           # take the tag off-screen above the camera's VFOV)

# Position→velocity gains (1/s, i.e. how aggressively position error maps to
# target velocity). With v_max as a cap, going higher just saturates sooner.
KP_LAT = 1.2
KP_FWD = 1.2
KP_UP = 1.4

# Velocity error → stick gains. KV is bounded by realized-lean lag (~150 ms on
# the FC's Angle loop) — too high and the loop self-oscillates regardless of
# how clean the position estimate is. These are the values from the 10:13
# flight that stayed airborne ~40 s with position-delta velocity. Earlier
# attempts at IMU-integrated velocity needed gains 3-4× higher because the
# IMU-derived v was small; restoring those high gains here with the
# (correct, larger) position-delta v would oscillate violently.
KV_LAT_US_PER_MPS = 70.0    # roll us per (m/s) of v-error
KV_FWD_US_PER_MPS = 50.0    # pitch us per (m/s) of v-error
KV_UP_US_PER_MPS = 40.0     # throttle us per (m/s) of v-error

# Vertical integral on velocity error (learns hover throttle for battery sag).
KI_UP_US_PER_MPS = 25.0
UP_INT_MAX_US = 60

# Yaw: simple P on bearing to keep the tag centered horizontally.
KP_YAW_US_PER_RAD = 280.0
MAX_YAW_US = 90

# Output clamps
MAX_PITCH_US = 200
MAX_ROLL_US = 200

# --- pose filter (LPF on tag position; velocity from filtered-position delta) ---
# Position: snap toward the vision measurement at ALPHA_POS each fresh update,
# hold on dropout. Velocity: derivative of the FILTERED position, then LPF'd
# again so per-frame PnP corner jitter doesn't read as 1-2 m/s on the ground.
# Decays toward zero on vision loss (never extrapolated — the snap-back
# failure mode was dead-reckoning velocity through a dropout).
ALPHA_POS = 0.40
ALPHA_VEL = 0.25
VISION_FRESH_S = 0.20        # consider vision usable up to this age
ESTIMATOR_MAX_DT_S = 0.05    # cap step (safety against stutters)
VEL_DECAY_TAU_S = 0.25       # velocity bleeds with this time constant on loss
# If the filter recovers fresh vision after this long blind, the position
# jump is not real motion — re-seed pos to the vision reading and zero v.
RESEED_AFTER_LOSS_S = 0.10
# While throttle is more than this far BELOW hover, the drone is still on
# the ground. Pin v=0 each update so pre-takeoff pose jitter can't saturate
# the sticks.
GROUND_PIN_BAND_US = 60

# --- tag-loss behavior ---
# When the pose filter hasn't had a fresh vision update for STATE_MAX_AGE_S,
# bleed the throttle down to land softly, then disarm after KILL_AFTER_LOST_S.
STATE_MAX_AGE_S = 0.35       # filter usable up to this much after last vision
LAND_DESCENT_US_PER_S = 130.0
KILL_AFTER_LOST_S = 3.5

# --- arming ---
# ARM_CONFIRM_S: max wait for the FC to clear its ARMING_DISABLED flags after
# we raise the arm switch. Long enough to cover a fresh power-on gyro
# calibration (~3-5 s) — otherwise mode shows e.g. 'STAB*' the whole time
# (the trailing '*' is Betaflight's "arming disabled" flag).
ARM_CONFIRM_S = 8.0
ARM_LOCK_MIN_S = 0.45

# --- battery cutoff (1S whoop; USB-only reads ~0.7V so we gate on BATT_PRESENT) ---
LOW_BATT_CUTOFF = True
BATT_PRESENT_V = 2.5
MIN_CELL_V = 3.3
CELLS = 1

# --- loop rates ---
TX_HZ = 50.0
DISPLAY_HZ = 30.0
LOG_HZ = 12.0

# Precompute static AprilTag corner coords for solvePnP.
_TH = TAG_SIZE_M / 2.0
TAG_LOCAL_CORNERS = np.array([
    [-_TH, -_TH, 0.0],
    [+_TH, -_TH, 0.0],
    [+_TH, +_TH, 0.0],
    [-_TH, +_TH, 0.0],
], dtype=np.float64)

# =============================================================================


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def load_calibration():
    if not os.path.exists(CALIB_FILE):
        sys.exit(f"no {CALIB_FILE} — run camera_setup/calibrate_camera.py first")
    c = np.load(CALIB_FILE)
    K = c["K"].astype(np.float64)
    dist = c["dist"].astype(np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    print(f"calibration loaded (RMS={float(c['rms']):.3f}px)")
    return K, dist, (fx, fy, cx, cy)


def cam_to_world(t_cam, tilt_deg, roll_deg, pitch_deg):
    """Rotate a point from the camera frame into the tag-facing, gravity-level
    world frame. Returns (fwd, lat, up): +fwd ahead, +lat right, +up up.

    Camera frame (OpenCV): +x right, +y down, +z forward. Camera is fixed-tilt
    up by tilt_deg on the drone. The drone's live roll/pitch adds to that —
    folding them in keeps the world axes decoupled from a body lean.
    """
    x, y, z = float(t_cam[0]), float(t_cam[1]), float(t_cam[2])
    # 1. Undo body roll about the optical axis. Camera +x = drone right; on
    # +roll (right wing down), the camera image rotates so that "down in
    # image" tilts toward image-right. Rz(-roll) on (x,y) undoes that.
    rr = -math.radians(SIGN_IMU_ROLL * roll_deg)
    cr, sr = math.cos(rr), math.sin(rr)
    x, y = cr * x + sr * y, -sr * x + cr * y
    # 2. Undo total pitch about the camera right axis: fixed mount + live nose
    # pitch (camera +y points down; +pitch nose-up tilts image-down toward back).
    a = math.radians(tilt_deg + SIGN_IMU_PITCH * pitch_deg)
    ca, sa = math.cos(a), math.sin(a)
    y_lvl = ca * y - sa * z
    z_lvl = sa * y + ca * z
    return float(z_lvl), float(x), float(-y_lvl)   # fwd, lat, up


class PoseFilter:
    """Per-axis (fwd, lat, up) state in the tag-facing world frame.

    Position: LPF toward fresh vision measurements; held on dropout
    (never extrapolated — dead-reckoning position across dropouts was
    the snap-back failure mode in earlier versions).

    Velocity: LPF of d(filtered position)/dt. Decays toward zero on vision
    loss. After a longer blind interval, a recovery measurement re-seeds
    pos and zeros v so the recovery jump isn't read as a velocity spike.

    Sign convention: self.p holds TAG positions in the drone's tag-facing
    world frame. Drone moving forward → tag fwd decreases → v_fwd < 0.
    """
    def __init__(self):
        self.p = [0.0, 0.0, 0.0]
        self.v = [0.0, 0.0, 0.0]
        self.last_t = None
        self.last_fresh_t = None
        self.inited = False

    def reset(self, p0, t_now):
        self.p = list(p0)
        self.v = [0.0, 0.0, 0.0]
        self.last_t = t_now
        self.last_fresh_t = t_now
        self.inited = True

    def update(self, vis_pos, vis_fresh, t_now, ground_pinned=False):
        if not self.inited:
            if vis_fresh:
                self.reset(vis_pos, t_now)
            return
        dt = clamp(t_now - (self.last_t or t_now), 1e-3, ESTIMATOR_MAX_DT_S)
        if vis_fresh:
            blind = (self.last_fresh_t is None
                     or (t_now - self.last_fresh_t) > RESEED_AFTER_LOSS_S)
            if blind:
                self.p = list(vis_pos)
                self.v = [0.0, 0.0, 0.0]
            else:
                for i in range(3):
                    p_new = (1.0 - ALPHA_POS) * self.p[i] + ALPHA_POS * vis_pos[i]
                    v_inst = (p_new - self.p[i]) / dt
                    self.v[i] = (1.0 - ALPHA_VEL) * self.v[i] + ALPHA_VEL * v_inst
                    self.p[i] = p_new
            self.last_fresh_t = t_now
        else:
            decay = math.exp(-dt / VEL_DECAY_TAU_S)
            for i in range(3):
                self.v[i] *= decay
        if ground_pinned:
            self.v = [0.0, 0.0, 0.0]
        self.last_t = t_now

    def age(self, now):
        if self.last_fresh_t is None:
            return 1e9
        return now - self.last_fresh_t


class Shared:
    """Latest vision output, written by the vision thread, read by main."""
    def __init__(self):
        self.lock = threading.Lock()
        self.pos = None                  # (fwd, lat, up) latest plausible
        self.pos_stamp = 0.0
        self.decision_margin = 0.0
        self.ids_seen = ""
        self.detect_label = ""
        self.has_tag = False
        self.frame = None
        self.running = True
        self.vision_status = "vision init"


class TxState:
    """Shared RC channels + telemetry. Control loop writes channels; TX thread
    sends them at TX_HZ; RX thread decodes telemetry and IMU into these fields.
    """
    def __init__(self):
        self.lock = threading.Lock()
        # Serial-write lock: TX and RX threads BOTH write to the same port
        # (TX sends RC frames, RX sends MSP requests + pings). pyserial.write
        # is not atomic for multi-byte payloads, so concurrent writes interleave
        # bytes and corrupt CRSF frames — which on the FC side blocks parsing
        # AND can cause the FC to drop the link briefly. Guard every ser.write
        # with this lock.
        self.ser_write_lock = threading.Lock()
        self.bytes_rx = 0
        self.frames_rx = 0
        self.ch = [NEUTRAL_US] * 16
        self.ch[CH_THR] = IDLE_THR_US
        self.ch[CH_ARM] = ARM_LOW_US
        self.ch[CH_PREARM] = PREARM_LOW_US
        self.ch[CH_MODE] = MODE_ANGLE_US
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


def vision_loop(shared, tx, K, dist, cam_params):
    detector = Detector(families=TAG_FAMILY, nthreads=4, quad_decimate=1.5,
                        quad_sigma=0.0, refine_edges=0, decode_sharpening=0.25)
    cap = cv2.VideoCapture(DEVICE_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("ERROR: cannot open camera; vision thread exiting")
        shared.running = False
        return
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    undist_map1 = undist_map2 = None
    if np.any(dist):
        undist_map1, undist_map2 = cv2.initUndistortRectifyMap(
            K, dist, None, K, (WIDTH, HEIGHT), cv2.CV_16SC2)

    last_tag_center_px = None
    last_good_stamp = None
    miss = 0
    stat_t0 = time.monotonic()
    n_read = n_hit = 0

    def pick_targets(tags):
        return [d for d in tags if d.tag_id in TARGET_TAG_IDS]

    def solve_pose(targets, offset_xy):
        if not targets:
            return None, 0.0
        ox, oy = offset_xy
        obj_pts, img_pts = [], []
        for d in targets:
            off = np.array(TAG_WORLD_OFFSETS[d.tag_id], dtype=np.float64)
            obj_pts.append(TAG_LOCAL_CORNERS + off)
            img_pts.append(d.corners.astype(np.float64) + np.array([ox, oy]))
        obj_pts = np.concatenate(obj_pts, axis=0)
        img_pts = np.concatenate(img_pts, axis=0)
        ok, _, tvec = cv2.solvePnP(obj_pts, img_pts, K, np.zeros(5),
                                   flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            return None, 0.0
        dm = max(float(getattr(d, "decision_margin", 0.0)) for d in targets)
        return tvec.ravel(), dm

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
            dt = now - stat_t0
            shared.vision_status = f"capFPS={n_read/dt:.0f} detFPS={n_hit/dt:.0f}"
            stat_t0, n_read, n_hit = now, 0, 0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Pass 1: full frame.
        with suppress_c_stderr():
            tags = detector.detect(gray, estimate_tag_pose=False)
        targets = pick_targets(tags)
        detect_label = "full" if targets else ""
        offset_xy = (0.0, 0.0)

        # Pass 2: contrast-boost fallback (only every few misses, expensive).
        if not targets and (miss % 3 == 0):
            with suppress_c_stderr():
                tags = detector.detect(clahe.apply(gray), estimate_tag_pose=False)
            targets = pick_targets(tags)
            if targets:
                detect_label = "clahe"

        # Pass 3: local ROI around last seen center.
        if (not targets and last_tag_center_px is not None
                and last_good_stamp is not None
                and (now - last_good_stamp) < VISION_ROI_RECOVERY_S):
            cx_px, cy_px = int(last_tag_center_px[0]), int(last_tag_center_px[1])
            r = VISION_ROI_HALF_SIZE_PX
            h, w = gray.shape[:2]
            x0, y0 = max(0, cx_px - r), max(0, cy_px - r)
            x1, y1 = min(w, cx_px + r), min(h, cy_px + r)
            if (x1 - x0) >= 80 and (y1 - y0) >= 80:
                roi = gray[y0:y1, x0:x1]
                with suppress_c_stderr():
                    tags = detector.detect(roi, estimate_tag_pose=False)
                targets = pick_targets(tags)
                if targets:
                    offset_xy = (float(x0), float(y0))
                    detect_label = "roi"

        if targets:
            miss = 0
            n_hit += 1
            tvec, dm = solve_pose(targets, offset_xy)
        else:
            miss += 1
            tvec, dm = None, 0.0

        if tvec is not None:
            fwd, lat, up = cam_to_world(tvec, CAMERA_TILT_DEG,
                                        tx.roll_deg, tx.pitch_deg)
            plausible = (dm >= MIN_TAG_DECISION_MARGIN
                         and 0.10 <= fwd <= MAX_TRACK_RANGE_M
                         and abs(lat) <= MAX_TRACK_LAT_M
                         and abs(up) <= MAX_TRACK_UP_M)
            ids_seen = "+".join(str(d.tag_id) for d in targets)
            if plausible:
                with shared.lock:
                    shared.pos = (fwd, lat, up)
                    shared.pos_stamp = now
                    shared.decision_margin = dm
                    shared.ids_seen = ids_seen
                    shared.detect_label = detect_label
                    shared.has_tag = True
                ox, oy = offset_xy
                centers = []
                for d in targets:
                    poly = (d.corners + np.array([ox, oy])).astype(int)
                    cv2.polylines(frame, [poly], True, (0, 255, 0), 2)
                    centers.append(poly.mean(axis=0))
                last_tag_center_px = np.mean(centers, axis=0)
                last_good_stamp = now
                cv2.putText(frame,
                            f"fwd={fwd*100:5.1f} lat={lat*100:+5.1f} up={up*100:+5.1f}cm "
                            f"dm={dm:.0f} [{detect_label} ids={ids_seen}]",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0), 2)
            else:
                with shared.lock:
                    shared.has_tag = False
                cv2.putText(frame, "TAG REJECTED", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        else:
            with shared.lock:
                shared.has_tag = False
            cv2.putText(frame, "NO TARGET", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        with shared.lock:
            shared.frame = frame
    cap.release()


def _safe_write(ser, tx, payload):
    """Write under the lock so RC + MSP/ping frames don't interleave on the wire."""
    with tx.ser_write_lock:
        try:
            ser.write(payload)
            return True
        except Exception as e:
            print(f"[serial] write failed: {e}", flush=True)
            return False


def tx_loop(ser, tx):
    """Push current channels at TX_HZ."""
    period = 1.0 / TX_HZ
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
    """Parse CRSF telemetry (flight mode, battery, attitude). Re-ping
    periodically so the FC keeps the link open."""
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


def is_armed(mode):
    return mode is not None and not mode.endswith("*") and not mode.startswith("!")


def main():
    K, dist, cam_params = load_calibration()
    port = autodetect_port()
    if not port:
        sys.exit("Ranger not found on USB. Plug it in and retry.")
    ser = open_ranger(port, 420000)
    os.makedirs(LOG_DIR, exist_ok=True)
    log_fp = open(LOG_FILE, "w", buffering=1)

    def log(msg):
        print(msg, flush=True)
        log_fp.write(msg + "\n")

    log(f"===== RUN START {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====")
    log(f"CRSF on {port} @ 420000")

    shared = Shared()
    tx = TxState()
    threading.Thread(target=vision_loop, args=(shared, tx, K, dist, cam_params),
                     daemon=True).start()
    threading.Thread(target=tx_loop, args=(ser, tx), daemon=True).start()
    threading.Thread(target=rx_loop, args=(ser, tx), daemon=True).start()

    cv2.namedWindow("hover", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow("hover", 1280, 720)

    pose_filter = PoseFilter()
    armed_cmd = False
    target_fwd_m = TARGET_FWD_FALLBACK_M
    target_lat_m = TARGET_LAT_M
    target_up_m = TARGET_UP_M
    yaw_at_arm = 0.0
    arm_lock_since = None
    up_integral_us = 0.0
    last_thr = float(IDLE_THR_US)
    last_state_t = time.monotonic()
    tx_period = 1.0 / TX_HZ
    next_loop = time.monotonic()
    last_display_t = 0.0
    last_log_t = 0.0

    def disarm_and_exit(reason):
        log(f"DISARM: {reason}")
        dis = [NEUTRAL_US] * 16
        dis[CH_THR] = IDLE_THR_US
        dis[CH_ARM] = ARM_LOW_US
        dis[CH_PREARM] = PREARM_LOW_US
        dis[CH_MODE] = MODE_ANGLE_US
        tx.set(dis)
        time.sleep(0.25)
        tx.stop = True
        shared.running = False
        time.sleep(0.05)
        ser.close()
        cv2.destroyAllWindows()
        if not log_fp.closed:
            log_fp.write(f"===== RUN END {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            log_fp.close()

    def axis_lean_us(pos, vel, target, kp, kv, vmax):
        """Velocity-loop PD for lat/fwd: position error → target velocity
        (capped at vmax), then a velocity-error term sets the lean command."""
        v_des = clamp(-kp * (pos - target), -vmax, vmax)
        return kv * (vel - v_des)   # vel here is TAG-frame d(pos)/dt

    def vertical_throttle(up, v_up, dt):
        """Velocity-loop with integral, ASYMMETRIC clamp (smaller upward
        authority than downward) — vision-only altitude must not run away
        upward, but can react to a fall while the tag stays in frame."""
        nonlocal up_integral_us
        e_pos = up - target_up_m
        v_des = clamp(-KP_UP * e_pos, -VMAX_UP_MPS, VMAX_UP_MPS)
        e_v = v_up - v_des             # > 0 means we're below target / not climbing fast enough
        up_integral_us = clamp(up_integral_us + KI_UP_US_PER_MPS * e_v * dt,
                               -UP_INT_MAX_US, UP_INT_MAX_US)
        # e_v > 0 → need more throttle. Positive thr_corr CLAMPED tighter
        # (THR_CLIMB_TRIM_US) than negative side (THR_DESC_TRIM_US).
        thr_corr = KV_UP_US_PER_MPS * e_v + up_integral_us
        thr_corr = clamp(thr_corr, -THR_DESC_TRIM_US, THR_CLIMB_TRIM_US)
        return int(clamp(HOVER_THROTTLE_US + thr_corr,
                         IDLE_THR_US, MAX_THROTTLE_US))

    def yaw_us(fwd, lat):
        # Point at the tag horizontally. fwd should always be positive in
        # flight (tag ahead); if it isn't, hold neutral yaw.
        if fwd < 0.10:
            return NEUTRAL_US
        bearing = math.atan2(lat, fwd)
        u = clamp(KP_YAW_US_PER_RAD * bearing, -MAX_YAW_US, MAX_YAW_US)
        return int(NEUTRAL_US + SIGN_YAW * u)

    def slew_thr(desired, prev):
        step = THR_SLEW_US_PER_S * tx_period
        return int(clamp(desired, prev - step, prev + step))

    if DRY_RUN:
        log("*** DRY RUN — will NOT arm ***")
        state = "DRYRUN"
    else:
        log("*** LIVE FLIGHT ***")
        ans = input("Area clear? Hand on battery? Type FLY to arm: ").strip()
        log(f"pilot confirmation input={ans!r}")
        if ans != "FLY":
            disarm_and_exit("not confirmed by user")
            return
        log("waiting for link + telemetry...")
        t0 = time.monotonic()
        last_diag = 0.0
        while tx.flight_mode is None and time.monotonic() - t0 < 8.0:
            now = time.monotonic()
            if now - last_diag >= 1.0:
                log(f"  waiting... rx_bytes={tx.bytes_rx} rx_frames={tx.frames_rx} "
                    f"att={tx.att_stamp > 0} batt={tx.pack_v}")
                last_diag = now
            time.sleep(0.1)
        if tx.flight_mode is None:
            disarm_and_exit(
                f"no telemetry (rx_bytes={tx.bytes_rx} rx_frames={tx.frames_rx}) "
                f"— FC may be busy or link down")
            return
        log(f"link up, mode={tx.flight_mode!r}")
        if tx.flight_mode.startswith("!"):
            disarm_and_exit(f"FC arm-blocked: {tx.flight_mode!r}")
            return
        state = "ARM_SETTLE"
    last_state_t = time.monotonic()

    prev_loop_t = time.monotonic()
    try:
        while shared.running:
            now = time.monotonic()
            dt = clamp(now - prev_loop_t, 0.0, ESTIMATOR_MAX_DT_S)
            prev_loop_t = now

            flight_mode = tx.flight_mode
            pack_v = tx.pack_v
            if (LOW_BATT_CUTOFF and pack_v is not None
                    and BATT_PRESENT_V < pack_v < MIN_CELL_V * CELLS):
                disarm_and_exit(f"battery low ({pack_v:.2f}V)")
                break
            if armed_cmd and flight_mode and flight_mode.startswith("!"):
                disarm_and_exit(f"FC refused/err mode {flight_mode!r}")
                break

            # --- pull latest vision pose ---
            with shared.lock:
                vis_pos = shared.pos
                vis_stamp = shared.pos_stamp
                vis_dm = shared.decision_margin
                vis_has = shared.has_tag
                vis_ids = shared.ids_seen
                vis_det = shared.detect_label
                frame = shared.frame
            vis_age = now - vis_stamp if vis_stamp else 1e9
            vis_fresh = vis_has and vis_age < VISION_FRESH_S

            # --- pose filter ---
            if state in ("DRYRUN", "ARM_SETTLE"):
                if vis_fresh:
                    pose_filter.reset(vis_pos, now)
                pose_filter.update(vis_pos, vis_fresh, now, ground_pinned=True)
            else:
                # Stay ground-pinned until throttle has built up to takeoff
                # thrust — pose jitter on the floor would otherwise feed
                # phantom velocity into the controller before liftoff.
                ground_pinned = last_thr < (HOVER_THROTTLE_US - GROUND_PIN_BAND_US)
                pose_filter.update(vis_pos, vis_fresh, now,
                                   ground_pinned=ground_pinned)

            est_fwd, est_lat, est_up = pose_filter.p
            v_fwd, v_lat, v_up = pose_filter.v
            est_age = pose_filter.age(now) if state == "FLY" else 0.0

            # --- compute outputs ---
            roll = pitch = yaw = NEUTRAL_US
            thr = IDLE_THR_US
            arm = ARM_LOW_US
            prearm = PREARM_LOW_US
            note = ""

            if state == "DRYRUN":
                if pose_filter.inited:
                    # Show what we'd command, but never arm.
                    u_roll = axis_lean_us(est_lat, v_lat, target_lat_m,
                                          KP_LAT, KV_LAT_US_PER_MPS, VMAX_LAT_MPS)
                    u_pitch = axis_lean_us(est_fwd, v_fwd, target_fwd_m,
                                           KP_FWD, KV_FWD_US_PER_MPS, VMAX_FWD_MPS)
                    u_roll = clamp(u_roll, -MAX_ROLL_US, MAX_ROLL_US)
                    u_pitch = clamp(u_pitch, -MAX_PITCH_US, MAX_PITCH_US)
                    roll = int(NEUTRAL_US + SIGN_ROLL * u_roll)
                    pitch = int(NEUTRAL_US + SIGN_PITCH * u_pitch)
                    yaw = yaw_us(est_fwd, est_lat)
                    thr = vertical_throttle(est_up, v_up, dt)
                    note = (f"DRY est=({est_fwd:+.2f},{est_lat:+.2f},{est_up:+.2f}) "
                            f"v=({v_fwd:+.2f},{v_lat:+.2f},{v_up:+.2f})")
                else:
                    note = "DRY waiting for vision"

            elif state == "ARM_SETTLE":
                prearm = PREARM_HIGH_US
                if now - last_state_t < 0.4:
                    arm = ARM_LOW_US
                    note = "prearm…"
                else:
                    arm = ARM_HIGH_US
                    armed_cmd = True
                    if flight_mode and flight_mode.startswith("!"):
                        disarm_and_exit(f"won't arm: {flight_mode!r}")
                        break
                    if is_armed(flight_mode):
                        # Wait for stable vision lock before lifting off.
                        strong = vis_fresh and vis_dm >= MIN_TAG_DECISION_MARGIN
                        if strong:
                            arm_lock_since = arm_lock_since or now
                            lock_s = now - arm_lock_since
                            if lock_s < ARM_LOCK_MIN_S:
                                note = (f"lock {lock_s:.2f}/{ARM_LOCK_MIN_S:.2f}s "
                                        f"dm={vis_dm:.0f}")
                            else:
                                target_fwd_m = float(vis_pos[0])   # hold takeoff distance
                                yaw_at_arm = tx.yaw_deg
                                pose_filter.reset(vis_pos, now)
                                up_integral_us = 0.0
                                last_thr = float(IDLE_THR_US)
                                state, last_state_t = "FLY", now
                                log(f">> FLY: target fwd={target_fwd_m:+.2f} lat=0 up=0  "
                                    f"yaw_arm={yaw_at_arm:+.1f}")
                        else:
                            arm_lock_since = None
                            note = (f"waiting for vision lock vis_age={vis_age*1000:.0f}ms "
                                    f"dm={vis_dm:.0f}")
                    elif now - last_state_t > ARM_CONFIRM_S:
                        disarm_and_exit(
                            f"arm not confirmed after {ARM_CONFIRM_S:.1f}s "
                            f"(mode={flight_mode!r}). FC may still be doing "
                            f"gyro cal — wait a few seconds after powering "
                            f"the drone before launching the script.")
                        break
                    else:
                        # FC has the asterisk set (ARMING_DISABLED). Keep
                        # logging so we can see whether it eventually clears
                        # or stays stuck.
                        note = (f"FC arm-blocked (mode={flight_mode!r}) "
                                f"t={now-last_state_t:.1f}/{ARM_CONFIRM_S:.1f}s "
                                f"att_stamp={tx.att_stamp > 0}")

            elif state == "FLY":
                arm = ARM_HIGH_US
                prearm = PREARM_HIGH_US
                if est_age < STATE_MAX_AGE_S:
                    u_roll = axis_lean_us(est_lat, v_lat, target_lat_m,
                                          KP_LAT, KV_LAT_US_PER_MPS, VMAX_LAT_MPS)
                    u_pitch = axis_lean_us(est_fwd, v_fwd, target_fwd_m,
                                           KP_FWD, KV_FWD_US_PER_MPS, VMAX_FWD_MPS)
                    u_roll = clamp(u_roll, -MAX_ROLL_US, MAX_ROLL_US)
                    u_pitch = clamp(u_pitch, -MAX_PITCH_US, MAX_PITCH_US)
                    roll = int(NEUTRAL_US + SIGN_ROLL * u_roll)
                    pitch = int(NEUTRAL_US + SIGN_PITCH * u_pitch)
                    yaw = yaw_us(est_fwd, est_lat)
                    thr = vertical_throttle(est_up, v_up, dt)
                    thr = slew_thr(thr, last_thr)
                    last_thr = thr
                    note = (f"fly fwd={est_fwd:+.2f}(t{target_fwd_m:+.2f}) "
                            f"lat={est_lat:+.2f} up={est_up:+.2f}  "
                            f"v=({v_fwd:+.2f},{v_lat:+.2f},{v_up:+.2f}) "
                            f"intI={up_integral_us:+.0f}")
                else:
                    # Lost tag too long — bleed throttle to a soft descent.
                    lost = est_age - STATE_MAX_AGE_S
                    if est_age > KILL_AFTER_LOST_S:
                        disarm_and_exit(f"tag lost {est_age:.1f}s")
                        break
                    bleed = LAND_DESCENT_US_PER_S * lost
                    t = int(clamp(HOVER_THROTTLE_US - bleed,
                                  IDLE_THR_US, HOVER_THROTTLE_US))
                    thr = slew_thr(t, last_thr)
                    last_thr = thr
                    up_integral_us = 0.0
                    note = f"LOST {est_age:.2f}s — descending thr={thr}"

            ch = [NEUTRAL_US] * 16
            ch[CH_ROLL] = int(clamp(roll, 1000, 2000))
            ch[CH_PITCH] = int(clamp(pitch, 1000, 2000))
            ch[CH_YAW] = int(clamp(yaw, 1000, 2000))
            ch[CH_THR] = int(clamp(thr, 1000, 2000))
            ch[CH_ARM] = arm
            ch[CH_MODE] = MODE_ANGLE_US
            ch[CH_PREARM] = prearm
            tx.set(ch)

            if frame is not None:
                cv2.putText(frame,
                            f"[{state}] R{ch[CH_ROLL]} P{ch[CH_PITCH]} "
                            f"Y{ch[CH_YAW]} T{ch[CH_THR]} "
                            f"{'ARM' if arm > 1500 else 'safe'}",
                            (10, HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 255), 2)
                if now - last_display_t >= (1.0 / max(DISPLAY_HZ, 1.0)):
                    cv2.imshow("hover", frame)
                    last_display_t = now
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27, ord(' ')):
                disarm_and_exit("kill key")
                break

            if note and (now - last_log_t >= (1.0 / max(LOG_HZ, 1.0))):
                mode_txt = flight_mode if flight_mode is not None else "n/a"
                batt_txt = f"{pack_v:.2f}V" if pack_v is not None else "n/a"
                vis_txt = (f"vis={'fresh' if vis_fresh else 'lost'} "
                           f"age={vis_age*1000:.0f}ms dm={vis_dm:.0f} "
                           f"det={vis_det or 'none'} ids={vis_ids or '-'}")
                imu_txt = (f"imu(r/p)=({tx.roll_deg:+.1f},{tx.pitch_deg:+.1f})deg")
                est_txt = f"est_age={est_age*1000:.0f}ms"
                log(f"[{state}] {note}  -> R{ch[CH_ROLL]} P{ch[CH_PITCH]} "
                    f"Y{ch[CH_YAW]} T{ch[CH_THR]}  {vis_txt}  {imu_txt}  {est_txt}  "
                    f"vis=[{shared.vision_status}] mode={mode_txt} batt={batt_txt}")
                last_log_t = now

            next_loop += tx_period
            sleep = next_loop - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_loop = time.monotonic()
    except KeyboardInterrupt:
        disarm_and_exit("Ctrl-C")
    finally:
        if shared.running:
            disarm_and_exit("loop exit")
        elif not log_fp.closed:
            log_fp.write(f"===== RUN END {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            log_fp.close()


if __name__ == "__main__":
    main()
