#!/usr/bin/env python3
"""Manual flight: TX12 USB joystick → laptop → Ranger → drone (transparent RC relay).

The TX12 is in EdgeTX "USB Joystick (HID)" mode, so the laptop sees it as
/dev/input/js0 (7 axes, 24 buttons on this radio). This script reads the
gimbals + arm switch from that joystick and re-emits them as a CRSF
RC_CHANNELS stream out the Ranger's USB-C serial — the same send path the
autonomous controller uses (controller_v2.threads.open_ranger + the CRSF
frame builders in setup/live_telemetry.py). Net effect: you fly the drone by
hand through the laptop, exactly as if the TX12 were transmitting directly.

WHY CALIBRATION: EdgeTX's channel→USB-axis assignment is configurable and
radio-specific, so which HID axis is roll vs pitch vs throttle vs yaw — and
which control is the arm switch — is NOT safe to hardcode. `--calibrate`
discovers it empirically and writes tx12_joystick_cal.json (next to this script).

Channel map and µs levels come from controller_v2.config (the Air75
Betaflight dump) so this stays in lockstep with the autonomous controller.
This FC has no prearm — a single arm switch gates arming (CH_ARM only).

RECORDING: the blackbox switch (relayed to AUX2 so the FC's blackbox starts)
also triggers laptop video capture of the drone feed. Video records while the
drone is ARMED and the switch is ON, and is saved when either drops (disarm or
switch off). Needs cv2 — run with the repo venv: .venv/bin/python.

Usage:
    .venv/bin/python joystick_flight.py --calibrate  # one-time, per radio config
    .venv/bin/python joystick_flight.py --dry-run    # read joystick, print sticks, never send
    .venv/bin/python joystick_flight.py              # fly: autodetect Ranger port
    .venv/bin/python joystick_flight.py /dev/ttyACM0 420000  # explicit Ranger port + baud

SAFETY
  - Starts disarmed, throttle idle.
  - Arm is edge-gated: the script refuses to pass ARM-high until it has first
    seen the arm switch in the DISARMED position (no arm-on-startup).
  - Joystick unplug / read error → forces disarm + idle and exits.
  - Ctrl-C → sends explicit disarm frames, then exits.
  - --dry-run never opens the Ranger and never sends anything.
"""
import argparse
import array
import datetime
import fcntl
import json
import os
import struct
import sys
import threading
import time

# Shared CRSF builders (live_telemetry) and the channel map (controller_v2)
# live in the sibling drone_control/ folder.
HERE = os.path.dirname(os.path.abspath(__file__))
_CONTROL = os.path.join(os.path.dirname(HERE), "drone_control")
sys.path.insert(0, HERE)                               # for `from vicon_recorder import ...`
sys.path.insert(0, _CONTROL)                          # for `from controller_v2 import ...`
sys.path.insert(0, os.path.join(_CONTROL, "setup"))   # for `from live_telemetry import ...`
import serial  # noqa: E402
from live_telemetry import (  # noqa: E402
    build_rc_channels_packed, build_device_ping, autodetect_port, CrsfParser,
    decode_flight_mode, decode_battery, T_FLIGHT_MODE, T_BATTERY,
)

# Channel map + µs levels + arm choreography — single source of truth.
from controller_v2 import config  # noqa: E402

# OpenCV is only needed for video recording. Import it optionally so the script
# still flies (recording disabled) if run under a Python without cv2 — e.g. the
# system python3.14, vs the repo's .venv (python3.12) which has cv2.
try:
    import cv2  # noqa: E402
    CV2_OK = True
except Exception:
    cv2 = None
    CV2_OK = False

# Vicon pose recorder deps (ViconRecorder class is defined below, next to
# VideoRecorder). Optional like cv2 — if numpy/scipy/the lab parser are missing,
# vicon recording disables itself (VICON_OK=False) and flight continues. The
# parser is the lab's, in the sibling pycode_ViCON/ folder.
_PYCODE_VICON = os.path.join(os.path.dirname(HERE), "pycode_ViCON")
if _PYCODE_VICON not in sys.path:
    sys.path.insert(0, _PYCODE_VICON)
try:
    import socket  # noqa: E402
    import numpy as np  # noqa: E402
    import scipy.io as sio  # noqa: E402
    # Same primitives UdpReceiver_datacollection.py uses — it is the base template
    # for ALL Vicon logging in this repo, so this path stays identical to it:
    # UdpRigidBodiesViCON (threaded receiver + the startup sample-rate
    # determination), DataProcessorViCON (parser), RealTimeSleeper (100 Hz loop),
    # Differentiator (b1 velocities).
    from UdpReceiver_datacollection import (  # noqa: E402
        DataProcessorViCON, UdpRigidBodiesViCON, RealTimeSleeper, Differentiator,
    )
    VICON_OK = True
    _VICON_ERR = None
except Exception as _e:  # noqa: BLE001 — missing dep just disables vicon
    VICON_OK = False
    _VICON_ERR = _e

VICON_UDP_IP = "0.0.0.0"
VICON_UDP_PORT = 51001
_VICON_BLOCK = 1024
_VICON_PROBE_TIMEOUT_S = 3.0   # no Vicon traffic within this at prepare() → disable, keep flying
_VICON_SAMPLE_DT = 0.01        # 100 Hz logging loop, same as the UdpReceiver template

CAL_FILE_DEFAULT = os.path.join(HERE, "tx12_joystick_cal.json")

# Betaflight refuses to arm while the throttle channel is above min_check
# (default 1050µs) — the THROTTLE arming-disable flag. Verified against
# Betaflight docs 2026-06-04. The throttle stick must be at the bottom to arm.
ARM_THROTTLE_MAX_US = 1050

# --- Air75 ARM channel (user-verified 2026-06-04) ---
# On THIS drone the ARM mode is on AUX3. In a CRSF frame: gimbals = array
# indices 0-3, AUX1=4, AUX2=5, AUX3=6. So arm lives at array index 6, armed at
# 1508µs and disarmed at 1000µs.
# NOTE: config.CH_ARM/CH_MODE come from a Meteor75 Pro dump (memory flags them
# unverified on the Air75) and are WRONG here — config.CH_ARM=5 (AUX2) is
# ignored by the FC, and config.CH_MODE=6 was clobbering AUX3 with 1500µs (below
# the arm band), which is why the drone never armed despite the "ARMED" display.
ARM_CH = 6
ARM_ARMED_US = 1508
ARM_DISARMED_US = 1000

# --- Blackbox / record switch (user-verified 2026-06-04) ---
# The blackbox switch is joystick axis 5, a 3-POSITION switch driving AUX2 =
# CRSF array index 5, where the FC's blackbox modes are configured:
#   HIGH (+32767) → AUX2 2000µs → START blackbox logging  (also starts video)
#   MID  (0)      → AUX2 1500µs → nothing
#   LOW  (-32767) → AUX2 1000µs → ERASE blackbox dataflash
# It must be relayed as three distinct µs values: a binary on/off relay sent the
# LOW (erase) value for the MIDDLE detent, so the middle wrongly erased. Note
# AUX2_LOW_US is the ERASE position — send_disarm() must NOT send it (it sends
# AUX2_MID_US instead) or every disarm/exit would erase the log.
AUX2_CH = 5
AUX2_HIGH_US = 2000
AUX2_MID_US = 1500
AUX2_LOW_US = 1000
REC_DIR = os.path.join(HERE, "recordings")

# --- Linux joystick API (dependency-free; /dev/input/js0) ---
# js_event is 8 bytes: __u32 time, __s16 value, __u8 type, __u8 number.
JS_EVENT_FMT = "IhBB"
JS_EVENT_SIZE = 8
JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JS_EVENT_INIT = 0x80          # OR'd into type for the synthetic open-time burst
JSIOCGAXES = 0x80016A11       # u8: number of axes
JSIOCGBUTTONS = 0x80016A12    # u8: number of buttons


def _jsiocgname(length):
    return 0x80006A13 | (length << 16)


class Joystick:
    """Non-blocking reader for /dev/input/jsN. Keeps the latest value of every
    axis and button. The kernel emits a synthetic INIT burst on open, so state
    is fully populated before the operator touches anything."""

    def __init__(self, path):
        self.path = path
        self.fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        b = bytearray(1)
        fcntl.ioctl(self.fd, JSIOCGAXES, b)
        self.n_axes = b[0]
        fcntl.ioctl(self.fd, JSIOCGBUTTONS, b)
        self.n_buttons = b[0]
        name = bytearray(128)
        fcntl.ioctl(self.fd, _jsiocgname(128), name)
        self.name = name.split(b"\x00", 1)[0].decode(errors="replace")
        self.axes = {}
        self.buttons = {}
        self.alive = True
        self.poll(settle=0.05)   # absorb the INIT burst

    def poll(self, settle=0.0):
        """Drain all pending events. If settle>0, keep reading for that long
        (used during calibration so a freshly-moved stick's final value lands)."""
        deadline = time.monotonic() + settle
        while True:
            try:
                data = os.read(self.fd, JS_EVENT_SIZE * 128)
            except BlockingIOError:
                data = b""
            except OSError:
                # ENODEV etc. — joystick unplugged.
                self.alive = False
                return
            for i in range(0, len(data) - JS_EVENT_SIZE + 1, JS_EVENT_SIZE):
                _t, val, typ, num = struct.unpack(
                    JS_EVENT_FMT, data[i:i + JS_EVENT_SIZE])
                base = typ & ~JS_EVENT_INIT
                if base == JS_EVENT_AXIS:
                    self.axes[num] = val
                elif base == JS_EVENT_BUTTON:
                    self.buttons[num] = val
            if settle <= 0.0 or time.monotonic() >= deadline:
                return
            if not data:
                time.sleep(0.005)

    def snapshot(self):
        return dict(self.axes), dict(self.buttons)

    def close(self):
        try:
            os.close(self.fd)
        except OSError:
            pass


# --- Custom-baud Ranger open (TCSETS2/BOTHER). Replicated from
# controller_v2.threads.open_ranger so this script doesn't drag in cv2/numpy.
# pyserial's normal path rejects 420000 (termios EINVAL); the kernel custom-baud
# ioctl is the fix — see memory feedback_crsf_custom_baud. ---
_TCGETS2, _TCSETS2 = 0x802C542A, 0x402C542B
_BOTHER, _CBAUD = 0o010000, 0o010017


def open_ranger(port, baud=420000):
    ser = serial.Serial(port, baudrate=115200, timeout=0.0)
    buf = array.array("i", [0] * 64)
    fcntl.ioctl(ser.fileno(), _TCGETS2, buf)
    buf[2] = (buf[2] & ~_CBAUD) | _BOTHER
    buf[9] = buf[10] = baud
    fcntl.ioctl(ser.fileno(), _TCSETS2, buf)
    return ser


# ===================== calibration =====================

GIMBALS = [
    # (key, low-label → 1000µs, high-label → 2000µs)
    ("throttle", "throttle to the BOTTOM (minimum)", "throttle to the TOP (maximum)"),
    ("roll",     "roll/aileron stick fully LEFT",    "roll/aileron stick fully RIGHT"),
    ("pitch",    "pitch/elevator stick fully BACK (toward you)",
                 "pitch/elevator stick fully FORWARD (away from you)"),
    ("yaw",      "yaw/rudder stick fully LEFT",       "yaw/rudder stick fully RIGHT"),
]


def _capture(js, prompt):
    input(f"  → {prompt}, hold it, then press Enter...")
    js.poll(settle=0.15)
    ax, btn = js.snapshot()
    return ax, btn


def calibrate(js, cal_path):
    print(f"\nCalibrating {js.name} ({js.n_axes} axes, {js.n_buttons} buttons)")
    print("Hold each control at the named position BEFORE pressing Enter.\n")

    cal = {"device": js.name, "axes": {}}

    for key, lo_label, hi_label in GIMBALS:
        lo_ax, _ = _capture(js, lo_label)
        hi_ax, _ = _capture(js, hi_label)
        deltas = {n: abs(hi_ax.get(n, 0) - lo_ax.get(n, 0))
                  for n in set(lo_ax) | set(hi_ax)}
        if not deltas or max(deltas.values()) < 4000:
            print(f"  !! no clear movement detected for {key} "
                  f"(max delta {max(deltas.values()) if deltas else 0}). "
                  f"Re-run calibration and move the {key} control fully.")
            return None
        axis = max(deltas, key=deltas.get)
        cal["axes"][key] = {
            "axis": axis,
            "lo_raw": lo_ax[axis], "hi_raw": hi_ax[axis],
            "lo_us": 1000, "hi_us": 2000,
        }
        print(f"     {key:8s} → axis[{axis}]  "
              f"lo_raw={lo_ax[axis]:+6d} hi_raw={hi_ax[axis]:+6d}\n")

    # Arm switch: detect whichever control (axis OR button) changes between
    # the two switch positions.
    print("ARM SWITCH:")
    dis_ax, dis_btn = _capture(js, "set the ARM switch to DISARMED / safe")
    arm_ax, arm_btn = _capture(js, "set the ARM switch to ARMED")
    btn_changed = [n for n in set(dis_btn) | set(arm_btn)
                   if dis_btn.get(n, 0) != arm_btn.get(n, 0)]
    ax_deltas = {n: abs(arm_ax.get(n, 0) - dis_ax.get(n, 0))
                 for n in set(dis_ax) | set(arm_ax)}
    best_ax = max(ax_deltas, key=ax_deltas.get) if ax_deltas else None
    if btn_changed:
        n = btn_changed[0]
        cal["arm"] = {"kind": "button", "index": n,
                      "disarmed_raw": dis_btn.get(n, 0), "armed_raw": arm_btn.get(n, 0)}
        print(f"     arm → button[{n}]  "
              f"disarmed={dis_btn.get(n, 0)} armed={arm_btn.get(n, 0)}\n")
    elif best_ax is not None and ax_deltas[best_ax] >= 8000:
        cal["arm"] = {"kind": "axis", "index": best_ax,
                      "disarmed_raw": dis_ax[best_ax], "armed_raw": arm_ax[best_ax]}
        print(f"     arm → axis[{best_ax}]  "
              f"disarmed_raw={dis_ax[best_ax]:+6d} armed_raw={arm_ax[best_ax]:+6d}\n")
    else:
        print("  !! could not detect an arm control. Make sure the ARM switch "
              "is mapped to a channel in the EdgeTX model, then re-run.")
        return None

    # Blackbox / record switch (optional). Set OFF then ON; if nothing changes,
    # leave it unconfigured (flying + arming still work, just no auto-record).
    print("BLACKBOX/RECORD SWITCH (optional — leave switch still + Enter twice to skip):")
    off_ax, off_btn = _capture(js, "set the BLACKBOX switch OFF / down")
    on_ax, on_btn = _capture(js, "set the BLACKBOX switch ON / recording")
    rb_changed = [n for n in set(off_btn) | set(on_btn)
                  if off_btn.get(n, 0) != on_btn.get(n, 0)]
    rax_deltas = {n: abs(on_ax.get(n, 0) - off_ax.get(n, 0))
                  for n in set(off_ax) | set(on_ax)}
    rbest = max(rax_deltas, key=rax_deltas.get) if rax_deltas else None
    if rb_changed:
        n = rb_changed[0]
        cal["record"] = {"kind": "button", "index": n,
                         "off_raw": off_btn.get(n, 0), "on_raw": on_btn.get(n, 0)}
        print(f"     record → button[{n}]  off={off_btn.get(n, 0)} on={on_btn.get(n, 0)}\n")
    elif rbest is not None and rax_deltas[rbest] >= 8000:
        cal["record"] = {"kind": "axis", "index": rbest,
                         "off_raw": off_ax[rbest], "on_raw": on_ax[rbest]}
        print(f"     record → axis[{rbest}]  "
              f"off_raw={off_ax[rbest]:+6d} on_raw={on_ax[rbest]:+6d}\n")
    else:
        print("     (no record switch detected — recording disabled)\n")

    with open(cal_path, "w") as f:
        json.dump(cal, f, indent=2)
    print(f"Saved calibration → {cal_path}")
    return cal


def load_cal(cal_path):
    if not os.path.exists(cal_path):
        sys.exit(f"No calibration file at {cal_path}. Run with --calibrate first.")
    with open(cal_path) as f:
        cal = json.load(f)
    for key in ("throttle", "roll", "pitch", "yaw"):
        if key not in cal.get("axes", {}):
            sys.exit(f"Calibration missing gimbal '{key}'. Re-run --calibrate.")
    if "arm" not in cal:
        sys.exit("Calibration missing arm switch. Re-run --calibrate.")
    return cal


# ===================== mapping =====================

def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def map_axis(raw, spec):
    """Linear map a raw axis value to µs using the calibrated endpoints.
    Handles inverted axes (lo_raw may exceed hi_raw) and clamps to [1000,2000]."""
    lo_raw, hi_raw = spec["lo_raw"], spec["hi_raw"]
    lo_us, hi_us = spec["lo_us"], spec["hi_us"]
    if hi_raw == lo_raw:
        return lo_us
    frac = (raw - lo_raw) / (hi_raw - lo_raw)
    us = lo_us + frac * (hi_us - lo_us)
    return int(round(_clamp(us, 1000, 2000)))


def arm_is_armed(js, arm_spec):
    """True iff the arm control currently reads in its ARMED position."""
    if arm_spec["kind"] == "button":
        return js.buttons.get(arm_spec["index"], 0) == arm_spec["armed_raw"]
    raw = js.axes.get(arm_spec["index"], arm_spec["disarmed_raw"])
    mid = (arm_spec["disarmed_raw"] + arm_spec["armed_raw"]) / 2.0
    if arm_spec["armed_raw"] >= arm_spec["disarmed_raw"]:
        return raw >= mid
    return raw <= mid


def record_switch_on(js, spec):
    """True iff the blackbox/record switch reads in its ON (HIGH) position. For a
    3-position switch we accept only the calibrated ON detent (within a half-step
    of on_raw), so the middle position doesn't count as ON. Drives video record."""
    if spec is None:
        return False
    if spec.get("kind") == "button":
        return js.buttons.get(spec["index"], 0) == spec.get("on_raw", 1)
    raw = js.axes.get(spec["index"])
    if raw is None:
        return False
    return abs(raw - spec["on_raw"]) < 16384


def aux2_us_for_switch(js, spec):
    """Map the 3-position blackbox switch to its AUX2 µs: HIGH→start logging,
    MID→nothing, LOW→erase. Each detent must get a distinct value — a binary
    relay sent the LOW/erase value for the middle detent, wrongly erasing.
    Defaults to MID (nothing) when unknown so it never accidentally erases."""
    if spec is None:
        return AUX2_MID_US
    if spec.get("kind") == "button":
        on = js.buttons.get(spec["index"], 0) == spec.get("on_raw", 1)
        return AUX2_HIGH_US if on else AUX2_MID_US
    raw = js.axes.get(spec["index"])
    if raw is None:
        return AUX2_MID_US
    if raw > 16384:
        return AUX2_HIGH_US      # high detent → start blackbox logging
    if raw < -16384:
        return AUX2_LOW_US       # low detent → erase blackbox
    return AUX2_MID_US           # middle → no blackbox mode


class VideoRecorder:
    """Records the drone's video feed to a file in a background thread.

    Camera open + per-frame read/write happen off the main loop so they can
    never stall the 50 Hz RC stream. start()/stop() are called from the main
    loop on switch edges; start() returns immediately (the thread opens the
    camera, ~0.5-1 s, so the first second of footage may be missed)."""

    def __init__(self, device_index, width, height):
        self.device_index = device_index
        self.width = width
        self.height = height
        self._thread = None
        self._stop = threading.Event()
        self.recording = False
        self.status = "idle" if CV2_OK else "disabled (no cv2)"
        self.path = None
        self.frames = 0
        self.fps = None
        # Wall-clock time the first frame was actually captured. The camera takes
        # ~0.5-1 s to open, so frame 0 lags the session t0 by this much — the
        # combine uses (first_frame_wall - t0) to align video to the data.
        self.first_frame_wall = None
        # Latest captured frame, shared (read-only) with the main loop so it can
        # show a live preview window. Guarded by a lock; capture stays in _run.
        self._frame_lock = threading.Lock()
        self._latest_frame = None

    def get_latest_frame(self):
        with self._frame_lock:
            return self._latest_frame

    def start(self, out_path):
        if self.recording or not CV2_OK:
            return
        # Make sure any prior recording's thread has fully finalized its file.
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self.path = out_path
        self.first_frame_wall = None
        self.recording = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        if not self.recording:
            return
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        self.recording = False

    def _run(self):
        cap = writer = None
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            name = os.path.basename(self.path)
            cap = cv2.VideoCapture(self.device_index, cv2.CAP_V4L2)
            if not cap.isOpened():
                self.status = f"camera /dev/video{self.device_index} open FAILED"
                return
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            rep = cap.get(cv2.CAP_PROP_FPS)
            self.fps = rep if rep and rep > 1 else 30.0
            writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"),
                                     self.fps, (self.width, self.height))
            if not writer.isOpened():
                self.status = "VideoWriter open FAILED"
                return
            self.frames = 0
            self.status = f"REC {name} @ {self.fps:.0f}fps"
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    continue
                if self.first_frame_wall is None:
                    self.first_frame_wall = time.time()
                writer.write(frame)
                self.frames += 1
                with self._frame_lock:
                    self._latest_frame = frame
            self.status = f"saved {name} ({self.frames} frames)"
        except Exception as e:  # never let a recorder fault take down flight
            self.status = f"recorder error: {e}"
        finally:
            if writer is not None:
                writer.release()
            if cap is not None:
                cap.release()
            self.recording = False


class ViconRecorder:
    """Records Vicon pose to a per-session .mat, built on the SAME primitives as
    UdpReceiver_datacollection.py (the base template for all Vicon logging here):
    `UdpRigidBodiesViCON` (threaded receiver + the original startup sample-rate
    determination), `DataProcessorViCON` (parser), a 100 Hz `RealTimeSleeper`
    loop, and `Differentiator` for b1 velocities. Output matches that template —
    `exptime`, `Abs_time` (from the shared trigger t0), `b1_x`..`b1_qw` (+ any
    extra bodies), `b1_x_dot`/`b1_y_dot`/`b1_z_dot`.

    Two adaptations for the flight context (vs the standalone template):
      - prepare() does the connect + sample-rate measurement ONCE at startup
        (not per session), behind a fail-soft probe so flight still works if
        Vicon isn't streaming. The lab `UdpRigidBodiesViCON` blocks with no
        timeout, so we probe first and only construct it once packets are seen.
      - The receiver runs continuously after prepare(); start()/stop() just gate
        recording, so each session begins logging instantly at t0 (no per-flick
        measurement delay) — which keeps the deterministic sync with video +
        blackbox intact. Mirrors VideoRecorder's start()/stop()/recording/status."""

    def __init__(self, port=VICON_UDP_PORT, ip=VICON_UDP_IP):
        self.port = port
        self.ip = ip
        self.udp = None          # UdpRigidBodiesViCON — built in prepare()
        self.dp = None           # DataProcessorViCON
        self.sample_rate = None  # measured by the startup determination
        self.num_bodies = None
        self._thread = None
        self._stop = threading.Event()
        self.recording = False
        self.status = "idle" if VICON_OK else f"disabled ({_VICON_ERR})"
        self.path = None
        self.samples = 0
        self.first_packet_wall = None
        self.t0_wall = None

    def _stream_present(self):
        """Fail-soft probe: is anything streaming on the Vicon port right now?
        Lets us skip the lab receiver's un-timeouted blocking get_sample_rate
        when Vicon is off, so the flight script never hangs at startup."""
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.ip, self.port))
            s.settimeout(_VICON_PROBE_TIMEOUT_S)
            s.recvfrom(_VICON_BLOCK)
            return True
        except socket.timeout:
            return False
        finally:
            s.close()

    def prepare(self):
        """Connect + run the original startup sample-rate determination ONCE.
        Returns True if Vicon is live and ready to record, False (disabled) if
        not — never raises, never hangs flight. Safe to call again; no-op once
        prepared."""
        if not VICON_OK or self.udp is not None:
            return self.udp is not None
        if not self._stream_present():
            self.status = f"no Vicon stream on :{self.port} — disabled"
            return False
        try:
            # UdpRigidBodiesViCON.__init__ runs get_sample_rate() (the original
            # startup processing: times ~1000 packets to determine the polling
            # rate); start_thread() then reads num_bodies from the header.
            self.udp = UdpRigidBodiesViCON(udp_ip=self.ip, udp_port=self.port)
            self.udp.start_thread()
            self.sample_rate = self.udp.sample_rate
            self.num_bodies = self.udp.num_bodies
            self.dp = DataProcessorViCON(self.num_bodies, self.sample_rate)
            self.status = (f"ready ({self.num_bodies} bodies, "
                           f"{self.sample_rate:.0f} Hz)")
            return True
        except Exception as e:  # noqa: BLE001 — never let vicon setup reach flight
            self.status = f"vicon prepare error: {e}"
            self.udp = None
            return False

    def start(self, t0_wall, out_path):
        if self.recording or self.udp is None:   # prepare() must have succeeded
            return
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self.t0_wall = t0_wall
        self.path = out_path
        self.samples = 0
        self.first_packet_wall = None
        self.recording = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        if not self.recording:
            return
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        self.recording = False

    def _run(self):
        # Same loop body as the UdpReceiver template: 100 Hz RealTimeSleeper,
        # get_data() (latest packet from the receiver thread), process, then
        # Differentiate b1 x/y/z for velocities. Records from the first tick, so
        # the session starts at t0 (receiver is already running from prepare()).
        names = list(self.dp.save_list_name) + ["b1_x_dot", "b1_y_dot", "b1_z_dot"]
        rts = RealTimeSleeper(_VICON_SAMPLE_DT)
        diff_x = Differentiator(diff_steps=2)
        diff_y = Differentiator(diff_steps=2)
        diff_z = Differentiator(diff_steps=2)
        abs_time = []
        rows = []
        try:
            rts.init()
            while not self._stop.is_set():
                data_raw, udp_time = self.udp.get_data()
                data, save_list_data = self.dp.process_data(data_raw)
                if self.first_packet_wall is None:
                    self.first_packet_wall = time.time()
                    self.status = f"REC vicon ({self.num_bodies} bodies)"
                diff_x.step(data[1]["x"], udp_time)
                diff_y.step(data[1]["y"], udp_time)
                diff_z.step(data[1]["z"], udp_time)
                abs_time.append(time.time() - self.t0_wall)
                rows.append(list(save_list_data)
                            + [diff_x.data_rate, diff_y.data_rate, diff_z.data_rate])
                self.samples = len(rows)
                rts.sleep()
            self._save(abs_time, rows, names)
        except Exception as e:  # noqa: BLE001 — never let vicon faults reach flight
            self.status = f"vicon recorder error: {e}"
        finally:
            self.recording = False

    def _save(self, abs_time, rows, names):
        if not rows:
            self.status = "no Vicon samples recorded"
            return
        arr = np.asarray(rows, dtype=float)
        out = {
            "exptime": (datetime.datetime.fromtimestamp(self.t0_wall)
                        .strftime("%Y%m%d_%H%M%S")),
            "Abs_time": np.asarray(abs_time, dtype=float),
            "t0_wall": float(self.t0_wall),
            "first_packet_wall": float(self.first_packet_wall or self.t0_wall),
            "num_samples": len(rows),
            "sample_rate_hz": float(self.sample_rate or 0.0),
        }
        for i, nm in enumerate(names):
            out[nm] = arr[:, i]
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        sio.savemat(self.path, out)
        self.status = f"saved {os.path.basename(self.path)} ({len(rows)} samples)"


def build_channels(js, cal, allow_arm):
    """Read joystick state → (16-channel µs list, armed flag, record_on flag).

    This FC has NO prearm — a single arm switch on AUX3 (ARM_CH) gates arming.
    Gimbals are the standard AETR layout on indices 0-3. The blackbox switch is
    relayed to AUX2 (AUX2_CH) so the FC's blackbox starts when it's high. No mode
    channel is driven (config.CH_MODE is the Meteor75 value and would clobber
    AUX3 — see ARM_CH note above)."""
    ch = [config.NEUTRAL_US] * 16
    ax = cal["axes"]
    ch[config.CH_ROLL] = map_axis(js.axes.get(ax["roll"]["axis"], 0), ax["roll"])
    ch[config.CH_PITCH] = map_axis(js.axes.get(ax["pitch"]["axis"], 0), ax["pitch"])
    ch[config.CH_YAW] = map_axis(js.axes.get(ax["yaw"]["axis"], 0), ax["yaw"])
    ch[config.CH_THR] = map_axis(
        js.axes.get(ax["throttle"]["axis"], ax["throttle"]["lo_raw"]), ax["throttle"])

    armed = arm_is_armed(js, cal["arm"]) and allow_arm
    ch[ARM_CH] = ARM_ARMED_US if armed else ARM_DISARMED_US

    record_on = record_switch_on(js, cal.get("record"))
    ch[AUX2_CH] = aux2_us_for_switch(js, cal.get("record"))
    return ch, armed, record_on


# ===================== run loop =====================

CSI = "\033["


def _write_session_json(session, recorder, vicon):
    """Write per-session metadata so combine_flight.py + analysis can align all
    three streams to the shared trigger t0."""
    meta = {
        "stamp": session["stamp"],
        "t0_wall": session["t0"],
        "t0_human": datetime.datetime.fromtimestamp(session["t0"]).isoformat(),
        "aux2_on_us": AUX2_HIGH_US,
        "note": ("All streams begin at the switch flick (t0). vicon Abs_time and "
                 "blackbox time each zero-base to their first sample for "
                 "deterministic alignment; video lags t0 by video.start_offset_s. "
                 "Drop the .bbl in data_logging/blackbox/ then run combine_flight.py."),
    }
    if recorder is not None:
        ff = recorder.first_frame_wall
        meta["video"] = {
            "file": "video.mp4", "frames": recorder.frames, "fps": recorder.fps,
            "first_frame_wall": ff,
            "start_offset_s": (ff - session["t0"]) if ff else None,
            "status": recorder.status,
        }
    if vicon is not None:
        fp = vicon.first_packet_wall
        meta["vicon"] = {
            "file": "vicon.mat", "samples": vicon.samples,
            "first_packet_wall": fp,
            "first_packet_offset_s": (fp - session["t0"]) if fp else None,
            "status": vicon.status,
        }
    with open(os.path.join(session["dir"], "session.json"), "w") as f:
        json.dump(meta, f, indent=2)


def render(ch, armed, allow_arm, raw_armed, record_on, recorder, vicon,
           flight_mode, pack_v, bytes_rx, dry):
    arm_txt = (f"{CSI}32mARMED{CSI}0m" if armed
               else (f"{CSI}33msafe (switch ARMED — toggle to DISARMED first){CSI}0m"
                     if raw_armed and not allow_arm else "safe"))
    fm = flight_mode if flight_mode is not None else "—"
    v = f"{pack_v:.2f}V" if pack_v is not None else "—"
    mode_lbl = "DRY-RUN (not sending)" if dry else "LIVE → Ranger"
    # When the arm switch is engaged but throttle is too high, Betaflight will
    # silently refuse to arm (THROTTLE flag) — call it out explicitly.
    thr_hint = ""
    if raw_armed and ch[config.CH_THR] > ARM_THROTTLE_MAX_US:
        thr_hint = (f" {CSI}31m[THROTTLE {ch[config.CH_THR]}us > {ARM_THROTTLE_MAX_US}"
                    f" — lower stick fully to arm]{CSI}0m")
    if record_on:
        vid = f"vid{recorder.frames}" if (recorder and recorder.recording) else "vid-"
        vic = f"vic{vicon.samples}" if (vicon and vicon.recording) else "vic-"
        rec_txt = f"{CSI}31m●REC{CSI}0m({vid},{vic})" if (armed) else \
                  f"{CSI}33mBB-on(arm to record){CSI}0m"
    else:
        rec_txt = "rec-off"
    sys.stdout.write(
        f"\r{CSI}K[{mode_lbl}] "
        f"R{ch[config.CH_ROLL]:4d} P{ch[config.CH_PITCH]:4d} "
        f"T{ch[config.CH_THR]:4d} Y{ch[config.CH_YAW]:4d} | {arm_txt} "
        f"AUX3={ch[ARM_CH]:4d} AUX2={ch[AUX2_CH]:4d} {rec_txt} | "
        f"FC:{fm} {v} rx={bytes_rx}{thr_hint}"
    )
    sys.stdout.flush()


def run(args):
    cal = load_cal(args.cal_file)
    js = Joystick(args.js)
    print(f"Joystick: {js.name}  ({args.js})")
    print(f"Mapping:  roll=axis[{cal['axes']['roll']['axis']}] "
          f"pitch=axis[{cal['axes']['pitch']['axis']}] "
          f"thr=axis[{cal['axes']['throttle']['axis']}] "
          f"yaw=axis[{cal['axes']['yaw']['axis']}] "
          f"arm={cal['arm']['kind']}[{cal['arm']['index']}]")

    # Recording: the blackbox switch starts THREE synchronized streams at the
    # flick — FC blackbox (AUX2, relayed), laptop video, laptop vicon — and all
    # stop at disarm. Each flight gets its own session folder.
    rec = cal.get("record")
    recorder = vicon = None
    if rec is None:
        print("Recording: no 'record' switch in calibration — AUX2 held off, "
              "no video/vicon.")
    else:
        if CV2_OK:
            recorder = VideoRecorder(config.DEVICE_INDEX, config.WIDTH, config.HEIGHT)
            video_msg = f"video /dev/video{config.DEVICE_INDEX}"
        else:
            video_msg = "video OFF (no cv2 — run with .venv/bin/python)"
        if VICON_OK and not args.dry_run:
            vicon = ViconRecorder()
            # Connect + run the original startup sample-rate determination ONCE
            # now (fail-soft: if Vicon isn't streaming this disables it and we
            # fly without it). Keeping it here — not per session — means each
            # recording starts instantly at the switch flick, synced to t0.
            print(f"Vicon:    probing UDP :{vicon.port} (determining sample rate)…")
            if not vicon.prepare():
                vicon_msg = f"vicon OFF ({vicon.status})"
                vicon = None
            else:
                vicon_msg = f"vicon {vicon.status}"
        elif VICON_OK and args.dry_run:
            vicon_msg = "vicon OFF (dry-run)"
        else:
            vicon_msg = "vicon OFF (deps missing)"
        print(f"Recording: blackbox switch {rec['kind']}[{rec['index']}] → "
              f"AUX2 + {video_msg} + {vicon_msg}")
        print(f"           sessions → {REC_DIR}/<timestamp>/  "
              f"(records while ARMED + switch ON)")

    ser = None
    if not args.dry_run:
        port = args.port or autodetect_port()
        if not port:
            js.close()
            sys.exit("No Ranger serial port found. Plug in the Ranger USB-C, or "
                     "pass the port explicitly: joystick_flight.py /dev/ttyACM0")
        ser = open_ranger(port, args.baud)
        print(f"Ranger:   {port} @ {args.baud} baud")
    else:
        print("Ranger:   (dry-run — no serial opened, nothing transmitted)")

    print("\nArm is edge-gated: flip the ARM switch to DISARMED once to enable "
          "arming.\nCtrl-C to stop (sends disarm).\n")

    parser = CrsfParser()
    flight_mode = None
    pack_v = None
    bytes_rx = 0
    seen_disarmed = False
    prev_want_record = False

    period = 1.0 / config.TX_HZ
    nxt = time.monotonic()
    last_ping = 0.0
    last_render = 0.0
    PREVIEW_WIN = "drone feed — recording status"
    window_open = False

    def send_disarm():
        if ser is None:
            return
        ch = [config.NEUTRAL_US] * 16
        ch[config.CH_THR] = config.IDLE_THR_US
        ch[ARM_CH] = ARM_DISARMED_US
        ch[AUX2_CH] = AUX2_MID_US   # MID = nothing; never send LOW (=erase) here
        frame = build_rc_channels_packed(ch)
        for _ in range(5):
            ser.write(frame)
            time.sleep(0.01)

    # A "session" = one switch-on→disarm window. begin/end start & stop the
    # laptop recorders together off a single shared wall-clock t0 (the FC
    # blackbox starts on its own when it sees AUX2 go high, ~one frame later).
    session = {"dir": None, "t0": None, "stamp": None}

    def begin_session():
        t0 = time.time()
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        sdir = os.path.join(REC_DIR, stamp)
        os.makedirs(sdir, exist_ok=True)
        session.update(dir=sdir, t0=t0, stamp=stamp)
        if recorder is not None:
            recorder.start(os.path.join(sdir, "video.mp4"))
        if vicon is not None:
            vicon.start(t0, os.path.join(sdir, "vicon.mat"))
        sys.stdout.write(f"\n{CSI}32m● REC SESSION {stamp}{CSI}0m → {sdir}\n")

    def end_session():
        if recorder is not None:
            recorder.stop()
        if vicon is not None:
            vicon.stop()
        if session["dir"]:
            _write_session_json(session, recorder, vicon)
            sys.stdout.write(f"\n{CSI}33m■ SESSION SAVED{CSI}0m {session['stamp']}\n")
            if recorder is not None:
                sys.stdout.write(f"   video: {recorder.status}\n")
            if vicon is not None:
                sys.stdout.write(f"   vicon: {vicon.status}\n")
        session.update(dir=None, t0=None, stamp=None)

    try:
        while True:
            js.poll()
            if not js.alive:
                print("\n!! joystick disconnected — disarming.", flush=True)
                send_disarm()
                break

            # Edge-gate: only enable arm passthrough after seeing DISARMED once.
            raw_armed = arm_is_armed(js, cal["arm"])
            if not raw_armed:
                seen_disarmed = True

            ch, armed, record_on = build_channels(js, cal, allow_arm=seen_disarmed)

            # Recording mirrors the drone's blackbox: capture while ARMED and the
            # blackbox switch is ON; stop + save when either drops (disarm or
            # switch off). Edge-triggered on the desired state so a failed start
            # isn't retried every loop.
            want_record = armed and record_on
            if want_record and not prev_want_record:
                begin_session()
            elif prev_want_record and not want_record:
                end_session()
            prev_want_record = want_record

            if ser is not None:
                # Drain telemetry (non-blocking).
                waiting = ser.in_waiting
                if waiting:
                    chunk = ser.read(waiting)
                    bytes_rx += len(chunk)
                    for ftype, payload in parser.feed(chunk):
                        if ftype == T_FLIGHT_MODE:
                            flight_mode = decode_flight_mode(payload)
                        elif ftype == T_BATTERY:
                            b = decode_battery(payload)
                            if b:
                                pack_v = b["voltage_V"]
                # Send RC frame.
                try:
                    ser.write(build_rc_channels_packed(ch))
                except Exception as e:
                    print(f"\n!! Ranger write failed: {e} — stopping.", flush=True)
                    break
                now = time.monotonic()
                if now - last_ping > 2.0:
                    ser.write(build_device_ping())
                    last_ping = now

            now = time.monotonic()
            if now - last_render > 0.066:   # ~15 Hz
                render(ch, armed, seen_disarmed, raw_armed, record_on, recorder,
                       vicon, flight_mode, pack_v, bytes_rx, args.dry_run)
                # Live preview window: the feed + recording status while a session
                # is recording; closed between sessions. cv2 GUI must run on the
                # main thread (capture stays in the recorder thread), so drive it
                # here, throttled with the text render so it can't slow the RC loop.
                if recorder is not None and CV2_OK:
                    if recorder.recording:
                        frame = recorder.get_latest_frame()
                        if frame is not None:
                            disp = frame.copy()
                            vic = vicon.samples if (vicon and vicon.recording) else 0
                            cv2.putText(disp, f"REC  {recorder.frames}f   vicon {vic}",
                                        (14, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                        (0, 0, 255), 2)
                            cv2.putText(disp, "ARMED" if armed else "DISARMED",
                                        (14, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                        (0, 200, 0) if armed else (0, 200, 255), 2)
                            cv2.imshow(PREVIEW_WIN, disp)
                            cv2.waitKey(1)
                            window_open = True
                    elif window_open:
                        cv2.destroyAllWindows()
                        window_open = False
                last_render = now

            nxt += period
            s = nxt - time.monotonic()
            if s > 0:
                time.sleep(s)
            else:
                nxt = time.monotonic()
    except KeyboardInterrupt:
        print("\nCtrl-C — disarming.", flush=True)
        send_disarm()
    finally:
        # If we exit mid-session (Ctrl-C / disconnect while recording), stop and
        # save both streams so nothing is lost.
        if session["dir"]:
            end_session()
        if window_open and CV2_OK:
            cv2.destroyAllWindows()
        js.close()
        if ser is not None:
            send_disarm()
            ser.close()
        print("Stopped.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("port", nargs="?", default=None,
                    help="Ranger serial port (default: autodetect)")
    ap.add_argument("baud", nargs="?", type=int, default=420000,
                    help="Ranger baud (default: 420000)")
    ap.add_argument("--js", default="/dev/input/js0", help="joystick device")
    ap.add_argument("--cal-file", default=CAL_FILE_DEFAULT,
                    help="calibration JSON path")
    ap.add_argument("--calibrate", action="store_true",
                    help="run interactive calibration and exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="read joystick + print sticks; never open Ranger / never send")
    args = ap.parse_args()

    if args.calibrate:
        js = Joystick(args.js)
        try:
            calibrate(js, args.cal_file)
        finally:
            js.close()
        return
    run(args)


if __name__ == "__main__":
    main()
