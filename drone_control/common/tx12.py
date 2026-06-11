"""TX12 USB-joystick input: reader, calibration, and switch interpreters.

The TX12 (EdgeTX) in "USB Joystick (HID)" mode appears as /dev/input/js0
(7 axes, 24 buttons on this radio). This module reads the gimbals + arm / record /
mode switches and maps them to CRSF µs. EdgeTX's channel→USB-axis assignment is
radio-specific, so the mapping is discovered by `--calibrate` and stored in
tx12_joystick_cal.json (shared by the data logger and the Vicon controller).

Extracted verbatim from data_logging/joystick_flight.py so both the manual data
logger and the autonomous Vicon controller read the TX12 the SAME way. µs levels
for AUX2/AUX4 come from common.channels (the single source of truth).
"""
import os
import sys
import json
import time
import struct
import fcntl

from . import channels

# Shared TX12 calibration file (next to the data logger). Both pipelines use the
# same radio config, so they share one calibration.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CAL_PATH = os.path.join(_REPO, "data_logging", "tx12_joystick_cal.json")


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

    mode_spec = _capture_mode_switch(js)
    if mode_spec:
        cal["mode"] = mode_spec

    with open(cal_path, "w") as f:
        json.dump(cal, f, indent=2)
    print(f"Saved calibration → {cal_path}")
    return cal


def _capture_mode_switch(js):
    """Capture the 3-position flight-mode switch (an axis). Returns a spec
    {kind, index, low_raw, high_raw} or None if no clear movement was seen. The
    switch drives AUX4: low → acro, mid → angle, high → horizon."""
    print("FLIGHT-MODE SWITCH (optional 3-position → AUX4: low=ACRO, mid=ANGLE, "
          "high=HORIZON — leave still + Enter twice to skip):")
    lo_ax, _ = _capture(js, "set the MODE switch to LOW (acro / nothing)")
    hi_ax, _ = _capture(js, "set the MODE switch to HIGH (horizon)")
    deltas = {n: abs(hi_ax.get(n, 0) - lo_ax.get(n, 0))
              for n in set(lo_ax) | set(hi_ax)}
    best = max(deltas, key=deltas.get) if deltas else None
    if best is None or deltas[best] < 8000:
        print("     (no mode switch detected — AUX4 will default to ANGLE)\n")
        return None
    print(f"     mode → axis[{best}]  low_raw={lo_ax[best]:+6d} "
          f"high_raw={hi_ax[best]:+6d}\n")
    return {"kind": "axis", "index": best,
            "low_raw": lo_ax[best], "high_raw": hi_ax[best]}


def calibrate_mode_only(js, cal_path):
    """Capture ONLY the flight-mode switch and merge it into the existing
    calibration (keeps gimbals/arm/record), for adding the mode switch to an
    already-calibrated setup without redoing everything."""
    cal = load_cal(cal_path)            # requires an existing valid calibration
    spec = _capture_mode_switch(js)
    if not spec:
        print("No mode switch captured — calibration unchanged.")
        return None
    cal["mode"] = spec
    with open(cal_path, "w") as f:
        json.dump(cal, f, indent=2)
    print(f"Saved (mode switch merged) → {cal_path}")
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
        return channels.AUX2_MID_US
    if spec.get("kind") == "button":
        on = js.buttons.get(spec["index"], 0) == spec.get("on_raw", 1)
        return channels.AUX2_HIGH_US if on else channels.AUX2_MID_US
    raw = js.axes.get(spec["index"])
    if raw is None:
        return channels.AUX2_MID_US
    if raw > 16384:
        return channels.AUX2_HIGH_US     # high detent → start blackbox logging
    if raw < -16384:
        return channels.AUX2_LOW_US      # low detent → erase blackbox
    return channels.AUX2_MID_US          # middle → no blackbox mode


def mode_us_for_switch(js, spec):
    """Map the 3-position flight-mode switch to AUX4 µs: LOW→acro, MID→angle,
    HIGH→horizon. Uses the calibrated low/high endpoints to place the reading on
    a 0→1 fraction, so it's correct even if the switch axis is inverted. Defaults
    to ACRO when no mode switch is calibrated / unreadable (per user: AUX4
    defaults to acro)."""
    if spec is None:
        return channels.MODE_ACRO_US
    raw = js.axes.get(spec["index"])
    if raw is None:
        return channels.MODE_ACRO_US
    lo, hi = spec["low_raw"], spec["high_raw"]
    if hi == lo:
        return channels.MODE_ACRO_US
    frac = (raw - lo) / (hi - lo)    # ~0 at the low detent, ~1 at the high detent
    if frac >= 0.66:
        return channels.MODE_HORIZON_US
    if frac <= 0.33:
        return channels.MODE_ACRO_US
    return channels.MODE_ANGLE_US
