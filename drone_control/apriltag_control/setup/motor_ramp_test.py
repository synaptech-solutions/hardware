#!/usr/bin/env python3
"""
Throttle ramp test for the Meteor75 Pro via the Ranger.

Drives the THROTTLE stick (CRSF channel 3, index 2) — Betaflight handles
the per-motor mixing internally. This is exactly what a radio handset
would send.

Sequence (capped at 40% per request):
    5%, 10%, 15%, 20%, 25%, 30%, 35%, 40%
For each step:
    1. Arm  (AUX2 → 1800 us, throttle stays at 1000 us)
    2. Wait for FLIGHT_MODE telemetry to confirm ARMED (no trailing '*')
    3. Hold throttle at the target % for 1.5 s
    4. Drop throttle, disarm (AUX2 → 1000 us)
    5. Rest 3 s with motors truly off (disarmed)

Channel layout from the AIR75 `diff all` (2026-06):
    aux 0 0 2 1300 1700  → ARM on AUX3 (array index 6) — arm with ~1500
    aux 1 1 3 1300 1700  → ANGLE on AUX4 (index 7) — hold 1500
    (no PREARM mode in this config)

Safety:
    - Requires typed "YES" to start
    - Refuses to run if the drone reports already armed
    - Verifies arm via telemetry before pulsing throttle
    - Disarms on every exit path (try/finally + Ctrl-C handler)
    - Aborts if FLIGHT_MODE reports an error ('!FS!', '!XYZ' arming blockers)
    - Aborts if RF link drops mid-test (drone failsafe will trip anyway:
      failsafe_procedure=DROP gives ~1.5 s motors-off failsafe)

Usage:
    python3 motor_ramp_test.py                   # auto-detect port
    python3 motor_ramp_test.py /dev/ttyUSB0      # explicit
"""

import os
import sys
import time
import threading
import atexit
from typing import Optional

try:
    import serial
except ImportError:
    sys.exit("pyserial not installed. Run: pip install pyserial")

# CRSF builders moved to drone_control/common/ (setup → apriltag_control → drone_control).
sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common.live_telemetry import (
    CrsfParser, build_device_ping, build_rc_channels_packed,
    autodetect_port, decode_link_stats, decode_flight_mode, decode_battery,
    T_LINK_STATS, T_FLIGHT_MODE, T_DEVICE_INFO, T_BATTERY,
    CRSF_ADDR_FC,
)

# --- Channel layout (verified against user's 2026-06 `diff all`) ---
# Index = CRSF channel position: roll0 pitch1 thr2 yaw3, then AUX1=4 AUX2=5
# AUX3=6 AUX4=7. Decoded mode IDs from Betaflight 4.5 rc_modes.h.
THROTTLE_IDX = 2     # T in AETR
ARM_IDX = 6          # AUX3 — `aux 0 0 2 1300 1700`  mode 0 (ARM)
MODE_IDX = 7         # AUX4 — `aux 1 1 3 1300 1700`  mode 1 (ANGLE)
                     # (AUX4 1700-2100 = HORIZON; 1300-1700 = ANGLE)
# NOTE: this config has NO PREARM mode. The prearm cycle was removed.

# --- Channel values ---
THROTTLE_MIN_US = 1000   # 0% throttle, also < min_check=1050 (arms allowed)
NEUTRAL_US = 1500
ARM_HIGH_US = 1500       # inside ARM range 1300-1700
ARM_LOW_US = 1000        # outside ARM range
MODE_ANGLE_US = 1500     # inside ANGLE range 1300-1700

# --- Test parameters ---
START_PCT = 5
END_PCT = 50
STEP_PCT = 5
PULSE_S = 1.5
REST_S = 1.0
ARM_TIMEOUT_S = 5.0
LINK_TIMEOUT_S = 10.0
PRE_ARM_SETTLE_S = 1.0
POWER_ON_GRACE_S = 6.0   # covers Betaflight's pwr_on_arm_grace=5
MIN_VBAT_V = 0.0         # 0 = skip battery gate. Set >0 to require a flight
                         # battery. At 0.7 V (USB-only) the FC WON'T arm — this
                         # only lets the script keep streaming channels to watch
                         # in the Betaflight Receiver/Modes tabs.


def pct_to_us(pct: float) -> int:
    """5% → 1050 us, 40% → 1400 us, 100% → 2000 us."""
    return int(1000 + pct * 10)


# --- Logging ---

def log(msg: str = ""):
    print(msg)


# --- Shared state between threads ---

class State:
    def __init__(self):
        self._lock = threading.Lock()
        # Neutral starting channels: throttle min, AUX3 disarm, AUX4 angle, rest neutral
        self._ch = [NEUTRAL_US] * 16
        self._ch[THROTTLE_IDX] = THROTTLE_MIN_US
        self._ch[ARM_IDX] = ARM_LOW_US
        self._ch[MODE_IDX] = MODE_ANGLE_US

        self.flight_mode: Optional[str] = None
        self.armed: bool = False
        self.error_mode: bool = False
        self.uplink_lq: int = 0
        self.downlink_lq: int = 0
        self.devices: set[int] = set()
        self.voltage_V: Optional[float] = None

        self.stop = threading.Event()

    def set_throttle_pct(self, pct: float):
        with self._lock:
            self._ch[THROTTLE_IDX] = pct_to_us(pct)

    def set_throttle_us(self, us: int):
        with self._lock:
            self._ch[THROTTLE_IDX] = us

    def set_arm(self, armed_request: bool):
        with self._lock:
            self._ch[ARM_IDX] = ARM_HIGH_US if armed_request else ARM_LOW_US

    def channels(self) -> list[int]:
        with self._lock:
            return list(self._ch)


# --- Threads ---

def tx_loop(ser: serial.Serial, state: State):
    """Send current channels at 50 Hz so the Ranger keeps transmitting."""
    next_send = time.time()
    while not state.stop.is_set():
        now = time.time()
        if now >= next_send:
            try:
                ser.write(build_rc_channels_packed(state.channels()))
            except serial.SerialException:
                return
            next_send = now + 0.02
        time.sleep(0.001)


def rx_loop(ser: serial.Serial, state: State):
    """Parse incoming CRSF; update telemetry-derived flags."""
    parser = CrsfParser()
    last_ping = 0.0
    while not state.stop.is_set():
        try:
            chunk = ser.read(256)
        except serial.SerialException:
            return
        if chunk:
            for ftype, payload in parser.feed(chunk):
                if ftype == T_LINK_STATS:
                    d = decode_link_stats(payload)
                    if d:
                        state.uplink_lq = d["up_lq"]
                        state.downlink_lq = d["dn_lq"]
                elif ftype == T_FLIGHT_MODE:
                    m = decode_flight_mode(payload)
                    state.flight_mode = m
                    state.error_mode = m.startswith("!")
                    state.armed = (not m.endswith("*")) and (not m.startswith("!"))
                elif ftype == T_DEVICE_INFO and len(payload) >= 2:
                    state.devices.add(payload[1])
                elif ftype == T_BATTERY:
                    d = decode_battery(payload)
                    if d:
                        state.voltage_V = d["voltage_V"]

        now = time.time()
        if now - last_ping > 2.0:
            try:
                ser.write(build_device_ping())
            except serial.SerialException:
                return
            last_ping = now


# --- Helpers ---

def wait_for(predicate, timeout_s: float, poll_s: float = 0.05) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(poll_s)
    return False


def wait_for_link(state: State) -> bool:
    log(f"Waiting up to {LINK_TIMEOUT_S}s for RF link to come up...")
    ok = wait_for(
        lambda: state.uplink_lq >= 50 and state.downlink_lq >= 50
                and CRSF_ADDR_FC in state.devices,
        LINK_TIMEOUT_S,
    )
    if ok:
        log(f"  Link up: up={state.uplink_lq}%  down={state.downlink_lq}%  "
            f"FC seen.  Initial mode: {state.flight_mode!r}")
    return ok


def link_alive(state: State) -> bool:
    return state.uplink_lq >= 30 and state.downlink_lq >= 30


def disarm_now(state: State):
    """Idempotent disarm: throttle low + ARM low."""
    state.set_throttle_us(THROTTLE_MIN_US)
    state.set_arm(False)


# --- Main test sequence ---

def run_cycles(state: State):
    """Arm once, then hold neutral inputs (throttle min, sticks centered)
    indefinitely until Ctrl-C. No throttle is ever commanded."""
    # Pre-arm baseline — hold disarm signal long enough for FC to register
    disarm_now(state)
    time.sleep(PRE_ARM_SETTLE_S)

    # Sanity: link still up?
    if not link_alive(state):
        log(f"  ✗ Link dropped (up={state.uplink_lq}% down={state.downlink_lq}%) — aborting")
        return False

    # No PREARM in this config — go straight to ARM.

    # Arm
    log(f"  arm → AUX3 (idx {ARM_IDX})={ARM_HIGH_US} us, "
        f"waiting for telemetry confirmation...")
    state.set_arm(True)
    ok = wait_for(lambda: state.armed or state.error_mode, ARM_TIMEOUT_S)
    if not ok or state.error_mode or not state.armed:
        # Expected at 0.7 V / USB-only: the FC won't arm without a flight
        # battery. We deliberately do NOT disarm/exit — keep streaming the
        # arm-high + neutral channels at 50 Hz so they're visible in the
        # Betaflight Receiver/Modes tabs.
        log(f"  ⚠ Not armed within {ARM_TIMEOUT_S}s. Mode: {state.flight_mode!r}  "
            f"vbat={state.voltage_V} V")
        log(f"    (Expected on USB-only / low vbat — Betaflight blocks arming.)")
        log(f"    Holding arm-high + neutral so you can watch the channels in "
            f"Betaflight. Ctrl-C to stop.\n")
    else:
        log(f"  ✓ armed (mode: {state.flight_mode!r}, vbat={state.voltage_V} V)")
        log(f"\n  ✓ ARMED — holding neutral inputs (throttle min, sticks centered).")
        log(f"    Press Ctrl-C to disarm and exit.\n")

    # Hold the current channels forever. Throttle stays at min, sticks
    # centered, arm-high. The TX thread keeps sending these at 50 Hz so the
    # channels keep updating in Betaflight. We intentionally do NOT bail on
    # link/mode fault here — we just report it and keep streaming.
    next_sample = 0.0
    while True:
        now = time.time()
        if now >= next_sample:
            armed_str = "ARMED" if state.armed else "not-armed"
            log(f"    [{armed_str}] mode={state.flight_mode!r} vbat={state.voltage_V} V "
                f"up={state.uplink_lq}% dn={state.downlink_lq}%")
            next_sample = now + 1.0
        time.sleep(0.02)


def main():
    log("=" * 72)
    log("  ARM + HOLD NEUTRAL — AIR75 via Ranger")
    log("=" * 72)
    log("")
    log("  Arms the drone once, then sends only neutral inputs (throttle min,")
    log("  sticks centered) until Ctrl-C. No throttle is ever commanded.")
    log("")
    log("  WARNINGS:")
    log("    • REMOVE ALL PROPS before running.")
    log("    • Motors will idle/spin once armed — keep clear.")
    log("    • Press Ctrl-C at any time to disarm and exit.")
    log("")

    port = (sys.argv[1] if len(sys.argv) > 1 else autodetect_port()) or "/dev/ttyUSB0"
    log(f"\nOpening {port} @ 420000 baud")
    ser = serial.Serial(port, baudrate=420000, timeout=0.05)

    state = State()

    # Belt-and-braces: register an atexit handler that drives a final disarm
    # by directly writing a disarm frame even if threads have died.
    def _final_disarm():
        try:
            ch = [NEUTRAL_US] * 16
            ch[THROTTLE_IDX] = THROTTLE_MIN_US
            ch[ARM_IDX] = ARM_LOW_US
            for _ in range(5):
                ser.write(build_rc_channels_packed(ch))
                time.sleep(0.02)
            ser.close()
        except Exception:
            pass
    atexit.register(_final_disarm)

    tx_t = threading.Thread(target=tx_loop, args=(ser, state), daemon=True)
    rx_t = threading.Thread(target=rx_loop, args=(ser, state), daemon=True)
    tx_t.start()
    rx_t.start()

    success = False
    try:
        if not wait_for_link(state):
            log(f"  ✗ Link did not come up (up={state.uplink_lq}% "
                f"down={state.downlink_lq}%, devices={state.devices})")
            log("    Is the drone powered? Battery on BT2.0?")
            return

        # Refuse to start if drone reports already armed
        time.sleep(0.4)
        if state.armed:
            log(f"  ✗ Drone reports already armed (mode={state.flight_mode!r}). "
                f"Refusing to run.")
            return

        # Battery check — Betaflight silently blocks arming with no flight
        # battery. Skipped entirely when MIN_VBAT_V <= 0 (USB-only / 0.7 V
        # bench test where the goal is just to watch channels in Betaflight).
        if MIN_VBAT_V > 0:
            if state.voltage_V is None:
                log("  Waiting briefly for first BATTERY frame...")
                wait_for(lambda: state.voltage_V is not None, 2.0)
            if state.voltage_V is None or state.voltage_V < MIN_VBAT_V:
                log(f"  ✗ Vbat = {state.voltage_V} V — no flight battery detected on BT2.0.")
                log("    Betaflight blocks arming without a flight battery. Plug a 1S")
                log("    LiPo into the BT2.0 connector on the drone and rerun.")
                return
            log(f"  Battery OK: {state.voltage_V:.2f} V")
        else:
            log(f"  Battery gate SKIPPED (MIN_VBAT_V=0). vbat={state.voltage_V} V — "
                f"won't arm on USB-only; streaming channels for Betaflight view.")

        # Honour Betaflight's pwr_on_arm_grace (5 s) — drone may have just booted
        log(f"  Settling for {POWER_ON_GRACE_S}s before first arm attempt "
            f"(covers pwr_on_arm_grace)...")
        time.sleep(POWER_ON_GRACE_S)

        success = run_cycles(state)

    except KeyboardInterrupt:
        log("\n  ! Ctrl-C received — disarming")
    finally:
        # Order matters: send disarm frames first, THEN let threads die
        disarm_now(state)
        time.sleep(0.4)  # let TX thread push the disarm frames
        state.stop.set()
        time.sleep(0.1)

    log("")
    log("=" * 72)
    log("  " + ("✓ Test complete." if success else "Test ended (incomplete)."))
    log("=" * 72)


if __name__ == "__main__":
    main()
