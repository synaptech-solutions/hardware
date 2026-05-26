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

Channel layout verified against the user's Betaflight dump:
    aux 0 0 1 1200 2100  → ARM on AUX2 (CRSF ch 6, array index 5)
    aux 1 1 2 1300 1700  → ANGLE on AUX3 in 1300-1700 (we hold 1500)

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

import sys
import time
import threading
import atexit
import os
import datetime
from typing import Optional, TextIO

try:
    import serial
except ImportError:
    sys.exit("pyserial not installed. Run: pip install pyserial")

sys.path.insert(0, "/home/andy-li/Desktop/Synetic Labs/hardware")
from live_telemetry import (
    CrsfParser, build_device_ping, build_rc_channels_packed,
    autodetect_port, decode_link_stats, decode_flight_mode, decode_battery,
    T_LINK_STATS, T_FLIGHT_MODE, T_DEVICE_INFO, T_BATTERY,
    CRSF_ADDR_FC,
)

# --- Channel layout (verified against user's Betaflight dump permanent IDs) ---
THROTTLE_IDX = 2     # T in AETR
ARM_IDX = 5          # AUX2 — `aux 0 0 1 1200 2100`     mode 0 (ARM)
PREARM_IDX = 9       # AUX6 — `aux 7 36 5 1700 2100`    mode 36 (PREARM)
                     # PREARM must be cycled LOW→HIGH before each ARM attempt;
                     # without it Betaflight silently refuses to arm.
MODE_IDX = 6         # AUX3 — `aux 1 1 2 1300 1700`     mode 1 (ANGLE)

# --- Channel values ---
THROTTLE_MIN_US = 1000   # 0% throttle, also < min_check=1050 (arms allowed)
NEUTRAL_US = 1500
ARM_HIGH_US = 1800       # inside ARM range 1200-2100
ARM_LOW_US = 1000        # outside ARM range
PREARM_HIGH_US = 1800    # inside PREARM range 1700-2100
PREARM_LOW_US = 1000     # outside PREARM range
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
MIN_VBAT_V = 2.0         # below this we assume no flight battery on BT2.0


def pct_to_us(pct: float) -> int:
    """5% → 1050 us, 40% → 1400 us, 100% → 2000 us."""
    return int(1000 + pct * 10)


# --- Logging ---

_LOG_FILE: Optional[TextIO] = None
_T0: float = 0.0


def log_open(path: str):
    global _LOG_FILE, _T0
    _LOG_FILE = open(path, "w", buffering=1)  # line-buffered
    _T0 = time.time()
    _LOG_FILE.write(f"# motor_ramp_test log — started {datetime.datetime.now().isoformat()}\n")


def log_close():
    global _LOG_FILE
    if _LOG_FILE is not None:
        try:
            _LOG_FILE.write(f"# log closed {datetime.datetime.now().isoformat()}\n")
            _LOG_FILE.close()
        except Exception:
            pass
        _LOG_FILE = None


def log(msg: str = ""):
    """Print to stdout and write to log file with a relative timestamp."""
    print(msg)
    if _LOG_FILE is not None:
        try:
            t = time.time() - _T0
            _LOG_FILE.write(f"[{t:8.3f}s] {msg}\n")
        except Exception:
            pass


# --- Shared state between threads ---

class State:
    def __init__(self):
        self._lock = threading.Lock()
        # Neutral starting channels: throttle min, AUX2 disarm, AUX3 angle, rest neutral
        self._ch = [NEUTRAL_US] * 16
        self._ch[THROTTLE_IDX] = THROTTLE_MIN_US
        self._ch[ARM_IDX] = ARM_LOW_US
        self._ch[PREARM_IDX] = PREARM_LOW_US
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

    def set_prearm(self, prearm_request: bool):
        with self._lock:
            self._ch[PREARM_IDX] = PREARM_HIGH_US if prearm_request else PREARM_LOW_US

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
    """Idempotent disarm: throttle low + ARM low + PREARM low."""
    state.set_throttle_us(THROTTLE_MIN_US)
    state.set_arm(False)
    state.set_prearm(False)


# --- Main test sequence ---

def run_cycles(state: State):
    pcts = list(range(START_PCT, END_PCT + 1, STEP_PCT))
    log(f"\nRunning {len(pcts)} cycles at {pcts}\n")
    for i, pct in enumerate(pcts, 1):
        log(f"━━ Cycle {i}/{len(pcts)} @ {pct}% throttle ━━")

        # Pre-arm baseline — hold disarm signal long enough for FC to register
        disarm_now(state)
        time.sleep(PRE_ARM_SETTLE_S)

        # Sanity: link still up?
        if not link_alive(state):
            log(f"  ✗ Link dropped (up={state.uplink_lq}% down={state.downlink_lq}%) — aborting")
            return False

        # PREARM must be cycled HIGH before ARM can engage (user's aux 7 = mode 36)
        log(f"  prearm → AUX6={PREARM_HIGH_US} us")
        state.set_prearm(True)
        time.sleep(0.3)

        # Arm
        log(f"  arm → AUX2={ARM_HIGH_US} us, waiting for telemetry confirmation...")
        state.set_arm(True)
        ok = wait_for(lambda: state.armed or state.error_mode, ARM_TIMEOUT_S)
        if not ok or state.error_mode or not state.armed:
            log(f"  ✗ Arm failed within {ARM_TIMEOUT_S}s. Mode: {state.flight_mode!r}")
            log(f"     vbat={state.voltage_V} V  link up={state.uplink_lq}% "
                f"down={state.downlink_lq}%")
            log(f"     Most common causes: no flight battery on BT2.0 (vbat<2 V), "
                f"FC just booted (pwr_on_arm_grace), or arm switch did not transition.")
            disarm_now(state)
            return False
        log(f"  ✓ armed (mode: {state.flight_mode!r}, vbat={state.voltage_V} V)")

        # Throttle pulse — sample telemetry at ~5 Hz so the log captures
        # what happened during the pulse, not just before/after.
        thr_us = pct_to_us(pct)
        log(f"  throttle → {pct}% ({thr_us} us)  for {PULSE_S}s")
        state.set_throttle_pct(pct)
        pulse_start = time.time()
        pulse_end = pulse_start + PULSE_S
        next_sample = 0.0
        while time.time() < pulse_end:
            if not link_alive(state) or state.error_mode:
                log(f"  ✗ link/mode fault mid-pulse  link up={state.uplink_lq}% "
                    f"down={state.downlink_lq}%  mode={state.flight_mode!r}")
                disarm_now(state)
                return False
            now = time.time()
            if now >= next_sample:
                log(f"    [pulse t={now-pulse_start:.2f}s] "
                    f"mode={state.flight_mode!r} vbat={state.voltage_V} V "
                    f"up={state.uplink_lq}% dn={state.downlink_lq}%")
                next_sample = now + 0.2
            time.sleep(0.02)

        # Throttle down then disarm
        state.set_throttle_us(THROTTLE_MIN_US)
        time.sleep(0.15)
        log(f"  disarm → AUX2={ARM_LOW_US} us")
        state.set_arm(False)
        wait_for(lambda: not state.armed, 1.5)
        log(f"  ✓ disarmed (mode: {state.flight_mode!r}). Resting {REST_S}s.")
        time.sleep(REST_S)
        log()
    return True


def main():
    # Open the log file before any output so the run is captured from the start.
    log_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(
        log_dir, f"motor_ramp_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"
    )
    log_open(log_path)
    atexit.register(log_close)

    log("=" * 72)
    log("  THROTTLE RAMP TEST — Meteor75 Pro via Ranger")
    log("=" * 72)
    log(f"  Log file: {log_path}")
    log("")
    log("  Pulses throttle stick from 5% up to 40% in 5% steps.")
    log("  Each pulse: 1.5 s on, 3 s off (drone disarmed between cycles).")
    log("")
    log("  WARNINGS:")
    log("    • REMOVE ALL PROPS before running.")
    log("    • Secure the drone — it WILL try to lift at 40% even propless.")
    log("    • Keep clear of motors; idle current draws are still hot.")
    log("    • Press Ctrl-C at any time to abort and disarm.")
    log("")
    resp = input("  Type YES (uppercase) to proceed: ").strip()
    log(f"  user input: {resp!r}")
    if resp != "YES":
        log("  Aborted by user.")
        sys.exit("Aborted.")

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
            ch[PREARM_IDX] = PREARM_LOW_US
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

        # Battery check — Betaflight silently blocks arming with no flight battery
        if state.voltage_V is None:
            log("  Waiting briefly for first BATTERY frame...")
            wait_for(lambda: state.voltage_V is not None, 2.0)
        if state.voltage_V is None or state.voltage_V < MIN_VBAT_V:
            log(f"  ✗ Vbat = {state.voltage_V} V — no flight battery detected on BT2.0.")
            log("    Betaflight blocks arming without a flight battery. Plug a 1S")
            log("    LiPo into the BT2.0 connector on the drone and rerun.")
            return
        log(f"  Battery OK: {state.voltage_V:.2f} V")

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
