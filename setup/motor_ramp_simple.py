#!/usr/bin/env python3
"""Dumb-simple throttle ramp with verbose telemetry. REMOVE PROPS. Ctrl-C to abort.

The flight mode telemetry is the key debug signal. Betaflight reports the
first blocking arming flag as the flight-mode string. Watch for things like
'!ERR', '!FS!', '!ARM' (arm-switch was high at boot — must toggle low first),
'!THR' (throttle not zero), '!ANG' (drone tilted past max_angle_inclination),
'!NOG' / '!CAL' (gyro / calibration), '!RXL' (RX lost), '!CRS' (CRSF link).
A trailing '*' on a mode (e.g. 'ANGL*') just means disarmed.
"""
import time, sys
sys.path.insert(0, "/home/andy-li/Desktop/Synetic Labs/hardware")
import serial
from live_telemetry import (build_rc_channels_packed, autodetect_port, CrsfParser,
                            decode_link_stats, decode_flight_mode,
                            T_LINK_STATS, T_FLIGHT_MODE, T_DEVICE_INFO, T_BATTERY)

THR, ARM = 2, 5
ARM_HIGH = 2000   # well inside Betaflight arm range 1200-2100
ARM_LOW  = 1000

t0 = time.time()
def log(msg): print(f"[{time.time()-t0:6.2f}s] {msg}", flush=True)

def send(ser, thr_us, arm_us):
    ch = [1500]*16; ch[THR] = thr_us; ch[ARM] = arm_us
    ser.write(build_rc_channels_packed(ch))

def hold(ser, parser, state, thr_us, arm_us, secs, label):
    log(f"  >> {label}  thr={thr_us}us  arm={arm_us}us  for {secs:.1f}s")
    end = time.time() + secs
    last_tx = 0.0
    last_mode_print = 0.0
    while time.time() < end:
        now = time.time()
        if now - last_tx >= 0.02:
            send(ser, thr_us, arm_us); last_tx = now
        chunk = ser.read(256)
        if chunk:
            for ftype, payload in parser.feed(chunk):
                if ftype == T_FLIGHT_MODE:
                    m = decode_flight_mode(payload)
                    if m != state["mode"]:
                        log(f"    FLIGHT_MODE: {state['mode']!r} -> {m!r}")
                        state["mode"] = m
                elif ftype == T_LINK_STATS:
                    d = decode_link_stats(payload)
                    if d and (d["up_lq"] != state["up"] or d["dn_lq"] != state["dn"]):
                        state["up"], state["dn"] = d["up_lq"], d["dn_lq"]
                        log(f"    LINK up={d['up_lq']}% dn={d['dn_lq']}%")
                elif ftype == T_DEVICE_INFO and len(payload) >= 2 and payload[1] not in state["devs"]:
                    state["devs"].add(payload[1])
                    log(f"    DEVICE seen src=0x{payload[1]:02X}")
                elif ftype == T_BATTERY and state["batt"] is None:
                    state["batt"] = True
                    log(f"    BATTERY telemetry first frame received")
        # heartbeat once per second so we know we're still alive
        if now - last_mode_print >= 1.0:
            last_mode_print = now
            log(f"    (still here — mode={state['mode']!r}  link up/dn={state['up']}/{state['dn']})")
        time.sleep(0.001)

input("REMOVE PROPS. Press ENTER to start, Ctrl-C to abort. ")
port = sys.argv[1] if len(sys.argv) > 1 else autodetect_port()
log(f"opening {port} @ 420000")
ser = serial.Serial(port, baudrate=420000, timeout=0.02)
parser = CrsfParser()
state = {"mode": None, "up": None, "dn": None, "devs": set(), "batt": None}

try:
    log("STEP 1: zero throttle + DISARM, 5s (link warm-up; FC must SEE the disarm switch first)")
    hold(ser, parser, state, ARM_LOW,  ARM_LOW,  5.0, "disarm/warmup")

    if state["mode"] is None:
        log("WARN: no FLIGHT_MODE telemetry yet — link may not be up. Continuing anyway.")
    else:
        log(f"baseline mode before arm: {state['mode']!r}")

    log("STEP 2: command ARM (throttle still zero), 4s — watch FLIGHT_MODE change")
    hold(ser, parser, state, ARM_LOW, ARM_HIGH, 4.0, "ARM request")

    armed = state["mode"] is not None and not state["mode"].endswith("*") and not state["mode"].startswith("!")
    if not armed:
        log(f"!! ARM REFUSED — FC reports mode={state['mode']!r}")
        log("!! Aborting ramp. See module docstring for what '!XXX' codes mean.")
    else:
        log(f"** ARMED — mode={state['mode']!r}")

        log("STEP 3: do nothing — throttle zero, armed, 3s")
        hold(ser, parser, state, ARM_LOW, ARM_HIGH, 3.0, "armed idle")

        log("STEP 4: takeoff ramp 5% -> 40% in 5% steps")
        for pct in range(5, 41, 5):
            us = 1000 + pct*10
            hold(ser, parser, state, us,      ARM_HIGH, 1.5, f"thr {pct}%")
            hold(ser, parser, state, ARM_LOW, ARM_HIGH, 1.0, "thr 0 (still armed)")
            if state["mode"] is not None and state["mode"].startswith("!"):
                log(f"!! FC went to error mode {state['mode']!r} mid-ramp — bailing")
                break
finally:
    log("STEP 5: DISARM, 1.5s")
    hold(ser, parser, state, ARM_LOW, ARM_LOW, 1.5, "disarm")
    ser.close()
    log(f"done. final mode={state['mode']!r}")
