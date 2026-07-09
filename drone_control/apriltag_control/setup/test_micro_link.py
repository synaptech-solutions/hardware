#!/usr/bin/env python3
"""Ranger Micro USB link test. Streams neutral CRSF RC at 115200 (the Micro's
remapped USB-CRSF rate) and prints live link stats. Rides through USB drops
(usbipd auto-attach) by reopening the port. Ctrl-C to stop.

Usage:  python3 setup/test_micro_link.py [/dev/ttyUSB0]
Run it IMMEDIATELY after the module powers up (within 60s, before WiFi mode).
"""
import os
import sys
import time
import array
import fcntl

import serial

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common.live_telemetry import crc8_dvb_s2, build_rc_channels_packed  # noqa: E402

_TCGETS2, _TCSETS2 = 0x802C542A, 0x402C542B
_BOTHER, _CBAUD = 0o010000, 0o010017
PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyUSB0"
BAUD = 115200
RC = build_rc_channels_packed([1500] * 16)


def set_baud(ser, baud):
    buf = array.array("i", [0] * 64)
    fcntl.ioctl(ser.fileno(), _TCGETS2, buf)
    buf[2] = (buf[2] & ~_CBAUD) | _BOTHER
    buf[9] = buf[10] = baud
    fcntl.ioctl(ser.fileno(), _TCSETS2, buf)


t0 = time.time()
last_print = 0.0
linked_once = False
print(f"streaming neutral RC to {PORT} @ {BAUD} — Ctrl-C to stop")
while True:
    if not os.path.exists(PORT):
        time.sleep(0.3)
        continue
    try:
        ser = serial.Serial()
        ser.port = PORT
        ser.baudrate = 115200
        ser.timeout = 0.005
        ser.dtr = True
        ser.rts = True
        ser.open()
        time.sleep(0.3)
        set_baud(ser, BAUD)
        ser.reset_input_buffer()
    except Exception:
        time.sleep(0.5)
        continue
    print(f"[{time.time()-t0:5.1f}s] port open, streaming...")
    buf = b""
    try:
        while True:
            ser.write(RC)
            buf += ser.read(2048)
            i = 0
            while i + 14 <= len(buf):
                if buf[i] in (0xC8, 0xEA, 0xEE) and buf[i+1] == 12 and buf[i+2] == 0x14:
                    body = buf[i+2:i+13]
                    if crc8_dvb_s2(body) == buf[i+13]:
                        p = body[1:]
                        up_lq, down_lq = p[2], p[8]
                        now = time.time()
                        if up_lq > 0 and not linked_once:
                            linked_once = True
                            print(f"[{now-t0:5.1f}s] *** RF LINK UP ***")
                        if now - last_print > 1.0:
                            last_print = now
                            state = "LINKED" if up_lq > 0 else "no RX link"
                            print(f"[{now-t0:5.1f}s] {state}  up_rssi={-p[0]} "
                                  f"up_lq={up_lq}  down_rssi={-p[7]} down_lq={down_lq}")
                        i += 14
                        continue
                i += 1
            buf = buf[max(0, len(buf)-32):]
            time.sleep(0.004)
    except KeyboardInterrupt:
        print("\nstopped." + ("  (RF link was seen)" if linked_once else "  (no RF link seen)"))
        sys.exit(0)
    except Exception:
        print(f"[{time.time()-t0:5.1f}s] port dropped — waiting for reattach")
        try:
            ser.close()
        except Exception:
            pass
        time.sleep(0.5)
