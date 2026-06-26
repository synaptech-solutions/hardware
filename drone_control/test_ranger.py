#!/usr/bin/env python3
"""Quick Ranger serial diagnostic.

Sends CRSF RC channel frames + device pings to the Ranger and prints
every byte received, decoded as CRSF where possible. Shows whether:
  - The Ranger sees a valid handset (RC frames at ~50 Hz)
  - The Ranger responds to pings (DEVICE_INFO)
  - The FC on the drone is forwarding telemetry (BATTERY, ATTITUDE, etc.)

Usage:
    python drone_control/test_ranger.py                       # autodetect
    python drone_control/test_ranger.py /dev/cu.usbserial-0001
"""
import sys
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DRONE_CONTROL = os.path.join(HERE, "Vicon_control")
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from common.ranger import open_ranger
from common.live_telemetry import (
    build_rc_channels_packed, build_device_ping, autodetect_port, CrsfParser,
    ADDR_NAMES, T_BATTERY, T_LINK_STATS, T_ATTITUDE, T_FLIGHT_MODE,
    T_DEVICE_INFO, T_DEVICE_PING, T_RC_CHANNELS,
)

FRAME_NAMES = {
    0x08: "BATTERY", 0x14: "LINK_STATS", 0x16: "RC_CHANNELS",
    0x1E: "ATTITUDE", 0x21: "FLIGHT_MODE", 0x28: "DEVICE_PING",
    0x29: "DEVICE_INFO", 0x7A: "MSP_REQ", 0x7B: "MSP_RESP",
}

def main():
    port = sys.argv[1] if len(sys.argv) > 1 else autodetect_port()
    if not port:
        sys.exit("No serial port found. Pass it explicitly or plug in the Ranger.")
    baud = int(sys.argv[2]) if len(sys.argv) > 2 else 420000

    print(f"Opening {port} @ {baud} ...")
    ser = open_ranger(port, baud)
    print(f"Port open. Sending RC frames (neutral sticks) + pings.\n")

    parser = CrsfParser()
    neutral = [1500] * 16
    neutral[4] = 1000  # AUX1 low (disarmed)

    tx_bytes = 0
    rx_bytes = 0
    rx_frames = 0
    t0 = time.monotonic()
    last_rc = 0.0
    last_ping = 0.0

    try:
        while True:
            now = time.monotonic()
            elapsed = now - t0

            # Send RC at ~50 Hz
            if now - last_rc >= 0.02:
                pkt = build_rc_channels_packed(neutral)
                ser.write(pkt)
                tx_bytes += len(pkt)
                last_rc = now

            # Send ping every 2s
            if now - last_ping >= 2.0:
                pkt = build_device_ping()
                ser.write(pkt)
                tx_bytes += len(pkt)
                last_ping = now

            # Read
            if ser.in_waiting:
                chunk = ser.read(ser.in_waiting)
                rx_bytes += len(chunk)
                for ftype, payload in parser.feed(chunk):
                    rx_frames += 1
                    name = FRAME_NAMES.get(ftype, f"0x{ftype:02X}")
                    extra = ""
                    if ftype == T_DEVICE_INFO and len(payload) > 2:
                        # payload starts with dest, origin, then null-terminated name
                        name_bytes = payload[2:] if len(payload) > 2 else payload
                        dev_name = name_bytes.split(b"\x00", 1)[0].decode(errors="replace")
                        extra = f"  device=\"{dev_name}\""
                    elif ftype == T_FLIGHT_MODE:
                        mode = payload.rstrip(b"\x00").decode(errors="replace")
                        extra = f"  mode=\"{mode}\""
                    elif ftype == T_BATTERY and len(payload) >= 8:
                        v = int.from_bytes(payload[0:2], "big") / 10.0
                        extra = f"  {v:.1f}V"
                    elif ftype == T_LINK_STATS and len(payload) >= 10:
                        extra = f"  upRSSI={payload[0]}dBm upLQ={payload[2]}%"
                    print(f"  [{elapsed:7.1f}s] RX frame #{rx_frames}: {name:16s} "
                          f"({len(payload):3d}B){extra}")

            # Status line every 1s
            if int(elapsed) > int(elapsed - 0.02):
                sys.stdout.write(
                    f"\r  >> {elapsed:.0f}s  tx={tx_bytes}B  rx={rx_bytes}B  "
                    f"frames={rx_frames}  ")
                sys.stdout.flush()

            time.sleep(0.005)
    except KeyboardInterrupt:
        print(f"\n\nStopped after {time.monotonic()-t0:.1f}s. "
              f"tx={tx_bytes}B rx={rx_bytes}B frames={rx_frames}")
    finally:
        ser.close()


if __name__ == "__main__":
    main()
