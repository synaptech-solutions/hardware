#!/usr/bin/env python3
"""Listen-only sniffer for the TX12 USB serial port.

EdgeTX exposes a generic CDC-ACM VCP whose output depends on the radio's
serial-port mode. We don't send anything (the radio owns the RF link); we
just open the port at a candidate baud, read for a couple seconds, and
report what we see: CRSF telemetry frames (Telem Mirror), printable CLI
text, or nothing.

Usage: python3 sniff_tx12.py [/dev/ttyACM1] [baud1 baud2 ...]
"""
import sys
import time

import serial

PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM1"
BAUDS = [int(b) for b in sys.argv[2:]] or [420000, 115200, 400000, 230400, 57600]

CRSF_SYNC = {0xC8, 0xEE, 0xEA, 0xEC}
CRSF_TYPE_NAMES = {
    0x08: "BATTERY", 0x14: "LINK_STATS", 0x16: "RC_CHANNELS",
    0x1E: "ATTITUDE", 0x21: "FLIGHT_MODE", 0x29: "DEVICE_INFO",
    0x02: "GPS", 0x07: "VARIO", 0x09: "BARO_ALT", 0x0B: "HEARTBEAT",
}


def crc8_dvb_s2(data: bytes) -> int:
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0xD5) & 0xFF if (crc & 0x80) else (crc << 1) & 0xFF
    return crc


def scan_crsf(buf: bytes):
    """Return dict of frame-type -> count for CRC-valid CRSF frames in buf."""
    types = {}
    i = 0
    n = len(buf)
    while i < n - 2:
        if buf[i] in CRSF_SYNC:
            length = buf[i + 1]
            if 2 <= length <= 62 and i + 2 + length <= n:
                body = buf[i + 2:i + 1 + length]
                crc = buf[i + 1 + length]
                if crc8_dvb_s2(body) == crc:
                    ftype = body[0]
                    types[ftype] = types.get(ftype, 0) + 1
                    i += 2 + length
                    continue
        i += 1
    return types


def main():
    print(f"Sniffing {PORT} (listen-only, no TX) ...")
    for baud in BAUDS:
        try:
            ser = serial.Serial(PORT, baudrate=baud, timeout=0.1)
        except Exception as e:
            print(f"  {baud:>7}: cannot open: {e}")
            continue
        buf = bytearray()
        t_end = time.time() + 2.0
        while time.time() < t_end:
            chunk = ser.read(1024)
            if chunk:
                buf.extend(chunk)
        ser.close()

        n = len(buf)
        if n == 0:
            print(f"  {baud:>7}: SILENT (0 bytes in 2s)")
            continue
        crsf = scan_crsf(bytes(buf))
        printable = sum(1 for b in buf if 0x20 <= b < 0x7F or b in (0x0A, 0x0D, 0x09))
        printable_pct = 100 * printable / n
        if crsf:
            named = ", ".join(
                f"{CRSF_TYPE_NAMES.get(t, f'0x{t:02X}')}x{c}"
                for t, c in sorted(crsf.items(), key=lambda kv: -kv[1])
            )
            print(f"  {baud:>7}: {n} bytes | CRSF FRAMES: {named}")
        elif printable_pct > 85:
            sample = bytes(buf[:120]).decode("ascii", errors="replace").replace("\r", " ")
            print(f"  {baud:>7}: {n} bytes | TEXT ({printable_pct:.0f}% printable): {sample!r}")
        else:
            print(f"  {baud:>7}: {n} bytes | binary/unknown, "
                  f"{printable_pct:.0f}% printable, first16={bytes(buf[:16]).hex()}")


if __name__ == "__main__":
    main()
