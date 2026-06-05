#!/usr/bin/env python3
"""
Bidirectional USB-serial test for a RadioMaster Ranger ELRS TX module.

What this verifies:
  1. The laptop sees the Ranger as a USB serial device.
  2. We can OPEN the port at the ELRS-standard baud rate (420000).
  3. We can WRITE bytes to it (a CRSF DEVICE_PING) without error.
  4. We can READ bytes back, and decode any CRSF frames present.

What this does NOT prove:
  - That the Ranger's USB-C accepts CRSF as RC-channel input and will
    transmit those channels over RF. Per ELRS docs the USB-C port is
    primarily for firmware flashing / configuration / MAVLink mode.
    Whether stick-channel CRSF over USB-C drives the RF link depends on
    firmware mode and is the actual unknown this script helps explore.

Usage:
    pip install pyserial
    python3 test_ranger_usb.py                 # auto-detect port
    python3 test_ranger_usb.py /dev/ttyUSB0    # explicit port
    python3 test_ranger_usb.py /dev/ttyUSB0 460800   # explicit baud

Power the Ranger via its XT30 (2-3S LiPo) so it is fully booted, not
only USB-powered, otherwise the RF stage may not initialise.
"""

import sys
import time
import threading
import glob
from typing import Optional

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    sys.exit("pyserial not installed. Run: pip install pyserial")


CRSF_SYNC = 0xC8
CRSF_ADDR_BROADCAST = 0x00
CRSF_ADDR_FC = 0xC8
CRSF_ADDR_RADIO = 0xEA
CRSF_ADDR_TX_MODULE = 0xEE
CRSF_ADDR_RX = 0xEC

CRSF_FRAMETYPE_GPS = 0x02
CRSF_FRAMETYPE_BATTERY = 0x08
CRSF_FRAMETYPE_LINK_STATISTICS = 0x14
CRSF_FRAMETYPE_RC_CHANNELS_PACKED = 0x16
CRSF_FRAMETYPE_ATTITUDE = 0x1E
CRSF_FRAMETYPE_FLIGHT_MODE = 0x21
CRSF_FRAMETYPE_DEVICE_PING = 0x28
CRSF_FRAMETYPE_DEVICE_INFO = 0x29
CRSF_FRAMETYPE_PARAMETER_SETTINGS_ENTRY = 0x2B
CRSF_FRAMETYPE_PARAMETER_READ = 0x2C
CRSF_FRAMETYPE_PARAMETER_WRITE = 0x2D
CRSF_FRAMETYPE_COMMAND = 0x32

FRAMETYPE_NAMES = {
    0x02: "GPS",
    0x08: "BATTERY",
    0x14: "LINK_STATS",
    0x16: "RC_CHANNELS",
    0x1E: "ATTITUDE",
    0x21: "FLIGHT_MODE",
    0x28: "DEVICE_PING",
    0x29: "DEVICE_INFO",
    0x2B: "PARAM_ENTRY",
    0x2C: "PARAM_READ",
    0x2D: "PARAM_WRITE",
    0x32: "COMMAND",
}


def crc8_dvb_s2(data: bytes, poly: int = 0xD5) -> int:
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFF if (crc & 0x80) else (crc << 1) & 0xFF
    return crc


def build_crsf_frame(frame_type: int, payload: bytes) -> bytes:
    # length = type(1) + payload + crc(1)
    length = len(payload) + 2
    body = bytes([frame_type]) + payload
    crc = crc8_dvb_s2(body)
    return bytes([CRSF_SYNC, length]) + body + bytes([crc])


def build_device_ping() -> bytes:
    # Extended-header ping: dest=broadcast, src=radio handset
    payload = bytes([CRSF_ADDR_BROADCAST, CRSF_ADDR_RADIO])
    return build_crsf_frame(CRSF_FRAMETYPE_DEVICE_PING, payload)


def build_rc_channels_packed(channels_us: list[int]) -> bytes:
    """Pack 16 channels (values 988-2012 us, center 1500) into CRSF RC frame.

    The Ranger may or may not act on this — that's part of what we are
    testing. Channel mapping (AETR by default): 0=A, 1=E, 2=T, 3=R.
    """
    assert len(channels_us) == 16
    # Convert microseconds to CRSF channel ticks: us 988->172, 2012->1811
    ticks = [int((us - 988) * (1811 - 172) / (2012 - 988) + 172) & 0x7FF
             for us in channels_us]
    bits = 0
    nbits = 0
    out = bytearray()
    for t in ticks:
        bits |= (t & 0x7FF) << nbits
        nbits += 11
        while nbits >= 8:
            out.append(bits & 0xFF)
            bits >>= 8
            nbits -= 8
    if nbits > 0:
        out.append(bits & 0xFF)
    return build_crsf_frame(CRSF_FRAMETYPE_RC_CHANNELS_PACKED, bytes(out))


def autodetect_port() -> Optional[str]:
    """Look for a USB-serial device that matches typical ELRS TX modules."""
    candidates = []
    for p in list_ports.comports():
        desc = f"{p.device} | vid:pid={p.vid:04x}:{p.pid:04x} | {p.description} | {p.manufacturer}" \
            if p.vid is not None else f"{p.device} | {p.description}"
        candidates.append((p, desc))
    if not candidates:
        return None

    print("Detected serial ports:")
    for _, desc in candidates:
        print(f"  {desc}")

    # Common USB-UART bridges used on ELRS TX modules
    known_vidpid = {
        (0x1A86, 0x7523),  # CH340
        (0x10C4, 0xEA60),  # CP210x
        (0x303A, 0x1001),  # Espressif native USB CDC
        (0x0403, 0x6001),  # FTDI
    }
    for p, _ in candidates:
        if p.vid is not None and (p.vid, p.pid) in known_vidpid:
            return p.device

    # Fallback: any ttyUSB* / ttyACM*
    for pattern in ("/dev/ttyUSB*", "/dev/ttyACM*"):
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]
    return None


class CrsfParser:
    """Stateful byte-stream → CRSF frame parser."""

    def __init__(self):
        self.buf = bytearray()

    def feed(self, data: bytes):
        self.buf.extend(data)
        while True:
            # Resync to a known sync byte
            while self.buf and self.buf[0] not in (CRSF_SYNC, 0xEE, 0xEA):
                self.buf.pop(0)
            if len(self.buf) < 2:
                return
            length = self.buf[1]
            if length < 2 or length > 62:
                self.buf.pop(0)
                continue
            total = length + 2  # sync + length + (length bytes)
            if len(self.buf) < total:
                return
            frame = bytes(self.buf[:total])
            del self.buf[:total]
            body = frame[2:-1]
            crc_got = frame[-1]
            crc_calc = crc8_dvb_s2(body)
            if crc_got != crc_calc:
                continue  # Bad frame, drop
            yield frame[0], frame[2], body[1:], frame.hex()


def reader_loop(ser: serial.Serial, stop: threading.Event, stats: dict):
    parser = CrsfParser()
    while not stop.is_set():
        try:
            chunk = ser.read(256)
        except serial.SerialException as e:
            print(f"[reader] serial error: {e}")
            return
        if not chunk:
            continue
        stats["bytes_in"] += len(chunk)
        for sync, ftype, payload, hex_str in parser.feed(chunk):
            stats["frames_in"] += 1
            name = FRAMETYPE_NAMES.get(ftype, f"0x{ftype:02X}")
            print(f"[recv] CRSF sync=0x{sync:02X} type={name} "
                  f"payload={payload.hex()} raw={hex_str}")


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else autodetect_port()
    baud = int(sys.argv[2]) if len(sys.argv) > 2 else 420000

    if not port:
        sys.exit("FAIL: no USB serial device found. Is the Ranger plugged in "
                 "and powered? Try: ls /dev/ttyUSB* /dev/ttyACM* ; dmesg | tail")

    print(f"\n--- Test 1: device detection ---")
    print(f"PASS: using port {port} at {baud} baud")

    print(f"\n--- Test 2: open port ---")
    try:
        ser = serial.Serial(port, baudrate=baud, timeout=0.1)
    except (serial.SerialException, PermissionError) as e:
        sys.exit(f"FAIL: cannot open {port}: {e}\n"
                 f"On Linux you likely need: sudo usermod -aG dialout $USER "
                 f"(then log out/in), or run with sudo to test quickly.")
    print(f"PASS: opened {port}")

    stop = threading.Event()
    stats = {"bytes_in": 0, "frames_in": 0, "bytes_out": 0}
    t = threading.Thread(target=reader_loop, args=(ser, stop, stats), daemon=True)
    t.start()

    print(f"\n--- Test 3: write CRSF DEVICE_PING and RC channels for 10 s ---")
    ping = build_device_ping()
    print(f"   ping bytes: {ping.hex()}")
    center = [1500] * 16
    rc_frame = build_rc_channels_packed(center)
    print(f"   rc bytes  : {rc_frame.hex()} ({len(rc_frame)} bytes)")

    deadline = time.time() + 10
    next_ping = 0.0
    next_rc = 0.0
    while time.time() < deadline:
        now = time.time()
        if now >= next_ping:
            ser.write(ping)
            stats["bytes_out"] += len(ping)
            next_ping = now + 1.0
        if now >= next_rc:
            ser.write(rc_frame)
            stats["bytes_out"] += len(rc_frame)
            next_rc = now + 0.02  # 50 Hz
        time.sleep(0.005)

    stop.set()
    t.join(timeout=1.0)
    ser.close()

    print(f"\n--- Results ---")
    print(f"  bytes written to Ranger : {stats['bytes_out']}")
    print(f"  bytes read from Ranger  : {stats['bytes_in']}")
    print(f"  valid CRSF frames parsed: {stats['frames_in']}")
    if stats["bytes_in"] == 0:
        print("\n  RX = 0 bytes. The Ranger isn't talking on this port at "
              f"{baud} baud. Try another baud (460800 for MAVLink mode, "
              f"115200 for debug log), or the module may only speak on USB "
              f"during bootloader/flashing.")
    elif stats["frames_in"] == 0:
        print("\n  Got bytes but no valid CRSF frames. The port is alive but "
              f"the data isn't CRSF — could be ELRS debug text, MAVLink, or "
              f"a different baud. Re-run with a different baud.")
    else:
        print("\n  Bidirectional CRSF link to the Ranger over USB-C works.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted")
