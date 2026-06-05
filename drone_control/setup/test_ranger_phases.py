#!/usr/bin/env python3
"""
Phase test: does the Ranger CHANGE STATE when we send CRSF RC frames?

Per ExpressLRS docs, the TX module does not transmit OTA without a CRSF
signal from a "handset". So if the Ranger is acting on the RC frames we
send (vs. parsing-and-discarding), we should observe a state change in
the telemetry it emits.

Method: cycle through three phases, count incoming CRSF frames per type
in each, plus inter-arrival times. Compare. If absolutely nothing
differs between RC-on and RC-off phases, the Ranger is ignoring our RC
input. If anything changes, it's acting on it.

  Phase A (5 s): no RC, only DEVICE_PING every 1 s
  Phase B (5 s): RC_CHANNELS_PACKED @ 50 Hz + DEVICE_PING every 1 s
  Phase C (5 s): no RC again (sanity — should match phase A)

Run after powering the Ranger via XT30 (2-3S) so RF stage is fully
initialised. Then:
    python3 test_ranger_phases.py
"""

import sys
import time
import threading
import statistics
from collections import defaultdict, deque
from typing import Optional

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    sys.exit("pyserial not installed. Run: pip install pyserial")

# --- CRSF helpers (copied from test_ranger_usb.py, kept inline for portability) ---

CRSF_SYNC = 0xC8
CRSF_ADDR_BROADCAST = 0x00
CRSF_ADDR_RADIO = 0xEA
CRSF_ADDR_TX_MODULE = 0xEE

CRSF_FRAMETYPE_LINK_STATISTICS = 0x14
CRSF_FRAMETYPE_RC_CHANNELS_PACKED = 0x16
CRSF_FRAMETYPE_DEVICE_PING = 0x28
CRSF_FRAMETYPE_DEVICE_INFO = 0x29

FRAMETYPE_NAMES = {
    0x02: "GPS", 0x08: "BATTERY", 0x14: "LINK_STATS",
    0x16: "RC_CHANNELS", 0x1E: "ATTITUDE", 0x21: "FLIGHT_MODE",
    0x28: "DEVICE_PING", 0x29: "DEVICE_INFO",
    0x2B: "PARAM_ENTRY", 0x2C: "PARAM_READ", 0x2D: "PARAM_WRITE",
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
    length = len(payload) + 2
    body = bytes([frame_type]) + payload
    return bytes([CRSF_SYNC, length]) + body + bytes([crc8_dvb_s2(body)])


def build_device_ping() -> bytes:
    return build_crsf_frame(CRSF_FRAMETYPE_DEVICE_PING,
                            bytes([CRSF_ADDR_BROADCAST, CRSF_ADDR_RADIO]))


def build_rc_channels_packed(channels_us: list[int]) -> bytes:
    assert len(channels_us) == 16
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
    known = {(0x1A86, 0x7523), (0x10C4, 0xEA60), (0x303A, 0x1001), (0x0403, 0x6001)}
    for p in list_ports.comports():
        if p.vid is not None and (p.vid, p.pid) in known:
            return p.device
    return None


class CrsfParser:
    def __init__(self):
        self.buf = bytearray()

    def feed(self, data: bytes):
        self.buf.extend(data)
        while True:
            while self.buf and self.buf[0] not in (CRSF_SYNC, 0xEE, 0xEA):
                self.buf.pop(0)
            if len(self.buf) < 2:
                return
            length = self.buf[1]
            if length < 2 or length > 62:
                self.buf.pop(0)
                continue
            total = length + 2
            if len(self.buf) < total:
                return
            frame = bytes(self.buf[:total])
            del self.buf[:total]
            body = frame[2:-1]
            if crc8_dvb_s2(body) != frame[-1]:
                continue
            yield frame[2], body[1:]  # frame_type, payload


# --- Phase test ---

class PhaseStats:
    def __init__(self, name: str):
        self.name = name
        self.frame_counts: dict[int, int] = defaultdict(int)
        self.frame_arrivals: dict[int, list[float]] = defaultdict(list)
        self.bytes_in = 0
        self.bytes_out = 0
        self.unique_payloads: dict[int, set] = defaultdict(set)
        self.last_link_stats_payload: Optional[bytes] = None
        self.start: float = 0.0
        self.end: float = 0.0

    def record(self, frame_type: int, payload: bytes, t: float):
        self.frame_counts[frame_type] += 1
        self.frame_arrivals[frame_type].append(t)
        self.unique_payloads[frame_type].add(payload)
        if frame_type == CRSF_FRAMETYPE_LINK_STATISTICS:
            self.last_link_stats_payload = payload

    def duration(self) -> float:
        return self.end - self.start

    def hz(self, frame_type: int) -> float:
        d = self.duration()
        return self.frame_counts[frame_type] / d if d > 0 else 0.0

    def jitter_ms(self, frame_type: int) -> Optional[float]:
        arrs = self.frame_arrivals[frame_type]
        if len(arrs) < 3:
            return None
        intervals = [(arrs[i+1] - arrs[i]) * 1000 for i in range(len(arrs)-1)]
        return statistics.stdev(intervals)


def decode_link_stats(payload: bytes) -> dict:
    if len(payload) < 10:
        return {}
    return {
        "uplink_rssi_ant1": payload[0],
        "uplink_rssi_ant2": payload[1],
        "uplink_lq": payload[2],
        "uplink_snr": payload[3] if payload[3] < 128 else payload[3] - 256,
        "active_antenna": payload[4],
        "rf_mode": payload[5],
        "uplink_tx_power": payload[6],
        "downlink_rssi": payload[7],
        "downlink_lq": payload[8],
        "downlink_snr": payload[9] if payload[9] < 128 else payload[9] - 256,
    }


def run_phase(ser: serial.Serial, name: str, duration_s: float,
              send_rc: bool, rc_frame: bytes, ping_frame: bytes,
              stop_event: threading.Event) -> PhaseStats:
    stats = PhaseStats(name)
    stats.start = time.time()
    deadline = stats.start + duration_s
    next_ping = stats.start
    next_rc = stats.start

    print(f"\n>>> {name}: {'SENDING RC' if send_rc else 'NO RC'} for {duration_s}s")
    parser = CrsfParser()

    while time.time() < deadline and not stop_event.is_set():
        now = time.time()
        # Read what's available
        chunk = ser.read(256)
        if chunk:
            stats.bytes_in += len(chunk)
            for ft, pl in parser.feed(chunk):
                stats.record(ft, pl, time.time())
        # Send
        if now >= next_ping:
            ser.write(ping_frame)
            stats.bytes_out += len(ping_frame)
            next_ping = now + 1.0
        if send_rc and now >= next_rc:
            ser.write(rc_frame)
            stats.bytes_out += len(rc_frame)
            next_rc = now + 0.02  # 50 Hz

    stats.end = time.time()
    return stats


def summarise(stats: PhaseStats):
    print(f"\n  Phase {stats.name} ({stats.duration():.2f}s):")
    print(f"    bytes out: {stats.bytes_out}, bytes in: {stats.bytes_in}")
    for ft, count in sorted(stats.frame_counts.items()):
        name = FRAMETYPE_NAMES.get(ft, f"0x{ft:02X}")
        n_unique = len(stats.unique_payloads[ft])
        jit = stats.jitter_ms(ft)
        jit_s = f", jitter={jit:.1f}ms" if jit is not None else ""
        print(f"    {name:<14} count={count:<4} hz={stats.hz(ft):>5.2f} "
              f"unique_payloads={n_unique}{jit_s}")
    if stats.last_link_stats_payload:
        decoded = decode_link_stats(stats.last_link_stats_payload)
        print(f"    last LINK_STATS: {decoded}")


def compare(a: PhaseStats, b: PhaseStats):
    print(f"\n=== Diff {a.name} vs {b.name} ===")
    all_types = set(a.frame_counts) | set(b.frame_counts)
    differs = False
    for ft in sorted(all_types):
        name = FRAMETYPE_NAMES.get(ft, f"0x{ft:02X}")
        ha, hb = a.hz(ft), b.hz(ft)
        ua = len(a.unique_payloads[ft])
        ub = len(b.unique_payloads[ft])
        flag = ""
        if abs(ha - hb) > 0.5 or ua != ub:
            flag = "  <-- CHANGED"
            differs = True
        print(f"    {name:<14} hz {ha:>5.2f} -> {hb:>5.2f}   "
              f"unique payloads {ua} -> {ub}{flag}")
    if not differs:
        print("    (no observable difference in CRSF telemetry — see notes)")


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else autodetect_port()
    baud = int(sys.argv[2]) if len(sys.argv) > 2 else 420000
    if not port:
        sys.exit("No USB serial device found")

    print(f"Opening {port} @ {baud} baud")
    ser = serial.Serial(port, baudrate=baud, timeout=0.05)

    ping = build_device_ping()
    rc = build_rc_channels_packed([1500] * 16)

    stop = threading.Event()
    try:
        a = run_phase(ser, "A_no_RC", 5.0, False, rc, ping, stop)
        summarise(a)
        b = run_phase(ser, "B_RC_50Hz", 5.0, True, rc, ping, stop)
        summarise(b)
        c = run_phase(ser, "C_no_RC", 5.0, False, rc, ping, stop)
        summarise(c)
    finally:
        ser.close()

    compare(a, b)
    compare(b, c)

    # Known ELRS air-rate indices (from src/include/common.h)
    rf_mode_names = {
        20: "LoRa 2G4 25Hz", 21: "LoRa 2G4 50Hz (init)",
        22: "LoRa 2G4 100Hz", 24: "LoRa 2G4 150Hz",
        25: "LoRa 2G4 200Hz", 27: "LoRa 2G4 250Hz",
        29: "LoRa 2G4 500Hz", 32: "FLRC 2G4 500Hz", 33: "FLRC 2G4 1000Hz",
    }

    print("\n=== Verdict ===")
    a_ls = decode_link_stats(a.last_link_stats_payload) if a.last_link_stats_payload else {}
    b_ls = decode_link_stats(b.last_link_stats_payload) if b.last_link_stats_payload else {}
    a_mode = a_ls.get("rf_mode")
    b_mode = b_ls.get("rf_mode")
    a_name = rf_mode_names.get(a_mode, f"unknown({a_mode})")
    b_name = rf_mode_names.get(b_mode, f"unknown({b_mode})")

    if a_mode is not None and b_mode is not None and a_mode != b_mode:
        print(f"  rf_mode changed: {a_mode} ({a_name}) -> {b_mode} ({b_name})")
        print("  → The Ranger transitioned RF state in response to your CRSF input.")
        print("  → It is acting on your RC frames, not silently discarding them.")
    elif any(a.unique_payloads[ft] != b.unique_payloads[ft]
             for ft in set(a.frame_counts) | set(b.frame_counts)):
        print("  Payload content changed between phases (not just frame rate).")
        print("  → The Ranger is reacting to your CRSF input.")
    else:
        print("  No detectable state change in CRSF telemetry.")
        print("  Could still be acting on RC internally. Verify via OLED, bind, or SDR.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted")
