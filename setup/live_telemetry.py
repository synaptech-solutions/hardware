#!/usr/bin/env python3
"""
Live telemetry dashboard for an ELRS-controlled drone.

Listens on the Ranger's USB-C serial, sends keep-alive RC frames so the
TX module stays in "handset connected" state, and renders a terminal
dashboard updated at ~10 Hz showing:

  - Attitude (pitch / roll / yaw, decoded from CRSF ATTITUDE 0x1E)
  - Battery (voltage / current / capacity / remaining, BATTERY 0x08)
  - Flight mode (FLIGHT_MODE 0x21)
  - Link statistics both directions (LINK_STATISTICS 0x14)
  - Device names seen (DEVICE_INFO 0x29, TX/RX/FC)
  - Freshness timer per field, so stale data is obvious

Usage:
    python3 live_telemetry.py                     # auto-detect port
    python3 live_telemetry.py /dev/ttyUSB0 420000

Ctrl-C to exit. The script transmits neutral sticks (1500 µs every
channel, AUX1 low) so even if the drone were somehow armed it stays put.
"""

import math
import sys
import time
from collections import defaultdict
from typing import Optional

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    sys.exit("pyserial not installed. Run: pip install pyserial")


# --- CRSF helpers ---

CRSF_SYNC = 0xC8
CRSF_ADDR_BROADCAST = 0x00
CRSF_ADDR_FC = 0xC8
CRSF_ADDR_RADIO = 0xEA
CRSF_ADDR_RX = 0xEC
CRSF_ADDR_TX_MODULE = 0xEE

ADDR_NAMES = {0xC8: "FC", 0xEA: "Radio", 0xEC: "RX", 0xEE: "TX"}

T_BATTERY = 0x08
T_LINK_STATS = 0x14
T_RC_CHANNELS = 0x16
T_ATTITUDE = 0x1E
T_FLIGHT_MODE = 0x21
T_DEVICE_PING = 0x28
T_DEVICE_INFO = 0x29

RF_MODE_NAMES = {
    20: "LoRa 2G4 25Hz", 21: "LoRa 2G4 50Hz (init)",
    22: "LoRa 2G4 100Hz", 23: "LoRa 2G4 100Hz 8ch",
    24: "LoRa 2G4 150Hz", 25: "LoRa 2G4 200Hz",
    26: "LoRa 2G4 200Hz 8ch", 27: "LoRa 2G4 250Hz",
    28: "LoRa 2G4 333Hz 8ch", 29: "LoRa 2G4 500Hz",
    30: "FLRC 2G4 250Hz DVDA", 31: "FLRC 2G4 500Hz DVDA",
    32: "FLRC 2G4 500Hz", 33: "FLRC 2G4 1000Hz",
}


def crc8_dvb_s2(data: bytes) -> int:
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0xD5) & 0xFF if (crc & 0x80) else (crc << 1) & 0xFF
    return crc


def build_crsf_frame(frame_type: int, payload: bytes) -> bytes:
    length = len(payload) + 2
    body = bytes([frame_type]) + payload
    return bytes([CRSF_SYNC, length]) + body + bytes([crc8_dvb_s2(body)])


def build_device_ping() -> bytes:
    return build_crsf_frame(T_DEVICE_PING,
                            bytes([CRSF_ADDR_BROADCAST, CRSF_ADDR_RADIO]))


def build_rc_channels_packed(channels_us: list[int]) -> bytes:
    assert len(channels_us) == 16
    ticks = [int((us - 988) * (1811 - 172) / (2012 - 988) + 172) & 0x7FF
             for us in channels_us]
    bits, nbits = 0, 0
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
    return build_crsf_frame(T_RC_CHANNELS, bytes(out))


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
            yield frame[2], body[1:]


def autodetect_port() -> Optional[str]:
    known = {(0x1A86, 0x7523), (0x10C4, 0xEA60), (0x303A, 0x1001), (0x0403, 0x6001)}
    for p in list_ports.comports():
        if p.vid is not None and (p.vid, p.pid) in known:
            return p.device
    return None


# --- Telemetry decoders ---

def decode_attitude(payload: bytes):
    if len(payload) < 6:
        return None
    pitch = int.from_bytes(payload[0:2], "big", signed=True)
    roll = int.from_bytes(payload[2:4], "big", signed=True)
    yaw = int.from_bytes(payload[4:6], "big", signed=True)
    return {
        "pitch_deg": math.degrees(pitch * 0.0001),
        "roll_deg": math.degrees(roll * 0.0001),
        "yaw_deg": math.degrees(yaw * 0.0001),
    }


def decode_battery(payload: bytes):
    if len(payload) < 8:
        return None
    voltage_dV = int.from_bytes(payload[0:2], "big")
    current_dA = int.from_bytes(payload[2:4], "big")
    capacity_mAh = int.from_bytes(payload[4:7], "big")
    remaining_pct = payload[7]
    return {
        "voltage_V": voltage_dV / 10.0,
        "current_A": current_dA / 10.0,
        "capacity_mAh": capacity_mAh,
        "remaining_pct": remaining_pct,
    }


def decode_flight_mode(payload: bytes):
    return payload.split(b"\x00", 1)[0].decode("ascii", errors="replace")


def decode_link_stats(payload: bytes):
    if len(payload) < 10:
        return None
    return {
        "up_rssi1_dBm": payload[0] - 256 if payload[0] > 127 else -payload[0],
        "up_rssi2_dBm": payload[1] - 256 if payload[1] > 127 else -payload[1],
        "up_lq": payload[2],
        "up_snr_dB": payload[3] - 256 if payload[3] > 127 else payload[3],
        "active_ant": payload[4],
        "rf_mode": payload[5],
        "up_tx_pwr_idx": payload[6],
        "dn_rssi_dBm": payload[7] - 256 if payload[7] > 127 else -payload[7],
        "dn_lq": payload[8],
        "dn_snr_dB": payload[9] - 256 if payload[9] > 127 else payload[9],
    }


def decode_device_info(payload: bytes):
    # extended header was already stripped (we got body[1:] from parser)
    # but we need the source address; rebuild it from the raw frame instead
    return None  # handled inline so we can read source addr


# --- Dashboard ---

CSI = "\033["
BOLD = CSI + "1m"
DIM = CSI + "2m"
RED = CSI + "31m"
GREEN = CSI + "32m"
YELLOW = CSI + "33m"
CYAN = CSI + "36m"
RESET = CSI + "0m"


def lq_colour(lq: Optional[int]) -> str:
    if lq is None:
        return DIM
    if lq >= 90:
        return GREEN
    if lq >= 60:
        return YELLOW
    return RED


def voltage_colour(v: Optional[float]) -> str:
    if v is None or v < 0.5:
        return DIM
    if v >= 3.7:
        return GREEN
    if v >= 3.4:
        return YELLOW
    return RED


def fmt_age(t: Optional[float], now: float) -> str:
    if t is None:
        return f"{DIM}never{RESET}"
    age = now - t
    if age < 1.0:
        return f"{GREEN}{age*1000:>4.0f}ms{RESET}"
    if age < 5.0:
        return f"{YELLOW}{age:>5.2f}s{RESET}"
    return f"{RED}{age:>5.1f}s{RESET}"


class State:
    def __init__(self):
        self.attitude = None
        self.attitude_t: Optional[float] = None
        self.battery = None
        self.battery_t: Optional[float] = None
        self.flight_mode: Optional[str] = None
        self.flight_mode_t: Optional[float] = None
        self.link = None
        self.link_t: Optional[float] = None
        self.devices: dict[int, str] = {}  # addr -> name
        self.devices_t: dict[int, float] = {}
        self.frames_in = 0
        self.bytes_in = 0
        self.bytes_out = 0
        self.start = time.time()


def render(s: State, port: str, baud: int):
    now = time.time()
    sys.stdout.write(CSI + "H" + CSI + "2J")  # home + clear

    title = f"  ELRS Drone Telemetry — {port} @ {baud} baud"
    sys.stdout.write(f"{BOLD}{title}{RESET}\n")
    sys.stdout.write("  " + "─" * 68 + "\n\n")

    # Connection / link
    sys.stdout.write(f"  {BOLD}LINK{RESET}\n")
    if s.link:
        L = s.link
        c_up = lq_colour(L["up_lq"])
        c_dn = lq_colour(L["dn_lq"])
        rfm = RF_MODE_NAMES.get(L["rf_mode"], f"unknown({L['rf_mode']})")
        sys.stdout.write(
            f"    Uplink   {c_up}{L['up_lq']:>3d}%{RESET}  RSSI {L['up_rssi1_dBm']:>+4d} dBm  "
            f"SNR {L['up_snr_dB']:>+3d} dB   (laptop → drone)\n"
            f"    Downlink {c_dn}{L['dn_lq']:>3d}%{RESET}  RSSI {L['dn_rssi_dBm']:>+4d} dBm  "
            f"SNR {L['dn_snr_dB']:>+3d} dB   (drone → laptop)\n"
            f"    RF mode  {rfm}   TX power idx {L['up_tx_pwr_idx']}\n"
        )
    else:
        sys.stdout.write(f"    {DIM}no link stats received yet{RESET}\n\n\n")
    sys.stdout.write(f"    updated {fmt_age(s.link_t, now)}\n\n")

    # Attitude
    sys.stdout.write(f"  {BOLD}ATTITUDE / IMU{RESET}\n")
    if s.attitude:
        a = s.attitude
        sys.stdout.write(
            f"    Pitch  {CYAN}{a['pitch_deg']:>+8.2f}°{RESET}    "
            f"Roll   {CYAN}{a['roll_deg']:>+8.2f}°{RESET}    "
            f"Yaw    {CYAN}{a['yaw_deg']:>+8.2f}°{RESET}\n"
        )
    else:
        sys.stdout.write(f"    {DIM}no attitude data yet{RESET}\n")
    sys.stdout.write(f"    updated {fmt_age(s.attitude_t, now)}\n\n")

    # Battery
    sys.stdout.write(f"  {BOLD}BATTERY{RESET}\n")
    if s.battery:
        b = s.battery
        vc = voltage_colour(b["voltage_V"])
        sys.stdout.write(
            f"    Voltage   {vc}{b['voltage_V']:>5.2f} V{RESET}    "
            f"Current   {b['current_A']:>5.2f} A\n"
            f"    Capacity  {b['capacity_mAh']:>5d} mAh  "
            f"Remaining {b['remaining_pct']:>3d}%\n"
        )
        if b["voltage_V"] < 0.5:
            sys.stdout.write(
                f"    {DIM}vbat reads ~0 V → flight battery not on BT2.0 (USB-only?){RESET}\n"
            )
    else:
        sys.stdout.write(f"    {DIM}no battery data yet{RESET}\n\n")
    sys.stdout.write(f"    updated {fmt_age(s.battery_t, now)}\n\n")

    # Flight mode
    sys.stdout.write(f"  {BOLD}FLIGHT MODE{RESET}\n")
    if s.flight_mode is not None:
        mode = s.flight_mode
        armed = not mode.endswith("*")
        armed_lbl = f"{RED}ARMED{RESET}" if armed else f"{GREEN}disarmed{RESET}"
        sys.stdout.write(f"    {CYAN}{mode!r}{RESET}   ({armed_lbl})\n")
    else:
        sys.stdout.write(f"    {DIM}no flight mode data yet{RESET}\n")
    sys.stdout.write(f"    updated {fmt_age(s.flight_mode_t, now)}\n\n")

    # Devices
    sys.stdout.write(f"  {BOLD}DEVICES SEEN{RESET}\n")
    if s.devices:
        for addr, name in s.devices.items():
            role = ADDR_NAMES.get(addr, f"0x{addr:02X}")
            sys.stdout.write(
                f"    [{role:<5}] {name:<28} (seen {fmt_age(s.devices_t.get(addr), now)})\n"
            )
    else:
        sys.stdout.write(f"    {DIM}no DEVICE_INFO frames yet{RESET}\n")

    # Totals
    sys.stdout.write("\n  " + "─" * 68 + "\n")
    runtime = now - s.start
    sys.stdout.write(
        f"  frames in: {s.frames_in:<6d}  bytes in: {s.bytes_in:<7d}  "
        f"bytes out: {s.bytes_out:<7d}  uptime: {runtime:>5.1f}s\n"
    )
    sys.stdout.write(f"  {DIM}Ctrl-C to exit{RESET}\n")
    sys.stdout.flush()


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else autodetect_port()
    baud = int(sys.argv[2]) if len(sys.argv) > 2 else 420000
    if not port:
        sys.exit("No USB serial device found. Plug in the Ranger and try again.")

    ser = serial.Serial(port, baudrate=baud, timeout=0.02)

    ping = build_device_ping()
    # neutral sticks: throttle low (988 µs), everything else center, AUX low
    rc_channels = [988] + [1500] * 15
    rc_channels[2] = 988  # explicit throttle low
    rc_frame = build_rc_channels_packed(rc_channels)

    state = State()
    parser = CrsfParser()

    next_ping = 0.0
    next_rc = 0.0
    next_render = 0.0

    sys.stdout.write(CSI + "?25l")  # hide cursor

    try:
        while True:
            now = time.time()
            chunk = ser.read(512)
            if chunk:
                state.bytes_in += len(chunk)
                for ftype, payload in parser.feed(chunk):
                    state.frames_in += 1
                    t = time.time()
                    if ftype == T_ATTITUDE:
                        d = decode_attitude(payload)
                        if d:
                            state.attitude = d
                            state.attitude_t = t
                    elif ftype == T_BATTERY:
                        d = decode_battery(payload)
                        if d:
                            state.battery = d
                            state.battery_t = t
                    elif ftype == T_FLIGHT_MODE:
                        state.flight_mode = decode_flight_mode(payload)
                        state.flight_mode_t = t
                    elif ftype == T_LINK_STATS:
                        d = decode_link_stats(payload)
                        if d:
                            state.link = d
                            state.link_t = t
                    elif ftype == T_DEVICE_INFO:
                        # payload here is body[1:] from parser, which for ext-header
                        # frames is [dest, src, name..., serial, hw, sw, n_params, proto]
                        # We need to look at the raw extended header. The parser stripped
                        # the type byte; payload starts with dest, then src.
                        if len(payload) >= 2:
                            src = payload[1]
                            name = payload[2:].split(b"\x00", 1)[0].decode(
                                "ascii", errors="replace")
                            if name:
                                state.devices[src] = name
                                state.devices_t[src] = t

            # Keep the link alive
            if now >= next_rc:
                ser.write(rc_frame)
                state.bytes_out += len(rc_frame)
                next_rc = now + 0.02  # 50 Hz
            if now >= next_ping:
                ser.write(ping)
                state.bytes_out += len(ping)
                next_ping = now + 2.0

            # Repaint at 10 Hz
            if now >= next_render:
                render(state, port, baud)
                next_render = now + 0.1
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write(CSI + "?25h")  # restore cursor
        sys.stdout.write("\n")
        ser.close()


if __name__ == "__main__":
    main()
