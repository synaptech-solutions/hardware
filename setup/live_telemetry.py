#!/usr/bin/env python3
"""
Live telemetry dashboard for an ELRS-controlled drone.

Listens on the Ranger's USB-C serial, sends keep-alive RC frames so the
TX module stays in "handset connected" state, and renders a terminal
dashboard updated at ~10 Hz showing:

  - Attitude (pitch / roll / yaw, decoded from CRSF ATTITUDE 0x1E)
  - Battery (voltage / current / capacity / remaining, BATTERY 0x08)
  - Flight mode (FLIGHT_MODE 0x21)
    - Raw IMU via MSP_RAW_IMU (acc/gyro/mag, 9x int16)
  - Link statistics both directions (LINK_STATISTICS 0x14)
  - Device names seen (DEVICE_INFO 0x29, TX/RX/FC)
  - Freshness timer per field, so stale data is obvious

Usage:
    python3 live_telemetry.py                     # auto-detect port
    python3 live_telemetry.py /dev/ttyUSB0 420000
                                                # (MSP_RAW_IMU probe on Ranger link)
    python3 live_telemetry.py /dev/ttyUSB0 420000 /dev/ttyACM0 115200

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
T_MSP_REQ = 0x7A
T_MSP_RESP = 0x7B

RF_MODE_NAMES = {
    20: "LoRa 2G4 25Hz", 21: "LoRa 2G4 50Hz (init)",
    22: "LoRa 2G4 100Hz", 23: "LoRa 2G4 100Hz 8ch",
    24: "LoRa 2G4 150Hz", 25: "LoRa 2G4 200Hz",
    26: "LoRa 2G4 200Hz 8ch", 27: "LoRa 2G4 250Hz",
    28: "LoRa 2G4 333Hz 8ch", 29: "LoRa 2G4 500Hz",
    30: "FLRC 2G4 250Hz DVDA", 31: "FLRC 2G4 500Hz DVDA",
    32: "FLRC 2G4 500Hz", 33: "FLRC 2G4 1000Hz",
}

# --- MSP (Betaflight) helpers ---
MSP_RAW_IMU_CMD = 102  # MSP_RAW_IMU: 9x int16 (acc xyz, gyro xyz, mag xyz)

# MSP_RAW_IMU scaling (common Betaflight MPU defaults).
# For many BF targets, accelerometer raw units are ~2048 LSB per 1 g.
ACC_LSB_PER_G = 2048.0
GYRO_LSB_PER_DPS = 16.4
G_MPS2 = 9.80665


def build_msp_request(cmd: int) -> bytes:
    size = 0
    chk = size ^ (cmd & 0xFF)
    return bytes([0x24, 0x4D, 0x3C, size, cmd & 0xFF, chk & 0xFF])


def build_crsf_msp_v2_request(cmd: int, payload: bytes = b"") -> bytes:
        """Build CRSF MSP_REQ (0x7A) carrying one MSPv2 request chunk.

        Encapsulated MSP payload format:
            <status><flags><fn_lo><fn_hi><len_lo><len_hi><payload...><msp_crc>
        where status 0x50 means MSPv2 + new frame + seq 0 + no error.
        """
        payload_len = len(payload)
        msp = bytearray()
        msp.append(0x50)  # status: v2 + start + seq0
        msp.append(0x00)  # flags
        msp.append(cmd & 0xFF)
        msp.append((cmd >> 8) & 0xFF)
        msp.append(payload_len & 0xFF)
        msp.append((payload_len >> 8) & 0xFF)
        msp.extend(payload)
        # MSP-over-CRSF checksum is over flags+function+length+payload (status excluded).
        msp.append(crc8_dvb_s2(bytes(msp[1:])))
        ext_payload = bytes([CRSF_ADDR_FC, CRSF_ADDR_RADIO]) + bytes(msp)
        return build_crsf_frame(T_MSP_REQ, ext_payload)


class MspParser:
    """MSP v1 response parser ($M> ...)."""
    def __init__(self):
        self.buf = bytearray()

    def feed(self, data: bytes):
        self.buf.extend(data)
        while True:
            i = self.buf.find(b"$M>")
            if i < 0:
                if len(self.buf) > 2048:
                    del self.buf[:-128]
                return
            if i > 0:
                del self.buf[:i]
            if len(self.buf) < 6:
                return
            size = self.buf[3]
            total = 6 + size
            if len(self.buf) < total:
                return
            cmd = self.buf[4]
            payload = bytes(self.buf[5:5 + size])
            chk = 0
            for b in self.buf[3:5 + size]:
                chk ^= b
            frame_chk = self.buf[5 + size]
            del self.buf[:total]
            if chk != frame_chk:
                continue
            yield cmd, payload


class CrsfMspParser:
    """Reassembles MSP-over-CRSF chunks (0x7A/0x7B) into complete MSP frames."""
    def __init__(self):
        self.buf = bytearray()
        self.expected = None
        self.version = None
        self.cmd = None
        self.last_seq = None

    def reset(self):
        self.buf.clear()
        self.expected = None
        self.version = None
        self.cmd = None
        self.last_seq = None

    def feed_chunk(self, payload: bytes):
        # payload: <dest><orig><status><msp_body_chunk...>
        if len(payload) < 3:
            return
        status = payload[2]
        chunk = payload[3:]
        seq = status & 0x0F
        start = bool(status & 0x10)
        version = (status >> 5) & 0x03
        error = bool(status & 0x80)
        if error:
            self.reset()
            return

        if start:
            self.buf = bytearray()
            self.expected = None
            self.version = version
            self.cmd = None
            self.last_seq = seq
        else:
            if self.version is None:
                return
            if version != self.version:
                self.reset()
                return
            if self.last_seq is None or seq != ((self.last_seq + 1) & 0x0F):
                self.reset()
                return
            self.last_seq = seq

        self.buf.extend(chunk)

        if self.expected is None:
            if self.version == 2 and len(self.buf) >= 5:
                self.cmd = self.buf[1] | (self.buf[2] << 8)
                data_len = self.buf[3] | (self.buf[4] << 8)
                self.expected = 5 + data_len
            elif self.version == 1 and len(self.buf) >= 2:
                self.cmd = self.buf[1]
                data_len = self.buf[0]
                self.expected = 2 + data_len

        if self.expected is None or len(self.buf) < self.expected:
            return

        frame = bytes(self.buf[:self.expected])
        if self.version == 2:
            data_len = frame[3] | (frame[4] << 8)
            out_payload = frame[5:5 + data_len]
        elif self.version == 1:
            data_len = frame[0]
            out_payload = frame[2:2 + data_len]
        else:
            self.reset()
            return

        out_cmd = self.cmd
        self.reset()
        yield out_cmd, out_payload


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


def decode_msp_raw_imu(payload: bytes):
    if len(payload) < 18:
        return None
    vals = [int.from_bytes(payload[i:i + 2], "little", signed=True)
            for i in range(0, 18, 2)]
    return {
        "ax": vals[0], "ay": vals[1], "az": vals[2],
        "gx": vals[3], "gy": vals[4], "gz": vals[5],
        "mx": vals[6], "my": vals[7], "mz": vals[8],
    }


def convert_msp_raw_imu_units(raw_imu: dict):
    ax_g = raw_imu["ax"] / ACC_LSB_PER_G
    ay_g = raw_imu["ay"] / ACC_LSB_PER_G
    az_g = raw_imu["az"] / ACC_LSB_PER_G

    gx_dps = raw_imu["gx"] / GYRO_LSB_PER_DPS
    gy_dps = raw_imu["gy"] / GYRO_LSB_PER_DPS
    gz_dps = raw_imu["gz"] / GYRO_LSB_PER_DPS

    return {
        "ax_g": ax_g,
        "ay_g": ay_g,
        "az_g": az_g,
        "ax_mps2": ax_g * G_MPS2,
        "ay_mps2": ay_g * G_MPS2,
        "az_mps2": az_g * G_MPS2,
        "gx_dps": gx_dps,
        "gy_dps": gy_dps,
        "gz_dps": gz_dps,
        "gx_rads": math.radians(gx_dps),
        "gy_rads": math.radians(gy_dps),
        "gz_rads": math.radians(gz_dps),
        "mag_norm": math.sqrt(
            raw_imu["mx"] * raw_imu["mx"]
            + raw_imu["my"] * raw_imu["my"]
            + raw_imu["mz"] * raw_imu["mz"]
        ),
    }


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
        self.raw_imu = None
        self.raw_imu_t: Optional[float] = None
        self.raw_imu_rx_count = 0
        self.msp_status = "MSP idle"
        self.link = None
        self.link_t: Optional[float] = None
        self.devices: dict[int, str] = {}  # addr -> name
        self.devices_t: dict[int, float] = {}
        self.frames_in = 0
        self.bytes_in = 0
        self.bytes_out = 0
        self.start = time.time()



def _fmt_float(v: Optional[float], unit: str = "", precision: int = 2) -> str:
    if v is None:
        return "NA"
    return f"{v:.{precision}f}{unit}"


def _fmt_int(v: Optional[int], unit: str = "") -> str:
    if v is None:
        return "NA"
    return f"{v}{unit}"


def render(s: State, port: str, baud: int, msp_port: Optional[str], msp_baud: int):
    now = time.time()
    a = s.attitude or {}
    b = s.battery or {}
    l = s.link or {}
    i = s.raw_imu or {}
    msp_src = f"{msp_port}@{msp_baud}" if msp_port else f"Ranger link {port}@{baud}"

    print("\n" + "=" * 72)
    print(f"ELRS Live Telemetry | {port}@{baud} | uptime {now - s.start:.1f}s")
    print("-" * 72)
    print(
        "ATTITUDE  "
        f"pitch={_fmt_float(a.get('pitch_deg'), 'deg')}  "
        f"roll={_fmt_float(a.get('roll_deg'), 'deg')}  "
        f"yaw={_fmt_float(a.get('yaw_deg'), 'deg')}"
    )
    print(
        "BATTERY   "
        f"voltage={_fmt_float(b.get('voltage_V'), 'V')}  "
        f"current={_fmt_float(b.get('current_A'), 'A')}  "
        f"remaining={_fmt_int(b.get('remaining_pct'), '%')}"
    )
    print(
        "LINK      "
        f"up_lq={_fmt_int(l.get('up_lq'), '%')}  "
        f"dn_lq={_fmt_int(l.get('dn_lq'), '%')}  "
        f"up_rssi={_fmt_int(l.get('up_rssi1_dBm'), 'dBm')}  "
        f"dn_rssi={_fmt_int(l.get('dn_rssi_dBm'), 'dBm')}"
    )
    print(
        "RAW_IMU   "
        f"ax={_fmt_int(i.get('ax'))}  ay={_fmt_int(i.get('ay'))}  az={_fmt_int(i.get('az'))}  "
        f"gx={_fmt_int(i.get('gx'))}  gy={_fmt_int(i.get('gy'))}  gz={_fmt_int(i.get('gz'))}"
    )
    print(f"MODE      {s.flight_mode if s.flight_mode is not None else 'NA'}")
    print(f"MSP       {s.msp_status} | source={msp_src} | frames={s.raw_imu_rx_count}")
    print(
        "RX/TX     "
        f"frames_in={s.frames_in}  bytes_in={s.bytes_in}  bytes_out={s.bytes_out}"
    )
    if s.devices:
        devices = ", ".join(
            f"{ADDR_NAMES.get(addr, f'0x{addr:02X}')}:{name}" for addr, name in s.devices.items()
        )
    else:
        devices = "none"
    print(f"DEVICES   {devices}")
    print("=" * 72)


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else autodetect_port()
    baud = int(sys.argv[2]) if len(sys.argv) > 2 else 420000
    msp_port = sys.argv[3] if len(sys.argv) > 3 else None
    msp_baud = int(sys.argv[4]) if len(sys.argv) > 4 else 115200
    if not port:
        sys.exit("No USB serial device found. Plug in the Ranger and try again.")

    ser = serial.Serial(port, baudrate=baud, timeout=0.02)
    msp_ser = ser
    msp_via_ranger = True
    if msp_port:
        try:
            msp_ser = serial.Serial(msp_port, baudrate=msp_baud, timeout=0.02)
            msp_via_ranger = False
        except Exception as e:
            print(f"WARN: cannot open MSP port {msp_port}@{msp_baud}: {e}")
            msp_port = None
            msp_ser = ser
            msp_via_ranger = True

    ping = build_device_ping()
    # neutral sticks: throttle low (988 µs), everything else center, AUX low
    rc_channels = [988] + [1500] * 15
    rc_channels[2] = 988  # explicit throttle low
    rc_frame = build_rc_channels_packed(rc_channels)

    state = State()
    parser = CrsfParser()
    msp_parser = MspParser()
    crsf_msp_parser = CrsfMspParser()
    msp_raw_imu_req = build_msp_request(MSP_RAW_IMU_CMD)
    crsf_msp_raw_imu_req = build_crsf_msp_v2_request(MSP_RAW_IMU_CMD)

    next_ping = 0.0
    next_rc = 0.0
    next_msp = 0.0
    next_render = 0.0

    try:
        while True:
            now = time.time()
            chunk = ser.read(512)
            if chunk:
                state.bytes_in += len(chunk)
                if msp_via_ranger:
                    # Ranger link carries CRSF frames. MSP arrives encapsulated in 0x7B.
                    pass
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
                    elif ftype == T_MSP_RESP and msp_via_ranger:
                        for cmd, msp_payload in crsf_msp_parser.feed_chunk(payload):
                            if cmd == MSP_RAW_IMU_CMD:
                                d = decode_msp_raw_imu(msp_payload)
                                if d:
                                    state.raw_imu = d
                                    state.raw_imu_t = t
                                    state.raw_imu_rx_count += 1
                                    state.msp_status = "OK (Ranger CRSF MSP)"

            # Read MSP from dedicated MSP link (FC USB), not CRSF link.
            if msp_ser is not None and not msp_via_ranger:
                msp_chunk = msp_ser.read(256)
                if msp_chunk:
                    for cmd, payload in msp_parser.feed(msp_chunk):
                        if cmd == MSP_RAW_IMU_CMD:
                            d = decode_msp_raw_imu(payload)
                            if d:
                                state.raw_imu = d
                                state.raw_imu_t = time.time()
                                state.raw_imu_rx_count += 1
                                state.msp_status = "OK (dedicated MSP port)"

            # Keep the link alive
            if now >= next_rc:
                ser.write(rc_frame)
                state.bytes_out += len(rc_frame)
                next_rc = now + 0.02  # 50 Hz
            if now >= next_ping:
                ser.write(ping)
                state.bytes_out += len(ping)
                next_ping = now + 2.0
            if now >= next_msp:
                if msp_ser is not None:
                    if msp_via_ranger:
                        msp_ser.write(crsf_msp_raw_imu_req)
                        state.bytes_out += len(crsf_msp_raw_imu_req)
                    else:
                        msp_ser.write(msp_raw_imu_req)
                        state.bytes_out += len(msp_raw_imu_req)
                    if state.raw_imu_t is None:
                        if msp_via_ranger:
                            state.msp_status = "waiting for MSP_RAW_IMU over CRSF (0x7B)"
                        else:
                            state.msp_status = "waiting for MSP_RAW_IMU response"
                else:
                    state.msp_status = "MSP disabled; provide FC MSP port"
                next_msp = now + 0.10  # 10 Hz MSP_RAW_IMU poll

            # Repaint at 10 Hz
            if now >= next_render:
                render(state, port, baud, msp_port, msp_baud)
                next_render = now + 0.1
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\n")
        ser.close()
        if msp_ser is not None and msp_ser is not ser:
            msp_ser.close()


if __name__ == "__main__":
    main()
