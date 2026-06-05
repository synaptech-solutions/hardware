#!/usr/bin/env python3
"""
Send one-by-one RC stick commands so Betaflight Configurator can verify direction.

Sequence:
  1) Throttle up pulse
  2) Roll right pulse
  3) Pitch forward pulse
  4) Yaw right pulse

Optional --both runs right/left, forward/back, yaw right/left for easier sign checks.

This script keeps ARM and PREARM low the entire time.

Usage:
  python3 setup/test_stick_directions.py
  python3 setup/test_stick_directions.py /dev/ttyUSB0
  python3 setup/test_stick_directions.py --both
"""

import argparse
import select
import sys
import termios
import time
import tty

try:
    import serial
except ImportError:
    sys.exit("pyserial not installed. Run: pip install pyserial")

from live_telemetry import build_rc_channels_packed, autodetect_port

# CRSF channel indices (AETR + AUX)
ROLL_IDX = 0
PITCH_IDX = 1
THROTTLE_IDX = 2
YAW_IDX = 3
ARM_IDX = 5      # AUX2 on this setup
PREARM_IDX = 9   # AUX6 on this setup
MODE_IDX = 6     # AUX3 (ANGLE in existing config)

NEUTRAL_US = 1500
THROTTLE_MIN_US = 1000
THROTTLE_PULSE_US = 1300
AXIS_POS_US = 1700
AXIS_NEG_US = 1300
ARM_LOW_US = 1000
PREARM_LOW_US = 1000
MODE_ANGLE_US = 1500

TX_HZ = 50.0

CSI = "\033["


def clear_screen():
    sys.stdout.write(CSI + "H" + CSI + "2J")
    sys.stdout.flush()


def render_hold_status(label: str, idx: int, active_us: int, elapsed_s: float):
    clear_screen()
    sys.stdout.write("REMOVE PROPS. Open Betaflight Configurator -> Receiver tab.\n")
    sys.stdout.write("This script never requests arm (ARM and PREARM forced LOW).\n\n")
    sys.stdout.write(f"Active test: {label}\n")
    sys.stdout.write(f"Channel: CH{idx + 1}    Value: {active_us} us\n")
    sys.stdout.write(f"Elapsed: {elapsed_s:5.1f}s\n\n")
    sys.stdout.write("Press any key to continue to the next test...\n")
    sys.stdout.flush()


def build_base_channels() -> list[int]:
    ch = [NEUTRAL_US] * 16
    ch[THROTTLE_IDX] = THROTTLE_MIN_US
    ch[ARM_IDX] = ARM_LOW_US
    ch[PREARM_IDX] = PREARM_LOW_US
    ch[MODE_IDX] = MODE_ANGLE_US
    return ch


def tx_hold(ser: serial.Serial, channels: list[int], seconds: float):
    period = 1.0 / TX_HZ
    end_t = time.time() + seconds
    while time.time() < end_t:
        ser.write(build_rc_channels_packed(channels))
        time.sleep(period)


def wait_for_key_while_sending(
    ser: serial.Serial,
    channels: list[int],
    label: str,
    idx: int,
    active_us: int,
    settle_s: float,
):
    period = 1.0 / TX_HZ
    channels[idx] = active_us
    start_t = time.time()
    next_ui = 0.0

    if not sys.stdin.isatty():
        input("STDIN is not a TTY. Press ENTER to continue... ")
        return

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            ser.write(build_rc_channels_packed(channels))
            now = time.time()
            if now >= next_ui:
                render_hold_status(label, idx, active_us, now - start_t)
                next_ui = now + 0.10

            ready, _, _ = select.select([sys.stdin], [], [], period)
            if ready:
                sys.stdin.read(1)
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        clear_screen()

    # Return to neutral (or throttle-min for throttle) between tests.
    channels[idx] = THROTTLE_MIN_US if idx == THROTTLE_IDX else NEUTRAL_US
    print(f"neutral: CH{idx + 1} reset for {settle_s:.1f}s", flush=True)
    tx_hold(ser, channels, settle_s)


def main():
    parser = argparse.ArgumentParser(description="Betaflight stick-direction checker")
    parser.add_argument("port", nargs="?", default=None, help="Serial port (auto-detect if omitted)")
    parser.add_argument("--baud", type=int, default=420000, help="Serial baud rate")
    parser.add_argument("--settle", type=float, default=1.0, help="Neutral settle duration (s)")
    parser.add_argument(
        "--both",
        action="store_true",
        help="Also send opposite direction for roll/pitch/yaw",
    )
    args = parser.parse_args()

    port = args.port or autodetect_port()
    if not port:
        sys.exit("No serial device found. Pass port manually, e.g. /dev/ttyUSB0")

    print("REMOVE PROPS. Open Betaflight Configurator -> Receiver tab.")
    print("This script never requests arm (ARM and PREARM forced LOW).")
    input("Press ENTER to start... ")

    ser = serial.Serial(port, baudrate=args.baud, timeout=0.05)
    channels = build_base_channels()

    try:
        print(f"\nConnected: {port} @ {args.baud}")
        print("Sending neutral baseline for 2.0s...", flush=True)
        tx_hold(ser, channels, 2.0)

        wait_for_key_while_sending(
            ser,
            channels,
            label="THROTTLE UP",
            idx=THROTTLE_IDX,
            active_us=THROTTLE_PULSE_US,
            settle_s=args.settle,
        )

        wait_for_key_while_sending(
            ser,
            channels,
            label="ROLL RIGHT",
            idx=ROLL_IDX,
            active_us=AXIS_POS_US,
            settle_s=args.settle,
        )

        if args.both:
            wait_for_key_while_sending(
                ser,
                channels,
                label="ROLL LEFT",
                idx=ROLL_IDX,
                active_us=AXIS_NEG_US,
                settle_s=args.settle,
            )

        wait_for_key_while_sending(
            ser,
            channels,
            label="PITCH FORWARD",
            idx=PITCH_IDX,
            active_us=AXIS_POS_US,
            settle_s=args.settle,
        )

        if args.both:
            wait_for_key_while_sending(
                ser,
                channels,
                label="PITCH BACK",
                idx=PITCH_IDX,
                active_us=AXIS_NEG_US,
                settle_s=args.settle,
            )

        wait_for_key_while_sending(
            ser,
            channels,
            label="YAW RIGHT",
            idx=YAW_IDX,
            active_us=AXIS_POS_US,
            settle_s=args.settle,
        )

        if args.both:
            wait_for_key_while_sending(
                ser,
                channels,
                label="YAW LEFT",
                idx=YAW_IDX,
                active_us=AXIS_NEG_US,
                settle_s=args.settle,
            )

        print("\nDone. Sending final neutral/disarm for 1.5s...", flush=True)
        tx_hold(ser, channels, 1.5)
    finally:
        ser.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
