"""open_ranger() — open the ELRS Ranger USB-C serial at a custom 420000 baud.

pyserial's normal open path calls tcsetattr with B-constants and rejects 420000
on this kernel/pyserial combo (termios EINVAL). Switching the line rate via the
kernel's custom-baud ioctl (TCSETS2 / BOTHER) is the fix. This was duplicated in
both joystick_flight.py and controller_v2/threads.py — it now lives here.

On non-Linux (macOS/Windows), pyserial handles 420000 baud natively, so the
ioctl path is skipped.

(See memory: feedback_crsf_custom_baud — do NOT strip this in favor of plain
pyserial Serial(420000) **on Linux**.)
"""
import sys

import serial


def open_ranger(port, baud=420000, timeout=0.0):
    """Open `port` and set a custom `baud`.

    On Linux, uses TCSETS2/BOTHER ioctl to set the non-standard baud rate.
    On other platforms, pyserial handles it directly.

    timeout: pyserial read timeout. The data logger uses 0.0 (pure non-blocking
    drain); the threaded apriltag rx loop used a small timeout — pass what you need.
    """
    if sys.platform == "linux":
        import array
        import fcntl
        _TCGETS2, _TCSETS2 = 0x802C542A, 0x402C542B
        _BOTHER, _CBAUD = 0o010000, 0o010017
        ser = serial.Serial(port, baudrate=115200, timeout=timeout)
        buf = array.array("i", [0] * 64)
        fcntl.ioctl(ser.fileno(), _TCGETS2, buf)
        buf[2] = (buf[2] & ~_CBAUD) | _BOTHER
        buf[9] = buf[10] = baud
        fcntl.ioctl(ser.fileno(), _TCSETS2, buf)
        return ser
    elif sys.platform == "darwin":
        import fcntl
        import struct
        import termios
        # macOS: termios rejects non-standard bauds. IOSSIOSPEED ioctl works,
        # but pyserial's _reconfigure_port can reset it. So we open with a
        # placeholder baud, then override via ioctl.
        IOSSIOSPEED = 0x80085402
        ser = serial.Serial()
        ser.port = port
        ser.timeout = timeout
        ser.baudrate = 9600  # placeholder — never actually used on the wire
        ser.open()
        fcntl.ioctl(ser.fileno(), IOSSIOSPEED, struct.pack("@L", baud))
        # verify the baud actually took
        attrs = termios.tcgetattr(ser.fileno())
        actual_in, actual_out = attrs[4], attrs[5]
        if actual_in != baud or actual_out != baud:
            print(f"WARNING: IOSSIOSPEED requested {baud} but termios reports "
                  f"ispeed={actual_in} ospeed={actual_out}")
        else:
            print(f"Ranger:   IOSSIOSPEED {baud} confirmed (ispeed={actual_in}, ospeed={actual_out})")
        return ser
    else:
        return serial.Serial(port, baudrate=baud, timeout=timeout)
