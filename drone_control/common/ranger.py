"""open_ranger() — open the ELRS Ranger USB-C serial at a custom 420000 baud.

pyserial's normal open path calls tcsetattr with B-constants and rejects 420000
on this kernel/pyserial combo (termios EINVAL). Switching the line rate via the
kernel's custom-baud ioctl (TCSETS2 / BOTHER) is the fix. This was duplicated in
both joystick_flight.py and controller_v2/threads.py — it now lives here.

(See memory: feedback_crsf_custom_baud — do NOT strip this in favor of plain
pyserial Serial(420000).)
"""
import array
import fcntl

import serial

_TCGETS2, _TCSETS2 = 0x802C542A, 0x402C542B
_BOTHER, _CBAUD = 0o010000, 0o010017


def open_ranger(port, baud=420000, timeout=0.0):
    """Open `port` and set a custom `baud` via TCSETS2/BOTHER.

    timeout: pyserial read timeout. The data logger uses 0.0 (pure non-blocking
    drain); the threaded apriltag rx loop used a small timeout — pass what you need.
    """
    ser = serial.Serial(port, baudrate=115200, timeout=timeout)
    buf = array.array("i", [0] * 64)
    fcntl.ioctl(ser.fileno(), _TCGETS2, buf)
    buf[2] = (buf[2] & ~_CBAUD) | _BOTHER
    buf[9] = buf[10] = baud
    fcntl.ioctl(ser.fileno(), _TCSETS2, buf)
    return ser
