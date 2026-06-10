#!/usr/bin/env python3
"""Live TX12 joystick monitor — shows EVERY input from the controller in real time.

Refreshes ~20 Hz and prints the raw value of every axis (with a bar + a "changed
recently" marker) and the state of every button. Use it to find which axis/button
a control maps to: flip a switch or move a stick and watch which line lights up.

Dependency-free — reads /dev/input/jsN directly via the same Joystick reader the
flight script uses (no cv2/numpy needed), so plain `python3` works too.

Usage:
  python3 data_logging/monitor_tx12.py            # default /dev/input/js0
  python3 data_logging/monitor_tx12.py /dev/input/js1
Ctrl-C to quit.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from joystick_flight import Joystick  # noqa: E402

CSI = "\033["
GREEN = CSI + "32m"
BOLD = CSI + "1m"
DIM = CSI + "2m"
RST = CSI + "0m"
RECENT_S = 0.7          # highlight an axis/button this long after it last changed


def bar(v, lo=-32767, hi=32767, width=34):
    frac = (v - lo) / (hi - lo) if hi != lo else 0.5
    frac = 0.0 if frac < 0 else 1.0 if frac > 1 else frac
    n = int(round(frac * width))
    return "[" + "#" * n + "-" * (width - n) + "]"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/dev/input/js0"
    js = Joystick(path)

    last_ax, last_bt = {}, {}
    changed_ax, changed_bt = {}, {}      # index -> monotonic time of last change

    sys.stdout.write(CSI + "2J")          # clear screen once
    try:
        while True:
            js.poll()
            if not js.alive:
                sys.stdout.write(CSI + "0m\n!! joystick disconnected.\n")
                break
            ax, bt = js.snapshot()
            now = time.monotonic()
            for i, v in ax.items():
                if last_ax.get(i) != v:
                    changed_ax[i] = now
                    last_ax[i] = v
            for i, v in bt.items():
                if last_bt.get(i) != v:
                    changed_bt[i] = now
                    last_bt[i] = v

            lines = [
                f"{BOLD}{js.name}{RST}   {js.n_axes} axes, {js.n_buttons} buttons"
                f"   ({GREEN}●{RST} = changed recently · Ctrl-C to quit)",
                "",
                f"{BOLD}AXES{RST}",
            ]
            for i in range(js.n_axes):
                v = ax.get(i, 0)
                recent = (now - changed_ax.get(i, -99)) < RECENT_S
                mark = f"{GREEN}●{RST}" if recent else " "
                val = f"{GREEN}{BOLD}{v:+7d}{RST}" if recent else f"{v:+7d}"
                lines.append(f" {mark} axis[{i:2d}]  {val}  {bar(v)}")

            lines += ["", f"{BOLD}BUTTONS{RST}"]
            pressed = [i for i in range(js.n_buttons) if bt.get(i, 0)]
            lines.append("  pressed: " + (
                "  ".join(f"{GREEN}btn{i}{RST}" for i in pressed) if pressed
                else f"{DIM}(none){RST}"))
            grid = "  ".join(
                (f"{GREEN}{BOLD}{i:2d}{RST}" if bt.get(i, 0)
                 else (f"{GREEN}{i:2d}{RST}" if (now - changed_bt.get(i, -99)) < RECENT_S
                       else f"{DIM}{i:2d}{RST}"))
                for i in range(js.n_buttons))
            lines.append("  all:     " + grid)

            sys.stdout.write(CSI + "H" + CSI + "J" + "\n".join(lines) + "\n")
            sys.stdout.flush()
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        js.close()
        sys.stdout.write(RST + "\nStopped.\n")


if __name__ == "__main__":
    main()
