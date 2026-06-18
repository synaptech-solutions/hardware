#!/usr/bin/env python3
"""DRY-RUN the RL hover policy on the LIVE Vicon feed — UNARMED, no Ranger.

Carry the drone around the Vicon volume by hand and watch what the policy WOULD
command. This NEVER opens the Ranger serial port, never builds an arm frame and
never transmits — there is physically no path to the motors, so the drone cannot
arm. (The policy only needs Vicon: position + attitude + velocity; body rates are
differentiated from the Vicon attitude, NOT the FC gyro — so the drone doesn't
even need to be powered on. Vicon tracks the passive markers.)

What to verify (this is the hand test the AxisMap / sign docstrings refer to):
  • THROTTLE vs height — set the launch reference at the floor, then LIFT the
    drone toward target_alt. Throttle (T) should DROP as you raise it through the
    target and KEEP dropping above it. If lifting it UP does not lower throttle,
    the down-axis / velocity sign is wrong — that is the "rockets into the
    ceiling" failure, caught on the ground.
  • pos_err(f,r,d) — at the launch point d≈-target_alt (target is target_alt
    "up", i.e. -down). Push the drone forward → f should go +; right → r +;
    up → d toward 0 then +.
  • tilt(f,r) — the world-down vector in body axes. Level ≈ (0,0). Nose-down →
    f component changes sign; roll right → r component. Confirms roll/pitch sense.
  • rates(f,r,d) — twist the drone; the matching rate should move and settle to 0.

Keys:  L or SPACE = (re)capture launch reference here   |   q / Ctrl-C = quit

Usage (repo venv):
  .venv/bin/python drone_control/Vicon_control/rl_dryrun.py
  .venv/bin/python drone_control/Vicon_control/rl_dryrun.py --policy models/hover_acro.npz
  .venv/bin/python drone_control/Vicon_control/rl_dryrun.py --target-alt 1.0 --mass-kg 0.034
"""
import argparse
import os
import select
import sys
import termios
import time
import tty

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_DRONE_CONTROL = os.path.dirname(HERE)
if _DRONE_CONTROL not in sys.path:
    sys.path.insert(0, _DRONE_CONTROL)

from Vicon_control import config                          # noqa: E402
from Vicon_control.vicon_source import ViconPoseSource    # noqa: E402
from Vicon_control.rl_policy import (                     # noqa: E402
    MLPPolicy, HoverPolicyController, AxisMap)

CSI = "\033["


def fmt_us(us):
    """Colour a CRSF µs value by how far it deflects from neutral (1500)."""
    d = abs(us - 1500)
    col = "0" if d < 60 else ("33" if d < 250 else "1;31")
    return f"{CSI}{col}m{us:4d}{CSI}0m"


def run(args):
    if not os.path.isfile(args.policy):
        sys.exit(f"Policy not found: {args.policy}")
    policy = MLPPolicy(args.policy)
    print(f"Policy:   {policy.describe()}")
    if policy.obs_dim not in (22, 23):
        print(f"{CSI}1;33mWARNING obs_dim={policy.obs_dim} (expected 22, or 23 for "
              f"the mass-conditioned hover task) — frames/layout may not match.{CSI}0m")

    control_dt = float(policy.meta.get("control_dt", 1.0 / config.TX_HZ))
    controller = HoverPolicyController(
        policy, axis_map=AxisMap(),
        target_alt=args.target_alt, control_dt=control_dt,
        mass_kg=args.mass_kg)
    print(f"Hover:    target_alt={controller.target_alt:.2f} m above launch  "
          f"(trained control_dt={control_dt*1000:.0f} ms = {1.0/control_dt:.0f} Hz)"
          + (f", mass={controller.mass_kg:.4f} kg" if controller._obs_has_mass else ""))

    # yaw_offset=0: the policy reads the RAW quaternion and builds its own body
    # frame via AxisMap — same as the flight script.
    source = ViconPoseSource(yaw_offset_deg=0.0, body_index=args.body_index)
    print(f"Vicon:    probing UDP :{source.port} for a stream …")
    if not source.prepare():
        sys.exit(f"No Vicon stream: {source.status}\n"
                 f"DRY-RUN needs Vicon live (it is the policy's only input).")
    print(f"Vicon:    {source.status}")

    print(f"\n{CSI}32m●  DRY RUN — Ranger NOT opened. Cannot arm, cannot transmit, "
          f"motors will not spin.{CSI}0m")
    print(f"{CSI}36mL / SPACE = capture launch reference here   q / Ctrl-C = quit{CSI}0m")
    print("Lift the drone toward the target: throttle T should DROP. "
          "If it doesn't, the sign/frame is inverted.\n")

    stdin_fd = sys.stdin.fileno() if sys.stdin.isatty() else None
    stdin_old = termios.tcgetattr(stdin_fd) if stdin_fd is not None else None
    if stdin_fd is not None:
        tty.setcbreak(stdin_fd)

    def keypress():
        if stdin_fd is None:
            return None
        k = None
        while select.select([sys.stdin], [], [], 0)[0]:
            k = sys.stdin.read(1)
        return k

    launched = False
    period = control_dt
    nxt = time.monotonic()
    prev = time.monotonic()
    last_render = 0.0
    try:
        while True:
            now = time.monotonic()
            dt = min(max(now - prev, 0.0), 0.05)
            prev = now

            k = keypress()
            if k in ("q", "\x03"):
                break
            pose = source.get_pose(time.time())

            if k in ("l", "L", " ") and pose is not None:
                controller.capture_launch(pose)
                launched = True
                sys.stdout.write(
                    f"\n{CSI}32m▶ launch captured{CSI}0m at "
                    f"({pose['x']:+.2f},{pose['y']:+.2f},{pose['z']:+.2f}) → "
                    f"target {controller.target_alt:.2f} m up\n")

            if now - last_render > 0.05:
                last_render = now
                if pose is None:
                    sys.stdout.write(f"\r{CSI}K{CSI}1;31mwaiting for Vicon packet…{CSI}0m")
                elif not launched:
                    sys.stdout.write(
                        f"\r{CSI}K xyz=({pose['x']:+.2f},{pose['y']:+.2f},{pose['z']:+.2f}) "
                        f"age={pose['age_s']*1000:4.0f}ms  "
                        f"{CSI}33mpress L/SPACE to set launch reference{CSI}0m")
                else:
                    out = controller.step(pose, dt)
                    us = out["us"]
                    pe, ti, rt = out["pos_err"], out["tilt"], out["rates"]
                    a = out["action"]
                    thr_pct = (us[2] - 1000) / 10.0
                    sys.stdout.write(
                        f"\r{CSI}K xyz=({pose['x']:+.2f},{pose['y']:+.2f},{pose['z']:+.2f}) "
                        f"err(f{pe[0]:+.2f} r{pe[1]:+.2f} d{pe[2]:+.2f}) "
                        f"tilt(f{ti[0]:+.2f} r{ti[1]:+.2f}) "
                        f"rate(f{rt[0]:+.1f} r{rt[1]:+.1f} d{rt[2]:+.1f}) | "
                        f"R{fmt_us(us[0])} P{fmt_us(us[1])} "
                        f"T{fmt_us(us[2])}({thr_pct:3.0f}%) Y{fmt_us(us[3])}")
                sys.stdout.flush()

            nxt += period
            s = nxt - time.monotonic()
            if s > 0:
                time.sleep(s)
            else:
                nxt = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        if stdin_old is not None:
            termios.tcsetattr(stdin_fd, termios.TCSADRAIN, stdin_old)
        source.stop()
        print("\nStopped (no command was ever sent).")
        # The Vicon receiver runs a NON-daemon thread blocked in a blocking
        # recvfrom(); if the stream has gone stale it never returns to see the
        # stop flag, so interpreter shutdown would hang joining it (and a second
        # Ctrl-C dumps a threading traceback). All our cleanup is done above, so
        # exit hard rather than wait on that thread.
        sys.stdout.flush()
        os._exit(0)


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default=os.path.join(HERE, "models", "hover_acro.npz"),
                    help="trained policy.npz (default: models/hover_acro.npz)")
    ap.add_argument("--target-alt", type=float, default=None,
                    help="hover altitude above launch (m); default = the policy's trained target_alt")
    ap.add_argument("--mass-kg", type=float, default=None,
                    help="measured airframe mass (kg) for a mass-conditioned policy; "
                         "default = the policy's exported nominal mass")
    ap.add_argument("--body-index", type=int, default=1, help="Vicon rigid-body index (b1 = drone)")
    return ap


if __name__ == "__main__":
    run(build_parser().parse_args())
