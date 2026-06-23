#!/usr/bin/env python3
"""Pre-flight PREVIEW of a Vicon mission's planned carrot path — no Ranger, no
Vicon, no arming. Builds the SAME mission object the flight loop flies (so the
geometry is exact, not a re-derivation) and draws it offline.

Two panels:
  LEFT  — top-down XY of the carrot path (the locus seg["at"](s) traces), with
          direction arrows, dwell/start points, and the crossover marked.
  RIGHT — signed curvature κ(s) and the lateral-acceleration demand a = v²·κ it
          implies at cruise speed, vs the drone's authority (FF + tilt limits).

WHY the second panel: a figure-8 is two tangent circles traced in OPPOSITE senses.
Position and heading are CONTINUOUS at the crossover, but the curvature FLIPS sign
(+1/R → -1/R), so the required lateral accel reverses instantaneously — the drone
can't, so the real path rounds off the centre. This panel shows that jump and
whether the demand exceeds the tilt limit (the figure-8's actual problem).

Usage (repo venv):
  .venv/bin/python drone_control/Vicon_control/preview.py            # figure8 (default)
  .venv/bin/python drone_control/Vicon_control/preview.py circle
  .venv/bin/python drone_control/Vicon_control/preview.py square --save /tmp/sq.png
  .venv/bin/python drone_control/Vicon_control/preview.py figure8 --speed 1.2
"""
import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_DRONE_CONTROL = os.path.dirname(HERE)
if _DRONE_CONTROL not in sys.path:
    sys.path.insert(0, _DRONE_CONTROL)

import matplotlib                                              # noqa: E402
# Default to the Agg (save-PNG) path like plot_flight.py; --show opts into the GUI.
INTERACTIVE = "--show" in sys.argv
matplotlib.use("QtAgg" if INTERACTIVE else "Agg")
import matplotlib.pyplot as plt                               # noqa: E402

from Vicon_control import config, mission                     # noqa: E402

G = 9.81

BUILDERS = {
    "figure8": (mission.build_figure8_mission, "FIG8_SPEED_MPS"),
    "circle":  (mission.build_circle_mission,  "CRUISE_SPEED_MPS"),
    "helix":   (mission.build_helix_mission,   "HELIX_SPEED_MPS"),
    "sinecircle": (mission.build_sine_circle_mission, "SINE_SPEED_MPS"),
    "waypoint": (mission.build_waypoint_mission, "CRUISE_SPEED_MPS"),
}


def _trace_pathmission(m):
    """PathMission → (moves, dwells). Each move samples seg["at"](s) along its arc
    length, with SIGNED curvature dφ/ds from seg["heading"](s): arcs read ±1/R (CCW
    +, CW −), lines read 0 — computed per-segment so the figure-8 crossover's sign
    flip stays a clean step, not a smeared spike."""
    moves, dwells = [], []
    n_per_m = 80.0
    for seg in m.segs:
        if seg["type"] == "dwell":
            dwells.append((seg["point"][0], seg["point"][1], seg["label"]))
            continue
        L = seg["len"]
        n = max(2, int(L * n_per_m))
        s = np.linspace(0.0, L, n)
        pts = np.array([seg["at"](float(si)) for si in s])
        phi = np.unwrap(np.array([seg["heading"](float(si)) for si in s]))
        kappa = np.gradient(phi, s) if L > 1e-6 else np.zeros_like(s)
        moves.append({"label": seg["label"], "xs": pts[:, 0], "ys": pts[:, 1],
                      "s": s, "kappa": kappa})
    return moves, dwells


def _trace_waypointmission(m):
    """WaypointMission → (moves, dwells). The carrot crawls in STRAIGHT legs between
    fixed waypoints, parking (dwelling) at each vertex — so every leg is a line
    (curvature 0) and the 'corners' are stops, not curves."""
    moves, dwells = [], []
    for i, w in enumerate(m.wps):
        lbl = m.labels[i] if m.labels else f"WP{i}"
        dwells.append((w[0], w[1], lbl))
        if i == 0:
            continue
        p0, p1 = m.wps[i - 1], m.wps[i]
        L = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        s = np.linspace(0.0, L, 2)
        moves.append({"label": f"leg {i}",
                      "xs": np.array([p0[0], p1[0]]),
                      "ys": np.array([p0[1], p1[1]]),
                      "s": s, "kappa": np.zeros(2)})
    return moves, dwells


def trace_mission(m):
    """Normalize any mission into (moves, dwells), agnostic to its class."""
    if hasattr(m, "segs"):
        return _trace_pathmission(m)
    if hasattr(m, "wps"):
        return _trace_waypointmission(m)
    sys.exit(f"don't know how to trace a {type(m).__name__}")


def build(mission_name, speed_override):
    if mission_name not in BUILDERS:
        sys.exit(f"unknown mission '{mission_name}' (choose: {', '.join(BUILDERS)})")
    builder, speed_attr = BUILDERS[mission_name]
    # Launch at the world origin, nose +Y (the usual VICON_YAW_OFFSET_DEG launch).
    # The figure-8/circle path shape is independent of launch yaw (it is laid out on
    # the world axes), so this only fixes where the square's body frame points.
    launch = (0.0, 0.0, 0.0, math.radians(config.VICON_YAW_OFFSET_DEG))
    m = builder(launch)
    speed = speed_override if speed_override else getattr(config, speed_attr)
    return m, speed


def plot(m, mission_name, speed, save_path):
    moves, dwells = trace_mission(m)
    fig, (ax, axk) = plt.subplots(1, 2, figsize=(13, 6))

    # ---- LEFT: XY path ----
    s_off = 0.0
    seg_bounds = [0.0]          # cumulative arc length at each move-segment boundary
    all_s, all_k = [], []
    kappa_peak = 0.0
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, max(1, len(moves))))
    for ci, mv in enumerate(moves):
        xs, ys, s, kappa = mv["xs"], mv["ys"], mv["s"], mv["kappa"]
        ax.plot(xs, ys, "-", color=colors[ci], lw=2, label=mv["label"])
        # a couple of travel-direction arrows along the segment
        for frac in (0.25, 0.6, 0.9):
            i = int(frac * (len(xs) - 1))
            j = min(i + 1, len(xs) - 1)
            if i == j:
                continue
            ax.annotate("", xy=(xs[j], ys[j]), xytext=(xs[i], ys[i]),
                        arrowprops=dict(arrowstyle="-|>", color=colors[ci], lw=1.5))
        all_s.append(s + s_off)
        all_k.append(kappa)
        s_off += float(s[-1])
        seg_bounds.append(s_off)
        kappa_peak = max(kappa_peak, float(np.max(np.abs(kappa))))

    for px, py, lbl in dwells:
        ax.plot(px, py, "ks", ms=7, zorder=5)
        ax.annotate(lbl, (px, py), textcoords="offset points",
                    xytext=(6, 6), fontsize=8)
    ax.plot(0.0, 0.0, "g*", ms=16, zorder=6, label="launch")
    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("world X (m)")
    ax.set_ylabel("world Y (m)")
    ax.set_title(f"{mission_name}: planned carrot path (top-down)")
    ax.legend(loc="best", fontsize=8)

    # ---- RIGHT: signed curvature + lateral-accel demand ----
    s_cat = np.concatenate(all_s)
    k_cat = np.concatenate(all_k)
    a_lat = speed * speed * k_cat                    # required lateral accel, m/s²
    axk.plot(s_cat, a_lat, "-", color="tab:red", lw=1.8)
    axk.axhline(0, color="k", lw=0.6)
    a_tilt = G * math.tan(math.radians(config.MAX_TILT_DEG))
    for lim, lbl, c in ((a_tilt, f"tilt limit g·tan({config.MAX_TILT_DEG:.0f}°)",
                         "tab:purple"),
                        (config.MAX_FF_ACCEL_MPS2, "MAX_FF_ACCEL", "tab:orange")):
        axk.axhline(lim, color=c, ls="--", lw=1, label=lbl)
        axk.axhline(-lim, color=c, ls="--", lw=1)
    for b in seg_bounds[1:-1]:
        axk.axvline(b, color="gray", ls=":", lw=1)
    axk.set_xlabel("arc length along path (m)")
    axk.set_ylabel(f"lateral accel demand v²·κ at v={speed:.2f} m/s  (m/s²)")
    a_peak = speed * speed * kappa_peak
    axk.set_title(f"peak |a_lat| = {a_peak:.1f} m/s²   (κ_peak = {kappa_peak:.2f} /m)")
    axk.grid(True, alpha=0.3)
    axk.legend(loc="best", fontsize=8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
        print(f"saved {save_path}")
    if INTERACTIVE:
        plt.show()


def main():
    ap = argparse.ArgumentParser(description="Preview a Vicon mission path offline.")
    ap.add_argument("mission", nargs="?", default="figure8",
                    choices=list(BUILDERS), help="which course (default figure8)")
    ap.add_argument("--speed", type=float, default=None,
                    help="override cruise speed for the accel panel (m/s)")
    ap.add_argument("--save", default=None,
                    help="PNG output path (default preview_<mission>.png here)")
    ap.add_argument("--show", action="store_true",
                    help="open an interactive window (needs a Qt backend)")
    args = ap.parse_args()

    m, speed = build(args.mission, args.speed)
    for line in m.describe():
        print(line)
    save = args.save or os.path.join(HERE, f"preview_{args.mission}.png")
    plot(m, args.mission, speed, save)


if __name__ == "__main__":
    main()
