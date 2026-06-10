#!/usr/bin/env python3
"""Visual proof of control-path latency: zoom on the biggest command steps and
overlay the laptop command, FC rcCommand, setpoint, motor output and gyro."""
import csv, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # data_logging/
_arg = sys.argv[1] if len(sys.argv) > 1 else "recordings/20260609_135349"
REC = _arg if os.path.isabs(_arg) else os.path.join(_BASE, _arg)
CSV = f"{REC}/flight_synced.csv"
want = ["Abs_time","cmd_ch00_us","cmd_ch02_us","cmd_ch03_us",
        "bb_rcCommand_0","bb_rcCommand_2","bb_rcCommand_3",
        "bb_setpoint_0","bb_setpoint_2","bb_gyroADC_0","bb_gyroADC_2",
        "bb_motor_0","bb_motor_1","bb_motor_2","bb_motor_3"]
with open(CSV) as fh:
    rd = csv.reader(fh); h = next(rd); idx = {n: h.index(n) for n in want if n in h}
    cols = {n: [] for n in idx}
    for row in rd:
        for n, i in idx.items():
            v = row[i]; cols[n].append(float(v) if v not in ("","nan","NaN") else np.nan)
d = {k: np.asarray(v, float) for k, v in cols.items()}
t = d["Abs_time"]
FS = 1000.0; tg = np.arange(t[0], t[-1], 1/FS)
def rs(n):
    y = d[n]; m = np.isfinite(y); return np.interp(tg, t[m], y[m])
def nz(y):
    y = y - np.nanmin(y); r = np.nanmax(y); return y/r if r > 1e-9 else y

def biggest_step(sig, guard=0.3):
    """index in tg of the largest level change (over 40ms), away from edges."""
    h = int(0.02*FS); j = np.zeros_like(sig); j[h:-h] = sig[2*h:] - sig[:-2*h]
    lo, hi = int(guard*len(sig)), int((1-guard*0.2)*len(sig))
    j2 = j.copy(); j2[:lo] = 0; j2[hi:] = 0
    return int(np.argmax(np.abs(j2)))

mot = np.nanmean([rs(f"bb_motor_{i}") for i in range(4)], axis=0)

fig, axes = plt.subplots(2, 1, figsize=(13, 9))

# --- Throttle step: cmd -> rcCommand -> motor ---
thr = rs("cmd_ch02_us"); e = biggest_step(thr)
w = int(0.25*FS); s0, s1 = e-w, e+w
ax = axes[0]; x = (tg[s0:s1]-tg[e])*1000
ax.plot(x, nz(thr[s0:s1]), label="laptop cmd (throttle)", lw=2)
ax.plot(x, nz(rs("bb_rcCommand_3")[s0:s1]), label="FC rcCommand", lw=2)
ax.plot(x, nz(mot[s0:s1]), label="motor output (avg)", lw=2)
ax.axvline(0, color="k", ls="--", alpha=0.5)
ax.set_title(f"{REC}  —  THROTTLE step: laptop command vs FC reception vs motors")
ax.set_xlabel("time relative to command step (ms)"); ax.set_ylabel("normalized")
ax.legend(loc="best"); ax.grid(alpha=0.3)

# --- Roll step: cmd -> setpoint -> gyro (actual rotation) ---
roll = rs("cmd_ch00_us"); e = biggest_step(roll)
s0, s1 = e-w, e+w; x = (tg[s0:s1]-tg[e])*1000
ax = axes[1]
ax.plot(x, nz(roll[s0:s1]), label="laptop cmd (roll)", lw=2)
ax.plot(x, nz(rs("bb_rcCommand_0")[s0:s1]), label="FC rcCommand", lw=2)
ax.plot(x, nz(rs("bb_setpoint_0")[s0:s1]), label="setpoint", lw=2, alpha=0.7)
ax.plot(x, nz(rs("bb_gyroADC_0")[s0:s1]), label="gyro (actual roll rate)", lw=2)
ax.axvline(0, color="k", ls="--", alpha=0.5)
ax.set_title("ROLL step: command vs actual airframe rotation (gyro)")
ax.set_xlabel("time relative to command step (ms)"); ax.set_ylabel("normalized")
ax.legend(loc="best"); ax.grid(alpha=0.3)

plt.tight_layout()
_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(_OUTDIR, exist_ok=True)
out = os.path.join(_OUTDIR, f"latency_overlay_{os.path.basename(REC.rstrip('/'))}.png")
plt.savefig(out, dpi=110)
print("saved", out)
