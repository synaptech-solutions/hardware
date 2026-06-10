#!/usr/bin/env python3
"""Latency as a function of time across a flight.

Slides a window over flight_synced.csv and, in each window, cross-correlates one
link of the control chain to get its lag (ms). Plots lag vs window-center time so
you can see how big each latency is and whether it drifts:

  transport   laptop cmd  -> FC rcCommand   (throttle: cleanest, edge xcorr)
  FC response FC rcCommand -> gyro           (roll+pitch: setpoint tracking)
  end-to-end  laptop cmd  -> gyro            (roll+pitch)

A point is plotted only where the in-window correlation clears a floor (faint
points = lower confidence). Marker area scales with correlation.
"""
import csv, sys, os
import numpy as np
from scipy import signal as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_arg = sys.argv[1] if len(sys.argv) > 1 else "recordings/20260610_115444"
REC = _arg if os.path.isabs(_arg) else os.path.join(_BASE, _arg)
CSV = f"{REC}/flight_synced.csv"

WIN_S, STEP_S, FS = 10.0, 2.5, 500.0          # window, hop, resample rate
CORR_FLOOR = 0.25

need = ["Abs_time", "cmd_ch02_us", "bb_rcCommand_3",
        "cmd_ch00_us", "bb_rcCommand_0", "bb_gyroADC_0",
        "cmd_ch01_us", "bb_rcCommand_1", "bb_gyroADC_1"]
with open(CSV) as fh:
    rd = csv.reader(fh); h = next(rd); idx = {n: h.index(n) for n in need if n in h}
    cols = {n: [] for n in idx}
    for row in rd:
        for n, i in idx.items():
            v = row[i]; cols[n].append(float(v) if v not in ("", "nan", "NaN") else np.nan)
d = {k: np.asarray(v, float) for k, v in cols.items()}
t = d["Abs_time"]; tg = np.arange(t[0], t[-1], 1 / FS)
def rs(n):
    y = d[n]; m = np.isfinite(y) & np.isfinite(t); return np.interp(tg, t[m], y[m])

# bandpass for signal-level comparisons (isolate maneuver content)
bb_bp, ab_bp = sp.butter(2, [0.3 / (FS / 2), 15 / (FS / 2)], "band")

def win_lag(a, b, t0, t1, deriv, max_lag_s=0.15):
    w = (tg >= t0) & (tg < t1)
    x, y = a[w].copy(), b[w].copy()
    if deriv:
        x, y = np.diff(x), np.diff(y)
    else:
        x = sp.filtfilt(bb_bp, ab_bp, x); y = sp.filtfilt(bb_bp, ab_bp, y)
    if x.std() < 1e-6 or y.std() < 1e-6:
        return None
    x = (x - x.mean()) / x.std(); y = (y - y.mean()) / y.std()
    f = sp.correlate(y, x, "full"); L = sp.correlation_lags(len(y), len(x), "full")
    s = np.abs(L) <= int(max_lag_s * FS); f, L = f[s], L[s]
    k = int(np.argmax(f)); lag = float(L[k]); q = f[k] / len(x)
    if 0 < k < len(f) - 1:
        y0, y1, y2 = f[k - 1], f[k], f[k + 1]; dd = y0 - 2 * y1 + y2
        if abs(dd) > 1e-12: lag += 0.5 * (y0 - y2) / dd
    return lag / FS * 1000.0, q

# pre-resample the channels we need
R = {n: rs(n) for n in idx}

def sweep(pairs, deriv):
    """pairs: list of (a_name, b_name); average their lag per window, weight by corr."""
    centers, lags = [], []
    x = t[0]
    while x + WIN_S <= t[-1]:
        vals, ws = [], []
        for an, bn in pairs:
            if an in R and bn in R:
                r = win_lag(R[an], R[bn], x, x + WIN_S, deriv)
                if r and abs(r[1]) > CORR_FLOOR:
                    vals.append(r[0]); ws.append(abs(r[1]))
        if vals:
            centers.append(x + WIN_S / 2 - t[0])
            lags.append((np.average(vals, weights=ws), max(ws)))
        x += STEP_S
    return np.array(centers), np.array(lags)  # lags[:,0]=ms, lags[:,1]=corr

transport = sweep([("cmd_ch02_us", "bb_rcCommand_3")], deriv=True)
response  = sweep([("bb_rcCommand_0", "bb_gyroADC_0"), ("bb_rcCommand_1", "bb_gyroADC_1")], deriv=False)
end2end   = sweep([("cmd_ch00_us", "bb_gyroADC_0"), ("cmd_ch01_us", "bb_gyroADC_1")], deriv=False)

fig, ax = plt.subplots(figsize=(13, 6))
series = [("laptop cmd → FC received (transport)", transport, "#1f77b4"),
          ("FC received → drone rotating (gyro)", response, "#2ca02c"),
          ("laptop cmd → drone rotating (end-to-end)", end2end, "#d62728")]
for name, (c, lg), color in series:
    if len(c) == 0:
        continue
    ms, corr = lg[:, 0], lg[:, 1]
    ax.plot(c, ms, "-", color=color, alpha=0.5, lw=1.3)
    ax.scatter(c, ms, s=8 + 60 * (corr - CORR_FLOOR).clip(0), color=color,
               label=f"{name}   (median {np.median(ms):.0f} ms)", zorder=3)

ax.axhline(0, color="k", lw=0.8, alpha=0.4)
ax.axhspan(0, 20, color="grey", alpha=0.06)
ax.text(0.5, 21, "one 50 Hz command frame (20 ms)", fontsize=8, color="grey")
ax.set_xlabel("flight time (s)"); ax.set_ylabel("latency (ms)")
meta = ""
mj = f"{REC}/flight_synced.meta.json"
if os.path.isfile(mj):
    import json; jm = json.load(open(mj))
    if jm.get("clock_drift_slope_s_per_s"):
        meta = f"   [drift-corrected: {jm['clock_drift_slope_s_per_s']*1000:+.2f} ms/s removed]"
ax.set_title(f"Control-path latency over the flight — {os.path.basename(REC)}{meta}\n"
             f"(window {WIN_S:.0f}s, hop {STEP_S:.1f}s; marker size = correlation confidence)")
ax.legend(loc="upper left", framealpha=0.9); ax.grid(alpha=0.3)
ax.set_ylim(-30, 80)
plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots",
                   f"latency_vs_time_{os.path.basename(REC.rstrip('/'))}.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=120)
print("saved", out)
for name, (c, lg), _ in series:
    if len(c): print(f"  {name:48} median {np.median(lg[:,0]):5.1f} ms   range [{lg[:,0].min():.0f}, {lg[:,0].max():.0f}]")
