#!/usr/bin/env python3
"""Independent blackbox<->laptop clock-offset check using Vicon.

combine_flight.py aligns the blackbox to the laptop streams by the shared trigger
flick, assuming offset = 0. Vicon measures the drone's ACTUAL rotation on the
LAPTOP clock; the blackbox gyro measures the same rotation on the FC clock.
Cross-correlating the two reveals the residual clock offset the sync left in --
i.e. how much of any apparent cmd->rcCommand "latency" is just sync misalignment
rather than real transport delay.

We derive Vicon body angular rate from the quaternion stream, low-pass both it and
the blackbox gyro to the maneuver band (<6 Hz, where pilot inputs live and SNR is
good), then try every Vicon-axis/sign pairing against each gyro axis and report the
offset from the best-correlated pair.
"""
import csv, sys, os
import numpy as np
from scipy import signal as sp_signal

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_arg = sys.argv[1] if len(sys.argv) > 1 else "recordings/20260610_115444"
REC = _arg if os.path.isabs(_arg) else os.path.join(_BASE, _arg)
CSV = f"{REC}/flight_synced.csv"

want = ["Abs_time", "b1_qx", "b1_qy", "b1_qz", "b1_qw",
        "bb_gyroADC_0", "bb_gyroADC_1", "bb_gyroADC_2"]
with open(CSV) as fh:
    rd = csv.reader(fh); h = next(rd)
    if "b1_qw" not in h:
        print(f"# {os.path.basename(REC)}: no Vicon pose in this flight -- clock check N/A"); sys.exit()
    idx = {n: h.index(n) for n in want if n in h}
    cols = {n: [] for n in idx}
    for row in rd:
        for n, i in idx.items():
            v = row[i]; cols[n].append(float(v) if v not in ("", "nan", "NaN") else np.nan)
d = {k: np.asarray(v, float) for k, v in cols.items()}
t = d["Abs_time"]
FS = 200.0
tg = np.arange(t[0], t[-1], 1/FS)
def rs(y):
    mm = np.isfinite(y) & np.isfinite(t); return np.interp(tg, t[mm], y[mm])

# Vicon body angular rate from quaternion: omega = 2 * (q^-1 * q_dot)
q = np.stack([rs(d["b1_qw"]), rs(d["b1_qx"]), rs(d["b1_qy"]), rs(d["b1_qz"])], axis=1)
q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
def qmul(a, b):
    w1,x1,y1,z1 = a.T; w2,x2,y2,z2 = b.T
    return np.stack([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2], axis=1)
qdot = np.gradient(q, 1/FS, axis=0)
omega = 2.0 * qmul(q*np.array([1,-1,-1,-1]), qdot)[:, 1:] * 180/np.pi  # deg/s, body
gyro = np.stack([rs(d["bb_gyroADC_0"]), rs(d["bb_gyroADC_1"]), rs(d["bb_gyroADC_2"])], axis=1)

# low-pass both to 6 Hz
bnum, anum = sp_signal.butter(2, 6.0/(FS/2), "low")
omega_f = sp_signal.filtfilt(bnum, anum, omega, axis=0)
gyro_f  = sp_signal.filtfilt(bnum, anum, gyro,  axis=0)

def xc(a, b, max_lag_s=0.4):
    a = (a-a.mean())/(a.std()+1e-9); b = (b-b.mean())/(b.std()+1e-9)
    full = sp_signal.correlate(b, a, "full"); lags = sp_signal.correlation_lags(len(b), len(a), "full")
    sel = np.abs(lags) <= int(max_lag_s*FS); full, lags = full[sel]/len(a), lags[sel]
    k = int(np.argmax(np.abs(full))); lag = float(lags[k])
    if 0 < k < len(full)-1:
        y0,y1,y2 = full[k-1],full[k],full[k+1]; dd = y0-2*y1+y2
        if abs(dd) > 1e-12: lag += 0.5*(y0-y2)/dd
    return lag/FS, full[k]

# best Vicon-axis/sign vs gyro-axis pairing
best = None
for vi in range(3):
    for gi in range(3):
        for sign in (1, -1):
            lag, corr = xc(sign*omega_f[:, vi], gyro_f[:, gi])
            if best is None or abs(corr) > abs(best[0]):
                best = (corr, lag, vi, gi, sign)
corr, lag, vi, gi, sign = best
ax = "RPY"
print(f"# {os.path.basename(REC)}  native ~{1/np.median(np.diff(t)):.0f}Hz")
print(f"#   best pair: Vicon {'+-'[sign<0]}omega[{ax[vi]}]  vs  blackbox gyro[{ax[gi]}]   corr={corr:.2f}")
print(f"#   blackbox lags Vicon by {lag*1000:+.1f} ms   <-- residual clock offset the sync left in")
print(f"#   => subtract ~{lag*1000:+.0f} ms from cmd->rcCommand / cmd->gyro to get TRUE transport-side lag")
