#!/usr/bin/env python3
"""Latency analysis: laptop commands -> FC rcCommand -> gyro -> motors.

flight_synced.csv shares one Abs_time clock, zero-based at the trigger flick.
We estimate each stage of the control-path delay with robust estimators:

  Stage A  laptop cmd_chXX_us -> bb_rcCommand   transport(USB+CRSF+RF+RX) + RC smoothing
  Stage B  bb_setpoint        -> bb_gyroADC     PID + motor + airframe response   [1 clock => clean]

Estimators:
  * GCC-PHAT  : phase-transform cross-corr, sharp peak, side-lobe resistant
  * joint lag : sum of normalized cross-corr across all channels -> the shared
                (clock-offset + transport) term, robust to per-channel artifacts
  * edge      : median time from a command step to the response moving (window based)
"""
import csv, sys
import numpy as np
from scipy import signal as sp_signal

REC = sys.argv[1] if len(sys.argv) > 1 else "recordings/20260609_135349"
CSV = f"{REC}/flight_synced.csv"
want = ["Abs_time","cmd_ch00_us","cmd_ch01_us","cmd_ch02_us","cmd_ch03_us",
        "bb_rcCommand_0","bb_rcCommand_1","bb_rcCommand_2","bb_rcCommand_3",
        "bb_setpoint_0","bb_setpoint_1","bb_setpoint_2","bb_setpoint_3",
        "bb_gyroADC_0","bb_gyroADC_1","bb_gyroADC_2","cmd_armed"]
with open(CSV) as fh:
    rd = csv.reader(fh); header = next(rd)
    idx = {nm: header.index(nm) for nm in want if nm in header}
    cols = {nm: [] for nm in idx}
    for row in rd:
        for nm, i in idx.items():
            v = row[i]; cols[nm].append(float(v) if v not in ("","nan","NaN") else np.nan)
data = {k: np.asarray(v, float) for k, v in cols.items()}
t = data["Abs_time"]; dt_med = np.median(np.diff(t))
print(f"# {REC}  rows={len(t)} span={t[-1]-t[0]:.1f}s  blackbox~{1/dt_med:.0f}Hz")

FS = 1000.0
tg = np.arange(t[0], t[-1], 1.0/FS)
def rs(nm):
    y = data[nm]; m = np.isfinite(y) & np.isfinite(t)
    return np.interp(tg, t[m], y[m]) if m.sum() >= 10 else np.full_like(tg, np.nan)
R = {nm: rs(nm) for nm in idx if nm != "Abs_time"}

def prep(a, b):
    m = np.isfinite(a) & np.isfinite(b); a, b = a[m].copy(), b[m].copy()
    if len(a) < 100 or a.std() < 1e-9 or b.std() < 1e-9: return None
    a = (a-a.mean())/a.std(); b = (b-b.mean())/b.std(); return a, b

def gcc_phat(a, b, max_lag_s=0.12):
    """lag (s) such that b is a delayed copy of a (positive => b lags a)."""
    p = prep(a, b)
    if p is None: return None
    a, b = p; nfft = 1 << int(np.ceil(np.log2(len(a)+len(b))))
    A = np.fft.rfft(a, nfft); B = np.fft.rfft(b, nfft)
    Rspec = B * np.conj(A); Rspec /= np.abs(Rspec) + 1e-9
    cc = np.fft.irfft(Rspec, nfft)
    mx = int(max_lag_s*FS)
    cc = np.concatenate((cc[-mx:], cc[:mx+1])); lags = np.arange(-mx, mx+1)
    k = int(np.argmax(cc)); lag = float(lags[k])
    if 0 < k < len(cc)-1:
        y0,y1,y2 = cc[k-1],cc[k],cc[k+1]; d = y0-2*y1+y2
        if abs(d) > 1e-12: lag += 0.5*(y0-y2)/d
    return lag/FS, cc[k]/np.max(np.abs(cc))

def ncc(a, b, max_lag_s=0.12):
    p = prep(a, b)
    if p is None: return None
    a, b = p
    full = sp_signal.correlate(b, a, mode="full"); lags = sp_signal.correlation_lags(len(b),len(a),"full")
    sel = np.abs(lags) <= int(max_lag_s*FS); full, lags = full[sel]/len(a), lags[sel]
    return full, lags

def edge_lag(cmd, resp, win_pre=0.08, win_post=0.15):
    """Detect command level-shifts on the resampled trace; measure delay until the
    response reaches 50% of its post-step change."""
    m = np.isfinite(cmd) & np.isfinite(resp); c, r = cmd[m], resp[m]
    Wp, Wq = int(win_pre*FS), int(win_post*FS)
    # step = command moves >2*its short-term noise over a 30ms window
    span = c[Wp:] - c[:-Wp] if len(c) > Wp else np.array([])
    thr = max(3*np.median(np.abs(np.diff(c)))*1, 0.15*np.std(c))
    cand = np.where(np.abs(np.diff(c)) > 0)[0]
    # find step centers: local windows where |c[i+15ms]-c[i-15ms]| is large & peaks
    h = int(0.015*FS)
    jump = np.full(len(c), 0.0)
    jump[h:-h] = np.abs(c[2*h:] - c[:-2*h])
    big = jump > max(2*np.std(np.diff(c))*np.sqrt(2*h), 0.12*np.std(c))
    events, i = [], 0
    while i < len(big):
        if big[i]:
            j = i
            while j < len(big) and big[j]: j += 1
            events.append((i+j)//2); i = j + int(0.05*FS)
        else: i += 1
    lags = []
    for e in events:
        if e-Wp < 0 or e+Wq >= len(r): continue
        base = np.median(r[e-Wp:e]); final = np.median(r[e+Wq-Wp:e+Wq])
        chg = final - base
        if abs(chg) < 4*np.std(r[e-Wp:e]) + 1e-6: continue
        seg = r[e:e+Wq] - base
        cr = np.where(seg*np.sign(chg) >= 0.5*abs(chg))[0]
        if len(cr): lags.append(cr[0]/FS)
    return (float(np.median(lags)), len(lags)) if len(lags) >= 3 else None

chans = [("ROLL","cmd_ch00_us","bb_rcCommand_0","bb_setpoint_0","bb_gyroADC_0"),
         ("PITCH","cmd_ch01_us","bb_rcCommand_1","bb_setpoint_1","bb_gyroADC_1"),
         ("YAW","cmd_ch03_us","bb_rcCommand_2","bb_setpoint_2","bb_gyroADC_2"),
         ("THROTTLE","cmd_ch02_us","bb_rcCommand_3","bb_setpoint_3",None)]

# --- joint shared lag across all Stage-A channels (robust clock+transport term) ---
accum = None; ref_lags = None
for nm,cmd,rc,sp,gy in chans:
    res = ncc(R[cmd], R[rc])
    if res is None: continue
    full, lags = res
    if np.nanstd(data[cmd]) < 5: continue   # skip dead sticks
    accum = full.copy() if accum is None else accum + full; ref_lags = lags
joint = None
if accum is not None:
    k = int(np.argmax(accum)); joint = ref_lags[k]/FS
print(f"# JOINT shared Stage-A lag (clock offset + transport, all live channels): "
      f"{joint*1000:+.1f} ms" if joint is not None else "# joint: n/a")

print("\n=== Stage A: laptop cmd -> FC rcCommand ===")
print(f"{'chan':9}{'std(us)':>9}{'GCC-PHAT':>12}{'edge(50%)':>16}")
for nm,cmd,rc,sp,gy in chans:
    s = np.nanstd(data[cmd]); g = gcc_phat(R[cmd], R[rc]); e = edge_lag(R[cmd], R[rc])
    gs = f"{g[0]*1000:6.1f}ms" if g else "n/a"
    es = f"{e[0]*1000:5.1f}ms (n={e[1]})" if e else "n/a"
    print(f"{nm:9}{s:8.1f} {gs:>12}{es:>16}")

print("\n=== Stage B: setpoint -> gyro  [single clock, clean] ===")
for nm,cmd,rc,sp,gy in chans:
    if gy is None: continue
    g = gcc_phat(R[sp], R[gy], 0.08); e = edge_lag(R[sp], R[gy], 0.05, 0.10)
    gs = f"{g[0]*1000:6.1f}ms" if g else "n/a"; es = f"{e[0]*1000:5.1f}ms(n={e[1]})" if e else "n/a"
    print(f"{nm:9} GCC {gs:>10}   edge {es}")
