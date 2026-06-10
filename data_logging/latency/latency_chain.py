#!/usr/bin/env python3
"""Consolidated control-path latency, link by link, aggregated over all clean
command steps. Uses 50%-crossing delay (robust, physically interpretable).

Chain:  laptop cmd --[transport+RC smoothing]--> rcCommand --[PID]--> motor
                                                              --> gyro (rotation)
"""
import csv, sys
import numpy as np
import os
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # data_logging/
_arg = sys.argv[1] if len(sys.argv) > 1 else "recordings/20260609_135349"
REC = _arg if os.path.isabs(_arg) else os.path.join(_BASE, _arg)
CSV = f"{REC}/flight_synced.csv"
want = ["Abs_time","cmd_ch00_us","cmd_ch01_us","cmd_ch02_us","cmd_ch03_us",
        "bb_rcCommand_0","bb_rcCommand_1","bb_rcCommand_2","bb_rcCommand_3",
        "bb_setpoint_0","bb_setpoint_1","bb_setpoint_2",
        "bb_gyroADC_0","bb_gyroADC_1","bb_gyroADC_2",
        "bb_motor_0","bb_motor_1","bb_motor_2","bb_motor_3"]
with open(CSV) as fh:
    rd = csv.reader(fh); h = next(rd); idx = {n: h.index(n) for n in want if n in h}
    cols = {n: [] for n in idx}
    for row in rd:
        for n, i in idx.items():
            v = row[i]; cols[n].append(float(v) if v not in ("","nan","NaN") else np.nan)
d = {k: np.asarray(v, float) for k, v in cols.items()}
t = d["Abs_time"]; FS = 1000.0; tg = np.arange(t[0], t[-1], 1/FS)
def rs(n):
    y = d[n]; m = np.isfinite(y); return np.interp(tg, t[m], y[m])
mot = np.nanmean([rs(f"bb_motor_{i}") for i in range(4)], axis=0)
print(f"# {REC}  native ~{1/np.median(np.diff(t)):.0f}Hz  (lag resolution ~ one sample)")

def step_events(sig, frac=0.4):
    """centers of level shifts that stand out above the signal's own activity,
    measured over a 40ms window."""
    h = int(0.02*FS); j = np.zeros_like(sig); j[h:-h] = sig[2*h:]-sig[:-2*h]
    thr = max(2*np.std(np.diff(sig))*np.sqrt(2*h), 0.20*np.std(sig))
    big = np.abs(j) > thr; ev, i = [], 0
    while i < len(big):
        if big[i]:
            k = i
            while k < len(big) and big[k]: k += 1
            ev.append((i+k)//2 - h); i = k + int(0.08*FS)
        else: i += 1
    return ev

def link_lag(src, dst, events, pre=0.06, post=0.16):
    Wp, Wq = int(pre*FS), int(post*FS); out = []
    for e in events:
        if e-Wp < 0 or e+Wq >= len(dst): continue
        sb = np.median(src[e-Wp:e]); sf = np.median(src[e+Wp:e+2*Wp]) if e+2*Wp < len(src) else src[e]
        sc = sf - sb
        if abs(sc) < 1e-9: continue
        # src 50% time
        ss = src[e-Wp:e+Wq]-sb; sc50 = np.where(ss*np.sign(sc) >= 0.5*abs(sc))[0]
        db = np.median(dst[e-Wp:e]); df = np.median(dst[e+Wq-Wp:e+Wq]); dc = df-db
        if abs(dc) < 4*np.std(dst[e-Wp:e])+1e-9: continue
        if np.sign(dc) != np.sign(sc): continue
        ds = dst[e-Wp:e+Wq]-db; dc50 = np.where(ds*np.sign(dc) >= 0.5*abs(dc))[0]
        if len(sc50) and len(dc50):
            out.append((dc50[0]-sc50[0])/FS)
    return out

def report(name, src, dst, events):
    L = link_lag(src, dst, events)
    if len(L) < 3:
        print(f"  {name:28} n/a ({len(L)} clean steps)"); return
    L = np.array(L)*1000
    print(f"  {name:28} median {np.median(L):5.1f} ms   IQR [{np.percentile(L,25):.0f},{np.percentile(L,75):.0f}]   n={len(L)}")

# THROTTLE: cleanest, largest-amplitude command -> best transport proxy
te = step_events(rs("cmd_ch02_us"))
print(f"\nTHROTTLE chain  ({len(te)} command steps detected)")
report("cmd -> rcCommand (transport)", rs("cmd_ch02_us"), rs("bb_rcCommand_3"), te)
report("cmd -> motor (full to actuator)", rs("cmd_ch02_us"), mot, te)
report("rcCommand -> motor (PID)", rs("bb_rcCommand_3"), mot, te)

# ATTITUDE axes: command -> actual rotation (gyro)
for nm, c, rc, sp, gy in [("ROLL","cmd_ch00_us","bb_rcCommand_0","bb_setpoint_0","bb_gyroADC_0"),
                          ("PITCH","cmd_ch01_us","bb_rcCommand_1","bb_setpoint_1","bb_gyroADC_1")]:
    ev = step_events(rs(c), 0.35)
    print(f"\n{nm} chain  ({len(ev)} command steps)")
    report("cmd -> rcCommand (transport)", rs(c), rs(rc), ev)
    report("setpoint -> gyro (loop+frame)", rs(sp), rs(gy), ev)
    report("cmd -> gyro (END-TO-END)", rs(c), rs(gy), ev)
