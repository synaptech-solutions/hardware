#!/usr/bin/env python3
"""Latency budget that AVOIDS the cmd<->rcCommand clock-sync circularity.

The standard sync (combine_flight) aligns the laptop command stream to the FC
rcCommand echo by cross-correlation. But rcCommand IS cmd after the uplink, so
that alignment folds the uplink latency into the clock offset -- you can no
longer read the uplink latency back out of the synced file. (This flight's fit
pushed the blackbox -156 ms; some of that is real clock offset, some is the
uplink we want to measure.)

Way out: only ever cross-correlate signals that share ONE clock, so the lag is a
pure physical latency with no clock-offset term mixed in. We use two single-clock
legs that share the same physical endpoint (the drone's actual yaw motion):

  LEG A  cmd_yaw  -> Vicon yaw-rate     (both LAPTOP clock)  = L_up + tau_resp + eps_vicon
  LEG B  rcCommand_yaw -> gyro yaw-rate (both FC clock)      = tau_resp
         (rcCommand & gyro are both bb_* -> remapped by the SAME map -> their
          relative timing is intact even though their absolute t was shifted)

  => uplink transport  L_up (+ Vicon pipeline latency eps_vicon) = LEG A - LEG B

Vicon is the neutral witness: it observes the motion directly, with NO RC link
between the motion and the observation, so it never inherits the uplink latency.
That is exactly why the pycode_Vicon yaw-sync works -- here we reuse the same
yaw-rate signal for latency instead of just clock offset.

Also reports whether the Vicon-FREE version (cmd_yaw -> attitude/IMU telemetry,
both laptop clock) is feasible on this link, by printing the downlink rates.

Usage:  roundtrip_latency.py [SESSION_DIR]   (default: newest with flight_synced.csv)
"""
import csv
import os
import sys
import numpy as np
import scipy.signal as sig
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
REC = os.path.join(os.path.dirname(HERE), "recordings")
FS = 200.0                     # uniform grid for cross-correlation (Hz)
MAX_LAG_S = 0.30               # physical latencies are well under this
LP_HZ = 4.0                    # low-pass before xcorr: yaw maneuvers are ~1 Hz;
                               # this keeps signal but kills the Vicon orientation
                               # differentiation noise that otherwise buries it.
MIN_CORR = 0.50                # below this a recovered lag is noise, not a latency


def _newest_session():
    cands = [os.path.join(REC, d) for d in os.listdir(REC)
             if os.path.isfile(os.path.join(REC, d, "flight_synced.csv"))]
    if not cands:
        sys.exit("no session with flight_synced.csv found")
    return max(cands, key=os.path.getmtime)


def _resolve(arg):
    """Accept an absolute path, a 'recordings/<stamp>' path, or a bare <stamp>."""
    for cand in (arg, os.path.join(REC, os.path.basename(arg.rstrip("/")))):
        if os.path.isfile(os.path.join(cand, "flight_synced.csv")):
            return os.path.abspath(cand)
    sys.exit(f"no flight_synced.csv under {arg!r}")


def _lowpass(x, fc=LP_HZ, fs=FS):
    b, a = sig.butter(2, fc / (fs / 2.0))
    return sig.filtfilt(b, a, x)                 # zero-phase -> does NOT bias lag


def load_synced(path, names):
    """Read just the wanted columns from flight_synced.csv -> {name: float array}."""
    with open(path) as fh:
        rd = csv.reader(fh)
        h = next(rd)
        idx = {n: h.index(n) for n in names if n in h}
        out = {n: [] for n in idx}
        for row in rd:
            for n, i in idx.items():
                v = row[i]
                out[n].append(float(v) if v not in ("", "nan", "NaN") else np.nan)
    return {n: np.asarray(v, float) for n, v in out.items()}


def find_lag(a, b, fs=FS, max_lag_s=MAX_LAG_S):
    """Delay of b relative to a, in seconds (positive => b happens AFTER a).

    Sign-robust (|peak|, so a flipped axis convention doesn't matter) with
    parabolic sub-sample refinement. Returns (lag_s, corr_coeff, sign)."""
    a = a - a.mean()
    b = b - b.mean()
    corr = sig.correlate(b, a, mode="full")
    lags = sig.correlation_lags(b.size, a.size, mode="full")
    sel = np.abs(lags) <= int(max_lag_s * fs)
    corr, lags = corr[sel], lags[sel]
    p = int(np.argmax(np.abs(corr)))
    sign = 1.0 if corr[p] >= 0 else -1.0
    y = sign * corr
    frac = 0.0
    if 0 < p < len(y) - 1:
        d = y[p - 1] - 2 * y[p] + y[p + 1]
        if d != 0:
            frac = float(np.clip(0.5 * (y[p - 1] - y[p + 1]) / d, -1, 1))
    lag = (lags[p] + frac) / fs
    coeff = float(np.abs(corr[p]) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return lag, coeff, sign


def vicon_body_yawrate(qx, qy, qz, qw, fs=FS):
    """BODY-frame yaw rate (rad/s) from the quaternion stream.

    NOT the d/dt of the zyx-Euler yaw -- that is the WORLD heading rate, which
    only equals the gyro's body-frame yaw rate when the drone is near level. For
    a drone banking around a large volume the two decouple, so we take the body
    z-component of angular velocity from the incremental rotation between frames:
        omega_body = rotvec( R_k^-1 * R_{k+1} ) * fs
    """
    quat = np.column_stack([qx, qy, qz, qw])
    n = np.linalg.norm(quat, axis=1)
    quat[n < 1e-6] = [0, 0, 0, 1]
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    R = Rotation.from_quat(quat)
    rotvec = (R[:-1].inv() * R[1:]).as_rotvec()
    wz = rotvec[:, 2] * fs
    return np.append(wz, wz[-1])                 # pad to length


def _fmt(lag, corr):
    """Lag in ms, flagged when the correlation is too weak to trust."""
    if corr < MIN_CORR:
        return f"  (untrustworthy: corr {corr:.2f} < {MIN_CORR}, signal too weak)"
    return f"{lag*1000:6.1f} ms   (corr {corr:.2f})"


def main():
    sess = _resolve(sys.argv[1]) if len(sys.argv) > 1 else _newest_session()
    csvp = os.path.join(sess, "flight_synced.csv")
    print(f"Session: {sess}   (low-pass {LP_HZ:.0f} Hz before xcorr)")

    d = load_synced(csvp, ["Abs_time", "cmd_ch03_us",
                           "bb_rcCommand_2", "bb_gyroADC_2",
                           "b1_qx", "b1_qy", "b1_qz", "b1_qw"])
    t = d["Abs_time"]
    have_vicon = all(k in d and np.isfinite(d[k]).any()
                     for k in ("b1_qx", "b1_qy", "b1_qz", "b1_qw"))

    finite = np.isfinite(d["cmd_ch03_us"]) & np.isfinite(d["bb_gyroADC_2"])
    t0, t1 = t[finite][0], t[finite][-1]
    g = np.arange(t0, t1, 1.0 / FS)

    def rs(name):
        y = d[name]
        m = np.isfinite(y)
        return _lowpass(np.interp(g, t[m], y[m]))

    cmd_yaw = rs("cmd_ch03_us")            # commanded yaw, LAPTOP clock
    rc_yaw = rs("bb_rcCommand_2")          # received yaw cmd, FC clock
    gyro_z = rs("bb_gyroADC_2")            # actual yaw rate, FC clock

    # FC-INTERNAL (one clock: FC): rcCommand -> gyro = PID + actuate + body response.
    tau_resp, cB, _ = find_lag(rc_yaw, gyro_z)

    print("\n--- can we measure latency WITHOUT the RC link in the path? ---")
    for typ in ("attitude", "imu"):
        tt = []
        for r in csv.DictReader(open(os.path.join(sess, "telemetry.csv"))):
            if r["type"] == typ:
                key = "att_yaw_deg" if typ == "attitude" else "imu_gz_dps"
                if r.get(key, "") not in ("", "nan"):
                    tt.append(float(r["t_rel"]))
        tt = np.array(tt)
        rate = 1.0 / np.median(np.diff(tt)) if tt.size > 2 else 0.0
        verdict = ("USABLE" if rate > 50 else
                   "marginal" if rate > 15 else "TOO SLOW for ms-scale timing")
        print(f"  downlink {typ:9} {tt.size:4d} frames  ~{rate:5.1f} Hz   -> {verdict}")
    print(f"  vicon (laptop clock)      ~{1/np.median(np.diff(t)):.0f} Hz"
          f"        -> {'present' if have_vicon else 'ABSENT'}")

    print("\n--- FC-internal control latency (clean: single FC clock) ---")
    print(f"  rcCommand -> gyro (PID+actuate+body)  {_fmt(tau_resp, cB)}")

    print("\n--- the circularity, made visible (on the synced grid) ---")
    cmd_to_gyro, cC, _ = find_lag(cmd_yaw, gyro_z)
    print(f"  cmd -> gyro                           {_fmt(cmd_to_gyro, cC)}")
    print(f"  rcCommand -> gyro                     {_fmt(tau_resp, cB)}")
    print("  cmd and rcCommand give the SAME lag because the standard sync glued")
    print("  them together -> the uplink was absorbed into the clock offset and is")
    print("  NOT readable from this file. (This is the problem you identified.)")

    print("\n--- recover the uplink: needs a witness OUTSIDE the link ---")
    if have_vicon:
        vic_rate = vicon_body_yawrate(d["b1_qx"], d["b1_qy"],
                                      d["b1_qz"], d["b1_qw"])
        # resample the body-rate (already a derivative) onto the grid + low-pass
        vic_rate = _lowpass(np.interp(g, t, vic_rate))
        witness, cW, _ = find_lag(gyro_z, vic_rate)   # same physical yaw, two sensors
        print(f"  Vicon-yaw vs gyro-yaw agreement: corr {cW:.2f}"
              f"  ({'usable' if cW >= MIN_CORR else 'TOO NOISY — yaw poorly tracked'})")
        if cW >= MIN_CORR:
            cmd_to_motion, cA, _ = find_lag(cmd_yaw, vic_rate)
            print(f"  cmd -> Vicon motion (L_up+PID+body)   {_fmt(cmd_to_motion, cA)}")
            print(f"  => uplink L_up (+ Vicon pipeline lag) "
                  f"{(cmd_to_motion - tau_resp)*1000:6.1f} ms")
        else:
            print("  -> Vicon position is excellent but its YAW is too noisy here to")
            print("     serve as the witness (smallest, least-observable axis on a")
            print("     whoop marker tree). Can't recover the uplink from this flight.")
    else:
        print("  no Vicon in this session, and telemetry downlink is too slow ->")
        print("  the uplink is not recoverable from this flight's data.")

    print("\n--- to actually measure the uplink, next flight needs ONE of: ---")
    print("  (a) clean Vicon yaw (better marker tree / orientation tracking), or")
    print("  (b) raise the CRSF telemetry ratio so attitude/IMU return >=50 Hz,")
    print("      then: uplink = (cmd->motion, one clock) - (rcCommand->gyro).")


if __name__ == "__main__":
    main()
