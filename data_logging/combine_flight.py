"""Deterministically merge one flight's laptop logs + Vicon pose + FC blackbox
into a single wide CSV on the Vicon clock.

Everything lives in the one session folder (recordings/<stamp>/):
  - vicon.mat      Vicon pose @100 Hz; Abs_time is already t_rel from trigger t0
  - commands.csv   outgoing RC frames (all 16 ch + joystick), stamped t_rel
  - telemetry.csv  incoming CRSF telemetry, stamped t_rel (laptop receive time)
  - blackbox/<log>.bbl   FC blackbox — drop the flight's .bbl into the session's
                   own blackbox/ subfolder (auto-created each session; kept with the
                   flight). The session root and the legacy shared data_logging/
                   blackbox/ are checked as fallbacks.

No cross-correlation. The blackbox switch starts every stream at the same flick,
so they share the trigger t0:
  - The laptop streams (vicon / commands / telemetry) all carry t_rel = wall - t0
    directly, so they align to the Vicon clock with ZERO offset.
  - Blackbox `time` is FC-uptime; we zero-base it to its first sample (= the
    flick). The only slack is the laptop->FC link latency, so ONLY the blackbox
    is queried at (Abs_time - offset). Pass --offset to nudge it, or measure it
    by cross-correlating cmd_ch00_us vs bb_rcCommand_0 in the output.

Everything is interpolated onto the Vicon Abs_time grid:
  - commands  -> cmd_*  (zero-order hold: they are held setpoints / flags)
  - telemetry -> tlm_*  (linear for measurements, hold for categorical fields)
  - blackbox  -> bb_*   (linear), plus derived motor_rpm_*/erpm_*/cmd_*

Output: <session>/flight_synced.csv  (+ flight_synced.meta.json for the scalars
and the column list). The session's video.mp4 + video_frames.csv sit alongside;
the video's t0 offset is in session.json / the meta file.

Usage:
  combine_flight.py                       # newest session + its own .bbl
  combine_flight.py SESSION_DIR           # that session + the .bbl in it
  combine_flight.py SESSION_DIR LOG.bbl   # explicit .bbl
  combine_flight.py --poles 12 --offset 0.015   # motor poles; latency nudge (s)
"""
import os
import re
import sys
import csv
import glob
import json
import argparse
import tempfile
import subprocess

import numpy as np
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
RECORDINGS = os.path.join(HERE, "recordings")
BLACKBOX_DIR = os.path.join(HERE, "blackbox")
# blackbox_decode built from Betaflight blackbox-tools (lives in pycode_ViCON).
DEFAULT_DECODER = os.path.join(
    os.path.dirname(HERE), "pycode_ViCON",
    "tools", "blackbox-tools", "obj", "blackbox_decode")

# Betaflight stores the blackbox eRPM field in units of 100 eRPM.
ERPM_FIELD_SCALE = 100.0

# Telemetry channels that are real measurements (linear-interp onto the clock)
# vs. categorical/discrete ones that must be zero-order-held (a fractional
# rf_mode is meaningless). flight_mode / device_* are strings and stay in the raw
# telemetry.csv — they are not carried into the numeric merged file.
TLM_LINEAR = {"att_pitch_deg", "att_roll_deg", "att_yaw_deg",
              "bat_v", "bat_a", "bat_mah", "bat_pct",
              "up_lq", "dn_lq", "up_rssi_dbm", "dn_rssi_dbm",
              "up_snr_db", "dn_snr_db",
              "imu_ax_g", "imu_ay_g", "imu_az_g",
              "imu_gx_dps", "imu_gy_dps", "imu_gz_dps", "imu_mag"}
TLM_HOLD = {"rf_mode", "active_ant", "up_tx_pwr_idx"}


def _newest(folder, pattern):
    hits = glob.glob(os.path.join(folder, pattern))
    if not hits:
        raise FileNotFoundError(f"no {pattern} in {folder}/")
    return max(hits, key=os.path.getmtime)


def newest_session():
    # A session is any recordings/* dir with a session.json (vicon.mat may be
    # absent if that flight was flown without Vicon).
    dirs = [d for d in glob.glob(os.path.join(RECORDINGS, "*")) if os.path.isdir(d)
            and os.path.exists(os.path.join(d, "session.json"))]
    if not dirs:
        raise FileNotFoundError(f"no session with session.json in {RECORDINGS}/")
    return max(dirs, key=os.path.getmtime)


def find_session_bbl(session):
    """Find this flight's blackbox .bbl kept WITH the flight: in its own
    blackbox/ subfolder (the layout the session creates), then the session root,
    then the legacy shared data_logging/blackbox/ as a last resort. Picks the
    largest .bbl when several were dropped in (e.g. btfl_all + btfl_001)."""
    for folder, label in ((os.path.join(session, "blackbox"), "blackbox/ subfolder"),
                          (session, "session folder")):
        hits = sorted(glob.glob(os.path.join(folder, "*.bbl")),
                      key=os.path.getsize, reverse=True)
        if hits:
            if len(hits) > 1:
                print(f"  note: {len(hits)} .bbl in the {label}; using the largest "
                      f"({os.path.basename(hits[0])}). One flight per .bbl keeps the "
                      "pairing unambiguous.")
            return hits[0]
    legacy = glob.glob(os.path.join(BLACKBOX_DIR, "*.bbl"))
    if legacy:
        b = max(legacy, key=os.path.getmtime)
        print(f"  note: no .bbl in {os.path.basename(session)}/blackbox/ — falling back "
              f"to the legacy shared blackbox/ ({os.path.basename(b)}). New workflow: "
              "drop the flight's .bbl into its session's blackbox/ subfolder.")
        return b
    raise FileNotFoundError(
        f"no .bbl found in {session}/blackbox/ (or {BLACKBOX_DIR}/). Copy the flight's "
        "blackbox .bbl into its session's blackbox/ subfolder, then re-run.")


def decode_bbl(bbl_path, decoder, out_dir):
    """Decode a .bbl to CSV (time in s); return the CSV path (largest session)."""
    if not os.path.isfile(decoder):
        raise FileNotFoundError(
            f"blackbox_decode not found at {decoder!r}. Build it (see "
            "pycode_ViCON/tools/blackbox-tools) or pass --decoder.")
    cmd = [decoder, "--unit-rotation", "rad/s", "--unit-frame-time", "s",
           "--output-dir", out_dir, bbl_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    csvs = [p for p in glob.glob(os.path.join(out_dir, "*.csv"))
            if "gps" not in os.path.basename(p).lower()]
    if not csvs:
        raise RuntimeError(f"blackbox_decode produced no CSV for {bbl_path!r}")
    csvs.sort(key=os.path.getsize, reverse=True)
    substantial = [p for p in csvs if os.path.getsize(p) > 50_000]
    if len(substantial) > 1:
        print(f"  note: {len(substantial)} flight sessions in this .bbl; using the "
              f"largest ({os.path.basename(csvs[0])}). One flight per download keeps "
              "the pairing unambiguous.")
    return csvs[0]


def _sanitize(col_name):
    """Blackbox header -> a clean field name: drop the unit suffix, turn
    `gyroADC[0] (rad/s)` into `gyroADC_0`, `rcCommand[3]` into `rcCommand_3`."""
    name = col_name.split("(")[0].strip()          # drop " (rad/s)" etc.
    name = name.replace("[", "_").replace("]", "")
    name = re.sub(r"[^0-9A-Za-z_]", "_", name)
    return name


def _mono(t, y):
    """Sort by t and drop non-increasing samples — np.interp / searchsorted need
    a strictly increasing x. Returns (t, y) cleaned."""
    if t.size == 0:
        return t, y
    order = np.argsort(t, kind="stable")
    t, y = t[order], y[order]
    keep = np.concatenate(([True], np.diff(t) > 0))
    return t[keep], y[keep]


def _to_float(cells):
    out = np.empty(len(cells), float)
    for i, c in enumerate(cells):
        try:
            out[i] = float(c)
        except (ValueError, TypeError):
            out[i] = np.nan
    return out


def _read_csv_columns(path):
    """Read a CSV -> (header list, {col_name: [str cells]}). Missing cells -> ''."""
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        cols = {h: [] for h in header}
        for row in r:
            if not row:
                continue
            for i, h in enumerate(header):
                cols[h].append(row[i] if i < len(row) else "")
    return header, cols


def load_blackbox(csv_path):
    """Parse the decoded CSV -> (t, cols) with EVERY column kept.

    `t` is the (seconds) time base; `cols` maps a sanitized name (e.g.
    `gyroADC_0`, `motor_2`, `eRPM_1`, `axisP_0`, `rcCommand_0`) to its float
    array — angular rate, accel, PID terms, setpoints, debug, battery, flags,
    etc., not just the motors. rcCommand_* is the ground truth for what the FC
    actually received (vs. what we sent in commands.csv)."""
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        names = [_sanitize(h) for h in header]
        ti = next((i for i, h in enumerate(header) if h.lower().startswith("time")), None)
        if ti is None:
            raise KeyError(f"no time column in {csv_path}")
        ncol = len(header)
        data = [[] for _ in range(ncol)]
        for row in reader:
            if not row:
                continue
            for i in range(ncol):
                cell = row[i].strip() if i < len(row) else ""
                try:
                    data[i].append(float(cell))
                except ValueError:        # text flag columns (e.g. 'IDLE') -> NaN
                    data[i].append(np.nan)

    t = np.asarray(data[ti], float)
    cols = {names[i]: np.asarray(data[i], float) for i in range(ncol) if i != ti}
    cols = {k: v for k, v in cols.items() if not np.isnan(v).all()}  # drop text-only cols
    # np.interp needs strictly increasing x.
    keep = np.concatenate(([True], np.diff(t) > 0))
    if not keep.all():
        t = t[keep]
        cols = {k: v[keep] for k, v in cols.items()}
    return t, cols


def load_vicon(mat_path):
    m = sio.loadmat(mat_path)
    get = lambda k: np.asarray(m[k]).ravel().astype(float)
    keys = ["Abs_time", "b1_x", "b1_y", "b1_z", "b1_qx", "b1_qy", "b1_qz", "b1_qw"]
    d = {k: get(k) for k in keys if k in m}
    if "Abs_time" not in d:
        raise KeyError(f"{mat_path} has no Abs_time field")
    return d


def load_commands(path):
    """commands.csv -> (t_rel, {col: float array}) for the 16 channels + the
    arm/record flags. The raw joystick axes/buttons stay in the file; the merged
    output carries what we SENT (channels) plus the flags."""
    header, cols = _read_csv_columns(path)
    t = _to_float(cols["t_rel"])
    order = np.argsort(t, kind="stable")
    t_sorted = t[order]
    keepinc = np.concatenate(([True], np.diff(t_sorted) > 0)) if t_sorted.size else \
        np.zeros(0, bool)
    keep = [c for c in header if c.startswith("ch") and c.endswith("_us")]
    keep += [c for c in ("armed", "record_on") if c in cols]
    series = {c: _to_float(cols[c])[order][keepinc] for c in keep}
    return t_sorted[keepinc], series


def load_telemetry(path):
    """telemetry.csv -> {col: (t_rel, value)} per numeric channel, taking only the
    rows where that channel was actually present (i.e. its frame type)."""
    _, cols = _read_csv_columns(path)
    t_all = _to_float(cols.get("t_rel", []))
    series = {}
    for c in (TLM_LINEAR | TLM_HOLD):
        if c not in cols:
            continue
        vals = _to_float(cols[c])
        m = ~np.isnan(vals)
        if m.any():
            series[c] = _mono(t_all[m], vals[m])
    return series


def _interp_lin(tq, t, y, lo, hi):
    """Linear interpolation; NaN outside [lo, hi] (the source's coverage)."""
    o = np.interp(tq, t, y)
    o[(tq < lo) | (tq > hi)] = np.nan
    return o


def _interp_hold(tq, t, y, lo, hi):
    """Zero-order hold (previous value); NaN outside [lo, hi]."""
    idx = np.clip(np.searchsorted(t, tq, side="right") - 1, 0, len(t) - 1)
    o = y[idx].astype(float)
    o[(tq < lo) | (tq > hi)] = np.nan
    return o


# laptop stick channel -> Betaflight rcCommand index (AETR vs RPYT). Yaw is omitted
# from drift estimation: it is logged sign-inverted vs the FC on this airframe, and
# its tiny amplitude makes it a poor reference anyway.
_DRIFT_PAIRS = [("ch00_us", "rcCommand_0"), ("ch01_us", "rcCommand_1"),
                ("ch02_us", "rcCommand_3")]


def estimate_bb_clock_map(cmd, bb_rel, bb_cols, fs=500.0, win_s=15.0, step_s=7.5):
    """Estimate the FC-clock -> laptop-clock map from the command echo.

    The laptop logs each RC frame it SENT (commands.csv, laptop clock); the FC logs
    what it RECEIVED (rcCommand, FC clock). They are the same signal, so the lag
    between them at any moment is the clock offset. The shared-trigger sync only
    pins t=0 -- the two crystals then drift (seen as a lag that grows linearly over
    the flight). We cross-correlate cmd vs rcCommand edges in sliding windows, fit a
    line offset(t)=slope*t+intercept, and return the correction so that

        laptop_time(bb_rel) = bb_rel*(1 - slope) - intercept

    Returns (slope_s_per_s, intercept_s, n_points, rmse_ms) or None if it can't get
    a confident fit (caller then falls back to a constant offset)."""
    if cmd is None:
        return None
    ct, cser = cmd
    if ct.size < 50:
        return None
    g = np.arange(max(ct[0], bb_rel[0]), min(ct[-1], bb_rel[-1]), 1.0 / fs)
    if g.size < int(2 * win_s * fs):
        return None

    def _xcorr_lag(a, b, max_lag_s=0.35):
        da, db = np.diff(a), np.diff(b)          # derivatives -> edge sensitive
        if da.std() < 1e-9 or db.std() < 1e-9:
            return None
        da = (da - da.mean()) / da.std(); db = (db - db.mean()) / db.std()
        full = np.correlate(db, da, "full")
        lags = np.arange(-(len(da) - 1), len(db))
        sel = np.abs(lags) <= int(max_lag_s * fs)
        full, lags = full[sel], lags[sel]
        k = int(np.argmax(full)); lag = float(lags[k]); q = full[k] / len(da)
        if 0 < k < len(full) - 1:                # parabolic refine
            y0, y1, y2 = full[k - 1], full[k], full[k + 1]; d = y0 - 2 * y1 + y2
            if abs(d) > 1e-12:
                lag += 0.5 * (y0 - y2) / d
        return lag / fs, q

    centers, offsets, weights = [], [], []
    for ch, rc in _DRIFT_PAIRS:
        if ch not in cser or rc not in bb_cols:
            continue
        c_g = _interp_hold(g, ct, cser[ch], ct[0], ct[-1])          # cmd: zero-order hold
        r_g = np.interp(g, bb_rel, bb_cols[rc])                     # rcCommand: linear
        x = g[0]
        while x + win_s <= g[-1]:
            w = (g >= x) & (g < x + win_s)
            res = _xcorr_lag(c_g[w], r_g[w])
            if res is not None and res[1] > 0.30:
                centers.append(x + win_s / 2.0)
                offsets.append(res[0] * 1000.0)                     # ms; +ve => rc lags cmd
                weights.append(res[1])
            x += step_s
    if len(centers) < 5:
        return None
    centers = np.asarray(centers); offsets = np.asarray(offsets); weights = np.asarray(weights)
    # weighted linear fit, one round of outlier rejection at 3*rmse
    for _ in range(2):
        A = np.polyfit(centers, offsets, 1, w=weights)
        resid = offsets - np.polyval(A, centers)
        rmse = float(np.sqrt(np.average(resid**2, weights=weights)))
        keep = np.abs(resid) <= max(3 * rmse, 15.0)
        if keep.all() or keep.sum() < 5:
            break
        centers, offsets, weights = centers[keep], offsets[keep], weights[keep]
    slope_ms, intercept_ms = float(A[0]), float(A[1])
    return slope_ms / 1000.0, intercept_ms / 1000.0, len(centers), rmse


def _write_csv(path, cols):
    """Write an ordered {name: 1-D array} dict as a wide CSV; NaN -> empty cell."""
    names = list(cols.keys())
    arr = np.column_stack([np.asarray(cols[k], float) for k in names])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(names)
        for row in arr:
            w.writerow(["" if np.isnan(v) else repr(float(v)) for v in row])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", nargs="?", help="session dir (default: newest)")
    ap.add_argument("bbl", nargs="?", help="blackbox .bbl (default: newest in blackbox/)")
    ap.add_argument("--out", help="output file (default: <session>/flight_synced.csv)")
    ap.add_argument("--poles", type=int, default=12, help="motor pole count (default 12)")
    ap.add_argument("--offset", type=float, default=0.0,
                    help="extra constant seconds the FC log lags the laptop trigger, "
                         "applied ON TOP of the auto drift correction")
    ap.add_argument("--no-drift-correct", action="store_true",
                    help="disable the cmd<->rcCommand clock-drift correction (revert to "
                         "the old constant-offset shared-trigger alignment)")
    ap.add_argument("--decoder", default=DEFAULT_DECODER, help="path to blackbox_decode")
    args = ap.parse_args()

    session = args.session or newest_session()
    vicon_mat = os.path.join(session, "vicon.mat")
    cmd_path = os.path.join(session, "commands.csv")
    # Vicon is preferred (it's the ground-truth pose + master clock), but optional:
    # if a flight was flown without Vicon, fall back to the commands.csv clock so
    # the rest of the streams still merge (no pose/trajectory in that case).
    if not os.path.isfile(vicon_mat) and not os.path.isfile(cmd_path):
        sys.exit(f"{session} has neither vicon.mat nor commands.csv — nothing to "
                 "build a timeline from.")
    # Blackbox is OPTIONAL: vicon + commands + telemetry all share the laptop t0
    # clock and merge without it (only the FC blackbox is on a separate clock). A
    # missing .bbl just omits the bb_* channels, so a flight still renders before
    # you've downloaded the log off the FC.
    try:
        bbl = args.bbl or find_session_bbl(session)
    except FileNotFoundError:
        bbl = None
    out = args.out or os.path.join(session, "flight_synced.csv")
    meta_out = os.path.splitext(out)[0] + ".meta.json"

    pole_pairs = args.poles / 2.0
    field_to_rpm = ERPM_FIELD_SCALE / pole_pairs

    meta = {}
    sj = os.path.join(session, "session.json")
    if os.path.isfile(sj):
        with open(sj) as f:
            meta = json.load(f)

    print(f"Session:  {session}")
    print(f"Vicon:    {vicon_mat if os.path.isfile(vicon_mat) else '(none — no pose)'}")
    print(f"Blackbox: {bbl if bbl else '(none — Vicon + commands + telemetry only)'}")
    print(f"Motors:   {args.poles} poles  ->  RPM = eRPM_field x {ERPM_FIELD_SCALE:.0f}"
          f" / {int(pole_pairs)}   |   offset {args.offset:+.3f}s\n")

    if bbl:
        with tempfile.TemporaryDirectory() as tmp:
            bb_t, bb_cols = load_blackbox(decode_bbl(bbl, args.decoder, tmp))
    else:
        bb_t, bb_cols = None, {}
    vc = load_vicon(vicon_mat) if os.path.isfile(vicon_mat) else None
    cmd = load_commands(cmd_path) if os.path.isfile(cmd_path) else None  # (ct, cser) | None

    # Master clock everything lands on: Vicon Abs_time if recorded, else the
    # commands.csv t_rel grid (same t0 clock). Both are zero-based at the trigger.
    if vc is not None:
        tq = vc["Abs_time"]
        master = "vicon"
    else:
        if cmd is None or cmd[0].size == 0:
            sys.exit("no vicon.mat and no usable commands.csv — cannot build a timeline.")
        tq = cmd[0]
        master = "commands"
        print("  note: no vicon.mat for this flight — using commands.csv as the master "
              "clock. This merge has NO pose/trajectory (Vicon wasn't recorded).")
    n = tq.size

    # Blackbox is on the FC clock: zero-base to its first sample (≈ the flick). The
    # FC and laptop crystals drift, so a single constant offset only aligns t≈0; we
    # map the FC clock onto the laptop clock via the cmd<->rcCommand echo (see
    # estimate_bb_clock_map). After mapping, bb_rel is in laptop-clock seconds and we
    # query it directly on the master grid. --offset is an extra constant nudge.
    if bb_t is not None:
        bb_rel = bb_t - bb_t[0]
        drift = None if args.no_drift_correct else \
            estimate_bb_clock_map(cmd, bb_rel, bb_cols)
        if drift is not None:
            slope, intercept, npts, rmse = drift
            bb_rel = bb_rel * (1.0 - slope) - intercept
            print(f"Clock drift: FC vs laptop = {slope*1000:+.3f} ms/s ({slope*100:+.3f}%), "
                  f"offset@t0 {intercept*1000:+.1f} ms  (fit over {npts} windows, rmse {rmse:.1f} ms)")
            print("  -> remapped blackbox onto the laptop clock (cmd<->rcCommand echo). "
                  "rcCommand should now land just AFTER its command.")
        else:
            print("Clock drift: not estimated (no command echo / low confidence) — using "
                  "constant shared-trigger alignment." + (
                      "" if args.no_drift_correct else " Pass --offset to nudge."))
        bb_q = tq - args.offset
        bb_lo, bb_hi = bb_rel[0], bb_rel[-1]
        in_cov = (bb_q >= bb_lo) & (bb_q <= bb_hi)
    else:
        bb_rel = None
        drift = None
        bb_q, bb_lo, bb_hi = tq, 0.0, 0.0
        in_cov = np.zeros(tq.shape, dtype=bool)

    cols = {"Abs_time": tq}

    # Pose + velocity (only when Vicon was recorded; central differences on tq).
    if vc is not None:
        for k in ("b1_x", "b1_y", "b1_z", "b1_qx", "b1_qy", "b1_qz", "b1_qw"):
            if k in vc:
                cols[k] = vc[k]
        for a in ("x", "y", "z"):
            if f"b1_{a}" in vc:
                p = vc[f"b1_{a}"]
                cols[f"b1_v{a}"] = np.gradient(p, tq) if p.size > 1 else np.zeros_like(p)

    # EVERY blackbox column -> bb_<name> (linear).
    for name, y in bb_cols.items():
        cols["bb_" + name] = _interp_lin(bb_q, bb_rel, y, bb_lo, bb_hi)

    # Derived motor arrays (flattened to _0.._3 columns for CSV).
    if all(f"eRPM_{m}" in bb_cols for m in range(4)):
        for m in range(4):
            e = bb_cols[f"eRPM_{m}"]
            cols[f"motor_rpm_{m}"] = _interp_lin(bb_q, bb_rel, e * field_to_rpm, bb_lo, bb_hi)
            cols[f"motor_erpm_{m}"] = _interp_lin(bb_q, bb_rel, e * ERPM_FIELD_SCALE, bb_lo, bb_hi)
    if all(f"motor_{m}" in bb_cols for m in range(4)):
        for m in range(4):
            cols[f"motor_cmd_{m}"] = _interp_lin(
                bb_q, bb_rel, bb_cols[f"motor_{m}"], bb_lo, bb_hi)

    # Outgoing commands -> cmd_<name>. If commands IS the master clock, the values
    # are already on tq (use as-is); otherwise zero-order-hold onto tq.
    n_cmd = 0
    if cmd is not None:
        ct, cser = cmd
        n_cmd = int(ct.size)
        if n_cmd:
            if master == "commands":
                for c, y in cser.items():
                    cols["cmd_" + c] = y
            else:
                clo, chi = ct[0], ct[-1]
                for c, y in cser.items():
                    cols["cmd_" + c] = _interp_hold(tq, ct, y, clo, chi)

    # Incoming telemetry -> tlm_<name> (linear measurements, hold for categorical).
    n_tlm_ch = 0
    tlm_path = os.path.join(session, "telemetry.csv")
    if os.path.isfile(tlm_path):
        tser = load_telemetry(tlm_path)
        n_tlm_ch = len(tser)
        for c, (tt, yy) in tser.items():
            if tt.size == 0:
                continue
            fn = _interp_hold if c in TLM_HOLD else _interp_lin
            cols["tlm_" + c] = fn(tq, tt, yy, tt[0], tt[-1])

    cov = int(in_cov.sum())
    span = bb_rel[-1] if bb_rel is not None else 0.0
    print(f"Merged {n} samples on the {master} clock; {cov} ({100*cov/max(n,1):.1f}%) "
          f"overlap the blackbox log ({span:.1f}s).  {len(bb_cols)} blackbox "
          f"channels, {n_cmd} command frames, {n_tlm_ch} telemetry channels.")
    if cov and "motor_rpm_0" in cols:
        rv = np.column_stack([cols[f"motor_rpm_{m}"] for m in range(4)])[in_cov]
        print(f"Motor RPM over overlap: min {np.nanmin(rv):.0f}  max {np.nanmax(rv):.0f}"
              f"  mean {np.nanmean(rv):.0f}")
    if bb_rel is not None and cov < n:
        print("  note: samples outside blackbox coverage are blank (the laptop streams "
              "ran longer than the FC log — expected near the very start/end).")

    _write_csv(out, cols)

    meta_out_data = {
        "sync_method": ("shared-trigger + cmd-rcCommand drift-correction"
                        if drift is not None else "deterministic-shared-trigger"),
        "master_clock": master,
        "has_pose": vc is not None,
        "sync_offset_s": args.offset,
        "clock_drift_slope_s_per_s": drift[0] if drift else 0.0,
        "clock_drift_intercept_s": drift[1] if drift else 0.0,
        "clock_drift_fit_windows": drift[2] if drift else 0,
        "clock_drift_rmse_ms": drift[3] if drift else None,
        "motor_poles": args.poles,
        "blackbox_file": os.path.basename(bbl) if bbl else "",
        "vicon_file": "vicon.mat" if vc is not None else "",
        "commands_file": "commands.csv" if os.path.isfile(cmd_path) else "",
        "telemetry_file": "telemetry.csv" if os.path.isfile(tlm_path) else "",
        "telemetry_raw_file": "telemetry_raw.csv",
        "session": os.path.basename(session),
        "video_file": meta.get("video", {}).get("file", ""),
        "video_frames_file": "video_frames.csv",
        "video_start_offset_s": meta.get("video", {}).get("start_offset_s", 0.0) or 0.0,
        "t0_human": meta.get("t0_human", ""),
        "n_samples": int(n),
        "blackbox_span_s": float(bb_rel[-1]) if bb_rel is not None else 0.0,
        "blackbox_coverage_samples": cov,
        "blackbox_coverage_frac": cov / max(n, 1),
        "n_command_frames": n_cmd,
        "n_telemetry_channels": n_tlm_ch,
        "columns": list(cols.keys()),
    }
    with open(meta_out, "w") as f:
        json.dump(meta_out_data, f, indent=2)

    print(f"\nSaved merged flight log: {out}")
    print(f"      + metadata/columns:  {meta_out}")
    vid = meta.get("video", {}).get("file")
    if vid and os.path.isfile(os.path.join(session, vid)):
        off = meta_out_data["video_start_offset_s"]
        print(f"Video alongside: {os.path.join(session, vid)}  "
              f"(starts {off:+.2f}s relative to data t=0; per-frame times in "
              f"video_frames.csv)")


if __name__ == "__main__":
    main()
