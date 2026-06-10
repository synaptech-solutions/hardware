"""Deterministically merge one flight's laptop logs + Vicon pose + FC blackbox
into a single wide CSV on the Vicon clock.

Sources (one session folder + the separately-downloaded FC blackbox):
  - vicon.mat      Vicon pose @100 Hz; Abs_time is already t_rel from trigger t0
  - commands.csv   outgoing RC frames (all 16 ch + joystick), stamped t_rel
  - telemetry.csv  incoming CRSF telemetry, stamped t_rel (laptop receive time)
  - <log>.bbl      FC blackbox, dropped into data_logging/blackbox/

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
  combine_flight.py                       # newest session + newest .bbl
  combine_flight.py SESSION_DIR           # that session + newest .bbl
  combine_flight.py SESSION_DIR LOG.bbl   # explicit
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
    dirs = [d for d in glob.glob(os.path.join(RECORDINGS, "*")) if os.path.isdir(d)
            and os.path.exists(os.path.join(d, "vicon.mat"))]
    if not dirs:
        raise FileNotFoundError(f"no session with vicon.mat in {RECORDINGS}/")
    return max(dirs, key=os.path.getmtime)


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
                    help="seconds the FC log lags the laptop trigger (laptop->FC "
                         "latency); blackbox query time = vicon Abs_time - offset")
    ap.add_argument("--decoder", default=DEFAULT_DECODER, help="path to blackbox_decode")
    args = ap.parse_args()

    session = args.session or newest_session()
    vicon_mat = os.path.join(session, "vicon.mat")
    if not os.path.isfile(vicon_mat):
        sys.exit(f"no vicon.mat in {session}")
    bbl = args.bbl or _newest(BLACKBOX_DIR, "*.bbl")
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
    print(f"Vicon:    {vicon_mat}")
    print(f"Blackbox: {bbl}")
    print(f"Motors:   {args.poles} poles  ->  RPM = eRPM_field x {ERPM_FIELD_SCALE:.0f}"
          f" / {int(pole_pairs)}   |   offset {args.offset:+.3f}s\n")

    with tempfile.TemporaryDirectory() as tmp:
        bb_t, bb_cols = load_blackbox(decode_bbl(bbl, args.decoder, tmp))
    vc = load_vicon(vicon_mat)

    tq = vc["Abs_time"]                 # the master clock everything lands on
    n = tq.size

    # Blackbox is on the FC clock: zero-base to its first sample (= the flick) and
    # query at Abs_time - offset (the laptop->FC link latency). Laptop streams
    # (commands/telemetry) share Abs_time directly, so they use offset 0.
    bb_rel = bb_t - bb_t[0]
    bb_q = tq - args.offset
    bb_lo, bb_hi = bb_rel[0], bb_rel[-1]
    in_cov = (bb_q >= bb_lo) & (bb_q <= bb_hi)

    cols = {"Abs_time": tq}

    # Pose + velocity (central differences on the Vicon clock).
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

    # Outgoing commands -> cmd_<name> (zero-order hold; same laptop clock as Vicon).
    n_cmd = 0
    cmd_path = os.path.join(session, "commands.csv")
    if os.path.isfile(cmd_path):
        ct, cser = load_commands(cmd_path)
        n_cmd = int(ct.size)
        if n_cmd:
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
    print(f"Merged {n} Vicon samples; {cov} ({100*cov/max(n,1):.1f}%) overlap the "
          f"blackbox log ({bb_rel[-1]:.1f}s).  {len(bb_cols)} blackbox channels, "
          f"{n_cmd} command frames, {n_tlm_ch} telemetry channels.")
    if cov and "motor_rpm_0" in cols:
        rv = np.column_stack([cols[f"motor_rpm_{m}"] for m in range(4)])[in_cov]
        print(f"Motor RPM over overlap: min {np.nanmin(rv):.0f}  max {np.nanmax(rv):.0f}"
              f"  mean {np.nanmean(rv):.0f}")
    if cov < n:
        print("  note: samples outside blackbox coverage are blank (Vicon ran longer "
              "than the FC log — expected near the very start/end).")

    _write_csv(out, cols)

    meta_out_data = {
        "sync_method": "deterministic-shared-trigger",
        "sync_offset_s": args.offset,
        "motor_poles": args.poles,
        "blackbox_file": os.path.basename(bbl),
        "vicon_file": "vicon.mat",
        "commands_file": "commands.csv" if os.path.isfile(cmd_path) else "",
        "telemetry_file": "telemetry.csv" if os.path.isfile(tlm_path) else "",
        "telemetry_raw_file": "telemetry_raw.csv",
        "session": os.path.basename(session),
        "video_file": meta.get("video", {}).get("file", ""),
        "video_frames_file": "video_frames.csv",
        "video_start_offset_s": meta.get("video", {}).get("start_offset_s", 0.0) or 0.0,
        "t0_human": meta.get("t0_human", ""),
        "n_samples": int(n),
        "blackbox_span_s": float(bb_rel[-1]),
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
