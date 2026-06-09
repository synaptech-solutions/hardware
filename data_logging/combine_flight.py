"""Deterministically merge one flight's Vicon pose + Betaflight blackbox.

No cross-correlation. The blackbox switch starts the FC log, the laptop video,
and the laptop Vicon recording at the same instant, so each stream's FIRST
sample marks that shared trigger:

  - Vicon  `Abs_time` is already measured from the trigger t0 (0 at the flick).
  - Blackbox `time` is FC-uptime; we zero-base it to its first sample, which is
    the moment logging started ≈ the same flick.

So the two align directly (Vicon Abs_time ≈ blackbox_time − blackbox_time[0]).
The only residual is the laptop→FC link latency (the FC starts logging ~one RC
frame after the laptop starts Vicon); pass --offset to nudge it out if needed.
Motor eRPM/RPM/cmd are interpolated onto every Vicon timestep → one merged file
with pose + motor data per timestep. The session's video.mp4 sits alongside it
(its t0 offset is in session.json).

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
    """Blackbox header → a clean .mat field name: drop the unit suffix, turn
    `gyroADC[0] (rad/s)` into `gyroADC_0`, `rcCommand[3]` into `rcCommand_3`."""
    name = col_name.split("(")[0].strip()          # drop " (rad/s)" etc.
    name = name.replace("[", "_").replace("]", "")
    name = re.sub(r"[^0-9A-Za-z_]", "_", name)
    return name


def load_blackbox(csv_path):
    """Parse the decoded CSV → (t, cols) with EVERY column kept.

    `t` is the (seconds) time base; `cols` maps a sanitized name (e.g.
    `gyroADC_0`, `motor_2`, `eRPM_1`, `axisP_0`) to its float array. This is how
    the full blackbox makes it into the merged file — angular rate, accel, PID
    terms, setpoints, debug, battery, flags, etc., not just the motors."""
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
                except ValueError:        # text flag columns (e.g. 'IDLE') → NaN
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


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", nargs="?", help="session dir (default: newest)")
    ap.add_argument("bbl", nargs="?", help="blackbox .bbl (default: newest in blackbox/)")
    ap.add_argument("--out", help="output file (default: <session>/flight_synced.mat)")
    ap.add_argument("--poles", type=int, default=12, help="motor pole count (default 12)")
    ap.add_argument("--offset", type=float, default=0.0,
                    help="seconds the FC log lags the laptop trigger (laptop→FC "
                         "latency); blackbox query time = vicon Abs_time − offset")
    ap.add_argument("--decoder", default=DEFAULT_DECODER, help="path to blackbox_decode")
    args = ap.parse_args()

    session = args.session or newest_session()
    vicon_mat = os.path.join(session, "vicon.mat")
    if not os.path.isfile(vicon_mat):
        sys.exit(f"no vicon.mat in {session}")
    bbl = args.bbl or _newest(BLACKBOX_DIR, "*.bbl")
    out = args.out or os.path.join(session, "flight_synced.mat")

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
    print(f"Motors:   {args.poles} poles  →  RPM = eRPM_field × {ERPM_FIELD_SCALE:.0f}"
          f" / {int(pole_pairs)}   |   offset {args.offset:+.3f}s\n")

    with tempfile.TemporaryDirectory() as tmp:
        bb_t, bb_cols = load_blackbox(decode_bbl(bbl, args.decoder, tmp))
    vc = load_vicon(vicon_mat)

    # Deterministic alignment: zero-base blackbox to its first sample (= trigger),
    # query at vicon Abs_time − offset. Outside blackbox coverage → NaN.
    bb_rel = bb_t - bb_t[0]
    tq = vc["Abs_time"] - args.offset
    in_cov = (tq >= bb_rel[0]) & (tq <= bb_rel[-1])

    def interp(y):
        o = np.interp(tq, bb_rel, y)
        o[~in_cov] = np.nan
        return o

    n = vc["Abs_time"].size

    # Velocity from pose (convenience), central differences on the vicon clock.
    def vel(p):
        return np.gradient(p, vc["Abs_time"]) if p.size > 1 else np.zeros_like(p)

    merged = {
        "Abs_time": vc["Abs_time"],
        "b1_x": vc["b1_x"], "b1_y": vc["b1_y"], "b1_z": vc["b1_z"],
        "b1_qx": vc["b1_qx"], "b1_qy": vc["b1_qy"], "b1_qz": vc["b1_qz"], "b1_qw": vc["b1_qw"],
        "b1_vx": vel(vc["b1_x"]), "b1_vy": vel(vc["b1_y"]), "b1_vz": vel(vc["b1_z"]),
    }

    # EVERY blackbox column, interpolated onto the vicon clock as bb_<name>.
    for name, y in bb_cols.items():
        merged["bb_" + name] = interp(y)

    # Derived (N,4) motor arrays on top of the raw bb_eRPM_*/bb_motor_* columns:
    # mechanical RPM (eRPM field × 100 / pole_pairs), electrical RPM, and the
    # commanded output. Kept so plot_flight.py's PNG panels still work.
    if all(f"eRPM_{m}" in bb_cols for m in range(4)):
        merged["motor_rpm"] = np.column_stack(
            [interp(bb_cols[f"eRPM_{m}"]) * field_to_rpm for m in range(4)])
        merged["motor_erpm"] = np.column_stack(
            [interp(bb_cols[f"eRPM_{m}"]) * ERPM_FIELD_SCALE for m in range(4)])
    if all(f"motor_{m}" in bb_cols for m in range(4)):
        merged["motor_cmd"] = np.column_stack(
            [interp(bb_cols[f"motor_{m}"]) for m in range(4)])

    merged.update({
        "sync_method": "deterministic-shared-trigger",
        "sync_offset_s": args.offset,
        "motor_poles": args.poles,
        "blackbox_file": os.path.basename(bbl),
        "vicon_file": "vicon.mat",
        "session": os.path.basename(session),
        "video_file": meta.get("video", {}).get("file", ""),
        "video_start_offset_s": meta.get("video", {}).get("start_offset_s", 0.0) or 0.0,
        "t0_human": meta.get("t0_human", ""),
    })

    cov = int(in_cov.sum())
    print(f"Merged {n} Vicon samples; {cov} ({100*cov/max(n,1):.1f}%) overlap the "
          f"blackbox log ({bb_rel[-1]:.1f}s).  {len(bb_cols)} blackbox channels carried.")
    if cov and "motor_rpm" in merged:
        rv = merged["motor_rpm"][in_cov]
        print(f"Motor RPM over overlap: min {np.nanmin(rv):.0f}  max {np.nanmax(rv):.0f}"
              f"  mean {np.nanmean(rv):.0f}")
    if cov < n:
        print("  note: samples outside blackbox coverage are NaN (Vicon ran longer "
              "than the FC log — expected near the very start/end).")

    sio.savemat(out, merged)
    print(f"\nSaved merged flight log: {out}")
    vid = meta.get("video", {}).get("file")
    if vid and os.path.isfile(os.path.join(session, vid)):
        off = merged["video_start_offset_s"]
        print(f"Video alongside: {os.path.join(session, vid)}  "
              f"(starts {off:+.2f}s relative to data t=0)")


if __name__ == "__main__":
    main()
