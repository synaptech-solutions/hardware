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


def load_blackbox(csv_path):
    """Parse decoded CSV → dict of float arrays: t, erpm0..3, motor0..3."""
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]

        def find(prefix):
            for i, name in enumerate(header):
                if name.startswith(prefix):
                    return i
            raise KeyError(f"column {prefix!r} not in {csv_path}")

        idx = {"t": find("time"),
               **{f"erpm{m}": find(f"eRPM[{m}]") for m in range(4)},
               **{f"motor{m}": find(f"motor[{m}]") for m in range(4)}}
        cols = {k: [] for k in idx}
        for row in reader:
            if not row:
                continue
            for k, i in idx.items():
                cell = row[i].strip() if i < len(row) else ""
                cols[k].append(float(cell) if cell else np.nan)

    out = {k: np.asarray(v, float) for k, v in cols.items()}
    # np.interp needs strictly increasing x.
    t = out["t"]
    keep = np.concatenate(([True], np.diff(t) > 0))
    if not keep.all():
        for k in out:
            out[k] = out[k][keep]
    return out


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
        bb = load_blackbox(decode_bbl(bbl, args.decoder, tmp))
    vc = load_vicon(vicon_mat)

    # Deterministic alignment: zero-base blackbox to its first sample (= trigger),
    # query at vicon Abs_time − offset. Outside blackbox coverage → NaN.
    bb_rel = bb["t"] - bb["t"][0]
    tq = vc["Abs_time"] - args.offset
    in_cov = (tq >= bb_rel[0]) & (tq <= bb_rel[-1])

    def interp(y):
        o = np.interp(tq, bb_rel, y)
        o[~in_cov] = np.nan
        return o

    n = vc["Abs_time"].size
    motor_erpm = np.column_stack([interp(bb[f"erpm{m}"]) * ERPM_FIELD_SCALE for m in range(4)])
    motor_rpm = np.column_stack([interp(bb[f"erpm{m}"]) * field_to_rpm for m in range(4)])
    motor_cmd = np.column_stack([interp(bb[f"motor{m}"]) for m in range(4)])

    # Velocity from pose (convenience), central differences on the vicon clock.
    def vel(p):
        return np.gradient(p, vc["Abs_time"]) if p.size > 1 else np.zeros_like(p)

    cov = int(in_cov.sum())
    print(f"Merged {n} Vicon samples; {cov} ({100*cov/max(n,1):.1f}%) overlap the "
          f"blackbox log ({bb_rel[-1]:.1f}s).")
    if cov:
        rv = motor_rpm[in_cov]
        print(f"Motor RPM over overlap: min {np.nanmin(rv):.0f}  max {np.nanmax(rv):.0f}"
              f"  mean {np.nanmean(rv):.0f}")
    if cov < n:
        print("  note: samples outside blackbox coverage are NaN (Vicon ran longer "
              "than the FC log — expected near the very start/end).")

    merged = {
        "Abs_time": vc["Abs_time"],
        "b1_x": vc["b1_x"], "b1_y": vc["b1_y"], "b1_z": vc["b1_z"],
        "b1_qx": vc["b1_qx"], "b1_qy": vc["b1_qy"], "b1_qz": vc["b1_qz"], "b1_qw": vc["b1_qw"],
        "b1_vx": vel(vc["b1_x"]), "b1_vy": vel(vc["b1_y"]), "b1_vz": vel(vc["b1_z"]),
        "motor_rpm": motor_rpm,      # (N,4) mechanical RPM, motors 0..3
        "motor_erpm": motor_erpm,    # (N,4) electrical RPM
        "motor_cmd": motor_cmd,      # (N,4) commanded throttle
        # provenance
        "sync_method": "deterministic-shared-trigger",
        "sync_offset_s": args.offset,
        "motor_poles": args.poles,
        "blackbox_file": os.path.basename(bbl),
        "vicon_file": "vicon.mat",
        "session": os.path.basename(session),
        "video_file": meta.get("video", {}).get("file", ""),
        "video_start_offset_s": meta.get("video", {}).get("start_offset_s", 0.0) or 0.0,
        "t0_human": meta.get("t0_human", ""),
    }
    sio.savemat(out, merged)
    print(f"\nSaved merged flight log: {out}")
    vid = meta.get("video", {}).get("file")
    if vid and os.path.isfile(os.path.join(session, vid)):
        off = merged["video_start_offset_s"]
        print(f"Video alongside: {os.path.join(session, vid)}  "
              f"(starts {off:+.2f}s relative to data t=0)")


if __name__ == "__main__":
    main()
