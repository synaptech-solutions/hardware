# Sync ViCON pose with Betaflight blackbox motor RPM into one merged file.
#
# The two streams are recorded on independent clocks (ViCON's Abs_time starts at
# 0 when the capture script launches; the blackbox `time` starts whenever logging
# armed and is *not* zero-based). We recover the constant offset between them by
# cross-correlating a signal both instruments observe: YAW RATE during a sharp
# in-air yaw maneuver. Yaw needs only a torque imbalance (differential motor RPM),
# not enough thrust to lift off, so both instruments register it simultaneously --
# this sidesteps the throttle "dead zone" where RPM rises but the drone can't move.
#
#   blackbox: gyroADC[2]  -> yaw rate, logged directly in rad/s
#   ViCON:    d/dt of the yaw Euler angle (from the quaternion) w.r.t. Abs_time
#
# Once the offset dt is known we shift the blackbox clock and interpolate the
# per-motor RPM onto every ViCON timestamp, producing one row per ViCON sample
# with pose + 4 motor RPMs.
#
# Usage:
#   python sync_log.py LOG.bbl POSE.mat                # explicit files
#   python sync_log.py                                 # newest .bbl + newest .mat
#   python sync_log.py LOG.bbl POSE.mat --out merged.mat
#   python sync_log.py LOG.bbl POSE.mat --poles 12     # motor pole count (default 12)
#   python sync_log.py LOG.bbl POSE.mat --plot         # also save a sync-check PNG
#
# eRPM is *electrical* RPM; mechanical RPM = eRPM / (poles / 2).

import os
import re
import sys
import csv
import glob
import argparse
import tempfile
import subprocess

import numpy as np
import scipy.io as sio
import scipy.signal as signal
from scipy.spatial.transform import Rotation

# blackbox_decode built from Betaflight blackbox-tools (see tools/blackbox-tools).
DEFAULT_DECODER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'tools', 'blackbox-tools', 'obj', 'blackbox_decode')

# Grid the two yaw-rate signals are resampled onto before cross-correlation.
# 1 kHz matches the blackbox rate and over-samples ViCON (~100 Hz); the lag is
# then refined to sub-sample precision by parabolic interpolation of the peak.
XCORR_FS = 1000.0

# Betaflight stores the blackbox eRPM field in units of 100 eRPM, so the decoded
# value is multiplied by this to recover true electrical RPM. (Confirmed against
# the RPM-filter debug[] motor frequencies on a real log, agreeing to ~0.5%.)
ERPM_FIELD_SCALE = 100.0


# --------------------------------------------------------------------------- #
# Calibrate the scipy cross-correlation lag convention with a known shift, so
# the sign of the recovered offset doesn't depend on the scipy version. With
# impulse-at-3 vs impulse-at-1, signal `a` is delayed by +2 samples relative to
# `b`; LAG_FACTOR maps the peak lag scipy reports back to that physical +2.
def _lag_factor():
    a = np.array([0, 0, 0, 1, 0, 0, 0, 0], float)
    b = np.array([0, 1, 0, 0, 0, 0, 0, 0], float)
    corr = signal.correlate(a, b, mode='full')
    lags = signal.correlation_lags(a.size, b.size, mode='full')
    peak_lag = lags[int(np.argmax(corr))]
    return 2.0 / peak_lag  # physical delay (+2) divided by reported lag -> +/-1


LAG_FACTOR = _lag_factor()


# --------------------------------------------------------------------------- #
def decode_bbl(bbl_path, decoder, out_dir):
    """Decode a .bbl to CSV (gyro in rad/s, time in s); return the CSV path.

    A .bbl may hold several arm/disarm sessions; the decoder writes one
    <name>.NN.csv per session. The production convention is one flight per file,
    so we pick the largest CSV and warn if more than one non-trivial session
    turns up.
    """
    if not os.path.isfile(decoder):
        raise FileNotFoundError(
            f'blackbox_decode not found at {decoder!r}. Build it (see '
            'tools/blackbox-tools) or pass --decoder.')

    cmd = [decoder, '--unit-rotation', 'rad/s', '--unit-frame-time', 's',
           '--output-dir', out_dir, bbl_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)

    csvs = [p for p in glob.glob(os.path.join(out_dir, '*.csv'))
            if 'gps' not in os.path.basename(p).lower()]
    if not csvs:
        raise RuntimeError(f'blackbox_decode produced no CSV for {bbl_path!r}')

    csvs.sort(key=os.path.getsize, reverse=True)
    substantial = [p for p in csvs if os.path.getsize(p) > 50_000]
    if len(substantial) > 1:
        print(f'  note: {len(substantial)} flight sessions in this .bbl; '
              f'using the largest ({os.path.basename(csvs[0])}). '
              'Production logs should hold one flight.')
    return csvs[0]


def _sanitize(col_name):
    """Blackbox header → a clean .mat field name: drop the unit suffix, turn
    `gyroADC[2] (rad/s)` into `gyroADC_2`, `rcCommand[3]` into `rcCommand_3`."""
    name = col_name.split('(')[0].strip()
    name = name.replace('[', '_').replace(']', '')
    name = re.sub(r'[^0-9A-Za-z_]', '_', name)
    return name


def load_blackbox(csv_path):
    """Parse the decoded blackbox CSV → (t, cols) with EVERY column kept.

    `t` is the (seconds) time base; `cols` maps a sanitized name (e.g.
    `gyroADC_2`, `motor_0`, `eRPM_1`, `axisP_0`) to its float array, so the full
    blackbox is carried into the merged file. `gyroADC_2` is also what the yaw
    cross-correlation uses for the clock offset.
    """
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        names = [_sanitize(h) for h in header]
        ti = next((i for i, h in enumerate(header) if h.lower().startswith('time')), None)
        if ti is None:
            raise KeyError(f'no time column in {csv_path}')
        ncol = len(header)
        data = [[] for _ in range(ncol)]
        for row in reader:
            if not row:
                continue
            for i in range(ncol):
                cell = row[i].strip() if i < len(row) else ''
                try:
                    data[i].append(float(cell))
                except ValueError:        # text flag columns (e.g. 'IDLE') → NaN
                    data[i].append(np.nan)

    t = np.asarray(data[ti], float)
    cols = {names[i]: np.asarray(data[i], float) for i in range(ncol) if i != ti}
    cols = {k: v for k, v in cols.items() if not np.isnan(v).all()}  # drop text-only cols
    # Blackbox time is monotonic; guard against any duplicate/!increasing stamps
    # so np.interp (which needs strictly increasing x) stays well-defined.
    keep = np.concatenate(([True], np.diff(t) > 0))
    if not keep.all():
        t = t[keep]
        cols = {k: v[keep] for k, v in cols.items()}
    return t, cols


def load_vicon(mat_path):
    """Load the ViCON .mat; return all original arrays plus the yaw angle."""
    m = sio.loadmat(mat_path)
    get = lambda k: np.asarray(m[k]).ravel().astype(float)

    keys = ['Abs_time', 'b1_x', 'b1_y', 'b1_z',
            'b1_qx', 'b1_qy', 'b1_qz', 'b1_qw',
            'b1_x_dot', 'b1_y_dot', 'b1_z_dot']
    d = {k: get(k) for k in keys if k in m}
    d['exptime'] = str(m['exptime'][0]) if 'exptime' in m else os.path.basename(mat_path)

    # Yaw (rotation about z) from the quaternion. Normalize; replace degenerate
    # (near-zero-norm) quaternions with identity so the conversion can't fail.
    quat = np.column_stack([d['b1_qx'], d['b1_qy'], d['b1_qz'], d['b1_qw']])
    norms = np.linalg.norm(quat, axis=1)
    bad = norms < 1e-6
    quat[bad] = [0.0, 0.0, 0.0, 1.0]
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    # 'zyx' -> first angle is yaw(z). Unwrap so differentiation sees no 2*pi jumps.
    yaw = Rotation.from_quat(quat).as_euler('zyx', degrees=False)[:, 0]
    d['yaw'] = np.unwrap(yaw)
    return d


# --------------------------------------------------------------------------- #
def _resample(t, y, fs):
    """Linearly resample (t, y) onto a uniform grid at fs Hz. Returns (t0, grid_y)."""
    grid_t = np.arange(t[0], t[-1], 1.0 / fs)
    return t[0], np.interp(grid_t, t, y)


def find_offset(vicon_t, vicon_yawrate, bb_t, bb_yawrate):
    """Cross-correlate the two yaw-rate signals; return (dt, corr_coeff, sign).

    dt satisfies  t_vicon = t_blackbox + dt  (add dt to blackbox time to land on
    the ViCON clock). `sign` is the yaw-axis polarity of the blackbox gyro
    relative to ViCON (+1 same, -1 flipped) -- determined from the peak's sign,
    so a mirrored axis convention doesn't fool the alignment.
    """
    v0, V = _resample(vicon_t, vicon_yawrate, XCORR_FS)
    b0, B = _resample(bb_t, bb_yawrate, XCORR_FS)
    V = V - V.mean()
    B = B - B.mean()

    corr = signal.correlate(V, B, mode='full')
    lags = signal.correlation_lags(V.size, B.size, mode='full')

    p = int(np.argmax(np.abs(corr)))           # |peak| -> robust to anti-correlation
    sign = 1.0 if corr[p] >= 0 else -1.0
    y = sign * corr                            # make the chosen peak a maximum

    # Parabolic sub-sample refinement of the peak location (in lag-index units).
    frac = 0.0
    if 0 < p < len(y) - 1:
        denom = y[p - 1] - 2 * y[p] + y[p + 1]
        if denom != 0:
            frac = 0.5 * (y[p - 1] - y[p + 1]) / denom
            frac = float(np.clip(frac, -1.0, 1.0))
    lag = (lags[p] + frac) * LAG_FACTOR        # fractional lag, sign-calibrated

    # t_vicon - t_blackbox = (v0 - b0) + lag/fs
    dt = (v0 - b0) + lag / XCORR_FS

    # Normalized correlation coefficient at the peak, for a quality readout.
    coeff = float(np.abs(corr[p]) / (np.linalg.norm(V) * np.linalg.norm(B) + 1e-12))
    return dt, coeff, sign


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description='Sync ViCON pose (.mat) with Betaflight blackbox motor RPM '
                    '(.bbl) into one merged .mat by yaw-rate cross-correlation.')
    ap.add_argument('bbl', nargs='?', help='blackbox .bbl (default: newest in blackbox_data/)')
    ap.add_argument('mat', nargs='?', help='ViCON .mat (default: newest in DataExchange/)')
    ap.add_argument('--out', help='output file (default: <mat>_synced.mat)')
    ap.add_argument('--poles', type=int, default=12, help='motor pole count (default 12)')
    ap.add_argument('--decoder', default=DEFAULT_DECODER, help='path to blackbox_decode')
    ap.add_argument('--plot', action='store_true', help='save a yaw-rate sync-check PNG')
    args = ap.parse_args()

    bbl = args.bbl or _newest('blackbox_data', '*.bbl')
    mat = args.mat or _newest('DataExchange', '*.mat')
    out = args.out or (os.path.splitext(mat)[0] + '_synced.mat')

    # The blackbox `eRPM` field is stored in units of 100 eRPM, so the decoded
    # value must be x100 to get true electrical RPM; mechanical RPM is then
    # electrical / (poles/2). Verified against the RPM-filter debug[] motor
    # frequencies (debug Hz x 60 == eRPM_field x 100 / (poles/2)) to within 0.5%.
    pole_pairs = args.poles / 2.0
    field_to_erpm = ERPM_FIELD_SCALE                 # decoded field -> electrical RPM
    field_to_rpm = ERPM_FIELD_SCALE / pole_pairs     # decoded field -> mechanical RPM

    print(f'Blackbox: {bbl}')
    print(f'ViCON:    {mat}')
    print(f'Motors:   {args.poles} poles  ->  RPM = eRPM_field x {ERPM_FIELD_SCALE:.0f} / {int(pole_pairs)}\n')

    # 1) Decode + load (keep every blackbox column).
    with tempfile.TemporaryDirectory() as tmp:
        bb_t, bb_cols = load_blackbox(decode_bbl(bbl, args.decoder, tmp))
    vc = load_vicon(mat)

    if 'gyroADC_2' not in bb_cols:
        sys.exit('blackbox has no gyroADC[2] column — cannot yaw-sync.')

    # 2) Yaw rate from each stream (blackbox: direct; ViCON: d(yaw)/dt).
    bb_yawrate = bb_cols['gyroADC_2']
    vc_yawrate = np.gradient(vc['yaw'], vc['Abs_time'])

    # 3) Recover the clock offset.
    dt, coeff, sign = find_offset(vc['Abs_time'], vc_yawrate, bb_t, bb_yawrate)
    print(f'Clock offset dt = {dt:+.4f} s   (t_vicon = t_blackbox + dt)')
    print(f'Peak correlation = {coeff:.3f}   yaw-axis sign = {sign:+.0f}')
    if coeff < 0.3:
        print('  WARNING: weak correlation -- was there a clear sharp-yaw event '
              'while hovering? The offset may be unreliable.')

    # 4) Map each ViCON timestamp onto the blackbox clock and interpolate.
    #    t_blackbox = t_vicon - dt. Samples outside blackbox coverage -> NaN.
    tb = vc['Abs_time'] - dt
    in_cov = (tb >= bb_t[0]) & (tb <= bb_t[-1])

    def interp(y):
        out = np.interp(tb, bb_t, y)
        out[~in_cov] = np.nan
        return out

    n = vc['Abs_time'].size
    bb_yaw_on_vicon = sign * interp(bb_yawrate)   # sign-corrected, for verification

    # 5) Save one merged file: ViCON pose + EVERY blackbox channel (bb_<name>) +
    #    derived mechanical RPM + sync provenance.
    merged = {
        'exptime': vc['exptime'],
        'Abs_time': vc['Abs_time'],
        'b1_x': vc['b1_x'], 'b1_y': vc['b1_y'], 'b1_z': vc['b1_z'],
        'b1_qx': vc['b1_qx'], 'b1_qy': vc['b1_qy'],
        'b1_qz': vc['b1_qz'], 'b1_qw': vc['b1_qw'],
        'vicon_yaw_rate': vc_yawrate,
        'blackbox_yaw_rate': bb_yaw_on_vicon,
        'sync_time_offset': dt,
        'sync_correlation': coeff,
        'sync_yaw_sign': sign,
        'motor_poles': args.poles,
        'blackbox_file': os.path.basename(bbl),
        'vicon_file': os.path.basename(mat),
    }
    for name, y in bb_cols.items():
        merged['bb_' + name] = interp(y)
    # Derived (N,4) motor arrays (kept so plot_synced.py's PNG panels still work).
    if all(f'eRPM_{m}' in bb_cols for m in range(4)):
        merged['motor_rpm'] = np.column_stack(
            [interp(bb_cols[f'eRPM_{m}']) * field_to_rpm for m in range(4)])
        merged['motor_erpm'] = np.column_stack(
            [interp(bb_cols[f'eRPM_{m}']) * field_to_erpm for m in range(4)])
    if all(f'motor_{m}' in bb_cols for m in range(4)):
        merged['motor_cmd'] = np.column_stack(
            [interp(bb_cols[f'motor_{m}']) for m in range(4)])
    for k in ('b1_x_dot', 'b1_y_dot', 'b1_z_dot'):
        if k in vc:
            merged[k] = vc[k]

    cov = int(in_cov.sum())
    print(f'\nMerged {n} ViCON samples; {cov} ({100*cov/n:.1f}%) overlap the blackbox log.'
          f'  {len(bb_cols)} blackbox channels carried.')
    if cov and 'motor_rpm' in merged:
        rpm_valid = merged['motor_rpm'][in_cov]
        print(f'Motor RPM over overlap: min {np.nanmin(rpm_valid):.0f}  '
              f'max {np.nanmax(rpm_valid):.0f}  mean {np.nanmean(rpm_valid):.0f}')

    sio.savemat(out, merged)
    print(f'\nSaved merged log: {out}')

    if args.plot:
        _plot_sync(vc['Abs_time'], vc_yawrate, bb_yaw_on_vicon, in_cov, coeff, dt,
                   os.path.splitext(out)[0] + '_sync.png')


def _newest(folder, pattern):
    hits = glob.glob(os.path.join(folder, pattern))
    if not hits:
        raise FileNotFoundError(f'no {pattern} in {folder}/')
    return max(hits, key=os.path.getmtime)


def _plot_sync(t, vicon_yawrate, bb_yawrate_on_vicon, in_cov, coeff, dt, png):
    """Overlay the two yaw-rate signals after alignment -- they should coincide."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t, vicon_yawrate, label='ViCON yaw rate (d yaw/dt)', lw=1.2)
    ax.plot(t, bb_yawrate_on_vicon, label='blackbox yaw rate (aligned)', lw=1.0, alpha=0.8)
    ax.set_title(f'Yaw-rate alignment check   dt={dt:+.3f}s   corr={coeff:.3f}')
    ax.set_xlabel('ViCON Abs_time (s)'); ax.set_ylabel('yaw rate (rad/s)')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(png, dpi=120)
    print(f'Sync-check plot: {png}')


if __name__ == '__main__':
    main()
