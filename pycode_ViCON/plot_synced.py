# Visualize a synced ViCON + blackbox log produced by sync_log.py.
#
# Plots everything in the merged file on a shared time axis so pose and motor
# RPM can be read off at the same instant: position, velocity, orientation,
# per-motor RPM, commanded motor output, the altitude<->thrust relationship,
# the yaw-rate sync-verification overlay, and a 3D trajectory.
#
# Usage:
#   python plot_synced.py                              # newest *_synced.mat in DataExchange/
#   python plot_synced.py --pick                       # list synced logs and choose
#   python plot_synced.py DataExchange/foo_synced.mat  # a specific file
#   python plot_synced.py --3d                         # also open an interactive,
#                                                      # rotatable 3D window (path
#                                                      # colored by mean motor RPM)

import sys
import glob
import os
import warnings

import numpy as np
import scipy.io as sio
from scipy.spatial.transform import Rotation

import matplotlib
# Interactive 3D needs a GUI backend; the static-PNG path uses Agg. Decide up
# front from the --3d flag so the chosen backend is consistent.
INTERACTIVE = '--3d' in sys.argv
if INTERACTIVE:
    os.environ.setdefault('QT_QPA_PLATFORM', 'wayland')
matplotlib.use('QtAgg' if INTERACTIVE else 'Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)


def list_synced(folder='DataExchange'):
    """Return *_synced.mat files in *folder*, newest first."""
    mats = glob.glob(os.path.join(folder, '*_synced.mat'))
    if not mats:
        raise FileNotFoundError(f'No *_synced.mat files in {folder}/ '
                                '(run sync_log.py first)')
    return sorted(mats, key=os.path.getmtime, reverse=True)


def select_synced(folder='DataExchange'):
    """Prompt the user to pick a synced log (newest first); Enter picks newest."""
    mats = list_synced(folder)
    if len(mats) == 1 or not sys.stdin.isatty():
        return mats[0]
    print('Select a synced log to plot:')
    for i, p in enumerate(mats):
        mb = os.path.getsize(p) / 1e6
        tag = '  (newest)' if i == 0 else ''
        print(f'  [{i}] {os.path.basename(p)}  {mb:6.2f} MB{tag}')
    while True:
        choice = input(f'Enter number [0-{len(mats)-1}, default 0]: ').strip()
        if choice == '':
            return mats[0]
        if choice.isdigit() and int(choice) < len(mats):
            return mats[int(choice)]
        print('  invalid selection, try again')


def load_synced(path):
    m = sio.loadmat(path)
    vec = lambda k: np.asarray(m[k]).ravel().astype(float)
    mat = lambda k: np.asarray(m[k]).astype(float)          # (N, 4) motor arrays
    scal = lambda k, d=None: (np.asarray(m[k]).ravel()[0] if k in m else d)
    text = lambda k, d='': (str(np.asarray(m[k]).ravel()[0]) if k in m else d)

    d = {
        't':  vec('Abs_time'),
        'x':  vec('b1_x'), 'y': vec('b1_y'), 'z': vec('b1_z'),
        'qx': vec('b1_qx'), 'qy': vec('b1_qy'), 'qz': vec('b1_qz'), 'qw': vec('b1_qw'),
    }
    for src, dst in (('b1_x_dot', 'vx'), ('b1_y_dot', 'vy'), ('b1_z_dot', 'vz')):
        if src in m:
            d[dst] = vec(src)
    for src, dst in (('motor_rpm', 'rpm'), ('motor_erpm', 'erpm'), ('motor_cmd', 'cmd')):
        if src in m:
            d[dst] = mat(src)
    if 'vicon_yaw_rate' in m:
        d['vicon_yaw_rate'] = vec('vicon_yaw_rate')
    if 'blackbox_yaw_rate' in m:
        d['blackbox_yaw_rate'] = vec('blackbox_yaw_rate')

    d['exptime'] = text('exptime', os.path.basename(path))
    d['vicon_file'] = text('vicon_file')
    d['blackbox_file'] = text('blackbox_file')
    d['offset'] = scal('sync_time_offset')
    d['corr'] = scal('sync_correlation')
    d['yaw_sign'] = scal('sync_yaw_sign')
    d['poles'] = scal('motor_poles')
    return d


def _euler_deg(d):
    """Yaw/pitch/roll (deg) from the quaternion; rows with bad norm -> identity."""
    quat = np.column_stack([d['qx'], d['qy'], d['qz'], d['qw']])
    norms = np.linalg.norm(quat, axis=1)
    quat[norms < 1e-6] = [0.0, 0.0, 0.0, 1.0]
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    return Rotation.from_quat(quat).as_euler('zyx', degrees=True)   # cols: yaw,pitch,roll


def summarize(d):
    t = d['t']
    dt = np.diff(t)
    rate = 1.0 / dt.mean() if dt.size else float('nan')
    euler = _euler_deg(d)

    print('=' * 64)
    print(f"Synced log:  {d['exptime']}")
    print(f"  ViCON:     {d['vicon_file']}")
    print(f"  blackbox:  {d['blackbox_file']}")
    if d['offset'] is not None:
        print(f"  sync:      offset {d['offset']:+.3f}s   correlation {d['corr']:.3f}"
              f"   yaw-sign {d['yaw_sign']:+.0f}   poles {int(d['poles'])}")
    print('-' * 64)
    print(f"Samples:     {t.size}    Duration: {t[-1]-t[0]:.2f} s"
          f"    Rate ~{rate:.1f} Hz")

    if 'rpm' in d:
        rpm = d['rpm']
        valid = np.isfinite(rpm).all(axis=1)
        nv = int(valid.sum())
        print(f"RPM coverage:{nv}/{t.size} samples ({100*nv/t.size:.1f}%)"
              + (f"   window {t[valid].min():.1f}..{t[valid].max():.1f}s" if nv else ""))
        if nv:
            for i in range(rpm.shape[1]):
                c = rpm[valid, i]
                print(f"  motor {i}:   mean {c.mean():7.0f}  min {c.min():6.0f}"
                      f"  max {c.max():6.0f} RPM")
    print('-' * 64)
    for ax in ('x', 'y', 'z'):
        v = d[ax]
        print(f"  pos {ax}:    min {v.min():8.3f}  max {v.max():8.3f}"
              f"  range {v.max()-v.min():6.3f} m")
    if 'vx' in d:
        speed = np.sqrt(d['vx']**2 + d['vy']**2 + d['vz']**2)
        print(f"  speed:    max {speed.max():8.3f}  mean {speed.mean():8.3f} m/s")
    for i, name in enumerate(('yaw', 'pitch', 'roll')):
        print(f"  {name:6s}:   min {euler[:,i].min():8.2f}  max {euler[:,i].max():8.2f} deg")
    print('=' * 64)
    return euler


def _mean_rpm(d):
    """Per-sample mean motor RPM (NaN where outside blackbox coverage)."""
    if 'rpm' not in d:
        return None
    with warnings.catch_warnings():           # all-NaN rows -> NaN, no warning
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(d['rpm'], axis=1)


def _break_wraps(angle, thresh=180.0):
    """Mask the sample after each +/-360 wrap so plots don't draw vertical lines."""
    a = np.ma.array(angle, copy=True)
    jumps = np.where(np.abs(np.diff(angle)) > thresh)[0] + 1
    a[jumps] = np.ma.masked
    return a


def plot(d, euler, out_path):
    t = d['t']
    tlim = (t[0], t[-1])
    fig, axes = plt.subplots(4, 2, figsize=(15, 15))
    title = f"Synced log {d['exptime']}"
    if d['offset'] is not None:
        title += f"   (sync corr {d['corr']:.3f}, offset {d['offset']:+.2f}s)"
    fig.suptitle(title, fontsize=14)

    def time_ax(ax):
        ax.set_xlim(tlim)
        ax.set_xlabel('time (s)')
        ax.grid(True, alpha=0.3)

    # (0,0) Position
    ax = axes[0, 0]
    for k in ('x', 'y', 'z'):
        ax.plot(t, d[k], label=k)
    ax.set_title('Position vs time'); ax.set_ylabel('position (m)')
    ax.legend(ncol=3); time_ax(ax)

    # (0,1) Motor RPM (mechanical) -- the key synced quantity
    ax = axes[0, 1]
    if 'rpm' in d:
        for i in range(d['rpm'].shape[1]):
            ax.plot(t, d['rpm'][:, i], label=f'motor {i}', lw=0.9)
        ax.legend(ncol=4, fontsize=8)
    ax.set_title('Motor RPM (mechanical) vs time'); ax.set_ylabel('RPM')
    time_ax(ax)

    # (1,0) Velocity
    ax = axes[1, 0]
    if 'vx' in d:
        for k in ('vx', 'vy', 'vz'):
            ax.plot(t, d[k], label=k, alpha=0.85)
        ax.legend(ncol=3)
    ax.set_title('Velocity vs time'); ax.set_ylabel('velocity (m/s)')
    time_ax(ax)

    # (1,1) Commanded motor output (raw, 0..2047) -- contrast with actual RPM
    ax = axes[1, 1]
    if 'cmd' in d:
        for i in range(d['cmd'].shape[1]):
            ax.plot(t, d['cmd'][:, i], label=f'cmd {i}', lw=0.9)
        ax.legend(ncol=4, fontsize=8)
    ax.set_title('Commanded motor output vs time'); ax.set_ylabel('command (raw)')
    time_ax(ax)

    # (2,0) Orientation. Yaw wraps at +/-180; break the wrap lines for clarity.
    ax = axes[2, 0]
    for i, name in enumerate(('yaw (z)', 'pitch (y)', 'roll (x)')):
        series = _break_wraps(euler[:, i]) if i == 0 else euler[:, i]
        ax.plot(t, series, label=name)
    ax.set_title('Orientation (Euler) vs time'); ax.set_ylabel('angle (deg)')
    ax.legend(ncol=3); time_ax(ax)

    # (2,1) Sync verification: the two yaw-rate signals should overlap
    ax = axes[2, 1]
    if 'vicon_yaw_rate' in d and 'blackbox_yaw_rate' in d:
        ax.plot(t, d['vicon_yaw_rate'], label='ViCON yaw rate', lw=1.1)
        ax.plot(t, d['blackbox_yaw_rate'], label='blackbox (aligned)', lw=0.9, alpha=0.8)
        ax.legend()
    ax.set_title('Sync check: yaw rate (should overlap)'); ax.set_ylabel('yaw rate (rad/s)')
    time_ax(ax)

    # (3,0) Altitude vs mean RPM -- thrust/motion relationship the sync unlocks
    ax = axes[3, 0]
    ax.plot(t, d['z'], color='tab:blue', label='altitude z')
    ax.set_ylabel('altitude z (m)', color='tab:blue')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    mean_rpm = _mean_rpm(d)
    if mean_rpm is not None:
        ax2 = ax.twinx()
        ax2.plot(t, mean_rpm, color='tab:red', alpha=0.8, label='mean RPM')
        ax2.set_ylabel('mean motor RPM', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
    ax.set_title('Altitude vs mean motor RPM'); time_ax(ax)

    # (3,1) 3D trajectory (color = time)
    axes[3, 1].remove()
    ax = fig.add_subplot(4, 2, 8, projection='3d')
    sc = ax.scatter(d['x'], d['y'], d['z'], c=t, s=4, cmap='viridis')
    ax.plot(d['x'], d['y'], d['z'], color='gray', lw=0.5, alpha=0.4)
    ax.scatter(d['x'][0], d['y'][0], d['z'][0], c='green', s=60, marker='o', label='start')
    ax.scatter(d['x'][-1], d['y'][-1], d['z'][-1], c='red', s=60, marker='X', label='end')
    ax.set_title('XYZ trajectory (color = time)')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_zlabel('z (m)')
    try:
        ax.set_box_aspect((np.ptp(d['x']), np.ptp(d['y']), np.ptp(d['z'])))
    except Exception:
        pass
    ax.legend(loc='upper left', fontsize=8)
    fig.colorbar(sc, ax=ax, label='time (s)', shrink=0.6)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=120)
    print(f'Plot saved: {out_path}')


def plot_3d(d):
    """Interactive 3D trajectory, colored by mean motor RPM where available."""
    mean_rpm = _mean_rpm(d)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    if mean_rpm is not None and np.isfinite(mean_rpm).any():
        finite = np.isfinite(mean_rpm)
        ax.plot(d['x'], d['y'], d['z'], color='gray', lw=0.5, alpha=0.4)
        sc = ax.scatter(d['x'][finite], d['y'][finite], d['z'][finite],
                        c=mean_rpm[finite], s=8, cmap='plasma')
        fig.colorbar(sc, ax=ax, label='mean motor RPM', shrink=0.6)
        # samples with no blackbox coverage shown faint
        nan = ~finite
        if nan.any():
            ax.scatter(d['x'][nan], d['y'][nan], d['z'][nan], c='lightgray',
                       s=4, alpha=0.4, label='no RPM data')
    else:
        sc = ax.scatter(d['x'], d['y'], d['z'], c=d['t'], s=6, cmap='viridis')
        fig.colorbar(sc, ax=ax, label='time (s)', shrink=0.6)

    ax.scatter(d['x'][0], d['y'][0], d['z'][0], c='green', s=80, marker='o', label='start')
    ax.scatter(d['x'][-1], d['y'][-1], d['z'][-1], c='red', s=80, marker='X', label='end')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_zlabel('z (m)')
    ax.set_title(f"3D trajectory  {d['exptime']}  (color = mean RPM; drag to rotate)")
    try:
        ax.set_box_aspect((np.ptp(d['x']), np.ptp(d['y']), np.ptp(d['z'])))
    except Exception:
        pass
    ax.legend()
    print('Opening interactive 3D window — drag to rotate, scroll to zoom. '
          'Close the window to exit.')
    plt.show()


def main():
    flags = {'--3d', '--pick'}
    args = [a for a in sys.argv[1:] if a not in flags]
    if args:
        path = args[0]
    elif '--pick' in sys.argv:
        path = select_synced()
    else:
        path = list_synced()[0]

    print(f'Processing: {path}\n')
    d = load_synced(path)
    euler = summarize(d)
    out_path = os.path.splitext(path)[0] + '_overview.png'
    plot(d, euler, out_path)
    if INTERACTIVE:
        plot_3d(d)


if __name__ == '__main__':
    main()
