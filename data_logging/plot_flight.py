"""Visualize a synced flight log produced by combine_flight.py.

combine_flight.py deterministically merges one flight's Vicon pose + Betaflight
blackbox into <session>/flight_synced.mat (pose + b1 velocity + per-motor
RPM/eRPM/command on the shared Vicon clock). This plots all of it on one time
axis — position, velocity, orientation, motor RPM (mechanical + electrical),
commanded output, the altitude<->thrust relationship, and a 3D trajectory — so
pose and motor behavior can be read off at the same instant. The session's
video.mp4 sits alongside the file this reads.

This is the data_logging counterpart to pycode_ViCON/plot_synced.py, adapted to
combine_flight's fields (deterministic sync — no yaw-rate cross-check panel).

Usage:
  plot_flight.py                                  # newest recordings/*/flight_synced.mat
  plot_flight.py --pick                           # list synced flights and choose
  plot_flight.py recordings/20260608_x/flight_synced.mat   # a specific file
  plot_flight.py --3d                             # also open an interactive 3D window
"""
import os
import sys
import glob
import warnings

import numpy as np
import scipy.io as sio
from scipy.spatial.transform import Rotation

import matplotlib
# Interactive 3D needs a GUI backend; the static-PNG path uses Agg. Decide up
# front from the --3d flag so the backend choice is consistent.
INTERACTIVE = '--3d' in sys.argv
if INTERACTIVE:
    os.environ.setdefault('QT_QPA_PLATFORM', 'wayland')
matplotlib.use('QtAgg' if INTERACTIVE else 'Agg')
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402  (registers 3d projection)

HERE = os.path.dirname(os.path.abspath(__file__))
RECORDINGS = os.path.join(HERE, 'recordings')


def list_synced(folder=RECORDINGS):
    """Return flight_synced.mat files under recordings/*/, newest first."""
    mats = glob.glob(os.path.join(folder, '*', 'flight_synced.mat'))
    if not mats:
        raise FileNotFoundError(
            f'No */flight_synced.mat in {folder}/ — run combine_flight.py first.')
    return sorted(mats, key=os.path.getmtime, reverse=True)


def select_synced(folder=RECORDINGS):
    """Prompt to pick a synced flight (newest first); Enter picks newest."""
    mats = list_synced(folder)
    if len(mats) == 1 or not sys.stdin.isatty():
        return mats[0]
    print('Select a synced flight to plot:')
    for i, p in enumerate(mats):
        tag = '  (newest)' if i == 0 else ''
        print(f'  [{i}] {os.path.basename(os.path.dirname(p))}{tag}')
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
    mat = lambda k: np.asarray(m[k]).astype(float)              # (N, 4) motor arrays

    def scal(k, d=None):
        if k not in m:
            return d
        a = np.asarray(m[k]).ravel()
        return a[0] if a.size else d

    def text(k, d=''):                          # savemat stores '' as a size-0 array
        if k not in m:
            return d
        a = np.asarray(m[k]).ravel()
        return str(a[0]) if a.size else d

    d = {
        't':  vec('Abs_time'),
        'x':  vec('b1_x'), 'y': vec('b1_y'), 'z': vec('b1_z'),
        'qx': vec('b1_qx'), 'qy': vec('b1_qy'), 'qz': vec('b1_qz'), 'qw': vec('b1_qw'),
    }
    for src, dst in (('b1_vx', 'vx'), ('b1_vy', 'vy'), ('b1_vz', 'vz')):
        if src in m:
            d[dst] = vec(src)
    for src, dst in (('motor_rpm', 'rpm'), ('motor_erpm', 'erpm'), ('motor_cmd', 'cmd')):
        if src in m:
            d[dst] = mat(src)

    d['exptime'] = text('t0_human', os.path.basename(os.path.dirname(path)))
    d['session'] = text('session', os.path.basename(os.path.dirname(path)))
    d['vicon_file'] = text('vicon_file')
    d['blackbox_file'] = text('blackbox_file')
    d['video_file'] = text('video_file')
    d['sync_method'] = text('sync_method')
    d['offset'] = scal('sync_offset_s')
    d['poles'] = scal('motor_poles')
    return d


def _euler_deg(d):
    """Yaw/pitch/roll (deg) from the quaternion; bad-norm rows -> identity."""
    quat = np.column_stack([d['qx'], d['qy'], d['qz'], d['qw']])
    norms = np.linalg.norm(quat, axis=1)
    quat[norms < 1e-6] = [0.0, 0.0, 0.0, 1.0]
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    return Rotation.from_quat(quat).as_euler('zyx', degrees=True)   # cols: yaw,pitch,roll


def _mean_rpm(d):
    """Per-sample mean motor RPM (NaN where outside blackbox coverage)."""
    if 'rpm' not in d:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(d['rpm'], axis=1)


def _break_wraps(angle, thresh=180.0):
    """Mask the sample after each +/-360 wrap so plots don't draw vertical lines."""
    a = np.ma.array(angle, copy=True)
    jumps = np.where(np.abs(np.diff(angle)) > thresh)[0] + 1
    a[jumps] = np.ma.masked
    return a


def summarize(d):
    t = d['t']
    dt = np.diff(t)
    rate = 1.0 / dt.mean() if dt.size else float('nan')
    euler = _euler_deg(d)

    print('=' * 64)
    print(f"Synced flight: {d['session']}   ({d['exptime']})")
    print(f"  Vicon:     {d['vicon_file']}")
    print(f"  blackbox:  {d['blackbox_file']}")
    print(f"  video:     {d['video_file']}")
    print(f"  sync:      {d['sync_method']}   offset {d['offset']:+.3f}s"
          f"   poles {int(d['poles']) if d['poles'] is not None else '?'}")
    print('-' * 64)
    print(f"Samples:     {t.size}    Duration: {t[-1]-t[0]:.2f} s    Rate ~{rate:.1f} Hz")

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


def plot(d, euler, out_path):
    t = d['t']
    tlim = (t[0], t[-1])
    fig, axes = plt.subplots(4, 2, figsize=(15, 15))
    title = f"Synced flight {d['session']}"
    if d['offset'] is not None:
        title += f"   ({d['sync_method']}, offset {d['offset']:+.2f}s)"
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

    # (0,1) Motor RPM (mechanical) — the key synced quantity
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

    # (1,1) Commanded motor output (raw) — contrast with actual RPM
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

    # (2,1) Motor eRPM (electrical) — what the blackbox logs directly
    ax = axes[2, 1]
    if 'erpm' in d:
        for i in range(d['erpm'].shape[1]):
            ax.plot(t, d['erpm'][:, i], label=f'motor {i}', lw=0.9)
        ax.legend(ncol=4, fontsize=8)
    ax.set_title('Motor eRPM (electrical) vs time'); ax.set_ylabel('eRPM')
    time_ax(ax)

    # (3,0) Altitude vs mean RPM — thrust/motion relationship the sync unlocks
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
    ax.set_title(f"3D trajectory  {d['session']}  (color = mean RPM; drag to rotate)")
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
    out_path = os.path.join(os.path.dirname(path), 'flight_overview.png')
    plot(d, euler, out_path)
    if INTERACTIVE:
        plot_3d(d)


if __name__ == '__main__':
    main()
