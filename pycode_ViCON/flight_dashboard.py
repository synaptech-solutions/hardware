"""Shared interactive flight dashboard (Plotly Dash).

Used by BOTH data_logging/dashboard_flight.py and pycode_ViCON/dashboard_synced.py
so the two have identical UI + functionality. Reads any synced .mat (combine_flight
or sync_log output) and derives panels from whatever fields the file contains —
pose, velocity, orientation, and every carried blackbox channel (bb_*: angular
velocity, accel, PID, rcCommand, setpoint, debug, battery, …).

Features:
  - sync/desync toggle (link x-axes for joint zoom/pan, or zoom each graph alone),
  - a checklist exposing every available channel as a panel,
  - per-graph colored legends (in each subplot title) + x-axis labelled in seconds,
  - a labelled master time-window slider that drives BOTH the time-series x-range
    AND which slice of the 3D trajectory is shown,
  - taller panels, per-graph + 3D fullscreen buttons,
  - a WebGL 3D trajectory (drag = rotate · right-click/ctrl-drag = pan · scroll =
    zoom) with the trace legend on the left, colorbar on the right, and a
    color-by dropdown (time / speed / altitude / vertical speed / mean RPM /
    angular rate / …).
"""
import csv
import json
import os
import re
import threading
import warnings
import webbrowser

import numpy as np
import scipy.io as sio
from scipy import signal as sp_signal
from scipy.spatial.transform import Rotation
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
           "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"]
SECONDARY_COLOR = "#d62728"

# Nice labels/units/series-names for known blackbox bases (bb_<base>_<i>).
BB_LABELS = {
    "gyroADC":    ("IMU — angular rate (gyro)", "rad/s", ["roll rate", "pitch rate", "yaw rate"]),
    "gyroUnfilt": ("IMU — angular rate (gyro, unfiltered)", "rad/s", ["roll", "pitch", "yaw"]),
    "accSmooth":  ("IMU — acceleration (accel)", "g", ["ax", "ay", "az"]),
    "motor":      ("Motor command (raw)", "cmd", ["m0", "m1", "m2", "m3"]),
    "eRPM":       ("Motor eRPM (electrical field)", "field", ["m0", "m1", "m2", "m3"]),
    "rcCommand":  ("RC command", "us", ["roll", "pitch", "yaw", "throttle"]),
    "setpoint":   ("Setpoint", "", ["roll", "pitch", "yaw", "throttle"]),
    "axisP":      ("PID — P term", "", ["roll", "pitch", "yaw"]),
    "axisI":      ("PID — I term", "", ["roll", "pitch", "yaw"]),
    "axisD":      ("PID — D term", "", ["roll", "pitch"]),
    "axisF":      ("PID — F term", "", ["roll", "pitch", "yaw"]),
    "debug":      ("Debug", "", None),
    "vbatLatest": ("Battery voltage", "V", None),
    "amperageLatest": ("Current", "A", None),
    "rssi":       ("RSSI", "", None),
}
# Order known bb groups by usefulness; unknowns fall after, alphabetical.
BB_ORDER = ["gyroADC", "accSmooth", "motor", "eRPM", "rcCommand", "setpoint",
            "axisP", "axisI", "axisD", "axisF", "gyroUnfilt", "debug",
            "vbatLatest", "amperageLatest", "rssi"]

DEFAULT_ON = ["pos", "vel", "orient", "motor_rpm", "cmd_sticks",
              "bb_gyroADC", "bb_accSmooth", "altrpm", "yawsync",
              "lat_timeline", "lat_throttle", "lat_roll"]


def _euler_deg(qx, qy, qz, qw):
    quat = np.column_stack([qx, qy, qz, qw])
    norms = np.linalg.norm(quat, axis=1)
    quat[norms < 1e-6] = [0.0, 0.0, 0.0, 1.0]
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    return Rotation.from_quat(quat).as_euler("zyx", degrees=True)   # yaw,pitch,roll


def _latency_timeline(m, t, win_s=10.0, step_s=2.5, fs=500.0, corr_floor=0.25):
    """Latency (ms) of each control-path link as a function of flight time.

    Slides a window over the flight; in each window cross-correlates one link to
    get its lag, then interpolates the per-window lags back onto the full time
    base t (so it plots as a normal panel). Returns {label: array(len(t))} or {}
    if the needed channels are absent. Links:
        transport   laptop cmd  -> FC rcCommand   (throttle, edge xcorr)
        FC response FC rcCommand -> gyro           (roll+pitch, signal xcorr)
        end-to-end  laptop cmd  -> gyro            (roll+pitch)
    """
    def has(k):
        return k in m and np.asarray(m[k]).ravel().size == t.size
    if not (has("cmd_ch02_us") and has("bb_rcCommand_3")):
        return {}
    tg = np.arange(float(t[0]), float(t[-1]), 1.0 / fs)
    def rs(k):
        y = np.asarray(m[k]).ravel().astype(float)
        ok = np.isfinite(y) & np.isfinite(t)
        return np.interp(tg, t[ok], y[ok]) if ok.sum() > 10 else None
    bp = sp_signal.butter(2, [0.3 / (fs / 2), 15 / (fs / 2)], "band")

    def win_lag(a, b, t0, t1, deriv, max_lag_s=0.15):
        w = (tg >= t0) & (tg < t1)
        x, y = a[w].copy(), b[w].copy()
        if deriv:
            x, y = np.diff(x), np.diff(y)
        else:
            x = sp_signal.filtfilt(*bp, x); y = sp_signal.filtfilt(*bp, y)
        if x.std() < 1e-6 or y.std() < 1e-6:
            return None
        x = (x - x.mean()) / x.std(); y = (y - y.mean()) / y.std()
        f = sp_signal.correlate(y, x, "full")
        lags = sp_signal.correlation_lags(len(y), len(x), "full")
        sel = np.abs(lags) <= int(max_lag_s * fs); f, lags = f[sel], lags[sel]
        k = int(np.argmax(f)); lag = float(lags[k]); q = f[k] / len(x)
        if 0 < k < len(f) - 1:
            y0, y1, y2 = f[k - 1], f[k], f[k + 1]; dd = y0 - 2 * y1 + y2
            if abs(dd) > 1e-12: lag += 0.5 * (y0 - y2) / dd
        return lag / fs * 1000.0, q

    cache = {}
    def get(k):
        if k not in cache: cache[k] = rs(k)
        return cache[k]

    def sweep(pairs, deriv, floor0=False):
        """Per-window lag (ms) -> smoothed, optionally floored at 0 (latency can't be
        negative), interpolated onto the full time base. Transport (~10 ms) sits at
        the ~10 ms blackbox-sample resolution, so the raw per-window estimate hops by
        ±one sample; a rolling median suppresses that quantization jitter."""
        centers, vals = [], []
        x = float(t[0])
        while x + win_s <= float(t[-1]):
            ms, ws = [], []
            for an, bn in pairs:
                A, B = get(an), get(bn)
                if A is None or B is None:
                    continue
                r = win_lag(A, B, x, x + win_s, deriv)
                if r and abs(r[1]) > corr_floor:
                    ms.append(r[0]); ws.append(abs(r[1]))
            if ms:
                centers.append(x + win_s / 2.0); vals.append(np.average(ms, weights=ws))
            x += step_s
        if len(centers) < 5:
            return None
        vals = np.asarray(vals, float)
        vals = sp_signal.medfilt(vals, kernel_size=5)        # kill ±1-sample hopping
        if floor0:
            vals = np.maximum(vals, 0.0)                     # transport >= 0 (physical)
        return np.interp(t, centers, vals, left=np.nan, right=np.nan)

    # Two independently-measured links, then end-to-end = their SUM (so it is
    # consistent by construction: a sub-link can never exceed the whole). The two
    # links are measured on the axes where each has the best signal -- transport
    # on throttle (large, clean; the comms+smoothing lag is axis-independent),
    # response on roll/pitch (that is where rotation actually happens).
    links = {}
    tr = sweep([("cmd_ch02_us", "bb_rcCommand_3")], True, floor0=True)   # transport (>=0)
    rr = None
    if has("bb_rcCommand_0") and has("bb_gyroADC_0"):
        rp = [("bb_rcCommand_0", "bb_gyroADC_0")]
        if has("bb_rcCommand_1") and has("bb_gyroADC_1"):
            rp.append(("bb_rcCommand_1", "bb_gyroADC_1"))
        rr = sweep(rp, False)                                     # FC+airframe response
    if tr is not None:
        links["cmd → received (transport, ~10 ms res. floor)"] = tr
    if rr is not None:
        links["received → rotating (response)"] = rr
    if tr is not None and rr is not None:
        links["cmd → rotating (end-to-end = sum)"] = tr + rr      # consistent: >= each part
    return links


def _load_synced_csv(path):
    """Read combine_flight.py's flight_synced.csv (+ .meta.json sidecar) into a
    loadmat-style field dict: each column -> 1-D float array, the motor_*_<i>
    columns folded back into the (N,4) arrays the panels expect, and the scalar
    metadata merged in. So the dashboard treats CSV and .mat identically."""
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        data = [[] for _ in header]
        for row in r:
            if not row:
                continue
            for i in range(len(header)):
                cell = row[i] if i < len(row) else ""
                try:
                    data[i].append(float(cell))
                except ValueError:
                    data[i].append(np.nan)
    m = {h: np.asarray(d, float) for h, d in zip(header, data)}
    for base in ("motor_rpm", "motor_erpm", "motor_cmd"):
        keys = [f"{base}_{i}" for i in range(4)]
        if all(k in m for k in keys):
            m[base] = np.column_stack([m.pop(k) for k in keys])
    meta_path = os.path.splitext(path)[0] + ".meta.json"
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            for k, v in json.load(f).items():
                if isinstance(v, (str, int, float)):   # skip lists (e.g. columns)
                    m[k] = v
    return m


def load_synced_fields(path):
    """Field dict for a synced file, reading EITHER combine_flight's CSV or a
    legacy/standalone .mat (sync_log). Lets one dashboard serve both pipelines."""
    if path.lower().endswith(".csv"):
        return _load_synced_csv(path)
    return sio.loadmat(path)


def load_channels(path):
    """Parse a synced file (.csv or .mat) into a dashboard data dict: time base,
    panel list (each with colored series), 3D pose + color-by options, metadata."""
    m = load_synced_fields(path)
    t = np.asarray(m["Abs_time"]).ravel().astype(float)
    N = t.size
    col = lambda k: np.asarray(m[k]).ravel().astype(float)
    has = lambda *ks: all(k in m for k in ks)

    def arr2(k):
        a = np.asarray(m[k]).astype(float)
        if a.ndim == 2 and a.shape[0] != N and a.shape[1] == N:
            a = a.T
        return a

    panels = []

    def mk(pid, label, unit, pairs, secondary=None):
        series = [(name, PALETTE[j % len(PALETTE)], y) for j, (name, y) in enumerate(pairs)]
        sec = [(name, SECONDARY_COLOR, y, u) for (name, y, u) in (secondary or [])] or None
        panels.append(dict(id=pid, label=label, unit=unit, series=series, secondary=sec))

    # --- curated pose panels ---------------------------------------------- #
    if has("b1_x", "b1_y", "b1_z"):
        mk("pos", "Position", "m",
           [("x", col("b1_x")), ("y", col("b1_y")), ("z", col("b1_z"))])
    vk = (["b1_vx", "b1_vy", "b1_vz"] if has("b1_vx")
          else (["b1_x_dot", "b1_y_dot", "b1_z_dot"] if has("b1_x_dot") else None))
    if vk:
        mk("vel", "Velocity", "m/s",
           [("vx", col(vk[0])), ("vy", col(vk[1])), ("vz", col(vk[2]))])
    if has("b1_qx", "b1_qy", "b1_qz", "b1_qw"):
        e = _euler_deg(col("b1_qx"), col("b1_qy"), col("b1_qz"), col("b1_qw"))
        mk("orient", "Orientation (Euler)", "deg",
           [("yaw", e[:, 0]), ("pitch", e[:, 1]), ("roll", e[:, 2])])
        mk("quat", "Quaternion", "",
           [("qx", col("b1_qx")), ("qy", col("b1_qy")),
            ("qz", col("b1_qz")), ("qw", col("b1_qw"))])
    if "motor_rpm" in m:
        a = arr2("motor_rpm")
        mk("motor_rpm", "Motor RPM (mechanical)", "RPM",
           [(f"m{j}", a[:, j]) for j in range(a.shape[1])])
    if has("vicon_yaw_rate", "blackbox_yaw_rate"):
        mk("yawsync", "Yaw-rate sync check", "rad/s",
           [("Vicon yaw rate", col("vicon_yaw_rate")),
            ("blackbox (aligned)", col("blackbox_yaw_rate"))])
    mean_rpm = None
    if "motor_rpm" in m and "b1_z" in m:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN rows → NaN
            mean_rpm = np.nanmean(arr2("motor_rpm"), axis=1)
        mk("altrpm", "Altitude vs mean RPM", "m",
           [("alt z", col("b1_z"))], secondary=[("mean RPM", mean_rpm, "RPM")])

    # --- every carried blackbox channel, grouped by base name ------------- #
    groups = {}
    for k in m:
        if not k.startswith("bb_"):
            continue
        v = np.asarray(m[k])
        if v.dtype.kind not in "fiu" or v.ravel().size != N:
            continue
        core = k[3:]
        mt = re.match(r"^(.*)_(\d+)$", core)
        base, idx = (mt.group(1), int(mt.group(2))) if mt else (core, -1)
        groups.setdefault(base, {})[idx] = v.ravel().astype(float)

    def order_key(b):
        return (BB_ORDER.index(b), "") if b in BB_ORDER else (len(BB_ORDER), b)

    for base in sorted(groups, key=order_key):
        items = sorted(groups[base].items())
        label, unit, names = BB_LABELS.get(base, (base, "", None))
        pairs = []
        for idx, y in items:
            if names and 0 <= idx < len(names):
                nm = names[idx]
            elif idx >= 0:
                nm = f"{base}[{idx}]"
            else:
                nm = base
            pairs.append((nm, y))
        mk("bb_" + base, label, unit, pairs)

    # --- outgoing commands (cmd_*) + incoming telemetry (tlm_*) ----------- #
    # The captured-everything streams. Compare against the FC blackbox: e.g.
    # cmd_sticks (what the laptop SENT) vs bb_rcCommand (what the FC RECEIVED).
    def present(*ks):
        return [k for k in ks if k in m and np.asarray(m[k]).ravel().size == N]

    # Stick commands SENT from the laptop, in µs. AETR channel order + the Air75
    # arm/blackbox switch channels (config.CH_ROLL.. / ARM_CH=6 / AUX2_CH=5).
    CMD_NAMES = {0: "roll", 1: "pitch", 2: "throttle", 3: "yaw",
                 5: "blackbox sw", 6: "arm sw"}
    trpy = [(CMD_NAMES[i], col(f"cmd_ch{i:02d}_us"))
            for i in (0, 1, 2, 3) if f"cmd_ch{i:02d}_us" in m]
    if trpy:
        mk("cmd_sticks", "Sticks sent — TRPY (laptop→drone)", "us", trpy)
    # Switches/aux: arm + blackbox always shown; other aux only if they moved.
    aux = []
    for i in range(4, 16):
        k = f"cmd_ch{i:02d}_us"
        if k not in m:
            continue
        y = col(k)
        if i in (5, 6) or (np.nanmax(y) - np.nanmin(y) > 1.0):
            aux.append((CMD_NAMES.get(i, f"ch{i:02d}"), y))
    if aux:
        mk("cmd_aux", "Command switches / aux (laptop→drone)", "us", aux)
    flags = present("cmd_armed", "cmd_record_on")
    if flags:
        mk("cmd_flags", "Command flags (armed / recording)", "",
           [(k[4:], col(k)) for k in flags])

    # Incoming CRSF telemetry (drone→laptop, downsampled back-channel).
    att = present("tlm_att_roll_deg", "tlm_att_pitch_deg", "tlm_att_yaw_deg")
    if att:
        mk("tlm_att", "Telemetry — attitude (FC estimate)", "deg",
           [(k.split("att_")[1].replace("_deg", ""), col(k)) for k in att])
    link = present("tlm_up_lq", "tlm_dn_lq", "tlm_up_rssi_dbm", "tlm_dn_rssi_dbm",
                   "tlm_up_snr_db", "tlm_dn_snr_db")
    if link:
        mk("tlm_link", "Telemetry — link stats (LQ/RSSI/SNR)", "",
           [(k[4:], col(k)) for k in link])
    bat = present("tlm_bat_v", "tlm_bat_a", "tlm_bat_pct")
    if bat:
        mk("tlm_bat", "Telemetry — battery", "", [(k[4:], col(k)) for k in bat])
    # Live IMU over telemetry (MSP_RAW_IMU) — only present if MSP polling was on.
    imu_a = present("tlm_imu_ax_g", "tlm_imu_ay_g", "tlm_imu_az_g")
    if imu_a:
        mk("tlm_imu_acc", "Telemetry — IMU accel (live, MSP)", "g",
           [(k.split("imu_")[1].replace("_g", ""), col(k)) for k in imu_a])
    imu_g = present("tlm_imu_gx_dps", "tlm_imu_gy_dps", "tlm_imu_gz_dps")
    if imu_g:
        mk("tlm_imu_gyro", "Telemetry — IMU gyro (live, MSP)", "deg/s",
           [(k.split("imu_")[1].replace("_dps", ""), col(k)) for k in imu_g])

    # --- latency overlays: command vs reception vs response --------------- #
    # Traces are scaled to a FIXED full-deflection reference (NOT each min-max'd),
    # so a tiny stick input stays tiny on screen instead of being stretched to
    # fill the axis (which makes RC-smoothing/quantization noise on a near-neutral
    # axis look like a huge -- even negative -- "latency"). Read the lag between
    # links straight off the x-axis on a step where the command actually MOVED:
    #   laptop cmd -> FC rcCommand (received) -> setpoint -> gyro / motor
    # Caveat: only meaningful where the command moved a real amount. A per-axis
    # "Δ" in the title flags how far the stick actually travelled; axes that
    # barely moved are noise-dominated and not a real latency reading.
    def _scale(y, center, full):
        return (np.asarray(y, float) - center) / full

    def _ptp_us(k):
        if k not in m:
            return 0.0
        y = col(k); return float(np.nanmax(y) - np.nanmin(y))

    # throttle: command -> received -> motor (the cleanest, largest-amplitude chain).
    # Throttle is a big clean signal, so each trace is min-max'd over its own span
    # -> they overlay at matched amplitude (cmd µs, rcCommand units and motor RPM
    # all have different native scales). Only TIMING matters here, and a large
    # signal won't stretch noise the way a near-neutral attitude axis would.
    def _mm(y):
        y = np.asarray(y, float); lo, hi = np.nanmin(y), np.nanmax(y)
        return (y - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(y)
    if "cmd_ch02_us" in m and "bb_rcCommand_3" in m:
        series = [("laptop cmd", _mm(col("cmd_ch02_us"))),
                  ("FC rcCommand", _mm(col("bb_rcCommand_3")))]
        if mean_rpm is not None:
            series.append(("motor (mean RPM)", _mm(mean_rpm)))
        mk("lat_throttle", f"Latency — Throttle: cmd → received → motor  (Δcmd {_ptp_us('cmd_ch02_us'):.0f}µs)",
           "norm 0–1", series)

    # attitude axes: cmd µs about neutral 1500 (±500 full); rcCommand/setpoint/gyro
    # scaled to comparable references. Yaw rcCommand logs inverted vs the FC, so we
    # flip the laptop cmd sign for the overlay (timing is unaffected by the flip).
    for nm, c_ch, rc, sp, gy, csign in [
            ("Roll", "cmd_ch00_us", "bb_rcCommand_0", "bb_setpoint_0", "bb_gyroADC_0", +1),
            ("Pitch", "cmd_ch01_us", "bb_rcCommand_1", "bb_setpoint_1", "bb_gyroADC_1", +1),
            ("Yaw", "cmd_ch03_us", "bb_rcCommand_2", "bb_setpoint_2", "bb_gyroADC_2", -1)]:
        if c_ch not in m or rc not in m:
            continue
        series = [("laptop cmd", csign * _scale(col(c_ch), 1500, 500)),
                  ("FC rcCommand", _scale(col(rc), 0, 500))]
        if sp in m:
            series.append(("setpoint", _scale(col(sp), 0, 500)))
        if gy in m:
            series.append(("gyro (actual rate)", _scale(col(gy), 0, 500)))
        flip = "  [cmd sign-flipped to match FC]" if csign < 0 else ""
        mk(f"lat_{nm.lower()}",
           f"Latency — {nm}: cmd → received → setpoint → gyro  (Δcmd {_ptp_us(c_ch):.0f}µs){flip}",
           "fraction of full deflection", series)

    # latency of each link AS A FUNCTION OF TIME (windowed cross-correlation):
    # how big the lag is and whether it drifts over the flight.
    tl = _latency_timeline(m, t)
    if tl:
        mk("lat_timeline", "Latency over time — per link (windowed, ms)", "ms",
           list(tl.items()))

    # --- 3D pose + color-by options --------------------------------------- #
    x = col("b1_x") if "b1_x" in m else np.zeros(N)
    y = col("b1_y") if "b1_y" in m else np.zeros(N)
    z = col("b1_z") if "b1_z" in m else np.zeros(N)
    color_opts = {"time": t, "altitude": z}
    if vk:
        vx, vy, vz = col(vk[0]), col(vk[1]), col(vk[2])
        color_opts["speed"] = np.sqrt(vx**2 + vy**2 + vz**2)
        color_opts["vertical speed"] = vz
    if mean_rpm is not None:
        color_opts["mean RPM"] = mean_rpm
    if has("bb_gyroADC_0", "bb_gyroADC_1", "bb_gyroADC_2"):
        color_opts["angular rate"] = np.sqrt(
            col("bb_gyroADC_0")**2 + col("bb_gyroADC_1")**2 + col("bb_gyroADC_2")**2)
    if "cmd_ch02_us" in m:
        color_opts["throttle cmd"] = col("cmd_ch02_us")

    meta = {}
    for k in ("session", "exptime", "sync_method", "blackbox_file", "vicon_file", "video_file"):
        if k in m and np.asarray(m[k]).ravel().size:
            meta[k] = str(np.asarray(m[k]).ravel()[0])
    for k in ("sync_offset_s", "sync_time_offset", "sync_correlation",
              "sync_yaw_sign", "motor_poles"):
        if k in m and np.asarray(m[k]).ravel().size:
            meta[k] = float(np.asarray(m[k]).ravel()[0])

    quat = ({q: col("b1_" + q) for q in ("qx", "qy", "qz", "qw")}
            if has("b1_qx", "b1_qy", "b1_qz", "b1_qw") else None)
    return dict(t=t, panels=panels, x=x, y=y, z=z, color_opts=color_opts,
                quat=quat, meta=meta, path=path)


def single_panel_fig(panel, window):
    """One panel → its own standalone figure (each 2D graph is separate).

    Per-graph legend at the top, x-axis labelled in seconds, autosizing height
    (the wrapper Div sets the box, so fullscreen fills the screen).

    Uses go.Scatter (SVG), NOT Scattergl: every Scattergl trace grabs its own
    WebGL context, and N panels + the 3D scene quickly exceed the browser/GPU
    context cap (low under software/XWayland GL). The browser then evicts the
    oldest context — the 3D trajectory — so it renders then blanks. Keeping 2D on
    SVG leaves the 3D scene as the ONLY WebGL context, so it can't be evicted."""
    if panel["secondary"]:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        for name, color, yv in panel["series"]:
            fig.add_trace(go.Scatter(x=None, y=yv, name=name, mode="lines",
                                       line=dict(color=color)), secondary_y=False)
        for name, color, yv, u in panel["secondary"]:
            fig.add_trace(go.Scatter(x=None, y=yv, name=name, mode="lines",
                                       line=dict(color=color)), secondary_y=True)
        fig.update_yaxes(title_text=panel["unit"], secondary_y=False)
        fig.update_yaxes(title_text=panel["secondary"][0][3], secondary_y=True)
    else:
        fig = go.Figure()
        for name, color, yv in panel["series"]:
            fig.add_trace(go.Scatter(x=None, y=yv, name=name, mode="lines",
                                       line=dict(color=color)))
        fig.update_yaxes(title_text=panel["unit"])
    # x supplied once (all traces share the time base)
    fig.update_traces(x=panel.get("_t"))
    fig.update_xaxes(title_text="time (s)", range=list(window))
    fig.update_layout(
        title=dict(text=panel["label"], x=0.0, xanchor="left", font=dict(size=15)),
        margin=dict(l=70, r=20, t=66, b=40), autosize=True,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    return fig


ORIENT_COLOR = "#000000"       # distinct from the Viridis trajectory


def _orientation_traces(data, mask):
    """A small weather-flag 'L' on EVERY shown point: long arm = body forward
    (×2), short arm = body up (×1), perpendicular. One trace (disconnected
    segments via None breaks), one distinct color. Each L is sized to the local
    point spacing so it sits right at its point rather than floating over the path."""
    q = data["quat"]
    X, Y, Z = data["x"][mask], data["y"][mask], data["z"][mask]
    n = X.size
    if n == 0:
        return None
    P = np.column_stack([X, Y, Z])
    # Long arm = 2× the median inter-point spacing (was 1×); short arm stays at
    # half the long arm, preserving the 2:1 length:height proportion.
    ARM_SCALE = 2.0
    if n > 1:
        d = np.linalg.norm(np.diff(P, axis=0), axis=1)
        d = d[d > 0]
        l_long = ARM_SCALE * (float(np.median(d)) if d.size else 1e-3)
    else:
        l_long = ARM_SCALE * 1e-3
    l_short = l_long / 2.0
    quats = np.column_stack([q["qx"][mask], q["qy"][mask],
                             q["qz"][mask], q["qw"][mask]])
    norms = np.linalg.norm(quats, axis=1)
    quats[norms < 1e-6] = [0.0, 0.0, 0.0, 1.0]
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    R = Rotation.from_quat(quats).as_matrix()    # body→world; cols = body x,y,z
    fwd, up = R[:, :, 0], R[:, :, 2]             # facing (long), up (short)
    F = P + fwd * l_long                          # long-arm tip (forward)
    U = F + up * l_short                          # short arm hangs off the FAR end
    # Per point: P, F, nan (long arm), F, U, nan (short arm off the tip) — vectorized.
    seg = np.empty((n * 6, 3))
    seg[0::6], seg[1::6], seg[2::6] = P, F, np.nan
    seg[3::6], seg[4::6], seg[5::6] = F, U, np.nan
    return go.Scatter3d(x=seg[:, 0], y=seg[:, 1], z=seg[:, 2], mode="lines",
                        line=dict(color=ORIENT_COLOR, width=2),
                        name="orientation (long=facing, short=up)")


def build_3d(data, color_by, window, show_orient=False):
    t = data["t"]
    lo, hi = window
    mask = (t >= lo) & (t <= hi)
    if not mask.any():
        mask = np.ones_like(t, bool)
    x, y, z = data["x"][mask], data["y"][mask], data["z"][mask]
    c = data["color_opts"].get(color_by, t)[mask]
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z, mode="markers+lines", name="trajectory",
        marker=dict(size=2, color=c, colorscale="Viridis", showscale=True,
                    colorbar=dict(title=color_by, thickness=14, x=1.0,
                                  xanchor="left", len=0.85)),
        line=dict(color="rgba(120,120,120,0.4)", width=2)))
    fig.add_trace(go.Scatter3d(x=[x[0]], y=[y[0]], z=[z[0]], mode="markers",
                  marker=dict(size=6, color="green"), name="start"))
    fig.add_trace(go.Scatter3d(x=[x[-1]], y=[y[-1]], z=[z[-1]], mode="markers",
                  marker=dict(size=6, color="red", symbol="x"), name="end"))
    if show_orient and data.get("quat"):
        ot = _orientation_traces(data, mask)
        if ot is not None:
            fig.add_trace(ot)
    fig.update_layout(
        autosize=True, margin=dict(l=0, r=0, t=36, b=0),
        scene=dict(xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z (m)",
                   aspectmode="data"),
        legend=dict(x=0.0, y=0.99, xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.7)"),
        uirevision="keep3d",
        title=f"3D trajectory — {color_by}   "
              "(drag = rotate · right-click/ctrl-drag = pan · scroll = zoom)")
    return fig


def _slider_marks(t0, t1):
    span = max(t1 - t0, 1e-6)
    step = max(round(span / 8.0), 1)
    marks = {}
    v = int(np.ceil(t0))
    while v <= t1:
        marks[v] = f"{v}s"
        v += step
    return marks


def _header(meta, path):
    import os
    bits = [f"file: {os.path.basename(path)}"]
    for k in ("session", "exptime", "sync_method", "blackbox_file", "vicon_file", "video_file"):
        if meta.get(k):
            bits.append(f"{k}: {meta[k]}")
    extra = []
    for k, fmt in (("sync_offset_s", "offset {:+.3f}s"), ("sync_time_offset", "offset {:+.3f}s"),
                   ("sync_correlation", "corr {:.3f}"), ("sync_yaw_sign", "yaw-sign {:+.0f}"),
                   ("motor_poles", "{:.0f} poles")):
        if k in meta:
            extra.append(fmt.format(meta[k]))
    if extra:
        bits.append(" · ".join(extra))
    return "   |   ".join(bits)


# Fullscreen a pattern-matching wrapper Div (its DOM id is the sorted-key JSON).
_FS_MATCH_JS = """function(n, id){
  if(n){ var domid = JSON.stringify({index:id.index, type:'pgwrap'});
    var el = document.getElementById(domid);
    if(el && el.requestFullscreen){ el.requestFullscreen();
      setTimeout(function(){ window.dispatchEvent(new Event('resize')); }, 300); } }
  return ''; }"""

_FS_ID_JS = """function(n){
  if(n){ var el = document.getElementById('%s');
    if(el && el.requestFullscreen){ el.requestFullscreen();
      setTimeout(function(){ window.dispatchEvent(new Event('resize')); }, 300); } }
  return ''; }"""

GCFG = {"scrollZoom": True, "displaylogo": False, "responsive": True}
_SLIDER_TIP = {"placement": "bottom", "always_visible": False}   # shows only while dragging


def _fs_button(bid):
    return html.Button("⛶ Fullscreen", id=bid, n_clicks=0,
                       style={"margin": "2px 0", "cursor": "pointer", "flex": "0 0 auto",
                              "fontSize": "12px"})


def _section_head(text):
    return html.H4(text, style={"margin": "14px 0 6px", "padding": "4px 0",
                                "borderTop": "2px solid #ccc"})


# 2D panels grouped by DATA SOURCE — drives both the sidebar checklist sections
# and the section headers above the rendered plots. Order = priority; the first
# matching predicate claims a panel, and a trailing "Other" catches the rest so
# nothing is ever hidden.
PANEL_SECTIONS = [
    ("Vicon — motion capture (ground truth)",
     lambda pid: pid in ("pos", "vel", "orient", "quat")),
    ("Commands sent — laptop → drone",
     lambda pid: pid.startswith("cmd_")),
    ("Latency — command vs reception vs response (normalized)",
     lambda pid: pid.startswith("lat_")),
    ("Telemetry — live from drone",
     lambda pid: pid.startswith("tlm_")),
    ("Blackbox — flight controller (FC)",
     lambda pid: pid.startswith("bb_") or pid.startswith("motor_")),
    ("Derived — cross-source",
     lambda pid: pid in ("altrpm", "yawsync")),
]


def _grouped_panels(avail):
    """Return [(section_header, [panel_id, ...]), ...] in display order, each
    panel assigned to exactly one section (first matching predicate wins)."""
    out, claimed = [], set()
    for header, pred in PANEL_SECTIONS:
        ids = [k for k in avail if k not in claimed and pred(k)]
        if ids:
            out.append((header, ids))
            claimed.update(ids)
    rest = [k for k in avail if k not in claimed]
    if rest:
        out.append(("Other", rest))
    return out


_PLOT_SEC_HDR = {"margin": "18px 0 8px", "padding": "5px 10px",
                 "background": "#eef3f8", "borderLeft": "4px solid #1f77b4",
                 "fontSize": "15px", "borderRadius": "3px"}
_SEL_SEC_HDR = {"fontWeight": "600", "fontSize": "12px", "color": "#1f77b4",
                "margin": "8px 0 2px", "borderBottom": "1px solid #ddd",
                "paddingBottom": "2px"}


def make_app(data, title):
    t = data["t"]
    t0, t1 = float(t[0]), float(t[-1])
    for p in data["panels"]:          # attach the shared time base for the figures
        p["_t"] = t
    avail = [p["id"] for p in data["panels"]]
    labels = {p["id"]: p["label"] for p in data["panels"]}
    panel_by_id = {p["id"]: p for p in data["panels"]}
    default = [k for k in DEFAULT_ON if k in avail] or avail[:4]
    color_choices = list(data["color_opts"].keys())

    # Group panels by data source for the sidebar (each section = one checklist,
    # pattern-matching id so one callback gathers them all).
    sections = _grouped_panels(avail)
    selector_blocks = []
    for idx, (header, ids) in enumerate(sections):
        selector_blocks.append(html.Div(children=[
            html.Div(header, style=_SEL_SEC_HDR),
            dcc.Checklist(
                id={"type": "psel", "section": idx},
                options=[{"label": " " + labels[k], "value": k} for k in ids],
                value=[k for k in ids if k in default],
                labelStyle={"display": "block", "fontSize": "13px"}),
        ]))

    app = dash.Dash(title, suppress_callback_exceptions=True)
    app.title = title

    def time_slider(sid, live=False):
        return dcc.RangeSlider(id=sid, min=t0, max=t1, value=[t0, t1],
                               step=max((t1 - t0) / 500.0, 1e-3),
                               marks=_slider_marks(t0, t1), allowCross=False,
                               updatemode="drag" if live else "mouseup",
                               tooltip=_SLIDER_TIP)

    app.layout = html.Div(style={"fontFamily": "system-ui, sans-serif", "margin": "0 14px"},
                          children=[
        html.H3(title, style={"marginBottom": "2px"}),
        html.Div(_header(data["meta"], data["path"]),
                 style={"color": "#555", "fontSize": "13px", "marginBottom": "8px"}),

        # ---------- 2D SECTION ----------
        _section_head("2D plots"),
        html.Div(style={"display": "flex", "gap": "18px", "alignItems": "flex-start"}, children=[
            html.Div(style={"flex": "0 0 250px"}, children=[
                html.Label("Panels — by data source", style={"fontWeight": "600"}),
                html.Div(selector_blocks,
                         style={"maxHeight": "560px", "overflowY": "auto",
                                "border": "1px solid #eee", "borderRadius": "4px",
                                "padding": "2px 8px"})]),
            html.Div(style={"flex": "1 1 auto"}, children=[
                html.Label("Time window — 2D graphs (s)",
                           style={"fontWeight": "600", "fontSize": "13px"}),
                time_slider("win2d"),
                html.Div(id="graphs2d", style={"marginTop": "16px"})]),
        ]),

        # ---------- 3D SECTION ----------
        _section_head("3D trajectory"),
        html.Div(style={"display": "flex", "gap": "18px", "alignItems": "flex-start"}, children=[
            html.Div(style={"flex": "0 0 220px"}, children=[
                dcc.Checklist(id="show3d",
                              options=[{"label": " show 3D", "value": "on"},
                                       {"label": " overlay orientation (L)", "value": "orient"}],
                              value=["on"]),
                html.Label("color by", style={"fontSize": "12px"}),
                dcc.Dropdown(id="color3d",
                             options=[{"label": c, "value": c} for c in color_choices],
                             value="time", clearable=False, style={"width": "180px"})]),
            html.Div(style={"flex": "1 1 auto"}, children=[
                html.Label("Time window — 3D trajectory (s) — live",
                           style={"fontWeight": "600", "fontSize": "13px"}),
                time_slider("win3d", live=True),
                html.Div(id="traj3d-wrap", style={
                    "height": "660px", "display": "flex", "flexDirection": "column",
                    "marginTop": "10px", "background": "#fff"}, children=[
                    _fs_button("td-fs"),
                    # Initial figure at build time (like the 2D panels) so the 3D
                    # renders on load — a callback-only Graph can come up 0-height
                    # inside a flex box and appear "not loading".
                    dcc.Graph(id="traj3d", config=GCFG,
                              figure=build_3d(data, "time", (t0, t1)),
                              style={"flexGrow": 1, "minHeight": 0})]),
                html.Div(id="_fs3", style={"display": "none"})]),
        ]),
        html.Br(), html.Br(), html.Br(), html.Br(),
    ])

    # Build one separate graph per selected panel, GROUPED under its data-source
    # header (Vicon / Commands / Telemetry / Blackbox / Derived). Rebuilt only when
    # the selection changes; the time window keeps its value via State. The section
    # checklists are pattern-matched, so this one callback gathers all of them.
    def _graph_block(p, window):
        return html.Div(
            id={"type": "pgwrap", "index": p["id"]},
            style={"height": "480px", "display": "flex", "flexDirection": "column",
                   "marginBottom": "14px", "background": "#fff",
                   "border": "1px solid #eee"},
            children=[
                _fs_button({"type": "pgfs", "index": p["id"]}),
                dcc.Graph(id={"type": "pg2d", "index": p["id"]},
                          figure=single_panel_fig(p, window),
                          style={"flexGrow": 1, "minHeight": 0}, config=GCFG),
                html.Div(id={"type": "pgfsout", "index": p["id"]},
                         style={"display": "none"}),
            ])

    @app.callback(Output("graphs2d", "children"),
                  Input({"type": "psel", "section": dash.ALL}, "value"),
                  dash.State("win2d", "value"))
    def _build2d(selected_lists, window):
        chosen = {k for lst in (selected_lists or []) for k in (lst or [])}
        if not chosen:
            return html.Div("No panels selected — pick some on the left.",
                            style={"color": "#888", "padding": "20px"})
        out = []
        for header, ids in sections:          # render grouped, in source order
            sel_ids = [k for k in ids if k in chosen and k in panel_by_id]
            if not sel_ids:
                continue
            out.append(html.H4(header, style=_PLOT_SEC_HDR))
            out.extend(_graph_block(panel_by_id[k], window) for k in sel_ids)
        return out

    # Move the time window on all 2D graphs at once — lightweight (range only).
    @app.callback(Output({"type": "pg2d", "index": dash.ALL}, "figure"),
                  Input("win2d", "value"),
                  dash.State({"type": "pg2d", "index": dash.ALL}, "id"))
    def _range2d(window, ids):
        out = []
        for _ in ids:
            patch = dash.Patch()
            patch["layout"]["xaxis"]["range"] = window
            out.append(patch)
        return out

    @app.callback(Output("traj3d", "figure"),
                  Input("show3d", "value"), Input("color3d", "value"), Input("win3d", "value"))
    def _t3(show, color, window):
        if "on" not in (show or []):
            return go.Figure(layout=dict(annotations=[dict(
                text="3D hidden", showarrow=False, font=dict(size=16))]))
        return build_3d(data, color, window, show_orient="orient" in (show or []))

    # Per-graph fullscreen (2D, pattern-matching) + 3D fullscreen.
    app.clientside_callback(_FS_MATCH_JS,
                            Output({"type": "pgfsout", "index": dash.MATCH}, "children"),
                            Input({"type": "pgfs", "index": dash.MATCH}, "n_clicks"),
                            dash.State({"type": "pgfs", "index": dash.MATCH}, "id"),
                            prevent_initial_call=True)
    app.clientside_callback(_FS_ID_JS % "traj3d-wrap",
                            Output("_fs3", "children"), Input("td-fs", "n_clicks"),
                            prevent_initial_call=True)
    return app


def serve(path, title, port=8050, open_browser=True):
    print(f"Loading: {path}")
    data = load_channels(path)
    app = make_app(data, title)
    url = f"http://127.0.0.1:{port}"
    print(f"Dashboard: {url}   ({len(data['panels'])} panels)   Ctrl-C to stop")
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(port=port, debug=False)
