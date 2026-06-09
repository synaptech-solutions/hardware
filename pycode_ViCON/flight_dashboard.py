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
import re
import threading
import warnings
import webbrowser

import numpy as np
import scipy.io as sio
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
    "gyroADC":    ("Angular velocity (gyro)", "rad/s", ["roll rate", "pitch rate", "yaw rate"]),
    "gyroUnfilt": ("Angular velocity (unfiltered)", "rad/s", ["roll", "pitch", "yaw"]),
    "accSmooth":  ("Acceleration", "blackbox units", ["ax", "ay", "az"]),
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

DEFAULT_ON = ["pos", "vel", "orient", "motor_rpm", "bb_gyroADC", "altrpm", "yawsync"]


def _euler_deg(qx, qy, qz, qw):
    quat = np.column_stack([qx, qy, qz, qw])
    norms = np.linalg.norm(quat, axis=1)
    quat[norms < 1e-6] = [0.0, 0.0, 0.0, 1.0]
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    return Rotation.from_quat(quat).as_euler("zyx", degrees=True)   # yaw,pitch,roll


def load_channels(path):
    """Parse a synced .mat into a dashboard data dict: time base, panel list
    (each with colored series), 3D pose + color-by options, and metadata."""
    m = sio.loadmat(path)
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
    (the wrapper Div sets the box, so fullscreen fills the screen)."""
    if panel["secondary"]:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        for name, color, yv in panel["series"]:
            fig.add_trace(go.Scattergl(x=None, y=yv, name=name, mode="lines",
                                       line=dict(color=color)), secondary_y=False)
        for name, color, yv, u in panel["secondary"]:
            fig.add_trace(go.Scattergl(x=None, y=yv, name=name, mode="lines",
                                       line=dict(color=color)), secondary_y=True)
        fig.update_yaxes(title_text=panel["unit"], secondary_y=False)
        fig.update_yaxes(title_text=panel["secondary"][0][3], secondary_y=True)
    else:
        fig = go.Figure()
        for name, color, yv in panel["series"]:
            fig.add_trace(go.Scattergl(x=None, y=yv, name=name, mode="lines",
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
    # Point-sized: one inter-point spacing for the long arm.
    if n > 1:
        d = np.linalg.norm(np.diff(P, axis=0), axis=1)
        d = d[d > 0]
        l_long = float(np.median(d)) if d.size else 1e-3
    else:
        l_long = 1e-3
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
            html.Div(style={"flex": "0 0 220px"}, children=[
                html.Label("Panels", style={"fontWeight": "600"}),
                dcc.Checklist(id="panels",
                    options=[{"label": " " + labels[k], "value": k} for k in avail],
                    value=default, labelStyle={"display": "block"},
                    style={"maxHeight": "360px", "overflowY": "auto"})]),
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
                    dcc.Graph(id="traj3d", config=GCFG,
                              style={"flexGrow": 1, "minHeight": 0})]),
                html.Div(id="_fs3", style={"display": "none"})]),
        ]),
        html.Br(), html.Br(), html.Br(), html.Br(),
    ])

    # Build one separate graph per selected panel (rebuilt only when the
    # selection changes; the time window keeps its value via State).
    @app.callback(Output("graphs2d", "children"),
                  Input("panels", "value"), dash.State("win2d", "value"))
    def _build2d(selected, window):
        sel = [panel_by_id[k] for k in (selected or []) if k in panel_by_id]
        if not sel:
            return html.Div("No panels selected — pick some on the left.",
                            style={"color": "#888", "padding": "20px"})
        out = []
        for p in sel:
            out.append(html.Div(
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
                ]))
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
