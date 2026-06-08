"""Interactive dashboard for sync_log.py output (*_synced.mat in DataExchange/).

A Plotly Dash web app — opens in your browser, nothing written to disk (use
plot_synced.py for static PNGs). Features:
  - linked-axis zoom/pan across all time-series + a range slider on the bottom
    plot to focus on a section of the flight,
  - a checklist to choose which panels are shown (incl. the yaw-rate sync-check
    overlay that verifies the cross-correlation alignment),
  - a smooth WebGL 3D trajectory you can rotate / pan / zoom (drag, shift-drag,
    scroll), colorable by time / mean RPM / speed.

Reuses plot_synced.py's loader + discovery so it reads the exact same files.
Counterpart to data_logging/dashboard_flight.py (deterministic-sync output).

Usage (run with the repo venv, ../.venv):
  ../.venv/bin/python dashboard_synced.py                       # newest *_synced.mat
  ../.venv/bin/python dashboard_synced.py --pick                # menu
  ../.venv/bin/python dashboard_synced.py DataExchange/foo_synced.mat
  ../.venv/bin/python dashboard_synced.py --port 8060 --no-browser
"""
import os
import sys
import argparse
import threading
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Single-source the .mat parsing + file discovery from the PNG script.
from plot_synced import load_synced, list_synced, select_synced, _euler_deg, _mean_rpm

DATAEXCHANGE = os.path.join(HERE, "DataExchange")

# Canonical panel order + labels/units. A panel only appears if its data exists.
PANELS = [
    ("pos",     "Position (m)",            "m"),
    ("vel",     "Velocity (m/s)",          "m/s"),
    ("orient",  "Orientation (deg)",       "deg"),
    ("rpm",     "Motor RPM (mechanical)",  "RPM"),
    ("erpm",    "Motor eRPM (electrical)", "eRPM"),
    ("cmd",     "Commanded motor output",  "cmd"),
    ("altrpm",  "Altitude vs mean RPM",    "m"),
    ("yawsync", "Yaw-rate sync check",     "rad/s"),
]
DEFAULT_ON = ["pos", "vel", "orient", "rpm", "altrpm", "yawsync"]


def available_panels(d):
    have = {"pos", "orient"}
    if "vx" in d:
        have.add("vel")
    if "rpm" in d:
        have |= {"rpm", "altrpm"}
    if "erpm" in d:
        have.add("erpm")
    if "cmd" in d:
        have.add("cmd")
    if "vicon_yaw_rate" in d and "blackbox_yaw_rate" in d:
        have.add("yawsync")
    return [k for k, _, _ in PANELS if k in have]


def _add_panel(fig, row, key, d, euler):
    t = d["t"]
    def line(y, name, **kw):
        fig.add_trace(go.Scattergl(x=t, y=y, name=name, mode="lines", **kw),
                      row=row, col=1)
    if key == "pos":
        for ax in ("x", "y", "z"):
            line(d[ax], ax)
    elif key == "vel":
        for ax in ("vx", "vy", "vz"):
            line(d[ax], ax)
    elif key == "orient":
        for j, name in enumerate(("yaw", "pitch", "roll")):
            line(euler[:, j], name)
    elif key in ("rpm", "erpm", "cmd"):
        arr = d[key]
        for m in range(arr.shape[1]):
            line(arr[:, m], f"m{m} {key}")
    elif key == "altrpm":
        fig.add_trace(go.Scattergl(x=t, y=d["z"], name="alt z", mode="lines"),
                      row=row, col=1, secondary_y=False)
        mr = _mean_rpm(d)
        if mr is not None:
            fig.add_trace(go.Scattergl(x=t, y=mr, name="mean RPM", mode="lines",
                                       line=dict(color="firebrick")),
                          row=row, col=1, secondary_y=True)
    elif key == "yawsync":
        line(d["vicon_yaw_rate"], "Vicon yaw rate")
        line(d["blackbox_yaw_rate"], "blackbox (aligned)")


def build_timeseries(d, selected):
    sel = [k for k, _, _ in PANELS if k in selected]
    if not sel:
        return go.Figure(layout=dict(height=200,
                         annotations=[dict(text="No panels selected",
                                           showarrow=False, font=dict(size=16))]))
    labels = {k: lbl for k, lbl, _ in PANELS}
    units = {k: u for k, _, u in PANELS}
    specs = [[{"secondary_y": k == "altrpm"}] for k in sel]
    euler = _euler_deg(d)
    fig = make_subplots(rows=len(sel), cols=1, shared_xaxes=True,
                        specs=specs, subplot_titles=[labels[k] for k in sel],
                        vertical_spacing=0.05)
    for i, k in enumerate(sel, start=1):
        _add_panel(fig, i, k, d, euler)
        fig.update_yaxes(title_text=units[k], row=i, col=1)
    if "altrpm" in sel:
        r = sel.index("altrpm") + 1
        fig.update_yaxes(title_text="mean RPM", row=r, col=1, secondary_y=True)
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.06),
                     row=len(sel), col=1)
    fig.update_xaxes(title_text="time (s)", row=len(sel), col=1)
    fig.update_layout(
        height=max(320, 240 * len(sel) + 80),
        margin=dict(l=60, r=20, t=40, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        uirevision="keep",
    )
    return fig


def build_3d(d, color_by):
    t = d["t"]
    if color_by == "rpm":
        c, label, cs = _mean_rpm(d), "mean RPM", "Plasma"
    elif color_by == "speed" and "vx" in d:
        c = np.sqrt(d["vx"]**2 + d["vy"]**2 + d["vz"]**2); label, cs = "speed (m/s)", "Turbo"
    else:
        c, label, cs = t, "time (s)", "Viridis"
    if c is None:
        c, label, cs = t, "time (s)", "Viridis"
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=d["x"], y=d["y"], z=d["z"], mode="markers+lines",
        marker=dict(size=2, color=c, colorscale=cs,
                    colorbar=dict(title=label, thickness=14)),
        line=dict(color="rgba(120,120,120,0.4)", width=2), name="trajectory"))
    fig.add_trace(go.Scatter3d(x=[d["x"][0]], y=[d["y"][0]], z=[d["z"][0]],
                  mode="markers", marker=dict(size=6, color="green"), name="start"))
    fig.add_trace(go.Scatter3d(x=[d["x"][-1]], y=[d["y"][-1]], z=[d["z"][-1]],
                  mode="markers", marker=dict(size=6, color="red", symbol="x"), name="end"))
    fig.update_layout(
        height=620, margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z (m)",
                   aspectmode="data"),
        uirevision="keep3d",
        title="3D trajectory  (drag = rotate · shift-drag = pan · scroll = zoom)")
    return fig


def header_text(d, path):
    bits = [f"file: {os.path.basename(path)}"]
    for k, lbl in (("exptime", "t0"), ("blackbox_file", "bbl"),
                   ("vicon_file", "vicon")):
        v = d.get(k)
        if v:
            bits.append(f"{lbl}: {v}")
    extra = []
    if d.get("offset") is not None:
        extra.append(f"offset {float(d['offset']):+.3f}s")
    if d.get("corr") is not None:
        extra.append(f"corr {float(d['corr']):.3f}")
    if d.get("yaw_sign") is not None:
        extra.append(f"yaw-sign {int(d['yaw_sign']):+d}")
    if d.get("poles") is not None:
        extra.append(f"{int(d['poles'])} poles")
    if extra:
        bits.append(" · ".join(extra))
    return "   |   ".join(bits)


def make_app(d, path, title):
    avail = available_panels(d)
    default = [k for k in DEFAULT_ON if k in avail]
    color_opts = [{"label": "time", "value": "time"}]
    if "rpm" in d:
        color_opts.append({"label": "mean RPM", "value": "rpm"})
    if "vx" in d:
        color_opts.append({"label": "speed", "value": "speed"})

    app = dash.Dash(title)
    app.title = title
    labels = {k: lbl for k, lbl, _ in PANELS}
    ctl = dict(padding="6px 12px", display="inline-block", verticalAlign="top")
    app.layout = html.Div(style={"fontFamily": "system-ui, sans-serif",
                                 "margin": "0 14px"}, children=[
        html.H3(title, style={"marginBottom": "2px"}),
        html.Div(header_text(d, path), style={"color": "#555", "fontSize": "13px",
                                              "marginBottom": "8px"}),
        html.Div(style={"borderBottom": "1px solid #ddd", "paddingBottom": "8px"},
                 children=[
            html.Div(style=ctl, children=[
                html.Label("Panels", style={"fontWeight": "600"}),
                dcc.Checklist(id="panels",
                    options=[{"label": " " + labels[k], "value": k} for k in avail],
                    value=default, labelStyle={"display": "block"})]),
            html.Div(style=ctl, children=[
                html.Label("3D trajectory", style={"fontWeight": "600"}),
                dcc.Checklist(id="show3d", options=[{"label": " show", "value": "on"}],
                              value=["on"]),
                html.Label("color by", style={"fontSize": "12px"}),
                dcc.Dropdown(id="color3d", options=color_opts, value="time",
                             clearable=False, style={"width": "140px"})]),
        ]),
        dcc.Graph(id="timeseries", config={"scrollZoom": True, "displaylogo": False}),
        dcc.Graph(id="traj3d", config={"scrollZoom": True, "displaylogo": False}),
    ])

    @app.callback(Output("timeseries", "figure"), Input("panels", "value"))
    def _ts(selected):
        return build_timeseries(d, set(selected or []))

    @app.callback(Output("traj3d", "figure"), Output("traj3d", "style"),
                  Input("show3d", "value"), Input("color3d", "value"))
    def _t3(show, color):
        if "on" not in (show or []):
            return go.Figure(), {"display": "none"}
        return build_3d(d, color), {"display": "block"}

    return app


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?", help="*_synced.mat (default: newest in DataExchange/)")
    ap.add_argument("--pick", action="store_true", help="choose from a menu")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--no-browser", action="store_true", help="don't auto-open the browser")
    args = ap.parse_args()

    if args.file:
        path = args.file
    elif args.pick:
        path = select_synced(DATAEXCHANGE)
    else:
        path = list_synced(DATAEXCHANGE)[0]
    print(f"Loading: {path}")
    d = load_synced(path)

    app = make_app(d, path, "Synced-log dashboard")
    url = f"http://127.0.0.1:{args.port}"
    print(f"Dashboard: {url}   (Ctrl-C to stop)")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(port=args.port, debug=False)


if __name__ == "__main__":
    main()
