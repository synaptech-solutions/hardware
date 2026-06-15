"""The PLANNED setpoint path of a flight, for the dashboard's 3D overlay.

A mission's path is a pure geometric function (lines + arcs + dwell points), so
the designed course can be drawn by simply EVALUATING that function — no pose, no
replay. This reads the flight's session.json and traces every segment into a dense
(x, y, z) polyline: the course as planned.

(The MOVING setpoint — where the carrot actually was at each instant — is NOT this;
it depends on the drone's pose through the leash + dwell-until-arrived gating, so it
is recorded live during flight as the sp_* columns. This module only draws the
designed path; the dashboard pairs it with the logged sp_* dot.)

Going forward, mission.summary() embeds each segment's lossless `geom` (line
endpoints / arc center+sweep / dwell point), so the trace is EXACT. Older flights
whose session.json predates `geom` fall back to inferring a straight line vs a
launch-centred circular arc from the segment's chord-vs-arc-length.
"""
import json
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_DRONE_CONTROL = os.path.dirname(_HERE)
if _DRONE_CONTROL not in sys.path:
    sys.path.insert(0, _DRONE_CONTROL)


def _yaw_spec(v):
    """session.json yaw field → the form the seg builders expect: None / "tangent"
    / radians (it is stored as degrees)."""
    if v is None or v == "tangent":
        return v
    return math.radians(float(v))


def _signed_sweep(center, r, th0, sweep, entry_yaw):
    """For an inferred (no-`geom`) arc, pick CCW (+) vs CW (-) by matching the start
    tangent to the heading the entry dwell pre-rotated to; default CCW if unknown."""
    from Vicon_control.mission import _arc_seg, _wrap_pi
    if not isinstance(entry_yaw, (int, float)):
        return +sweep                                # config default (CIRCLE_CW=False)
    best, best_err = +sweep, math.inf
    for cand in (+sweep, -sweep):
        h0 = _arc_seg(center, r, th0, cand, "")["heading"](0.0)
        err = abs(_wrap_pi(h0 - entry_yaw))
        if err < best_err:
            best, best_err = cand, err
    return best


def _rebuild_segments(segments):
    """session.json `segments` → mission seg dicts (with the `at`/`len` callables).
    Prefers the lossless `geom` block; infers line-vs-launch-centred-arc otherwise."""
    from Vicon_control.mission import _line_seg, _arc_seg, _dwell_seg, _lemniscate_seg

    origin = next((tuple(s["point"]) for s in segments if s["type"] == "dwell"), None)
    out, cur, prev_dwell_yaw = [], origin, None
    for s in segments:
        yaw = _yaw_spec(s.get("yaw"))
        geom = s.get("geom")
        if s["type"] == "dwell":
            pt = tuple(geom["point"]) if geom else tuple(s["point"])
            out.append(_dwell_seg(pt, float(s.get("dwell_s", 0.0)), s.get("label", ""), yaw=yaw))
            cur, prev_dwell_yaw = pt, yaw
            continue
        # ---- move segment ----
        if geom and geom.get("kind") == "line":
            seg = _line_seg(tuple(geom["p0"]), tuple(geom["p1"]), s.get("label", ""), yaw=yaw)
        elif geom and geom.get("kind") == "arc":
            seg = _arc_seg(tuple(geom["center"]), float(geom["radius"]),
                           float(geom["theta0"]), float(geom["dtheta"]),
                           s.get("label", ""), yaw=yaw)
        elif geom and geom.get("kind") == "lemniscate":
            seg = _lemniscate_seg(tuple(geom["center"]), float(geom["scale"]),
                                  float(geom["t0"]), float(geom["t1"]),
                                  s.get("label", ""), yaw=yaw)
        else:                                        # no geom → infer from the chord
            end = tuple(s["end"])
            start = cur if cur is not None else end
            chord = math.hypot(end[0] - start[0], end[1] - start[1])
            length = float(s["len_m"])
            if length <= chord + 0.05:               # straight: arc length ≈ chord
                seg = _line_seg(start, end, s.get("label", ""), yaw=yaw)
            else:                                    # closed/curved → launch-centred arc
                center = origin if origin is not None else start
                r = math.hypot(start[0] - center[0], start[1] - center[1])
                if r < 1e-6:
                    seg = _line_seg(start, end, s.get("label", ""), yaw=yaw)
                else:
                    laps = max(1, round(length / (2.0 * math.pi * r)))
                    th0 = math.atan2(start[1] - center[1], start[0] - center[0])
                    dtheta = _signed_sweep(center, r, th0, 2.0 * math.pi * laps, prev_dwell_yaw)
                    seg = _arc_seg(center, r, th0, dtheta, s.get("label", ""), yaw=yaw)
        out.append(seg)
        cur = seg["at"](seg["len"])
    return out


def _trace_segments(segs, z, n_per_m=40.0):
    """Densely sample move segments (and include dwell vertices) → (x, y, z) arrays."""
    xs, ys = [], []
    for seg in segs:
        if seg["type"] == "dwell":
            xs.append(seg["point"][0]); ys.append(seg["point"][1])
        else:
            n = max(2, int(seg["len"] * n_per_m))
            for k in range(n + 1):
                px, py = seg["at"](seg["len"] * k / n)
                xs.append(px); ys.append(py)
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    return xs, ys, np.full(xs.shape, float(z))


def planned_path(flight_path):
    """Trace the planned setpoint path for a synced flight (its dir must hold
    session.json). Returns {ref_x, ref_y, ref_z, kind} or None (no/unsupported
    mission, or hand-flown flight). Best-effort: never raises."""
    try:
        sess_path = os.path.join(os.path.dirname(os.path.abspath(flight_path)),
                                 "session.json")
        if not os.path.isfile(sess_path):
            return None
        with open(sess_path) as f:
            sess = json.load(f)
        mission = sess.get("mission")
        if not mission:
            return None
        kind = mission.get("kind")
        controller = sess.get("controller") or {}
        tgt = controller.get("target") or {}

        if kind == "path":
            segs = _rebuild_segments(mission["segments"])
            if not segs:
                return None
            z = mission.get("z", tgt.get("z", 0.0))
            rx, ry, rz = _trace_segments(segs, z)
        elif kind == "waypoint":
            wps = mission["waypoints_world"]
            rx = np.array([w["x"] for w in wps], float)
            ry = np.array([w["y"] for w in wps], float)
            rz = np.array([w["z"] for w in wps], float)
        elif kind == "hold":
            t = mission.get("target", tgt)
            rx = np.array([t["x"]], float); ry = np.array([t["y"]], float)
            rz = np.array([t["z"]], float)
        else:
            return None
        return {"ref_x": rx, "ref_y": ry, "ref_z": rz, "kind": kind}
    except Exception:                                # never break the dashboard
        return None
