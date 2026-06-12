"""Mission target providers for the Vicon flight loop.

A Mission converts the captured LAUNCH pose into a time-varying world-frame
setpoint (x, y, z, yaw) that the shared flight loop (vicon_hover.run) feeds to the
controller each tick via `controller.set_setpoint`. A Mission only SHAPES THE
REFERENCE — it never touches serial / arming / recording / safety. When `done`
goes True the loop lands exactly as if SPACEBAR were pressed (descend → cut →
disarm → save → exit), so every existing failsafe still wraps the flight.

  HoldMission     — fixed hover at the launch x/y/heading, CLIMB_M up. done is
                    False forever, so the flight ends only on SPACEBAR / low batt /
                    disarm / Vicon loss. This reproduces the original vicon_hover
                    behavior byte-for-byte (vicon_hover.run uses it by default).
  WaypointMission — walk a list of FIXED world waypoints with a moving setpoint
                    ("carrot") at cruise speed and a per-waypoint dwell; done after
                    the final hold. build_square_mission() builds the course.

WHY A MOVING SETPOINT (not a step): the position PID is KP≈15 deg/m, so stepping
the target 2 m would command 30° → clamp to MAX_TILT → the drone slams to max tilt,
races across the leg and overshoots. Instead the carrot crawls from the previous
waypoint toward the next at CRUISE_SPEED_MPS, so the position error — and thus the
commanded tilt — stays small and the motion is gentle. This is the horizontal
analog of the altitude loop's VMAX_UP_MPS cap, just done in the reference generator
instead of inside the controller.

FRAME MATH (the 2026-06-11 flyaway was a 90° frame error — get this right): the
square is specified in the LAUNCH BODY FRAME (forward = nose, right = starboard)
and converted to FIXED world points ONCE, at launch, using the captured launch yaw.
We reuse the controller's own world↔body transform so no NEW convention is
introduced. controller._body_errors uses M = [[c, s], [s, -c]] (c=cos yaw, s=sin
yaw) to send world→body; M is an involution (M·M = I), so body→world is the SAME
matrix:
    dx = cos(yaw)*fwd + sin(yaw)*right
    dy = sin(yaw)*fwd - cos(yaw)*right
At the usual launch (nose +Y ⇒ true heading yaw0 = 90°, since
config.VICON_YAW_OFFSET_DEG = 90 and the controller's heading 0 is nose +X), this
gives (dx, dy) = (right, fwd): forward → +Y, right → +X — the expected square.
Heading is HELD at yaw0 for the whole course (the legs are strafes, not turns).
"""
import math

from . import config


# ----------------------------------------------------------------------------- #
class HoldMission:
    """Static hover at the launch x/y/heading, CLIMB_M above launch altitude.
    Identical to the original vicon_hover target; never finishes on its own."""

    def __init__(self, launch):
        x0, y0, z0, yaw0 = launch
        self._t = (x0, y0, z0 + config.CLIMB_M, yaw0)

    def update(self, pose, dt, airborne):
        x, y, z, yaw = self._t
        return x, y, z, yaw, False

    def status(self):
        return "hold"

    def describe(self):
        x, y, z, yaw = self._t
        return [f"  Mission: static hover at world=({x:+.2f},{y:+.2f},{z:+.2f}) "
                f"yaw={math.degrees(yaw):+.0f}°"]

    def summary(self):
        x, y, z, yaw = self._t
        return {"kind": "hold",
                "target": {"x": x, "y": y, "z": z, "yaw_rad": yaw}}


# ----------------------------------------------------------------------------- #
class WaypointMission:
    """Moving-setpoint sequencer over FIXED world waypoints.

    waypoints: list of (x, y, z, yaw, dwell_s) in the WORLD frame.
    Per waypoint, a two-phase sub-state machine:
      GOTO — once airborne, crawl the carrot (sx, sy) from the previous waypoint
             toward this one at `cruise` m/s, leashed so it never gets more than
             `leash` m ahead of the drone (bounds the error if the drone lags).
             z + yaw are commanded at the waypoint's values throughout.
      HOLD — once the carrot is AT the waypoint AND the drone is within
             `arrive_tol` (horizontal + vertical), park the carrot and dwell for
             dwell_s. An `arrive_timeout` (counted only while airborne) is a
             backstop so a drone that never quite settles can't hang the course.
    `done` goes True after the final waypoint's hold completes; the returned
    setpoint then stays parked on that last waypoint while the loop lands.
    """

    def __init__(self, launch, waypoints, *, cruise_mps, leash_m,
                 arrive_tol_m, arrive_timeout_s, labels=None):
        if not waypoints:
            raise ValueError("WaypointMission needs at least one waypoint")
        self.wps = list(waypoints)
        self.labels = list(labels) if labels else None
        self.cruise = float(cruise_mps)
        self.leash = float(leash_m)
        self.arrive_tol = float(arrive_tol_m)
        self.arrive_timeout = float(arrive_timeout_s)
        # The carrot starts on the first waypoint (== the launch hover); index 0
        # is that takeoff/hover point, so its GOTO is just the vertical climb.
        self.idx = 0
        self.sx, self.sy = launch[0], launch[1]
        self.phase = "GOTO"
        self.t_in_phase = 0.0
        self.dwell_elapsed = 0.0
        self.done = False

    def _advance_carrot(self, wx, wy, pose, dt):
        """Crawl the carrot toward (wx, wy) at cruise speed, leashed to the drone."""
        dx, dy = wx - self.sx, wy - self.sy
        dist = math.hypot(dx, dy)
        if dist < 1e-9:
            return
        step = min(self.cruise * dt, dist)        # never overshoot the waypoint
        ux, uy = dx / dist, dy / dist
        nx, ny = self.sx + ux * step, self.sy + uy * step
        # Leash: if advancing would put the carrot more than leash_m from the
        # drone, pause it this tick and let the drone catch up (bounds the error).
        if self.leash > 0.0 and math.hypot(nx - pose["x"], ny - pose["y"]) > self.leash:
            return
        self.sx, self.sy = nx, ny

    def update(self, pose, dt, airborne):
        """One tick. Returns (x, y, z, yaw, done): the (x, y) is the moving carrot;
        z + yaw are the active waypoint's."""
        wx, wy, wz, wyaw, dwell = self.wps[self.idx]
        if not self.done:
            self.t_in_phase += dt
            if self.phase == "GOTO":
                if airborne:
                    self._advance_carrot(wx, wy, pose, dt)
                carrot_at_wp = math.hypot(wx - self.sx, wy - self.sy) < 1e-3
                drone_close = (
                    math.hypot(wx - pose["x"], wy - pose["y"]) < self.arrive_tol
                    and abs(wz - pose["z"]) < self.arrive_tol)
                timed_out = airborne and self.t_in_phase > self.arrive_timeout
                if (carrot_at_wp and drone_close) or timed_out:
                    self.phase, self.dwell_elapsed = "HOLD", 0.0
            elif self.phase == "HOLD":
                self.sx, self.sy = wx, wy          # park the carrot on the vertex
                self.dwell_elapsed += dt
                if self.dwell_elapsed >= dwell:
                    if self.idx + 1 < len(self.wps):
                        self.idx += 1
                        self.phase, self.t_in_phase = "GOTO", 0.0
                    else:
                        self.done = True
        return self.sx, self.sy, wz, wyaw, self.done

    def status(self):
        if self.done:
            return "MISSION done"
        wp = self.wps[self.idx]
        if self.phase == "GOTO":
            d = math.hypot(wp[0] - self.sx, wp[1] - self.sy)
            return f"WP{self.idx}/{len(self.wps) - 1} goto carrot{d:.2f}m"
        return (f"WP{self.idx}/{len(self.wps) - 1} hold "
                f"{self.dwell_elapsed:.1f}/{wp[4]:.0f}s")

    def describe(self):
        lines = [f"  Mission: {len(self.wps)}-waypoint course "
                 f"(cruise {self.cruise:.2f} m/s, leash {self.leash:.2f} m, "
                 f"arrive tol {self.arrive_tol:.2f} m):"]
        for i, w in enumerate(self.wps):
            lbl = self.labels[i] if self.labels else f"WP{i}"
            lines.append(f"    WP{i} {lbl:13s} world=({w[0]:+.2f},{w[1]:+.2f},"
                         f"{w[2]:+.2f}) yaw={math.degrees(w[3]):+.0f}°  hold {w[4]:.0f}s")
        return lines

    def summary(self):
        return {
            "kind": "waypoint",
            "cruise_mps": self.cruise, "leash_m": self.leash,
            "arrive_tol_m": self.arrive_tol, "arrive_timeout_s": self.arrive_timeout,
            "waypoints_world": [
                {"x": w[0], "y": w[1], "z": w[2], "yaw_rad": w[3], "dwell_s": w[4],
                 "label": (self.labels[i] if self.labels else None)}
                for i, w in enumerate(self.wps)],
        }


# ----------------------------------------------------------------------------- #
def build_square_mission(launch):
    """Build the default course from config: take off + hover, then a LEG_M square
    (forward → right → back → left, holding DWELL_S at each vertex) returning over
    the origin, then land. Body-frame spec → fixed world waypoints via the launch
    yaw (see module docstring for the transform)."""
    x0, y0, z0, yaw0 = launch
    z = z0 + config.CLIMB_M
    c, s = math.cos(yaw0), math.sin(yaw0)
    leg = config.LEG_M
    # Cumulative (forward, right) vertex positions in the LAUNCH BODY FRAME, with
    # the dwell at each: takeoff hover, then the four square corners.
    body = [
        ("takeoff/hover", 0.0, 0.0, config.INITIAL_HOVER_S),
        ("forward",       leg, 0.0, config.DWELL_S),
        ("right",         leg, leg, config.DWELL_S),
        ("back",          0.0, leg, config.DWELL_S),
        ("left/home",     0.0, 0.0, config.DWELL_S),
    ]
    wps, labels = [], []
    for label, fwd, right, dwell in body:
        dx = c * fwd + s * right
        dy = s * fwd - c * right
        wps.append((x0 + dx, y0 + dy, z, yaw0, dwell))
        labels.append(label)
    return WaypointMission(
        launch, wps,
        cruise_mps=config.CRUISE_SPEED_MPS, leash_m=config.LEASH_M,
        arrive_tol_m=config.ARRIVE_TOL_M, arrive_timeout_s=config.ARRIVE_TIMEOUT_S,
        labels=labels)


# ----------------------------------------------------------------------------- #
# Path segments for PathMission. A "move" segment is parametrized by ARC LENGTH s
# (so the carrot keeps a constant ground speed on straights AND curves); a "dwell"
# holds a point for a duration (the carrot waits there).
def _line_seg(p0, p1, label):
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = math.hypot(dx, dy)

    def at(s):
        if length < 1e-9:
            return (p1[0], p1[1])
        f = s / length
        return (p0[0] + dx * f, p0[1] + dy * f)

    return {"type": "move", "at": at, "len": length, "label": label}


def _arc_seg(center, radius, theta0, dtheta, label):
    """Arc of `radius` about `center`, from angle theta0 sweeping dtheta (signed;
    negative = clockwise viewed from above, since world yaw is CCW-positive about
    +Z and the bird's-eye view looks down -Z). Length = radius*|dtheta|."""
    length = radius * abs(dtheta)
    sgn = 1.0 if dtheta >= 0.0 else -1.0

    def at(s):
        th = theta0 + sgn * (s / radius)        # angle swept = s/radius
        return (center[0] + radius * math.cos(th),
                center[1] + radius * math.sin(th))

    return {"type": "move", "at": at, "len": length, "label": label}


def _dwell_seg(point, dur, label):
    return {"type": "dwell", "point": (point[0], point[1]), "dur": dur, "label": label}


class PathMission:
    """Continuous path follower: a carrot crawls along a concatenated parametric
    path (line + arc segments) at `cruise` m/s — leashed to the drone exactly like
    WaypointMission — with optional dwells. Unlike WaypointMission (stop-and-settle
    at each vertex) the carrot FLOWS through move segments without waiting for the
    drone, so curves are smooth; z + yaw are held (config.CLIMB_M above launch, at
    the launch heading). A `dwell` waits for the drone to arrive (within arrive_tol,
    or arrive_timeout) before counting down, so the takeoff hover and the final
    home-settle still gate on the drone actually being there.

    NOTE (from the square flight): at cruise the drone trails the carrot ~0.3 m,
    so on a curve it flies a slightly smaller, phase-lagged path. Fine at 0.4 m/s;
    add velocity feedforward when tightening.
    """

    def __init__(self, launch, segments, *, cruise_mps, leash_m,
                 arrive_tol_m, arrive_timeout_s):
        if not segments:
            raise ValueError("PathMission needs at least one segment")
        self.segs = list(segments)
        self.z = launch[2] + config.CLIMB_M
        self.yaw = launch[3]
        self.cruise = float(cruise_mps)
        self.leash = float(leash_m)
        self.arrive_tol = float(arrive_tol_m)
        self.arrive_timeout = float(arrive_timeout_s)
        self.i = 0                 # current segment index
        self.s = 0.0               # arc length into the current move
        self.cx, self.cy = launch[0], launch[1]   # carrot (starts at launch x/y)
        self.dwell_elapsed = 0.0
        self.t_in_seg = 0.0
        self.done = False

    def _drone_close(self, pose, px, py):
        return (math.hypot(px - pose["x"], py - pose["y"]) < self.arrive_tol
                and abs(self.z - pose["z"]) < self.arrive_tol)

    def _next_seg(self):
        self.i += 1
        self.s = self.dwell_elapsed = self.t_in_seg = 0.0
        if self.i >= len(self.segs):
            self.done = True

    def update(self, pose, dt, airborne):
        if self.done or self.i >= len(self.segs):
            self.done = True
            return self.cx, self.cy, self.z, self.yaw, True
        seg = self.segs[self.i]
        self.t_in_seg += dt
        if seg["type"] == "dwell":
            self.cx, self.cy = seg["point"]
            # Start counting only once the drone is here (or the timeout backstop).
            if (self.dwell_elapsed > 0.0 or self._drone_close(pose, self.cx, self.cy)
                    or self.t_in_seg > self.arrive_timeout):
                self.dwell_elapsed += dt
            if self.dwell_elapsed >= seg["dur"]:
                self._next_seg()
        else:  # move — flow the carrot along the path at cruise, leashed
            if airborne:
                ns = min(self.s + self.cruise * dt, seg["len"])
                nx, ny = seg["at"](ns)
                if not (self.leash > 0.0
                        and math.hypot(nx - pose["x"], ny - pose["y"]) > self.leash):
                    self.s = ns
            self.cx, self.cy = seg["at"](self.s)
            if self.s >= seg["len"] - 1e-9:
                self._next_seg()
        return self.cx, self.cy, self.z, self.yaw, self.done

    def status(self):
        if self.done or self.i >= len(self.segs):
            return "MISSION done"
        seg = self.segs[self.i]
        if seg["type"] == "dwell":
            return f"{seg['label']} {self.dwell_elapsed:.1f}/{seg['dur']:.0f}s"
        pct = 100.0 * self.s / seg["len"] if seg["len"] > 1e-9 else 100.0
        return f"{seg['label']} {pct:3.0f}%"

    def describe(self):
        lines = [f"  Mission: path follow ({len(self.segs)} segments, "
                 f"cruise {self.cruise:.2f} m/s, leash {self.leash:.2f} m) "
                 f"at z={self.z:+.2f} yaw={math.degrees(self.yaw):+.0f}°:"]
        for k, seg in enumerate(self.segs):
            if seg["type"] == "dwell":
                lines.append(f"    [{k}] dwell {seg['label']:11s} "
                             f"at ({seg['point'][0]:+.2f},{seg['point'][1]:+.2f}) "
                             f"for {seg['dur']:.0f}s")
            else:
                end = seg["at"](seg["len"])
                lines.append(f"    [{k}] move  {seg['label']:11s} "
                             f"len={seg['len']:.2f}m → ({end[0]:+.2f},{end[1]:+.2f})")
        return lines

    def summary(self):
        segs = []
        for seg in self.segs:
            if seg["type"] == "dwell":
                segs.append({"type": "dwell", "label": seg["label"],
                             "point": list(seg["point"]), "dwell_s": seg["dur"]})
            else:
                end = seg["at"](seg["len"])
                segs.append({"type": "move", "label": seg["label"],
                             "len_m": seg["len"], "end": list(end)})
        return {"kind": "path", "cruise_mps": self.cruise, "leash_m": self.leash,
                "arrive_tol_m": self.arrive_tol, "z": self.z, "yaw_rad": self.yaw,
                "segments": segs}


# ----------------------------------------------------------------------------- #
def build_circle_mission(launch):
    """Build the circle course from config: take off + hover, fly forward
    CIRCLE_RADIUS_M to the circle (centered on the launch origin, so the forward
    point lands exactly on it), trace one full circle (CIRCLE_CW), return to the
    origin, settle, and land. Heading is held at the launch yaw throughout (the
    circle is flown by translating, not turning)."""
    x0, y0, z0, yaw0 = launch
    c, s = math.cos(yaw0), math.sin(yaw0)
    r = config.CIRCLE_RADIUS_M
    # Forward r in the launch body frame → the circle's start point (body→world via
    # the controller's transform, same as the square).
    start = (x0 + c * r, y0 + s * r)        # (fwd=r, right=0)
    center = (x0, y0)                        # circle centered on the launch origin
    th0 = math.atan2(start[1] - center[1], start[0] - center[0])
    # CW (viewed from above) = negative sweep; CCW = positive (world yaw is CCW+).
    dtheta = (-2.0 * math.pi) if config.CIRCLE_CW else (2.0 * math.pi)
    segs = [
        _dwell_seg((x0, y0), config.INITIAL_HOVER_S, "takeoff/hover"),
        _line_seg((x0, y0), start, "forward"),
        _dwell_seg(start, config.SETTLE_S, "circle-entry"),
        _arc_seg(center, r, th0, dtheta, "circle"),
        _line_seg(start, (x0, y0), "return"),
        _dwell_seg((x0, y0), config.SETTLE_S, "home/settle"),
    ]
    return PathMission(
        launch, segs,
        cruise_mps=config.CRUISE_SPEED_MPS, leash_m=config.LEASH_M,
        arrive_tol_m=config.ARRIVE_TOL_M, arrive_timeout_s=config.ARRIVE_TIMEOUT_S)
