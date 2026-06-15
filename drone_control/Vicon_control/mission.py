"""Mission target providers for the Vicon flight loop.

A Mission converts the captured LAUNCH pose into a time-varying world-frame
setpoint (x, y, z, yaw) PLUS the setpoint's own velocity (svx, svy) that the
shared flight loop (vicon_hover.run) feeds to the controller each tick via
`controller.set_setpoint`. The velocity is the carrot's, known exactly (the
mission moves it), and feeds the controller's D term as velocity FEEDFORWARD:
D acts on (v_carrot - v_drone) instead of (-v_drone), so pacing a moving carrot
no longer reads as "rushing at a fixed target" and the KD*v standing lag
(~1.07 m at 0.8 m/s in flight 20260612_121316 — 55° of phase lag on the circle)
goes away. A Mission only SHAPES THE REFERENCE — it never touches serial /
arming / recording / safety. When `done` goes True the loop lands exactly as if
SPACEBAR were pressed (descend → cut → disarm → save → exit), so every existing
failsafe still wraps the flight.

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
import bisect
import math

from . import config


def _carrot_vel(px, py, cx, cy, dt, vmax):
    """Carrot velocity this tick: the step it just took / dt. Zero during dwells
    and leash pauses (the carrot didn't move). Clamped to vmax because a
    timeout-park can JUMP the carrot a large distance in one tick — the P term
    should see that step, but an unbounded one-tick velocity would kick the
    D term hard."""
    if dt <= 1e-6:
        return 0.0, 0.0
    vx, vy = (cx - px) / dt, (cy - py) / dt
    speed = math.hypot(vx, vy)
    if speed > vmax > 0.0:
        k = vmax / speed
        vx, vy = vx * k, vy * k
    return vx, vy


def _carrot_acc(pvx, pvy, vx, vy, dt):
    """Carrot acceleration this tick: diff of the (already clamped) velocity.
    With the trapezoid this is the ramp accel on straights plus the centripetal
    v²/r on arcs — the signals the ACCELERATION FEEDFORWARD turns directly into
    tilt. Clamped to MAX_FF_ACCEL_MPS2 because a leash engage zeroes the
    velocity in one tick (a real but instant -v/dt spike the FF shouldn't
    relay); the controller LPFs it as well."""
    if dt <= 1e-6:
        return 0.0, 0.0
    ax, ay = (vx - pvx) / dt, (vy - pvy) / dt
    mag = math.hypot(ax, ay)
    amax = config.MAX_FF_ACCEL_MPS2
    if mag > amax > 0.0:
        k = amax / mag
        ax, ay = ax * k, ay * k
    return ax, ay


# ----------------------------------------------------------------------------- #
class HoldMission:
    """Static hover at the launch x/y/heading, CLIMB_M above launch altitude.
    Identical to the original vicon_hover target; never finishes on its own."""

    def __init__(self, launch):
        x0, y0, z0, yaw0 = launch
        self._t = (x0, y0, z0 + config.CLIMB_M, yaw0)

    def update(self, pose, dt, airborne):
        x, y, z, yaw = self._t
        return x, y, z, yaw, 0.0, 0.0, 0.0, 0.0, False

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
        self.accel = float(config.CARROT_ACCEL_MPS2)
        # The carrot starts on the first waypoint (== the launch hover); index 0
        # is that takeoff/hover point, so its GOTO is just the vertical climb.
        self.idx = 0
        self.sx, self.sy = launch[0], launch[1]
        self.v = 0.0               # carrot speed (trapezoid state)
        self.svx, self.svy = 0.0, 0.0   # last reported carrot velocity (for accel)
        self.phase = "GOTO"
        self.t_in_phase = 0.0
        self.dwell_elapsed = 0.0
        self.done = False

    def _advance_carrot(self, wx, wy, pose, dt):
        """Crawl the carrot toward (wx, wy), leashed to the drone. Trapezoid speed:
        accelerate at `accel`, capped by cruise AND by the braking parabola
        sqrt(2*a*dist) so the carrot arrives at the waypoint with zero speed (the
        old instant 0↔cruise steps slammed the velocity feedforward into the tilt
        clamp at every leg transition — flight 20260612_132718)."""
        dx, dy = wx - self.sx, wy - self.sy
        dist = math.hypot(dx, dy)
        if dist < 1e-9:
            self.v = 0.0
            return
        v_next = min(self.cruise, self.v + self.accel * dt,
                     math.sqrt(2.0 * self.accel * dist))
        step = min(v_next * dt, dist)             # never overshoot the waypoint
        ux, uy = dx / dist, dy / dist
        nx, ny = self.sx + ux * step, self.sy + uy * step
        # Leash: if advancing would put the carrot more than leash_m from the
        # drone, pause it this tick and let the drone catch up (bounds the error).
        if self.leash > 0.0 and math.hypot(nx - pose["x"], ny - pose["y"]) > self.leash:
            self.v = 0.0                          # paused → re-ramp on release
            return
        self.sx, self.sy = nx, ny
        self.v = v_next

    def update(self, pose, dt, airborne):
        """One tick. Returns (x, y, z, yaw, svx, svy, done): the (x, y) is the
        moving carrot, (svx, svy) its velocity (for the controller's velocity
        feedforward); z + yaw are the active waypoint's."""
        wx, wy, wz, wyaw, dwell = self.wps[self.idx]
        px, py = self.sx, self.sy
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
                self.v = 0.0
                self.dwell_elapsed += dt
                if self.dwell_elapsed >= dwell:
                    if self.idx + 1 < len(self.wps):
                        self.idx += 1
                        self.phase, self.t_in_phase = "GOTO", 0.0
                    else:
                        self.done = True
        svx, svy = _carrot_vel(px, py, self.sx, self.sy, dt, self.cruise)
        sax, say = _carrot_acc(self.svx, self.svy, svx, svy, dt)
        self.svx, self.svy = svx, svy
        return self.sx, self.sy, wz, wyaw, svx, svy, sax, say, self.done

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
#
# Heading (`yaw` key): move segs take None (hold the current heading target) or
# "tangent" (nose follows the direction of travel — `heading`(s) gives the world
# heading of the path tangent at arc length s). Dwell segs take None or a fixed
# heading (rad) to rotate to during the dwell; the rotation is SLEWED (no step)
# and the dwell countdown waits for the drone's nose to actually get there.
def _wrap_pi(rad):
    while rad > math.pi:
        rad -= 2.0 * math.pi
    while rad < -math.pi:
        rad += 2.0 * math.pi
    return rad


def _line_seg(p0, p1, label, yaw=None):
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = math.hypot(dx, dy)
    hdg = math.atan2(dy, dx)

    def at(s):
        if length < 1e-9:
            return (p1[0], p1[1])
        f = s / length
        return (p0[0] + dx * f, p0[1] + dy * f)

    return {"type": "move", "at": at, "len": length, "label": label,
            "yaw": yaw, "heading": lambda s: hdg,
            # Lossless geometry for offline reconstruction (the dashboard's carrot
            # overlay) — `len`+`end` alone can't tell a line from an arc.
            "geom": {"kind": "line", "p0": [p0[0], p0[1]], "p1": [p1[0], p1[1]]}}


def _arc_seg(center, radius, theta0, dtheta, label, yaw=None):
    """Arc of `radius` about `center`, from angle theta0 sweeping dtheta (signed;
    negative = clockwise viewed from above, since world yaw is CCW-positive about
    +Z and the bird's-eye view looks down -Z). Length = radius*|dtheta|."""
    length = radius * abs(dtheta)
    sgn = 1.0 if dtheta >= 0.0 else -1.0

    def at(s):
        th = theta0 + sgn * (s / radius)        # angle swept = s/radius
        return (center[0] + radius * math.cos(th),
                center[1] + radius * math.sin(th))

    def heading(s):
        # d(at)/ds = sgn*(-sin th, cos th): the travel direction along the arc.
        th = theta0 + sgn * (s / radius)
        return math.atan2(sgn * math.cos(th), -sgn * math.sin(th))

    return {"type": "move", "at": at, "len": length, "label": label,
            "yaw": yaw, "heading": heading,
            "geom": {"kind": "arc", "center": [center[0], center[1]],
                     "radius": radius, "theta0": theta0, "dtheta": dtheta}}


def _dwell_seg(point, dur, label, yaw=None):
    return {"type": "dwell", "point": (point[0], point[1]), "dur": dur,
            "label": label, "yaw": yaw,
            "geom": {"kind": "dwell", "point": [point[0], point[1]]}}


def _param_seg(fn, t0, t1, label, yaw=None, geom=None, n=1000):
    """A general smooth parametric curve — fn(t)->(x, y) for t in [t0, t1] — as a
    MOVE segment, RE-PARAMETRIZED BY ARC LENGTH so the carrot holds a constant
    ground speed along it (a curve's natural parameter usually isn't arc length).
    Densely samples the curve ONCE at build time, builds a cumulative-length table,
    and serves at(s) by interpolation and heading(s) as the smooth path tangent
    (central difference) — so the nose can follow the direction of travel on an
    arbitrary curve, not just lines/arcs. `geom` is the lossless spec the dashboard
    re-traces from (see planned_path). Lines and circles still use _line_seg/_arc_seg
    (closed-form, exact); this is for curves with no closed-form arc length (the
    figure-8 lemniscate)."""
    ts = [t0 + (t1 - t0) * k / n for k in range(n + 1)]
    pts = [fn(t) for t in ts]
    cum = [0.0] * (n + 1)
    for i in range(1, n + 1):
        cum[i] = cum[i - 1] + math.hypot(pts[i][0] - pts[i - 1][0],
                                         pts[i][1] - pts[i - 1][1])
    length = cum[n]

    def _idx(s):
        """Sample interval (i) + fraction (f) for arc length s, clamped to the span."""
        if s <= 0.0:
            return 0, 0.0
        if s >= length:
            return n - 1, 1.0
        i = bisect.bisect_right(cum, s) - 1
        span = cum[i + 1] - cum[i]
        return i, ((s - cum[i]) / span if span > 1e-12 else 0.0)

    def at(s):
        i, f = _idx(s)
        return (pts[i][0] + (pts[i + 1][0] - pts[i][0]) * f,
                pts[i][1] + (pts[i + 1][1] - pts[i][1]) * f)

    def _tan(j):                             # travel tangent at sample j (central chord)
        lo, hi = max(0, j - 1), min(n, j + 1)
        return pts[hi][0] - pts[lo][0], pts[hi][1] - pts[lo][1]

    def heading(s):
        # Interpolate the tangent ACROSS the interval (not piecewise-constant per
        # sample) so the heading is continuous — else the yaw setpoint sees a 1-cm
        # staircase and jitters. Central-chord tangent at each end, blended by f.
        i, f = _idx(s)
        ax, ay = _tan(i)
        bx, by = _tan(i + 1)
        return math.atan2(ay + (by - ay) * f, ax + (bx - ax) * f)

    return {"type": "move", "at": at, "len": length, "label": label,
            "yaw": yaw, "heading": heading, "geom": geom}


def _lemniscate_seg(center, scale, t0, t1, label, yaw=None):
    """Bernoulli lemniscate (the classic ∞) centred at `center`, peaks at
    center ± (scale, 0) on world X, self-crossing at the centre. Its CURVATURE is
    CONTINUOUS — zero at the crossing (the path runs straight through the centre)
    and greatest at the peak tips (radius ≈ scale/3) — so there is no instantaneous
    bank/yaw reversal at the middle the way two tangent circles have. Parametrized
    x = cos t/(1+sin²t), y = sin t cos t/(1+sin²t) (crossing at t=π/2, 3π/2; peaks at
    t=0, π); wrapped in _param_seg for constant-speed arc-length travel + tangent
    heading."""
    cx, cy = center

    def fn(t):
        d = 1.0 + math.sin(t) ** 2
        return (cx + scale * math.cos(t) / d,
                cy + scale * math.sin(t) * math.cos(t) / d)

    geom = {"kind": "lemniscate", "center": [cx, cy], "scale": scale,
            "t0": t0, "t1": t1}
    return _param_seg(fn, t0, t1, label, yaw=yaw, geom=geom)


class PathMission:
    """Continuous path follower: a carrot crawls along a concatenated parametric
    path (line + arc segments) at `cruise` m/s — leashed to the drone exactly like
    WaypointMission — with optional dwells. Unlike WaypointMission (stop-and-settle
    at each vertex) the carrot FLOWS through move segments without waiting for the
    drone, so curves are smooth; z is held at config.CLIMB_M above launch, and the
    heading follows each segment's yaw spec (see HEADING below; default = hold the
    launch heading). A `dwell` waits for the drone to arrive (within arrive_tol,
    or arrive_timeout) before counting down, so the takeoff hover and the final
    home-settle still gate on the drone actually being there.

    NOTE (from the square flight): at cruise the drone used to trail the carrot
    by ~KD*v/KP (0.3 m at 0.4 m/s; ~1 m at 0.8 — flight 20260612_121316 flew a
    0.8 m oval of the 1.0 m circle and was 60° short of closing the lap when the
    carrot moved on). Fixed by the velocity feedforward: update() now also
    returns the carrot velocity and the controller's D term acts on
    (v_carrot - v_drone), so pacing the carrot no longer generates braking tilt.

    SPEED PROFILE (after flight 20260612_132718): the carrot's speed is a
    TRAPEZOID, not a step — it accelerates at `accel` from rest and brakes to
    arrive at every move-segment end with ZERO speed (brake point at v²/2a from
    the end). The old instant 0↔cruise steps made the velocity feedforward slam
    the tilt command into the MAX_TILT clamp at every segment transition and the
    drone overspeed to 1.4 m/s catching up. A leash pause zeroes the carrot
    speed (it re-ramps when released). NOTE: a move that chains directly into
    another move still brakes to zero between them — insert dwells (as the
    circle course does) or extend this if a future course needs flowing joints.

    HEADING: `self.yaw_sp` is the commanded heading, slewed at `yaw_slew` rad/s
    (never stepped, so the yaw P loop is never kicked). Move segs with
    yaw="tangent" track the path's travel direction; dwell segs with a fixed
    heading rotate to it and their countdown additionally WAITS until the
    drone's nose is within `yaw_tol` of the target (same arrive_timeout
    backstop), so e.g. the circle can't start until the pre-rotation finished.
    """

    def __init__(self, launch, segments, *, cruise_mps, leash_m,
                 arrive_tol_m, arrive_timeout_s, accel_mps2=None,
                 yaw_slew_dps=None, yaw_tol_deg=None):
        if not segments:
            raise ValueError("PathMission needs at least one segment")
        self.segs = list(segments)
        self.z = launch[2] + config.CLIMB_M
        self.yaw_sp = launch[3]    # commanded heading (slewed, never stepped)
        self.cruise = float(cruise_mps)
        self.leash = float(leash_m)
        self.arrive_tol = float(arrive_tol_m)
        self.arrive_timeout = float(arrive_timeout_s)
        self.accel = float(accel_mps2 if accel_mps2 is not None
                           else config.CARROT_ACCEL_MPS2)
        self.yaw_slew = math.radians(yaw_slew_dps if yaw_slew_dps is not None
                                     else config.YAW_SLEW_DPS)
        self.yaw_tol = math.radians(yaw_tol_deg if yaw_tol_deg is not None
                                    else config.YAW_ARRIVE_TOL_DEG)
        self.i = 0                 # current segment index
        self.s = 0.0               # arc length into the current move
        self.v = 0.0               # carrot speed (trapezoid state)
        self.svx, self.svy = 0.0, 0.0   # last reported carrot velocity (for accel)
        self.cx, self.cy = launch[0], launch[1]   # carrot (starts at launch x/y)
        self.dwell_elapsed = 0.0
        self.t_in_seg = 0.0
        self.done = False

    def _drone_close(self, pose, px, py):
        return (math.hypot(px - pose["x"], py - pose["y"]) < self.arrive_tol
                and abs(self.z - pose["z"]) < self.arrive_tol)

    def _slew_yaw(self, target, dt):
        """Move the commanded heading toward `target` (shortest way), rate-capped."""
        if target is None:
            return
        err = _wrap_pi(target - self.yaw_sp)
        step = max(-self.yaw_slew * dt, min(self.yaw_slew * dt, err))
        self.yaw_sp = _wrap_pi(self.yaw_sp + step)

    def _yaw_arrived(self, pose, target):
        """Heading setpoint finished slewing AND the drone's nose followed it."""
        if target is None:
            return True
        if abs(_wrap_pi(target - self.yaw_sp)) > 1e-3:
            return False
        pyaw = pose.get("yaw")
        return pyaw is None or abs(_wrap_pi(pyaw - target)) < self.yaw_tol

    def _next_seg(self):
        # Carry the carrot speed straight into the next segment when BOTH this
        # segment and the next are moves (a continuous-tangent joint, e.g. the
        # figure-8's crossover) — flow through instead of braking to a stop.
        # Otherwise (move→dwell, dwell→move, or the course end) reset to rest, as
        # before — so the circle/square (every move is followed by a dwell) are
        # byte-for-byte unchanged.
        flowing = (self.segs[self.i]["type"] == "move"
                   and self.i + 1 < len(self.segs)
                   and self.segs[self.i + 1]["type"] == "move")
        self.i += 1
        self.s = self.dwell_elapsed = self.t_in_seg = 0.0
        if not flowing:
            self.v = 0.0
        if self.i >= len(self.segs):
            self.done = True

    def _exit_speed(self):
        """Carrot speed to AIM FOR at the end of the current move: cruise if the
        next segment is also a move (flow through the joint at speed), else 0 (brake
        to a stop for a dwell / the course end). Flowing is only smooth when the
        joint is tangent-continuous — the course builder owns that (build_figure8_
        mission alternates loop senses so the crossover tangent matches)."""
        nxt = self.i + 1
        if nxt < len(self.segs) and self.segs[nxt]["type"] == "move":
            return self.cruise
        return 0.0

    def update(self, pose, dt, airborne):
        if self.done or self.i >= len(self.segs):
            self.done = True
            return self.cx, self.cy, self.z, self.yaw_sp, 0.0, 0.0, 0.0, 0.0, True
        seg = self.segs[self.i]
        px, py = self.cx, self.cy
        self.t_in_seg += dt
        if seg["type"] == "dwell":
            self.cx, self.cy = seg["point"]
            self._slew_yaw(seg["yaw"], dt)
            # Start counting once the drone is here AND facing the dwell's heading
            # (if it has one) — or the timeout backstop.
            if (self.dwell_elapsed > 0.0
                    or (self._drone_close(pose, self.cx, self.cy)
                        and self._yaw_arrived(pose, seg["yaw"]))
                    or self.t_in_seg > self.arrive_timeout):
                self.dwell_elapsed += dt
            if self.dwell_elapsed >= seg["dur"]:
                self._next_seg()
        else:  # move — flow the carrot along the path, trapezoid speed, leashed
            if airborne:
                # Speed up at `accel`, capped by cruise AND by the braking parabola
                # sqrt(v_exit² + 2*a*remaining) so it reaches the segment end at
                # v_exit: 0 before a dwell/end (brake to a stop, the old behavior),
                # or cruise before another move (flow straight through, e.g. the
                # figure-8 crossover — no tilt-clamp punch from a stop-and-go).
                remaining = max(seg["len"] - self.s, 0.0)
                v_exit = self._exit_speed()
                v_next = min(self.cruise, self.v + self.accel * dt,
                             math.sqrt(v_exit * v_exit + 2.0 * self.accel * remaining))
                ns = min(self.s + v_next * dt, seg["len"])
                nx, ny = seg["at"](ns)
                if (self.leash > 0.0
                        and math.hypot(nx - pose["x"], ny - pose["y"]) > self.leash):
                    self.v = 0.0          # paused by the leash → re-ramp on release
                else:
                    self.s, self.v = ns, v_next
            self.cx, self.cy = seg["at"](self.s)
            if seg["yaw"] == "tangent":
                self._slew_yaw(seg["heading"](self.s), dt)
            else:
                self._slew_yaw(seg["yaw"], dt)
            if self.s >= seg["len"] - 1e-9:
                self._next_seg()
        svx, svy = _carrot_vel(px, py, self.cx, self.cy, dt, self.cruise)
        sax, say = _carrot_acc(self.svx, self.svy, svx, svy, dt)
        self.svx, self.svy = svx, svy
        return self.cx, self.cy, self.z, self.yaw_sp, svx, svy, sax, say, self.done

    def _yaw_label(self, seg):
        if seg["yaw"] is None:
            return "yaw hold"
        if seg["yaw"] == "tangent":
            return "yaw tangent"
        return f"yaw→{math.degrees(seg['yaw']):+.0f}°"

    def status(self):
        if self.done or self.i >= len(self.segs):
            return "MISSION done"
        seg = self.segs[self.i]
        if seg["type"] == "dwell":
            base = f"{seg['label']} {self.dwell_elapsed:.1f}/{seg['dur']:.0f}s"
            if seg["yaw"] is not None and self.dwell_elapsed == 0.0:
                base += f" (rotating {math.degrees(self.yaw_sp):+.0f}°" \
                        f"→{math.degrees(seg['yaw']):+.0f}°)"
            return base
        pct = 100.0 * self.s / seg["len"] if seg["len"] > 1e-9 else 100.0
        return f"{seg['label']} {pct:3.0f}% v={self.v:.2f}"

    def describe(self):
        lines = [f"  Mission: path follow ({len(self.segs)} segments, "
                 f"cruise {self.cruise:.2f} m/s, accel {self.accel:.2f} m/s², "
                 f"leash {self.leash:.2f} m, yaw slew "
                 f"{math.degrees(self.yaw_slew):.0f}°/s) at z={self.z:+.2f}, "
                 f"launch yaw={math.degrees(self.yaw_sp):+.0f}°:"]
        for k, seg in enumerate(self.segs):
            if seg["type"] == "dwell":
                lines.append(f"    [{k}] dwell {seg['label']:11s} "
                             f"at ({seg['point'][0]:+.2f},{seg['point'][1]:+.2f}) "
                             f"for {seg['dur']:.0f}s  [{self._yaw_label(seg)}]")
            else:
                end = seg["at"](seg["len"])
                lines.append(f"    [{k}] move  {seg['label']:11s} "
                             f"len={seg['len']:.2f}m → ({end[0]:+.2f},{end[1]:+.2f})"
                             f"  [{self._yaw_label(seg)}]")
        return lines

    def summary(self):
        segs = []
        for seg in self.segs:
            yaw_spec = (seg["yaw"] if seg["yaw"] in (None, "tangent")
                        else math.degrees(seg["yaw"]))
            # `geom` is the lossless shape (line endpoints / arc center+sweep / dwell
            # point) so an offline tool can re-trace the exact carrot path; `end`/
            # `len_m` stay for human-readability and backward compatibility.
            if seg["type"] == "dwell":
                segs.append({"type": "dwell", "label": seg["label"],
                             "point": list(seg["point"]), "dwell_s": seg["dur"],
                             "yaw": yaw_spec, "geom": seg.get("geom")})
            else:
                end = seg["at"](seg["len"])
                segs.append({"type": "move", "label": seg["label"],
                             "len_m": seg["len"], "end": list(end),
                             "yaw": yaw_spec, "geom": seg.get("geom")})
        return {"kind": "path", "cruise_mps": self.cruise,
                "accel_mps2": self.accel, "leash_m": self.leash,
                "arrive_tol_m": self.arrive_tol, "z": self.z,
                "launch_yaw_rad": self.yaw_sp,
                "yaw_slew_dps": math.degrees(self.yaw_slew),
                "segments": segs}


# ----------------------------------------------------------------------------- #
def build_circle_mission(launch):
    """Build the circle course from config: take off + hover, fly forward
    CIRCLE_RADIUS_M to the circle (centered on the launch origin, so the forward
    point lands exactly on it), trace one full circle (CIRCLE_CW), return to the
    origin, settle, and land.

    Heading: with CIRCLE_FACE_TANGENT the nose follows the direction of travel
    around the circle — the entry dwell pre-rotates to the first tangent (90° to
    the right of the launch heading for CW) and GATES on the nose getting there,
    so the lap never starts with a standing yaw error; the exit dwell rotates
    back to the launch heading before the (strafed) return leg. With the flag
    off, the whole course strafes at the launch yaw like before."""
    x0, y0, z0, yaw0 = launch
    c, s = math.cos(yaw0), math.sin(yaw0)
    r = config.CIRCLE_RADIUS_M
    # Forward r in the launch body frame → the circle's start point (body→world via
    # the controller's transform, same as the square).
    start = (x0 + c * r, y0 + s * r)        # (fwd=r, right=0)
    center = (x0, y0)                        # circle centered on the launch origin
    th0 = math.atan2(start[1] - center[1], start[0] - center[0])
    # CW (viewed from above) = negative sweep; CCW = positive (world yaw is CCW+).
    # CIRCLE_LAPS consecutive laps = one continuous arc (the trapezoid ramps once
    # at the start and once at the end; the laps in between are constant-speed).
    laps = max(1, int(config.CIRCLE_LAPS))
    sweep = 2.0 * math.pi * laps
    dtheta = -sweep if config.CIRCLE_CW else sweep
    arc = _arc_seg(center, r, th0, dtheta, f"circle x{laps}",
                   yaw=("tangent" if config.CIRCLE_FACE_TANGENT else None))
    # Entry/exit headings for the pre-/de-rotation dwells (None = keep current).
    yaw_entry = arc["heading"](0.0) if config.CIRCLE_FACE_TANGENT else None
    yaw_exit = yaw0 if config.CIRCLE_FACE_TANGENT else None
    segs = [
        _dwell_seg((x0, y0), config.INITIAL_HOVER_S, "takeoff/hover"),
        _line_seg((x0, y0), start, "forward"),
        _dwell_seg(start, config.SETTLE_S, "circle-entry", yaw=yaw_entry),
        arc,
        # Exit dwell: dwells gate on the DRONE arriving (arrive_tol/timeout), so
        # any phase lag closes the lap here instead of being cut off when the
        # carrot heads home (flight 20260612_121316 lost the last 60° to this).
        # Also rotates the nose back to the launch heading before heading home.
        _dwell_seg(start, config.SETTLE_S, "circle-exit", yaw=yaw_exit),
        _line_seg(start, (x0, y0), "return"),
        _dwell_seg((x0, y0), config.SETTLE_S, "home/settle"),
    ]
    return PathMission(
        launch, segs,
        cruise_mps=config.CRUISE_SPEED_MPS, leash_m=config.LEASH_M,
        arrive_tol_m=config.ARRIVE_TOL_M, arrive_timeout_s=config.ARRIVE_TIMEOUT_S)


# ----------------------------------------------------------------------------- #
def build_figure8_mission(launch):
    """Build the figure-8 course from config: take off + hover AT the figure-8's
    crossover (which IS the launch origin), trace FIG8_LAPS smooth figure-8s, settle
    back at the origin, land. Structurally the circle's twin — same PathMission, same
    carrot/feedforward/leash/dwell machinery, nose following the travel direction —
    just a different path.

    The path is a BERNOULLI LEMNISCATE (_lemniscate_seg) centred at the launch
    origin, peaks at (x0±FIG8_PEAK_M, y0) on world X, crossing at the origin. Its
    curvature is CONTINUOUS (zero at the crossing → straight through the centre,
    greatest at the peak tips), so there's no instantaneous bank/yaw reversal at the
    middle — that reversal was the awkward, untrackable transition of the old
    two-tangent-circles design (flight 20260615_160611, which lapped the drone at
    2 m/s). It's ONE continuous move segment, so the carrot ramps up once from the
    takeoff hover and brakes once into the home-settle dwell; mid-path (including the
    centre crossing) it flows at cruise.

    Heading: with FIG8_FACE_TANGENT (default ON, like the circle) the nose follows
    the travel direction — the takeoff dwell pre-rotates to the path's start tangent
    and GATES on the nose getting there; the home dwell rotates back to the launch
    heading. The lemniscate + the modest FIG8_SPEED_MPS keep the peak yaw rate within
    authority (see config), which is what makes tangent-facing feasible here.

    Like the old figure-8 there is NO forward approach leg: the crossing is the
    launch point, so the drone is already on the path at takeoff."""
    x0, y0, z0, yaw0 = launch
    laps = max(1, int(config.FIG8_LAPS))
    # The lemniscate parametrization crosses the centre at t=π/2; start there so the
    # course begins at the launch origin. Sweep ±2π·laps — sign = traversal direction
    # (which lobe is flown first / the nose's sweep sense); the two mirror in time.
    t0 = math.pi / 2.0
    span = (-1.0 if config.FIG8_CW else 1.0) * 2.0 * math.pi * laps
    face = config.FIG8_FACE_TANGENT
    lem = _lemniscate_seg((x0, y0), config.FIG8_PEAK_M, t0, t0 + span,
                          f"figure-8 x{laps}", yaw=("tangent" if face else None))
    # Tangent-facing: pre-rotate to the path's start tangent during takeoff, rotate
    # back to the launch heading during the home settle (None = hold heading).
    yaw_start = lem["heading"](0.0) if face else None
    yaw_home = yaw0 if face else None
    segs = [
        _dwell_seg((x0, y0), config.INITIAL_HOVER_S, "takeoff/hover", yaw=yaw_start),
        lem,
        _dwell_seg((x0, y0), config.SETTLE_S, "home/settle", yaw=yaw_home),
    ]
    return PathMission(
        launch, segs,
        cruise_mps=config.FIG8_SPEED_MPS, leash_m=config.LEASH_M,
        arrive_tol_m=config.ARRIVE_TOL_M, arrive_timeout_s=config.ARRIVE_TIMEOUT_S)
