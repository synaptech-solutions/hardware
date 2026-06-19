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
                    the final hold. build_waypoint_mission() builds it from the
                    absolute-world-coordinate course in config.WAYPOINTS.

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

    def update(self, pose, dt, airborne, land_requested=False):
        x, y, z, yaw = self._t
        # A static hover never finishes on its own; a controlled-land request
        # (SPACEBAR) winds it down immediately by reporting done.
        return x, y, z, yaw, 0.0, 0.0, 0.0, 0.0, bool(land_requested)

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
class SyncSpinMission:
    """Bookend any inner Mission with a deliberate 360° flat yaw SPIN for post-flight
    clock + latency sync. Sequence (all at the launch-hover height):

        settle → SPIN → settle → [inner mission] → settle → SPIN → settle → done

    Each spin slews the heading setpoint at config.SPIN_RATE_DPS while holding x/y/z,
    so the controller's yaw-rate FF turns it into a clean constant-rate rotation that
    BOTH Vicon (laptop clock) and the FC gyro (FC clock) record — the loud, sharp
    event combine.py cross-correlates to recover the laptop↔drone uplink latency and,
    from the start-vs-end spins, the FC↔laptop clock drift. See config's sync-spin
    section and the latency analysis notes.

    The EXIT bookend runs on a CONTROLLED land (inner done, or SPACEBAR via
    land_requested) — never on an emergency (low batt / Vicon loss / disarm), which
    the flight loop services immediately and bypasses this. `done` goes True only
    after the exit bookend, so the loop's land-on-done path wraps the whole thing.

    Disabled (config.SYNC_SPIN_ENABLED=False) → the loop flies the inner mission
    directly, so this composes transparently with hover / waypoint / path flights.
    The exit bookend spins where the inner mission FINISHED (its last setpoint), not
    back at launch, so a course that ends downrange doesn't fly home first.
    """

    def __init__(self, launch, inner):
        x0, y0, z0, yaw0 = launch
        self.inner = inner
        self.settle = config.SPIN_SETTLE_S
        self.sweep = 2.0 * math.pi * config.SPIN_TURNS
        self.rate = math.radians(config.SPIN_RATE_DPS) * (1.0 if config.SPIN_DIR >= 0 else -1.0)
        self.hold = (x0, y0, z0 + config.CLIMB_M)   # hold point for the current bookend
        self.yaw_base = yaw0          # heading the active spin rotates about / settles on
        self.yaw_sp = yaw0            # commanded heading (slewed during a spin)
        self.last_inner = None        # inner's last (x, y, z, yaw) → the exit hold point
        self.phase = "settle_entry"
        self.t_in_phase = 0.0
        self.spun = 0.0               # |rad| swept in the active spin
        self.climb_done = False       # entry spin waits for the climb to finish
        self.t_climb = 0.0            # time spent waiting to reach climb height
        self.done = False

    def _hover(self):
        hx, hy, hz = self.hold
        return hx, hy, hz, self.yaw_sp, 0.0, 0.0, 0.0, 0.0, self.done

    def _begin_spin(self):
        self.t_in_phase, self.spun = 0.0, 0.0
        self.yaw_base = self.yaw_sp                  # spin about the currently-held heading

    def _to_exit(self):
        """Enter the exit bookend, holding where the inner mission left off."""
        if self.last_inner is not None:
            x, y, z, yaw = self.last_inner
            self.hold, self.yaw_sp = (x, y, z), yaw
        self.phase, self.t_in_phase, self.spun = "settle_exit", 0.0, 0.0

    def _advance_spin(self, dt):
        """Slew the heading at the spin rate; True when the full sweep is done,
        snapping the setpoint exactly onto the start heading (whole turns) so no
        fractional overshoot remains and the FF rate drops cleanly to zero."""
        self.spun += abs(self.rate) * dt
        if self.spun >= self.sweep:
            self.yaw_base += math.copysign(self.sweep, self.rate)   # ≡ start heading (mod 2π)
            self.yaw_sp = self.yaw_base
            return True
        self.yaw_sp = self.yaw_base + math.copysign(self.spun, self.rate)
        return False

    def update(self, pose, dt, airborne, land_requested=False):
        if self.done:
            return self._hover()
        # Hold level hover through the straight-up takeoff; the entry spin only starts
        # once airborne at the climb height.
        if not airborne:
            return self._hover()
        # Wait for the climb to (near-)complete before the entry bookend — spinning
        # mid-climb (flight 20260617_131652: spun at 0.6 m of a 1.0 m target) tilts
        # the still-rising drone. Hold the climb target until within tol (or timeout).
        if self.phase == "settle_entry" and not self.climb_done:
            if abs(pose["z"] - self.hold[2]) <= config.SPIN_CLIMB_TOL_M:
                self.climb_done = True
            else:
                self.t_climb += dt
                if self.t_climb >= config.SPIN_CLIMB_TIMEOUT_S:
                    self.climb_done = True       # backstop: proceed anyway
                else:
                    return self._hover()         # keep climbing; don't start the spin
        # A controlled land before/while the inner mission runs → jump to the exit
        # bookend, so even an aborted course still gets a closing spin to sync on.
        if land_requested and self.phase in ("settle_entry", "spin_entry",
                                             "settle_mid", "inner"):
            self._to_exit()

        self.t_in_phase += dt
        if self.phase == "settle_entry":
            if self.t_in_phase >= self.settle:
                self.phase = "spin_entry"
                self._begin_spin()
        elif self.phase == "spin_entry":
            if self._advance_spin(dt):
                self.phase, self.t_in_phase = "settle_mid", 0.0
        elif self.phase == "settle_mid":
            if self.t_in_phase >= self.settle:
                self.phase, self.t_in_phase = "inner", 0.0
        elif self.phase == "inner":
            r = self.inner.update(pose, dt, airborne)
            self.last_inner = (r[0], r[1], r[2], r[3])
            if r[8]:                                  # inner finished → exit bookend
                self._to_exit()
            else:
                return r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], False
        elif self.phase == "settle_exit":
            if self.t_in_phase >= self.settle:
                self.phase = "spin_exit"
                self._begin_spin()
        elif self.phase == "spin_exit":
            if self._advance_spin(dt):
                self.phase, self.t_in_phase = "settle_end", 0.0
        elif self.phase == "settle_end":
            if self.t_in_phase >= self.settle:
                self.done = True
        return self._hover()

    def status(self):
        if self.done:
            return "sync-spin done"
        if self.phase == "inner":
            return f"inner:{self.inner.status()}"
        if self.phase in ("spin_entry", "spin_exit"):
            return f"{self.phase} {math.degrees(self.spun):.0f}/{math.degrees(self.sweep):.0f}°"
        return self.phase

    def describe(self):
        lines = [f"  Sync-spin bookends: settle {self.settle:.0f}s → "
                 f"{config.SPIN_TURNS:.0f}×360° @ {config.SPIN_RATE_DPS:.0f}°/s "
                 f"({'CCW' if self.rate >= 0 else 'CW'}) → settle, before AND after."]
        lines += self.inner.describe()
        return lines

    def summary(self):
        return {"kind": "sync_spin",
                "rate_dps": config.SPIN_RATE_DPS, "turns": config.SPIN_TURNS,
                "settle_s": self.settle, "dir": "ccw" if self.rate >= 0 else "cw",
                "inner": self.inner.summary()}


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
                 arrive_tol_m, arrive_timeout_s, labels=None, face_path=False,
                 yaw_slew_dps=None, yaw_tol_deg=None):
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
        # Heading: hold the launch yaw (strafe), OR follow the path — nose points
        # along the leg of travel, pre-rotating to the next leg during each dwell
        # (same slew + arrival-gate machinery PathMission uses for the circle).
        self.launch_x, self.launch_y, self.launch_yaw = launch[0], launch[1], launch[3]
        self.face_path = bool(face_path)
        self.yaw_sp = launch[3]                       # commanded heading (slewed, never stepped)
        self.yaw_slew = math.radians(yaw_slew_dps if yaw_slew_dps is not None
                                     else config.YAW_SLEW_DPS)
        self.yaw_tol = math.radians(yaw_tol_deg if yaw_tol_deg is not None
                                    else config.YAW_ARRIVE_TOL_DEG)

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

    def _heading_target(self):
        """The world heading the nose should hold. Strafe → launch yaw. Face-path →
        the current leg's travel direction while moving; while holding at a vertex,
        PRE-ROTATE to the next leg's direction (so the next leg starts nose-aligned).
        Returns None (= hold current yaw_sp) when there's no leg to face."""
        if not self.face_path:
            return self.launch_yaw
        i = self.idx
        here = (self.wps[i][0], self.wps[i][1])
        if self.phase == "GOTO":
            prev = ((self.wps[i - 1][0], self.wps[i - 1][1]) if i >= 1
                    else (self.launch_x, self.launch_y))
            if math.hypot(here[0] - prev[0], here[1] - prev[1]) > 1e-6:
                return math.atan2(here[1] - prev[1], here[0] - prev[0])
        # HOLD, or a zero-length GOTO (e.g. takeoff at the launch point): face the
        # NEXT leg if there is one, else just hold the current heading.
        if i + 1 < len(self.wps):
            nxt = (self.wps[i + 1][0], self.wps[i + 1][1])
            if math.hypot(nxt[0] - here[0], nxt[1] - here[1]) > 1e-6:
                return math.atan2(nxt[1] - here[1], nxt[0] - here[0])
        return None

    def update(self, pose, dt, airborne, land_requested=False):
        """One tick. Returns (x, y, z, yaw, svx, svy, sax, say, done): the (x, y) is
        the moving carrot, (svx, svy) its velocity (controller D feedforward); z is
        the active waypoint's; yaw is the held launch heading (strafe) or the slewed
        path-following heading (face_path)."""
        if land_requested:
            self.done = True                   # SPACEBAR → abandon the course, land
        wx, wy, wz, wyaw, dwell = self.wps[self.idx]
        px, py = self.sx, self.sy
        if not self.done:
            self.t_in_phase += dt
            tgt = self._heading_target()       # launch yaw (strafe) or the leg/next-leg dir
            self._slew_yaw(tgt, dt)
            if self.phase == "GOTO":
                if airborne:
                    self._advance_carrot(wx, wy, pose, dt)
                carrot_at_wp = math.hypot(wx - self.sx, wy - self.sy) < 1e-3
                drone_close = (
                    math.hypot(wx - pose["x"], wy - pose["y"]) < self.arrive_tol
                    and abs(wz - pose["z"]) < self.arrive_tol)
                timed_out = airborne and self.t_in_phase > self.arrive_timeout
                if (carrot_at_wp and drone_close) or timed_out:
                    self.phase, self.dwell_elapsed, self.t_in_phase = "HOLD", 0.0, 0.0
            elif self.phase == "HOLD":
                self.sx, self.sy = wx, wy          # park the carrot on the vertex
                self.v = 0.0
                # Face-path: count the dwell only once the nose has pre-rotated to the
                # next leg (same gate PathMission uses); timeout backstop so a never-
                # quite-aligned nose can't hang the course. Strafe: yaw_ok is always
                # True, so this is byte-identical to the old behavior.
                yaw_ok = (not self.face_path) or self._yaw_arrived(pose, tgt)
                if self.dwell_elapsed > 0.0 or yaw_ok or self.t_in_phase > self.arrive_timeout:
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
        return self.sx, self.sy, wz, self.yaw_sp, svx, svy, sax, say, self.done

    def status(self):
        if self.done:
            return "MISSION done"
        wp = self.wps[self.idx]
        head = f" hd{math.degrees(self.yaw_sp):+.0f}°" if self.face_path else ""
        if self.phase == "GOTO":
            d = math.hypot(wp[0] - self.sx, wp[1] - self.sy)
            return f"WP{self.idx}/{len(self.wps) - 1} goto carrot{d:.2f}m{head}"
        return (f"WP{self.idx}/{len(self.wps) - 1} hold "
                f"{self.dwell_elapsed:.1f}/{wp[4]:.0f}s{head}")

    def describe(self):
        mode = ("nose follows path" if self.face_path
                else f"heading held at launch yaw {math.degrees(self.launch_yaw):+.0f}°")
        lines = [f"  Mission: {len(self.wps)}-waypoint course "
                 f"(cruise {self.cruise:.2f} m/s, leash {self.leash:.2f} m, "
                 f"arrive tol {self.arrive_tol:.2f} m; {mode}):"]
        for i, w in enumerate(self.wps):
            lbl = self.labels[i] if self.labels else f"WP{i}"
            tail = f"  yaw {math.degrees(w[3]):+.0f}°"
            if self.face_path and i + 1 < len(self.wps):
                nxt = self.wps[i + 1]
                if math.hypot(nxt[0] - w[0], nxt[1] - w[1]) > 1e-6:
                    tail = f"  → leg head {math.degrees(math.atan2(nxt[1]-w[1], nxt[0]-w[0])):+.0f}°"
            lines.append(f"    WP{i} {lbl:13s} world=({w[0]:+.2f},{w[1]:+.2f},"
                         f"{w[2]:+.2f})  hold {w[4]:.0f}s{tail}")
        return lines

    def summary(self):
        return {
            "kind": "waypoint", "face_path": self.face_path,
            "cruise_mps": self.cruise, "leash_m": self.leash,
            "arrive_tol_m": self.arrive_tol, "arrive_timeout_s": self.arrive_timeout,
            "waypoints_world": [
                {"x": w[0], "y": w[1], "z": w[2], "yaw_rad": w[3], "dwell_s": w[4],
                 "label": (self.labels[i] if self.labels else None)}
                for i, w in enumerate(self.wps)],
        }


# ----------------------------------------------------------------------------- #
def build_waypoint_mission(launch):
    """Build a WaypointMission from config.WAYPOINTS — points in ABSOLUTE VICON WORLD
    coordinates (x, y, z), used AS-IS (no launch rotation or offset), so a waypoint at
    world (−2, 0) is exactly there regardless of where/which-way the drone launched.
    Per-point z (meters); z=None → CLIMB_M above the launch altitude. Heading is HELD
    at the captured launch yaw throughout (the drone strafes, nose fixed). The carrot
    starts at the launch position and crawls to WP0 first (make WP0 the takeoff point)."""
    x0, y0, z0, yaw0 = launch
    if not config.WAYPOINTS:
        raise ValueError("config.WAYPOINTS is empty — define at least the WP0 takeoff point")
    wps, labels = [], []
    for x, y, z, dwell, label in config.WAYPOINTS:
        zw = z0 + config.CLIMB_M if z is None else float(z)
        wps.append((float(x), float(y), zw, yaw0, float(dwell)))
        labels.append(label)
    return WaypointMission(
        launch, wps,
        cruise_mps=config.CRUISE_SPEED_MPS, leash_m=config.LEASH_M,
        arrive_tol_m=config.ARRIVE_TOL_M, arrive_timeout_s=config.ARRIVE_TIMEOUT_S,
        labels=labels, face_path=config.WAYPOINT_FACE_PATH)


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


def _arc_seg(center, radius, theta0, dtheta, label, yaw=None, z0=None, z1=None,
             z_sine=None):
    """Arc of `radius` about `center`, from angle theta0 sweeping dtheta (signed;
    negative = clockwise viewed from above, since world yaw is CCW-positive about
    +Z and the bird's-eye view looks down -Z). Length = radius*|dtheta|.

    Optional VERTICAL profile along the arc length s (turns the flat circle into a
    3D path; the (x,y) is unchanged either way). At most one of:
      z0/z1   linear ramp z0→z1 over the arc — a HELIX.
      z_sine  (z_mid, amp, cycles, phase): z = z_mid + amp·sin(2π·cycles·(s/len)+phase)
              — the height oscillates `cycles` full sine periods over the whole arc
              (so the carrot rides up and down while it laps the circle).
    Omit both → flat arc (the plain circle/figure-8, unchanged)."""
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

    def _u(s):                                  # normalized arc position in [0, 1]
        return (min(s, length) / length) if length > 1e-9 else 0.0

    geom = {"kind": "arc", "center": [center[0], center[1]],
            "radius": radius, "theta0": theta0, "dtheta": dtheta}
    seg = {"type": "move", "at": at, "len": length, "label": label,
           "yaw": yaw, "heading": heading, "geom": geom}
    if z_sine is not None:
        z_mid, amp, cycles, phase = z_sine
        geom["z_sine"] = {"z_mid": z_mid, "amp": amp, "cycles": cycles, "phase": phase}
        seg["z_at"] = lambda s: z_mid + amp * math.sin(
            2.0 * math.pi * cycles * _u(s) + phase)
    elif z0 is not None and z1 is not None:
        geom["z0"], geom["z1"] = z0, z1
        seg["z_at"] = lambda s: z0 + (z1 - z0) * _u(s)
    return seg


def _dwell_seg(point, dur, label, yaw=None):
    return {"type": "dwell", "point": (point[0], point[1]), "dur": dur,
            "label": label, "yaw": yaw,
            "geom": {"kind": "dwell", "point": [point[0], point[1]]}}


def _param_seg(p, u0, u1, label, yaw, geom, samples=2000):
    """Arc-length-parameterized segment for an arbitrary smooth planar curve p(u),
    u in [u0, u1]. Densely samples the curve once to build a cumulative arc-length
    table, so at(s) advances the carrot at UNIT speed in arc length (like the line
    and arc segs — the trapezoid speed profile assumes that) and heading(s) is the
    local travel direction. No analytic derivative needed: heading comes from the
    sample spacing, which is plenty smooth at the default density."""
    n = max(2, int(samples))
    us = [u0 + (u1 - u0) * k / n for k in range(n + 1)]
    pts = [p(u) for u in us]
    cum = [0.0] * (n + 1)
    for k in range(1, n + 1):
        cum[k] = cum[k - 1] + math.hypot(pts[k][0] - pts[k - 1][0],
                                         pts[k][1] - pts[k - 1][1])
    length = cum[n]

    def _idx(s):
        # bracketing sample index i and fraction f into [pts[i], pts[i+1]]
        if s <= 0.0:
            return 0, 0.0
        if s >= length:
            return n - 1, 1.0
        i = bisect.bisect_right(cum, s) - 1
        span = cum[i + 1] - cum[i]
        return i, (0.0 if span < 1e-12 else (s - cum[i]) / span)

    def at(s):
        i, f = _idx(s)
        (x0, y0), (x1, y1) = pts[i], pts[i + 1]
        return (x0 + (x1 - x0) * f, y0 + (y1 - y0) * f)

    def heading(s):
        i, _ = _idx(s)                       # forward diff = travel direction
        (x0, y0), (x1, y1) = pts[i], pts[i + 1]
        return math.atan2(y1 - y0, x1 - x0)

    return {"type": "move", "at": at, "len": length, "label": label,
            "yaw": yaw, "heading": heading, "geom": geom}


def _lemniscate_seg(center, a, label, yaw=None, cw=False, laps=1):
    """Smooth figure-8 — the lemniscate of BERNOULLI — centred at `center`, long
    axis along world X, half-span `a` (the far ends sit at center ± (a, 0)). Started
    at the CROSSOVER (parameter t0 = π/2) so the carrot begins and ends at `center`,
    and traced for `laps` full loops.

    Unlike two tangent circles (whose curvature FLIPS sign +1/R → -1/R at the
    crossover — an instant lateral-accel reversal the drone can't track), this is a
    single C-∞ curve: curvature is CONTINUOUS, zero at the crossover and peaking at
    3/a at the far ends. cw flips the loop sense (sign of y)."""
    cx, cy = center
    sgn = -1.0 if cw else 1.0

    def p(t):
        d = 1.0 + math.sin(t) ** 2
        return (cx + a * math.cos(t) / d, cy + sgn * a * math.sin(t) * math.cos(t) / d)

    laps = max(1, int(laps))
    t0 = 0.5 * math.pi
    return _param_seg(p, t0, t0 + 2.0 * math.pi * laps, label, yaw,
                      {"kind": "lemniscate", "center": [cx, cy], "a": a,
                       "cw": cw, "laps": laps},
                      samples=2000 * laps)


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
        # Current carrot altitude. Tracks self.z unless a move segment carries a
        # z-ramp (a HELIX arc), which crawls cz from its z0 to z1 over the segment;
        # cz then HOLDS that value through the following segments (so e.g. the helix
        # exit dwell + return leg stay at the climbed-to height). With no z-ramp
        # anywhere, cz == self.z forever → circle/figure-8/square are unchanged.
        self.cz = self.z
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
                and abs(self.cz - pose["z"]) < self.arrive_tol)

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

    def update(self, pose, dt, airborne, land_requested=False):
        if land_requested:
            self.done = True                   # SPACEBAR → abandon the path, land
        if self.done or self.i >= len(self.segs):
            self.done = True
            return self.cx, self.cy, self.cz, self.yaw_sp, 0.0, 0.0, 0.0, 0.0, True
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
            if "z_at" in seg:                  # helix arc: crawl the carrot's altitude
                self.cz = seg["z_at"](self.s)
            if seg["yaw"] == "tangent":
                self._slew_yaw(seg["heading"](self.s), dt)
            else:
                self._slew_yaw(seg["yaw"], dt)
            if self.s >= seg["len"] - 1e-9:
                self._next_seg()
        svx, svy = _carrot_vel(px, py, self.cx, self.cy, dt, self.cruise)
        sax, say = _carrot_acc(self.svx, self.svy, svx, svy, dt)
        self.svx, self.svy = svx, svy
        return self.cx, self.cy, self.cz, self.yaw_sp, svx, svy, sax, say, self.done

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
                zr = ""
                if "z_at" in seg:              # helix ramp / sine bob: show the z span
                    zv = [seg["z_at"](seg["len"] * j / 24.0) for j in range(25)]
                    zr = f"  z[{min(zv):+.2f},{max(zv):+.2f}]m"
                lines.append(f"    [{k}] move  {seg['label']:11s} "
                             f"len={seg['len']:.2f}m → ({end[0]:+.2f},{end[1]:+.2f})"
                             f"{zr}  [{self._yaw_label(seg)}]")
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
        cruise_mps=config.CIRCLE_SPEED_MPS, leash_m=config.LEASH_M,
        arrive_tol_m=config.ARRIVE_TOL_M, arrive_timeout_s=config.ARRIVE_TIMEOUT_S)


# ----------------------------------------------------------------------------- #
def build_helix_mission(launch):
    """Build the HELIX course: the circle, but the carrot CLIMBS as it laps. Take
    off + hover at config.CLIMB_M, fly forward config.HELIX_RADIUS_M to the circle
    (centred on the launch origin), trace config.HELIX_LAPS turns while rising
    config.HELIX_HEIGHT_M total (linearly with arc length), settle at the top,
    return, land. SAME PathMission / carrot / feedforward / leash / dwell machinery
    as the circle — the ONLY difference is the arc carries a z-ramp (z0→z1), so the
    bird's-eye (x,y) spiral is identical to the circle and only the altitude changes.

    Altitude: the laps span world z from z_base = launch + CLIMB_M up to
    z_top = z_base + HELIX_HEIGHT_M. Forward/entry legs sit at z_base, the carrot
    holds z_top through the exit/return/home legs, and the landing descends from
    z_top. The climb rate the drone must track is
        HELIX_HEIGHT_M · HELIX_SPEED_MPS / (2π·HELIX_RADIUS_M·HELIX_LAPS);
    keep it under the altitude loop's VMAX_UP_MPS (else the drone lags the rising
    carrot and catches up only at the top dwell). Heading + direction behave exactly
    like the circle (HELIX_FACE_TANGENT / HELIX_CW). DRY-RUN + preview.py first."""
    x0, y0, z0, yaw0 = launch
    c, s = math.cos(yaw0), math.sin(yaw0)
    r = config.HELIX_RADIUS_M
    start = (x0 + c * r, y0 + s * r)         # forward r in the launch body frame
    center = (x0, y0)                         # spiral centered on the launch origin
    th0 = math.atan2(start[1] - center[1], start[0] - center[0])
    laps = max(1, int(config.HELIX_LAPS))
    dtheta = (-1.0 if config.HELIX_CW else 1.0) * 2.0 * math.pi * laps
    z_base = z0 + config.CLIMB_M              # == PathMission's self.z (hover height)
    z_top = z_base + config.HELIX_HEIGHT_M
    arc = _arc_seg(center, r, th0, dtheta, f"helix x{laps}",
                   yaw=("tangent" if config.HELIX_FACE_TANGENT else None),
                   z0=z_base, z1=z_top)
    yaw_entry = arc["heading"](0.0) if config.HELIX_FACE_TANGENT else None
    yaw_exit = yaw0 if config.HELIX_FACE_TANGENT else None
    segs = [
        _dwell_seg((x0, y0), config.INITIAL_HOVER_S, "takeoff/hover"),
        _line_seg((x0, y0), start, "forward"),
        _dwell_seg(start, config.SETTLE_S, "helix-entry", yaw=yaw_entry),
        arc,
        _dwell_seg(start, config.SETTLE_S, "helix-exit", yaw=yaw_exit),
        _line_seg(start, (x0, y0), "return"),
        _dwell_seg((x0, y0), config.SETTLE_S, "home/settle"),
    ]
    return PathMission(
        launch, segs,
        cruise_mps=config.HELIX_SPEED_MPS, leash_m=config.LEASH_M,
        arrive_tol_m=config.ARRIVE_TOL_M, arrive_timeout_s=config.ARRIVE_TIMEOUT_S)


# ----------------------------------------------------------------------------- #
def build_sine_circle_mission(launch):
    """Build the SINE-CIRCLE course: a circle whose ALTITUDE oscillates like a sine
    wave while it laps. Take off + hover at CLIMB_M, fly forward SINE_RADIUS_M to the
    circle (centred on the launch origin), trace SINE_LAPS laps while z rides
    z_mid + SINE_AMP_M·sin(...) with SINE_CYCLES_PER_LAP humps per lap, settle, return,
    land. SAME PathMission / carrot / leash / dwell machinery as the circle and helix
    — the ONLY difference is the arc's z-profile is a sine instead of flat (circle) or
    a ramp (helix); the bird's-eye (x,y) is the plain circle.

    Altitude: z oscillates about z_mid = launch + CLIMB_M with amplitude SINE_AMP_M,
    so it spans [z_mid - SINE_AMP_M, z_mid + SINE_AMP_M]. KEEP SINE_AMP_M < CLIMB_M so
    the trough stays well above the ground. The sine starts at z_mid (phase 0, rising)
    and — because cycles = SINE_CYCLES_PER_LAP·SINE_LAPS is a whole/half number — ends
    back at z_mid, so the exit/return/landing are at the hover height. Peak vertical
    speed is SINE_AMP_M·SINE_CYCLES_PER_LAP·SINE_SPEED_MPS / SINE_RADIUS_M; keep it
    under the altitude loop's VMAX_UP_MPS or the drone lags the bobbing carrot.
    Heading + direction behave exactly like the circle. DRY-RUN + preview.py first."""
    x0, y0, z0, yaw0 = launch
    c, s = math.cos(yaw0), math.sin(yaw0)
    r = config.SINE_RADIUS_M
    start = (x0 + c * r, y0 + s * r)          # forward r in the launch body frame
    center = (x0, y0)                          # circle centered on the launch origin
    th0 = math.atan2(start[1] - center[1], start[0] - center[0])
    laps = max(1, int(config.SINE_LAPS))
    dtheta = (-1.0 if config.SINE_CW else 1.0) * 2.0 * math.pi * laps
    z_mid = z0 + config.CLIMB_M               # == PathMission's self.z (hover height)
    cycles = config.SINE_CYCLES_PER_LAP * laps    # total sine periods over the arc
    arc = _arc_seg(center, r, th0, dtheta, f"sine-circle x{laps}",
                   yaw=("tangent" if config.SINE_FACE_TANGENT else None),
                   z_sine=(z_mid, config.SINE_AMP_M, cycles, 0.0))
    yaw_entry = arc["heading"](0.0) if config.SINE_FACE_TANGENT else None
    yaw_exit = yaw0 if config.SINE_FACE_TANGENT else None
    segs = [
        _dwell_seg((x0, y0), config.INITIAL_HOVER_S, "takeoff/hover"),
        _line_seg((x0, y0), start, "forward"),
        _dwell_seg(start, config.SETTLE_S, "sine-entry", yaw=yaw_entry),
        arc,
        _dwell_seg(start, config.SETTLE_S, "sine-exit", yaw=yaw_exit),
        _line_seg(start, (x0, y0), "return"),
        _dwell_seg((x0, y0), config.SETTLE_S, "home/settle"),
    ]
    return PathMission(
        launch, segs,
        cruise_mps=config.SINE_SPEED_MPS, leash_m=config.LEASH_M,
        arrive_tol_m=config.ARRIVE_TOL_M, arrive_timeout_s=config.ARRIVE_TIMEOUT_S)


# ----------------------------------------------------------------------------- #
def build_figure8_mission(launch):
    """Build the figure-8 course from config: take off + hover AT the figure-8's
    crossover (which IS the launch origin), trace FIG8_LAPS smooth figure-8s, settle
    back at the origin, land. Structurally the circle's twin — same PathMission, same
    carrot/feedforward/leash/dwell machinery, nose following the travel direction —
    just a different path.

    The path is a single smooth lemniscate of Bernoulli (see _lemniscate_seg), long
    axis along WORLD X, half-span FIG8_END_X_M (far ends at (x0±FIG8_END_X_M, y0)),
    crossover at the launch origin. It replaces the old two-tangent-circles ∞, whose
    curvature flipped sign (+1/R → -1/R) at the crossover — an instant lateral-accel
    reversal the drone couldn't track. The lemniscate's curvature is CONTINUOUS:
    zero at the crossover, peaking at 3/FIG8_END_X_M at the far ends. The whole ∞ is
    ONE move segment, so the carrot ramps up once at takeoff, flows through the
    crossover at cruise, and brakes once into the home-settle dwell.

    Heading: with FIG8_FACE_TANGENT the nose follows the travel direction — the
    takeoff dwell pre-rotates to the path's start tangent and GATES on the nose
    getting there, the home dwell rotates back to the launch heading. Default OFF
    (strafe at the launch heading): see config for the yaw-rate budget.

    Like the old figure-8 there is NO forward approach leg: the crossing is the
    launch point, so the drone is already on the path at takeoff."""
    x0, y0, z0, yaw0 = launch
    a = config.FIG8_END_X_M                  # half-span: far ends at (x0±a, y0)
    laps = max(1, int(config.FIG8_LAPS))
    face = config.FIG8_FACE_TANGENT
    yaw_arc = "tangent" if face else None
    lem = _lemniscate_seg((x0, y0), a, f"figure-8 x{laps}",
                          yaw=yaw_arc, cw=config.FIG8_CW, laps=laps)
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
