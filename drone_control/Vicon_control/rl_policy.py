"""
Run a trained betaflight-gym RL hover policy on the live Vicon feed (ACRO).
"""
import json
import math
import re
from dataclasses import dataclass

import numpy as np

# World up in the Vicon frame (Z-up). Gravity "down" is the negative of this.
_WORLD_UP = np.array([0.0, 0.0, 1.0])


def quat_to_matrix(qx, qy, qz, qw):
    """Rotation matrix R (body→world) from a quaternion. Columns are the body
    axes expressed in the (Vicon) world frame. Normalised defensively."""
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw) or 1.0
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


@dataclass
class AxisMap:
    """Maps the Vicon rigid-body axes (columns of the quaternion's R) to the
    drone BODY FRD axes (forward/right/down). Defaults encode the calibrated
    Air75 mapping (config.VICON_YAW_OFFSET_DEG's basis): nose=+Y_vicon,
    right=+X_vicon, down=-Z_vicon. Each is (column_index, sign) — flip the sign
    (or swap the index) if a DRY-RUN hand test shows that axis inverted."""
    forward: tuple[int, int] = (1, +1)   # drone nose  = +Vicon-body-Y
    right: tuple[int, int] = (0, +1)     # drone right = +Vicon-body-X
    down: tuple[int, int] = (2, -1)      # drone down  = -Vicon-body-Z (Z is up)

    def frd_world_axes(self, R):
        """The drone forward/right/down unit axes, expressed in the world frame,
        as the columns of a body→world rotation R_frd. Returns [3,3] with columns
        (forward_w, right_w, down_w)."""
        def col(spec):
            idx, sgn = spec
            return sgn * R[:, idx]
        return np.column_stack([col(self.forward), col(self.right), col(self.down)])


class MLPPolicy:
    """Deterministic numpy inference for a rejax GaussianPolicy exported by
    betaflight-gym rl/export.py. ``act(obs)`` returns the clipped mean action."""

    def __init__(self, npz_path):
        d = np.load(npz_path, allow_pickle=True)
        self.meta = dict(d["meta"].item()) if "meta" in d else {}
        self.obs_mean = d["obs_mean"].astype(np.float64)
        self.obs_var = d["obs_var"].astype(np.float64)
        n = int(d["n_layers"])
        self.layers = [(d[f"W{i}"].astype(np.float64), d[f"b{i}"].astype(np.float64))
                       for i in range(n)]
        self.w_mean = d["W_mean"].astype(np.float64)
        self.b_mean = d["b_mean"].astype(np.float64)
        self.action_low = float(self.meta.get("action_low", -1.0))
        self.action_high = float(self.meta.get("action_high", 1.0))
        self.obs_dim = int(self.meta.get("obs_dim", self.obs_mean.shape[0]))
        self.act_dim = int(self.meta.get("act_dim", self.b_mean.shape[0]))
        act = self.meta.get("activation", "tanh")
        self._act = {"tanh": np.tanh,
                     "relu": lambda x: np.maximum(x, 0.0),
                     "swish": lambda x: x / (1.0 + np.exp(-x))}[act]

    def normalize(self, obs):
        return (np.asarray(obs, np.float64) - self.obs_mean) / np.sqrt(self.obs_var + 1e-8)

    def act(self, obs):
        """obs [obs_dim] -> action [act_dim] in [low, high] (deterministic mean)."""
        h = self.normalize(obs)
        for w, b in self.layers:
            h = self._act(h @ w + b)
        mean = h @ self.w_mean + self.b_mean
        return np.clip(mean, self.action_low, self.action_high)

    def describe(self):
        return (f"MLPPolicy obs_dim={self.obs_dim} act_dim={self.act_dim} "
                f"layers={[w.shape for w, _ in self.layers]} "
                f"act={self.meta.get('activation')} "
                f"target_alt={self.meta.get('target_alt')} "
                f"meta={json.dumps(self.meta, default=str)}")


class HoverPolicyController:
    """Stateful flight-side hover controller backed by an MLPPolicy.

    Mirrors the role of ViconHoverController but for the ACRO policy: capture the
    launch pose (anchors the NED frame + the hover target), then each tick build
    the egocentric obs, run the policy, and return AETR stick commands as CRSF µs.

    Reproduces the sim's deployment pipeline EXACTLY — a pure obs->policy->action
    loop. The policy commands all four AETR channels directly: no throttle prior,
    no scripted takeoff (PPO discovered hover throttle in training). ``last_action``
    in the obs is the previous policy output, as tasks/hover.py threads it. The
    flight loop only chooses WHEN to start stepping this (after arm + launch) and
    when to cut — it never hand-flies the sticks.

    Body rates come from differentiating the Vicon attitude (true angular velocity
    — what the sim obs used, not a noisy gyro), low-passed.

    Landing: ``begin_landing()`` walks the hover target's altitude down at
    ``land_speed`` so the SAME policy flies a gentle descent (x/y held); the flight
    loop cuts throttle near the ground.
    """

    def __init__(self, policy, *, axis_map=None, target_alt=None, control_dt=0.01,
                 sign_roll=1, sign_pitch=1, sign_yaw=1,
                 rate_lpf_s=0.03, land_speed_mps=0.25, land_cut_m=0.30,
                 mass_kg=None):
        self.policy = policy
        self.axis = axis_map or AxisMap()
        self.target_alt = (policy.meta.get("target_alt", 1.0)
                           if target_alt is None else target_alt)
        self.control_dt = control_dt
        # Mass-conditioned policy: training appends the (domain-randomized) mass
        # to the obs (layout term "mass"); here we feed the MEASURED real mass so
        # the policy picks the right hover throttle for this airframe. Defaults
        # to the nominal the policy was centred on (exported as meta mass_kg).
        self._obs_has_mass = "mass" in str(policy.meta.get("obs_layout", ""))
        # Action history: training feeds the last K commanded actions (newest
        # first) as the "action_hist" obs term so the policy can compensate for
        # transport latency (a single last_action can't span a multi-step delay).
        # Recover K from the exported layout; fall back to 1 (legacy single
        # last_action(4)) for older policies.
        _layout = str(policy.meta.get("obs_layout", ""))
        _m = re.search(r"action_hist\((\d+)\)", _layout)
        self._act_hist_len = (max(1, int(_m.group(1)) // policy.act_dim) if _m else 1)
        # velocity+gyro frame stack: past H frames of [vel_b(3), rates(3)] (the
        # "state_hist" obs term), newest first. 0 if the policy didn't train with it.
        _ms = re.search(r"state_hist\((\d+)\)", _layout)
        self._sr_hist_len = (max(0, int(_ms.group(1)) // 6) if _ms else 0)
        self.mass_kg = (mass_kg if mass_kg is not None
                        else policy.meta.get("mass_kg"))
        if self._obs_has_mass and self.mass_kg is None:
            raise ValueError("policy obs includes a 'mass' term but no mass_kg "
                             "given and none in policy meta — pass mass_kg=<measured kg>.")
        # AETR output signs — a last-resort flip if the link/airframe wiring
        # inverts a channel vs the sim (normally all +1; verify in DRY-RUN).
        self.sign = np.array([sign_roll, sign_pitch, 1.0, sign_yaw])  # T never flips
        self.rate_lpf_s = rate_lpf_s
        self.land_speed = land_speed_mps
        self.land_cut_m = land_cut_m
        self.reset()

    def reset(self):
        self.launch = None            # (x0, y0, z0)
        self.target_w = None          # [3] hover point, Vicon world (Z-up)
        self.north_w = None           # anchored NED basis (world vectors)
        self.east_w = None
        self.down_w = np.array([0.0, 0.0, -1.0])
        self.last_action = np.zeros(self.policy.act_dim)   # prev RAW policy output
        # newest-first ring buffer of the last K commanded actions (the obs
        # "action_hist" term); K=1 reduces to the old single last_action.
        self.act_hist = np.zeros((self._act_hist_len, self.policy.act_dim))
        # past H frames of [vel_b(3), body_rates(3)] (newest first); H=0 -> unused
        self.sr_hist = np.zeros((self._sr_hist_len, 6))
        self._prev_Rfrd = None        # for body-rate differencing
        self._rate_f = np.zeros(3)    # LPF'd body rates (rad/s, FRD)
        self.landing = False

    # -- launch frame -------------------------------------------------------

    def capture_launch(self, pose):
        """Anchor the NED frame + hover target at the current pose (call once at
        takeoff). North = the drone's horizontal heading now; target = here, but
        ``target_alt`` higher."""
        R = quat_to_matrix(pose["qx"], pose["qy"], pose["qz"], pose["qw"])
        fwd_w = self.axis.frd_world_axes(R)[:, 0]
        north = fwd_w - np.dot(fwd_w, _WORLD_UP) * _WORLD_UP    # horizontalize
        nn = np.linalg.norm(north)
        self.north_w = north / nn if nn > 1e-6 else np.array([1.0, 0.0, 0.0])
        self.down_w = -_WORLD_UP
        self.east_w = np.cross(self.down_w, self.north_w)       # NED: E = D x N
        self.launch = (pose["x"], pose["y"], pose["z"])
        self.target_w = np.array([pose["x"], pose["y"], pose["z"] + self.target_alt])
        self._prev_Rfrd = None
        self._rate_f = np.zeros(3)
        self.landing = False

    def begin_landing(self):
        self.landing = True

    # -- observation --------------------------------------------------------

    def build_obs(self, pose, dt):
        """The egocentric obs (body FRD), matching tasks/hover.py. Width depends
        on the policy's layout: pos_err(3)+self_vel(3)+rot(9)+body_rates(3)+
        action_hist(4·K)+[mass(1)], where K = self._act_hist_len."""
        R = quat_to_matrix(pose["qx"], pose["qy"], pose["qz"], pose["qw"])
        Rfrd = self.axis.frd_world_axes(R)        # cols = forward/right/down (world)
        # project a world vector onto the body FRD axes
        bp = lambda w: Rfrd.T @ np.asarray(w, float)

        pos_w = np.array([pose["x"], pose["y"], pose["z"]])
        vel_w = np.array([pose["vx"], pose["vy"], pose["vz"]])
        pos_err = bp(self.target_w - pos_w)
        vel_b = bp(vel_w)
        # rotation matrix obs = world-NED axes expressed in body FRD (R_frd^T·axis)
        rot = np.concatenate([bp(self.north_w), bp(self.east_w), bp(self.down_w)])

        # body rates from attitude differencing: omega_body = vee(R_frd^T Ṙ_frd)
        if self._prev_Rfrd is not None and dt > 1e-6:
            S = self._prev_Rfrd.T @ ((Rfrd - self._prev_Rfrd) / dt)
            omega = np.array([S[2, 1] - S[1, 2], S[0, 2] - S[2, 0],
                              S[1, 0] - S[0, 1]]) * 0.5
            a = min(dt / self.rate_lpf_s, 1.0)
            self._rate_f += (omega - self._rate_f) * a
        self._prev_Rfrd = Rfrd

        # Layout matches tasks/hover.py byte-for-byte: current frame in vel_b +
        # self._rate_f, then the state_hist block (past vel+gyro frames, newest
        # first), then action_hist (newest first, == sim act_buf[:, 0] most recent).
        obs = np.concatenate([pos_err, vel_b, rot, self._rate_f,
                              self.sr_hist.reshape(-1), self.act_hist.reshape(-1)])
        if self._obs_has_mass:
            obs = np.concatenate([obs, [self.mass_kg]])   # measured real mass (kg)
        # push this frame's [vel_b, rates] for the NEXT obs (after using the old
        # stack above), mirroring the sim's post-step state push.
        if self._sr_hist_len > 0:
            self.sr_hist = np.roll(self.sr_hist, 1, axis=0)
            self.sr_hist[0] = np.concatenate([vel_b, self._rate_f])
        return obs

    # -- one control step ---------------------------------------------------

    def step(self, pose, dt):
        """Build obs, run the policy, return the AETR command as CRSF µs + the
        intermediates for logging / the DRY-RUN dashboard.

        Pure obs->policy->action, identical to the sim: the policy commands all
        four AETR channels directly (no bias, no scripted takeoff). The flight
        loop only decides WHEN to start running this (after arm + launch) and
        when to cut; it never hand-flies the sticks.
        """
        if self.landing and dt > 0.0:
            # walk the hover target down so the policy flies the descent itself
            floor = self.launch[2] + self.land_cut_m
            self.target_w[2] = max(self.target_w[2] - self.land_speed * dt, floor)

        obs = self.build_obs(pose, dt)
        action = self.policy.act(obs)              # the policy commands AETR directly
        self.last_action = action                  # most recent (compat / logging)
        # push into the newest-first history AFTER building this tick's obs, so
        # next tick sees [a_now, a_prev, ...] — exactly the sim's post-step act_buf.
        self.act_hist = np.roll(self.act_hist, 1, axis=0)
        self.act_hist[0] = action
        sent = action * self.sign

        us = self._action_to_us(sent)
        return {
            "us": us,                               # [roll, pitch, thr, yaw] µs
            "action": action, "sent": sent, "obs": obs,
            "pos_err": obs[0:3], "vel_b": obs[3:6],
            "tilt": obs[12:15],                     # gravity/down in body frame
            "rates": obs[15:18],
        }

    @staticmethod
    def _action_to_us(a):
        """AETR [-1,1] -> CRSF µs, the SAME map the firmware harness applies:
        us = 1500 + 500*clip(a,-1,1). Order AETR = (roll, pitch, throttle, yaw),
        which is exactly channels CH_ROLL/CH_PITCH/CH_THR/CH_YAW (0..3)."""
        return (1500.0 + 500.0 * np.clip(a, -1.0, 1.0)).astype(int)
