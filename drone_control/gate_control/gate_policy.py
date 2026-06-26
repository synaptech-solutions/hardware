"""Run a trained betaflight-gym recurrent (vision-GRU) gate policy — ACRO.

``GatePolicy`` deserializes a ``jax.export`` StableHLO artifact (``gate_actor.exp``,
weights + obs-norm baked in) and threads the GRU hidden state — no network code is
copied from the training repo. ``GatePolicyController`` builds the actor obs and
returns AETR µs.

The actor sees mask + onboard proprioception only (attitude, body rates, action
history, mass) — never position. It is gate-relative by design, so the controller
needs no gate location: the attitude frame is anchored to the launch heading (like
the hover policy). Attitude/rates come from differentiated Vicon attitude because
Betaflight does not stream gyro fast enough over CRSF.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

# CPU matches GateNet's pin (perception/predictor.py); JAX_PLATFORMS=cuda wins.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp          # noqa: E402
from jax import export           # noqa: E402

_WORLD_UP = np.array([0.0, 0.0, 1.0])


def quat_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Body->world rotation matrix from a quaternion (columns are the body axes in
    the Vicon world frame), normalised defensively."""
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw) or 1.0
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


class AxisMap:
    """Maps the Vicon rigid-body axes (columns of the quaternion's R) to the drone
    BODY FRD axes. Defaults encode the calibrated Air75 mapping (see rl_policy):
    nose=+Vicon-Y, right=+Vicon-X, down=-Vicon-Z. Flip a sign if a DRY-RUN hand
    test shows that axis inverted."""

    def __init__(self, forward: "tuple[int, int]" = (1, +1),
                 right: "tuple[int, int]" = (0, +1),
                 down: "tuple[int, int]" = (2, -1)) -> None:
        self.forward, self.right, self.down = forward, right, down

    def frd_world_axes(self, R: np.ndarray) -> np.ndarray:
        """Drone forward/right/down unit axes in the world frame = columns of a
        body->world rotation R_frd."""
        def col(spec: "tuple[int, int]") -> np.ndarray:
            idx, sgn = spec
            return sgn * R[:, idx]
        return np.column_stack([col(self.forward), col(self.right), col(self.down)])


class GatePolicy:
    """Deterministic recurrent inference for the exported gate artifact.

    ``reset()`` zeroes the GRU hidden state (call once at launch). ``act(mask_hw,
    proprio_raw)`` steps the GRU on the RAW mask + RAW proprio (the graph holds the
    obs-norm) and returns the clipped mean action in [-1, 1]."""

    def __init__(self, bundle_dir: str | Path) -> None:
        d = Path(bundle_dir)
        if d.is_file():
            d = d.parent
        self.contract: dict[str, Any] = json.loads((d / "gate_policy.json").read_text())
        c = self.contract
        self.mask_h, self.mask_w = int(c["mask_h"]), int(c["mask_w"])
        self.proprio_dim = int(c["proprio_dim"])
        self.act_dim = int(c["action_dim"])
        self.act_hist_len = int(c["act_hist_len"])
        self.control_dt = float(c["control_dt"])
        self.mass_kg = float(c["mass_kg"])

        self._exp = export.deserialize((d / "gate_actor.exp").read_bytes())
        # the carry shape is self-described by the artifact's first input aval
        self._carry_shape = tuple(self._exp.in_avals[0].shape)
        self.reset()
        # warm up so the first live step isn't stalled by lowering
        self.act(np.zeros((self.mask_h, self.mask_w), np.float32),
                 np.zeros(self.proprio_dim, np.float32))
        self.reset()

    def reset(self) -> None:
        self.carry = jnp.zeros(self._carry_shape, jnp.float32)

    def act(self, mask_hw: np.ndarray, proprio_raw: np.ndarray) -> np.ndarray:
        """mask_hw [H,W] in [0,1] (RAW), proprio_raw [proprio_dim] (RAW) -> [4] in [-1,1]."""
        mask = jnp.asarray(mask_hw, jnp.float32).reshape(1, self.mask_h, self.mask_w, 1)
        proprio = jnp.asarray(proprio_raw, jnp.float32).reshape(1, self.proprio_dim)
        self.carry, mean = self._exp.call(self.carry, mask, proprio)
        return np.clip(np.asarray(mean)[0], -1.0, 1.0)

    def describe(self) -> str:
        c = self.contract
        return (f"GatePolicy mask={self.mask_h}x{self.mask_w} proprio_dim={self.proprio_dim} "
                f"carry={self._carry_shape} act_hist={self.act_hist_len} "
                f"control_dt={self.control_dt} mass_kg={self.mass_kg} "
                f"platforms={self._exp.platforms} step={c.get('global_step')}")


class GatePolicyController:
    """Builds the actor obs and returns AETR µs (recurrent analog of
    ``rl_policy.HoverPolicyController``). Holds no gate location; the attitude frame
    is anchored to the launch heading by ``capture_launch``."""

    def __init__(self, policy: GatePolicy, *,
                 axis_map: AxisMap | None = None, mass_kg: float | None = None,
                 sign_roll: int = 1, sign_pitch: int = 1, sign_yaw: int = 1,
                 rate_lpf_s: float = 0.03) -> None:
        self.policy = policy
        self.axis = axis_map or AxisMap()
        self.mass_kg = float(policy.mass_kg if mass_kg is None else mass_kg)
        self.sign = np.array([sign_roll, sign_pitch, 1.0, sign_yaw])   # T never flips
        self.rate_lpf_s = rate_lpf_s
        self.reset()

    def reset(self) -> None:
        # frame defaults until capture_launch anchors it to the launch heading
        self.north_w = np.array([1.0, 0.0, 0.0])
        self.down_w = -_WORLD_UP
        self.east_w = np.cross(self.down_w, self.north_w)     # NED: E = D x N
        self.launch_w = None
        self.act_hist = np.zeros((self.policy.act_hist_len, self.policy.act_dim))
        self.last_action = np.zeros(self.policy.act_dim)
        self._prev_Rfrd = None
        self._rate_f = np.zeros(3)
        self.policy.reset()

    def capture_launch(self, pose: dict[str, float]) -> None:
        """Anchor the attitude frame to the launch heading + reset the GRU. North =
        the drone's horizontal forward bearing now (so rot_matrix ~ identity at
        takeoff, matching the sim's drone-faces-gate spawn). Records the launch
        position purely so the flight loop can geofence relative to it."""
        self.reset()
        R = quat_to_matrix(pose["qx"], pose["qy"], pose["qz"], pose["qw"])
        fwd_w = self.axis.frd_world_axes(R)[:, 0]
        north = fwd_w - np.dot(fwd_w, _WORLD_UP) * _WORLD_UP   # horizontalise
        nn_ = np.linalg.norm(north)
        self.north_w = north / nn_ if nn_ > 1e-6 else np.array([1.0, 0.0, 0.0])
        self.down_w = -_WORLD_UP
        self.east_w = np.cross(self.down_w, self.north_w)
        self.launch_w = np.array([pose["x"], pose["y"], pose["z"]])

    def build_proprio(self, pose: dict[str, float], dt: float) -> np.ndarray:
        """Onboard proprio matching tasks/gate.py's vision actor obs (body FRD):
        rot_matrix(9) + body_rates(3) + action_hist(4*K) + mass(1)."""
        R = quat_to_matrix(pose["qx"], pose["qy"], pose["qz"], pose["qw"])
        Rfrd = self.axis.frd_world_axes(R)                # cols = forward/right/down (world)
        bp = lambda w: Rfrd.T @ np.asarray(w, float)      # world vector -> body FRD

        rot = np.concatenate([bp(self.north_w), bp(self.east_w), bp(self.down_w)])

        # body rates from attitude differencing: omega_body = vee(R_frd^T Ṙ_frd), LPF'd
        if self._prev_Rfrd is not None and dt > 1e-6:
            S = self._prev_Rfrd.T @ ((Rfrd - self._prev_Rfrd) / dt)
            omega = np.array([S[2, 1] - S[1, 2], S[0, 2] - S[2, 0],
                              S[1, 0] - S[0, 1]]) * 0.5
            a = min(dt / self.rate_lpf_s, 1.0)
            self._rate_f += (omega - self._rate_f) * a
        self._prev_Rfrd = Rfrd

        return np.concatenate([rot, self._rate_f, self.act_hist.reshape(-1),
                               [self.mass_kg]])

    def step(self, pose: dict[str, float], mask_hw: np.ndarray,
             dt: float) -> dict[str, Any]:
        """Build the obs, step the recurrent policy, return AETR µs + intermediates.
        mask_hw is the GateNet mask resized to the policy's (H, W), values in [0,1]."""
        proprio = self.build_proprio(pose, dt)
        action = self.policy.act(mask_hw, proprio)            # commands AETR directly
        self.last_action = action
        self.act_hist = np.roll(self.act_hist, 1, axis=0)     # newest-first (post-step)
        self.act_hist[0] = action
        sent = action * self.sign
        return {
            "us": self._action_to_us(sent),
            "action": action, "sent": sent,
            "rot": proprio[0:9], "rates": proprio[9:12],
        }

    @staticmethod
    def _action_to_us(a: np.ndarray) -> np.ndarray:
        """AETR [-1,1] -> CRSF µs: us = 1500 + 500*clip(a,-1,1), order (R,P,T,Y)."""
        return (1500.0 + 500.0 * np.clip(a, -1.0, 1.0)).astype(int)
