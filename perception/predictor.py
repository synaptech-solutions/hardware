"""Run the trained GateNet gate-segmentation model on the laptop CPU.

This is the self-contained, hardware-side counterpart to ``navnet`` in the GateNet
repo: the model architecture (:mod:`perception.model`), the C03 undistortion
(:mod:`perception.calib`) and the trained weights (``weights/gatenet_flight.msgpack``,
exported from the ``checkpoints_flight`` Orbax checkpoint) are vendored here so the
flight machine needs only ``jax``/``jaxlib``/``flax`` on CPU -- no GPU, no GateNet venv.

The CPU forward pass is ~30 ms/frame for the (small, ``f=4``) network, i.e. live-rate
for the analog feed. Preprocessing matches whatever the checkpoint was trained on,
recorded in ``weights/infer.json`` (input resolution, pencil filter, undistortion).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# The CPU XLA backend is all we need; pick it before jax initialises a device.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import cv2
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from perception.calib import Undistorter
from perception.model import GateNetUNet
from perception.pencil import pencil_filter

_WEIGHTS_DIR = Path(__file__).parent / "weights"
_WEIGHTS_FILE = "gatenet_flight.msgpack"
_INFER_CONFIG = "infer.json"


class GatePredictor:
    """Segment gates in BGR camera frames; returns a per-pixel gate-probability mask.

    ``__call__(frame_bgr)`` returns ``(proc_bgr, prob)`` where ``proc_bgr`` is the frame
    actually fed to the model (rectified if the checkpoint expects undistortion, so an
    overlay drawn from ``prob`` lines up) and ``prob`` is a float32 ``(size, size)`` map
    in [0, 1]. The undistorter is built lazily from the first frame's resolution.
    """

    def __init__(self, weights_dir: str | Path = _WEIGHTS_DIR, *, size: int | None = None) -> None:
        wdir = Path(weights_dir)
        cfg = json.loads((wdir / _INFER_CONFIG).read_text())
        self.size = int(size if size is not None else cfg.get("size", 384))
        self.pencil = bool(cfg.get("pencil", False))
        self.undistort = bool(cfg.get("undistort", False))

        model = GateNetUNet()
        template = model.init(jax.random.PRNGKey(0), jnp.zeros((1, self.size, self.size, 3)), train=False)
        variables = {"params": template["params"], "batch_stats": template["batch_stats"]}
        variables = serialization.from_bytes(variables, (wdir / _WEIGHTS_FILE).read_bytes())

        @jax.jit
        def _infer(images: jnp.ndarray) -> jnp.ndarray:
            return jax.nn.sigmoid(model.apply(variables, images, train=False)[-1])

        self._infer = _infer
        self._und: Undistorter | None = None
        # Warm up the JIT so the first live frame is not stalled by compilation.
        self._infer(jnp.zeros((1, self.size, self.size, 3), jnp.float32)).block_until_ready()

    def __call__(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.undistort:
            if self._und is None:
                h, w = frame_bgr.shape[:2]
                self._und = Undistorter(w, h)
            proc = self._und.image(frame_bgr)
        else:
            proc = frame_bgr

        rgb = cv2.cvtColor(cv2.resize(proc, (self.size, self.size)), cv2.COLOR_BGR2RGB)
        if self.pencil:
            rgb = pencil_filter(rgb)
        batch = (rgb.astype(np.float32) / 255.0)[None]
        prob = np.asarray(self._infer(jnp.asarray(batch)))[0, ..., 0]
        return proc, prob
