"""Threaded GateNet mask source — the sole owner of the Cam Link device.

GateNet (~30 Hz / ~30 ms) is slower than the 50 Hz control loop, so it runs in its
own thread and publishes the latest mask; the loop reads the most recent one. The
flight loop must not also start a ``VideoRecorder`` on the same device.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from perception import GatePredictor
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from perception import GatePredictor


def open_camera(device: int, width: int, height: int, fps: int) -> "tuple[cv2.VideoCapture, int]":
    """Open /dev/video<device>, falling back to device+1 (Cam Link 4<->5 re-enum)."""
    for dev in (device, device + 1):
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap, dev
        cap.release()
    raise RuntimeError(f"could not open /dev/video{device} (or {device + 1}) — "
                       "is the Cam Link plugged in and free? (close mpv/live_infer)")


class MaskSource:
    """Background camera + GateNet thread. ``get_mask()`` returns ``(mask_hw, age_s)``:
    the latest size×size mask in [0, 1] (INTER_AREA down to the policy's input, giving
    soft coverage like the sim renderer), all-zeros before the first frame."""

    def __init__(self, mask_size: int, *, device: int = 4, width: int = 720,
                 height: int = 480, fps: int = 30, weights_dir: Optional[str] = None,
                 conf: Optional[float] = None) -> None:
        self.mask_size = int(mask_size)
        self.device = device
        self.width, self.height, self.fps = width, height, fps
        self.weights_dir = weights_dir
        self.conf = conf                           # threshold the prob, or None for soft

        self._lock = threading.Lock()
        self._mask = np.zeros((self.mask_size, self.mask_size), np.float32)
        self._frame: Optional[np.ndarray] = None
        self._stamp = 0.0
        self._frames = 0
        self._infer_ms = 0.0                       # EMA inference time
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._predictor: Optional[GatePredictor] = None
        self.status = "idle"

    def start(self) -> None:
        kw = {} if self.weights_dir is None else {"weights_dir": self.weights_dir}
        self._predictor = GatePredictor(**kw)
        cap, dev = open_camera(self.device, self.width, self.height, self.fps)
        self.status = f"/dev/video{dev} {self.width}x{self.height}"
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(cap,), daemon=True)
        self._thread.start()

    def _run(self, cap: "cv2.VideoCapture") -> None:
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.005)
                    continue
                t0 = time.perf_counter()
                _proc, prob = self._predictor(frame)        # prob (size,size) in [0,1]
                if self.conf is not None:
                    prob = (prob > self.conf).astype(np.float32)
                mask = cv2.resize(prob.astype(np.float32),
                                  (self.mask_size, self.mask_size),
                                  interpolation=cv2.INTER_AREA)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                with self._lock:
                    self._mask = mask
                    self._frame = frame
                    self._stamp = time.time()
                    self._frames += 1
                    self._infer_ms = (dt_ms if self._infer_ms == 0.0
                                      else 0.9 * self._infer_ms + 0.1 * dt_ms)
        finally:
            cap.release()

    def get_mask(self, now: Optional[float] = None) -> "tuple[np.ndarray, float]":
        now = time.time() if now is None else now
        with self._lock:
            age = (now - self._stamp) if self._frames > 0 else 1e9
            return self._mask.copy(), age

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    @property
    def frames(self) -> int:
        return self._frames

    @property
    def infer_ms(self) -> float:
        return self._infer_ms

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
