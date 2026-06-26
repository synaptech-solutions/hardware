"""Pencil-filter preprocessing (PencilNet, Pham et al., arXiv:2207.14131).

A color- and illumination-invariant edge/sketch representation: grayscale, then divide
by a dilated (local-max) copy of itself. Flat regions go white, edges/outlines go dark.
Applied identically at training and inference, it collapses differently-colored gates
(orange A2RL vs. cardboard) and different cameras/lighting toward a common edge sketch.
"""

from __future__ import annotations

import cv2
import numpy as np


def pencil_filter(img_rgb: np.ndarray, ksize: int = 5) -> np.ndarray:
    """RGB uint8 (H,W,3) -> pencil-sketch uint8 (H,W,3, channels replicated)."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dilated = cv2.dilate(gray, kernel)
    ratio = 255.0 * gray.astype(np.float32) / np.where(dilated == 0, 1, dilated)
    pencil = np.where(dilated == 0, 255, ratio).astype(np.uint8)
    return cv2.cvtColor(pencil, cv2.COLOR_GRAY2RGB)
