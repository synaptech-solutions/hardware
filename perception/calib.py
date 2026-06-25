"""BetaFPV C03 camera calibration and undistortion.

The C03 is an ultra-wide FPV lens; its strong barrel distortion curves straight gate
edges, which breaks the planar-homography assumption in mask generation (``masks.py``)
and in the downstream corner/PnP geometry. Following MonoRace (which undistorts as the
first preprocessing step), we rectify to a pinhole frame at the data-build and inference
boundaries; labeling stays on the raw distorted frames (corners are points, unaffected by
edge curvature) and is remapped with :meth:`Undistorter.points`.

The calibration (``calib_c03.npz``: OpenCV rational model ``K``, ``dist``, ``image_size``,
rms 0.62 px) was measured at 1280x720 and is vendored from the apriltag_control project;
``Undistorter`` rescales the intrinsics to whatever resolution it is given.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_CALIB_PATH = Path(__file__).parent / "calib_c03.npz"


class Undistorter:
    """Rectifies images and points for a given frame resolution (intrinsics auto-scaled).

    ``alpha`` is passed to :func:`cv2.getOptimalNewCameraMatrix` (0 keeps the tightest
    valid region, 1 keeps all source pixels with black borders).
    """

    def __init__(self, width: int, height: int, alpha: float = 0.0) -> None:
        data = np.load(_CALIB_PATH)
        k = data["K"].astype(np.float64).copy()
        cal_w, cal_h = (int(v) for v in data["image_size"])
        k[0] *= width / cal_w   # scale fx, cx
        k[1] *= height / cal_h  # scale fy, cy
        self.K = k
        self.dist = data["dist"].astype(np.float64)
        self.size = (width, height)
        self.new_K, self.roi = cv2.getOptimalNewCameraMatrix(
            self.K, self.dist, self.size, alpha, self.size
        )
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            self.K, self.dist, None, self.new_K, self.size, cv2.CV_16SC2
        )

    def image(self, img: np.ndarray) -> np.ndarray:
        """Undistort an image of the configured resolution."""
        return cv2.remap(img, self._map1, self._map2, cv2.INTER_LINEAR)

    def points(self, pts) -> np.ndarray:
        """Undistort pixel coordinates: ``(N, 2)`` distorted -> ``(N, 2)`` rectified."""
        p = np.asarray(pts, np.float32).reshape(-1, 1, 2)
        return cv2.undistortPoints(p, self.K, self.dist, P=self.new_K).reshape(-1, 2)
