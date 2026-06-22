"""Calibration handling for the Raven front camera.

``calibration/calib.json`` provides, per camera:
  * ``K``      3x3 pinhole intrinsics (row-major, 9 numbers)
  * ``coeff``  4 OpenCV *fisheye* distortion coefficients (k1..k4)
and a camera<->lidar 4x4 ``transform_matrix`` (under ``out_put``).

Orientation note: ``K`` has cx~=1586, cy~=2099, implying a **portrait** sensor
(~3172x4197), while the JPEGs decode as **landscape 4000x3000**. The captured
frames are the sensor rotated 90deg. :func:`rotate_to_sensor` rotates a decoded
JPEG back to portrait, and :func:`scaled_K` rescales ``K`` to a chosen
resolution so the principal point stays consistent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Resolution the provided K is implicitly defined at (2*cx, 2*cy), portrait.
DEFAULT_CALIB = "calibration/calib.json"


@dataclass
class CameraCalib:
    K: np.ndarray            # 3x3 intrinsics at calib resolution
    dist: np.ndarray         # (4,) fisheye k1..k4
    calib_size: tuple[int, int]   # (w, h) the K is defined at (portrait)
    cam_from_lidar: np.ndarray    # 4x4 transform (lidar point -> camera frame)

    # ---- intrinsics helpers -------------------------------------------------
    def scaled_K(self, size: tuple[int, int]) -> np.ndarray:
        """Return K rescaled from ``calib_size`` to ``size`` (w, h)."""
        sx = size[0] / self.calib_size[0]
        sy = size[1] / self.calib_size[1]
        K = self.K.copy()
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy
        return K

    def colmap_params(self, size: tuple[int, int]) -> list[float]:
        """OPENCV_FISHEYE params (fx, fy, cx, cy, k1, k2, k3, k4) for ``size``."""
        K = self.scaled_K(size)
        return [K[0, 0], K[1, 1], K[0, 2], K[1, 2], *self.dist.tolist()]

    # ---- image helpers ------------------------------------------------------
    def rotate_to_sensor(self, img: np.ndarray) -> np.ndarray:
        """Rotate a landscape JPEG to the calib (portrait) orientation if needed."""
        h, w = img.shape[:2]
        cw, ch = self.calib_size
        portrait = ch > cw
        if portrait and w > h:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if (not portrait) and h > w:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        return img

    def undistort_sample(self, img: np.ndarray) -> np.ndarray:
        """Undistort an image with the fisheye model (orientation auto-fixed)."""
        img = self.rotate_to_sensor(img)
        h, w = img.shape[:2]
        K = self.scaled_K((w, h))
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, self.dist, (w, h), np.eye(3), balance=0.0
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, self.dist, np.eye(3), new_K, (w, h), cv2.CV_16SC2
        )
        return cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)


def load_calib(path: str | Path = DEFAULT_CALIB, cam: str = "front") -> CameraCalib:
    data = json.loads(Path(path).read_text())
    info = data["camera_info"][cam]
    K = np.asarray(info["K"], dtype=np.float64).reshape(3, 3)
    dist = np.asarray(info["coeff"], dtype=np.float64).reshape(-1)[:4]
    # Calib resolution implied by principal point (portrait).
    calib_size = (int(round(2 * K[0, 2])), int(round(2 * K[1, 2])))

    cam_from_lidar = np.eye(4)
    try:
        tm = data["out_put"][cam]["transform_matrix"]
        cam_from_lidar = np.asarray(tm, dtype=np.float64).reshape(4, 4)
    except (KeyError, TypeError):
        pass

    return CameraCalib(K=K, dist=dist, calib_size=calib_size, cam_from_lidar=cam_from_lidar)


if __name__ == "__main__":
    import sys

    c = load_calib(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CALIB)
    print("K=\n", c.K)
    print("dist=", c.dist)
    print("calib_size (portrait w,h)=", c.calib_size)
    print("cam_from_lidar=\n", c.cam_from_lidar)
