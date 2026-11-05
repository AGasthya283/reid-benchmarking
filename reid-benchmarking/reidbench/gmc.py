"""Global (camera) motion compensation, computed once per sequence and shared by the trackers
that use it (Deep OC-SORT, TrackTrack).

Method: BoT-SORT's ``sparseOptFlow`` GMC (the method TrackTrack's released GMC files were produced
with): Shi-Tomasi corners on the previous frame (downscale 2), pyramidal LK flow, partial-affine
RANSAC fit, translation rescaled to full resolution. Output: 2x3 warp matrices mapping frame
t-1 -> t (identity for the first frame or on failure).
"""
from __future__ import annotations

import cv2
import numpy as np

FEATURE_PARAMS = dict(maxCorners=1000, qualityLevel=0.01, minDistance=1, blockSize=3,
                      useHarrisDetector=False, k=0.04)


class SparseOptFlowGMC:
    """Online version: call ``apply(frame_bgr)`` once per frame, in order."""

    def __init__(self, downscale: int = 2):
        self.downscale = downscale
        self.prev_gray, self.prev_pts = None, None

    def apply(self, img: np.ndarray) -> np.ndarray:
        H = np.eye(2, 3, dtype=np.float64)
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.downscale > 1:
            gray = cv2.resize(gray, (w // self.downscale, h // self.downscale))
        pts = cv2.goodFeaturesToTrack(gray, mask=None, **FEATURE_PARAMS)
        if self.prev_gray is not None and self.prev_pts is not None and len(self.prev_pts) > 0:
            nxt, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_pts, None)
            ok = status.reshape(-1).astype(bool)
            p0, p1 = self.prev_pts[ok], nxt[ok]
            if len(p0) > 4:
                A, _ = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC)
                if A is not None:
                    H = A.astype(np.float64)
                    H[:, 2] *= self.downscale
        self.prev_gray, self.prev_pts = gray, pts
        return H


def sparse_optflow_gmc(seq, downscale: int = 2) -> np.ndarray:
    g = SparseOptFlowGMC(downscale)
    return np.stack([g.apply(seq.image(f)) for f in range(1, seq.length + 1)])
