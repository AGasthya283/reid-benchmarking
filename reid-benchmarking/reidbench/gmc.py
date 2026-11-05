"""Global (camera) motion compensation, computed once per sequence and shared by the trackers
that use it (Deep OC-SORT, TrackTrack).

Method: BoT-SORT's ``sparseOptFlow`` GMC (the method TrackTrack's released GMC files were produced
with): Shi-Tomasi corners on the previous frame (downscale 2), pyramidal LK flow, partial-affine
RANSAC fit, translation rescaled to full resolution. Output: (N_frames, 2, 3) warp matrices
mapping frame t-1 -> t (identity for frame 1 or on failure).
"""
from __future__ import annotations

import cv2
import numpy as np


def sparse_optflow_gmc(seq, downscale: int = 2) -> np.ndarray:
    warps = np.tile(np.eye(2, 3, dtype=np.float64), (seq.length, 1, 1))
    prev_gray, prev_pts = None, None
    feature_params = dict(maxCorners=1000, qualityLevel=0.01, minDistance=1, blockSize=3,
                          useHarrisDetector=False, k=0.04)
    for fid in range(1, seq.length + 1):
        img = seq.image(fid)
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if downscale > 1:
            gray = cv2.resize(gray, (w // downscale, h // downscale))
        pts = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)
        if prev_gray is not None and prev_pts is not None and len(prev_pts) > 0:
            nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
            ok = status.reshape(-1).astype(bool)
            p0, p1 = prev_pts[ok], nxt[ok]
            if len(p0) > 4:
                H, _ = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC)
                if H is not None:
                    H = H.astype(np.float64)
                    H[:, 2] *= downscale
                    warps[fid - 1] = H
        prev_gray, prev_pts = gray, pts
    return warps
