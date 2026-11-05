"""Frame-by-frame (online) tracking: the same detector -> ReID -> tracker chain as the batch
benchmark, but driven one frame at a time. Used by the Streamlit platform for live replay of the
MOT17 sequences and for user-supplied test videos.

Two input modes:
  * ``CachedSource``   - a benchmark sequence; detections / embeddings / GMC come from the caches,
                         so replay is fast and bit-identical to the benchmark run.
  * ``LiveSource``     - any video; detector, ReID and GMC run on each frame.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .runners import RUNNERS


@dataclass
class VideoSequence:
    """Minimal ``Sequence``-compatible wrapper around a video file (frames decoded on load)."""
    name: str
    frames_bgr: list
    fps: float

    @property
    def length(self):
        return len(self.frames_bgr)

    @property
    def width(self):
        return self.frames_bgr[0].shape[1]

    @property
    def height(self):
        return self.frames_bgr[0].shape[0]

    def image(self, frame_id: int):
        return self.frames_bgr[frame_id - 1]

    @classmethod
    def from_file(cls, path, max_frames=None, max_side=1920, stride=1):
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames, i = [], 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i % stride == 0:
                s = max_side / max(fr.shape[:2])
                if s < 1:
                    fr = cv2.resize(fr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                frames.append(fr)
                if max_frames and len(frames) >= max_frames:
                    break
            i += 1
        cap.release()
        if not frames:
            raise ValueError(f"could not decode any frame from {path}")
        return cls(Path(path).stem, frames, fps / stride)


class CachedSource:
    def __init__(self, dets: dict, emb: np.ndarray | None, warps: np.ndarray):
        self.box, self.score, self.std = dets["box"].astype(np.float64), dets["score"].astype(np.float64), dets["std"]
        starts = np.searchsorted(dets["frame"], np.arange(1, len(warps) + 2))
        self.sl = [slice(starts[i], starts[i + 1]) for i in range(len(warps))]
        self.emb = emb
        self.warps = warps

    def __call__(self, fid: int, img=None):
        s = self.sl[fid - 1]
        loose = np.concatenate([self.box[s], self.score[s, None]], axis=1)
        e = self.emb[s].astype(np.float64) if self.emb is not None else np.zeros((len(loose), 1))
        m = self.std[s]
        return loose[m], e[m], loose, e, self.warps[fid - 1], {}


class LiveSource:
    """Runs detector + ReID + GMC on each frame (CPU: detector dominates)."""

    def __init__(self, detector, det_kw: dict, embedder=None, std_iou=0.80, embed_min_score=0.0):
        from .detect import nms_mask
        from .gmc import SparseOptFlowGMC

        self.det, self.det_kw, self.emb = detector, det_kw, embedder
        self.std_iou, self.embed_min_score = std_iou, embed_min_score
        self.gmc = SparseOptFlowGMC()
        self._nms = nms_mask

    def __call__(self, fid: int, img):
        t = {}
        t0 = time.perf_counter()
        r = self.det.predict(img, verbose=False, **self.det_kw)[0]
        b = r.boxes.xyxy.cpu().numpy().astype(np.float64)
        s = r.boxes.conf.cpu().numpy().astype(np.float64)
        o = np.argsort(-s)
        b, s = b[o], s[o]
        loose = np.concatenate([b, s[:, None]], axis=1) if len(b) else np.zeros((0, 5))
        m = self._nms(b, s, self.std_iou) if len(b) else np.zeros(0, bool)
        t["detector_ms"] = 1000 * (time.perf_counter() - t0)
        t0 = time.perf_counter()
        if self.emb is not None and len(b):
            need = m | (s > self.embed_min_score)
            e = np.zeros((len(b), 1), np.float64)
            idx = np.where(need)[0]
            if len(idx):
                f = self.emb(img, b[idx]).astype(np.float64)
                e = np.zeros((len(b), f.shape[1]))
                e[idx] = f
        else:
            e = np.zeros((len(b), 1))
        t["reid_ms"] = 1000 * (time.perf_counter() - t0)
        t0 = time.perf_counter()
        warp = self.gmc.apply(img)
        t["gmc_ms"] = 1000 * (time.perf_counter() - t0)
        return loose[m], e[m], loose, e, warp, t


class OnlineTracker:
    """Wraps a benchmark runner for frame-by-frame use. ``step`` returns MOT rows
    (frame, id, x, y, w, h, score), the std detections, and per-stage timings in ms."""

    def __init__(self, tracker: str, tracker_cfg: dict, seq, source, use_reid: bool, out_filter: dict):
        self.runner = RUNNERS[tracker](tracker_cfg, seq, use_reid)
        self.source, self.use_reid = source, use_reid
        self.min_area, self.max_ar = out_filter["min_box_area"], out_filter["max_aspect_ratio"]

    def step(self, fid: int, img=None):
        std, std_e, loose, loose_e, warp, t = self.source(fid, img)
        if not self.use_reid:
            std_e = loose_e = None
        t0 = time.perf_counter()
        out = self.runner.step(std, std_e, loose, loose_e, warp)
        t["tracker_ms"] = 1000 * (time.perf_counter() - t0)
        rows = []
        for tid, x, y, w, h, sc in out:
            if h <= 0 or w * h <= self.min_area or w / h > self.max_ar:
                continue
            rows.append((fid, tid, x, y, w, h, sc))
        return np.array(rows, dtype=np.float64).reshape(-1, 7), std, t
