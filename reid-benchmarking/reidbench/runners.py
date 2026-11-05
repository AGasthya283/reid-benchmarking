"""Stage 4 - run a tracker over one sequence from cached detections / embeddings / GMC.

All trackers receive exactly the same inputs:
  * std set  : detections after NMS 0.80 (x1, y1, x2, y2, score), conf >= 0.10
  * loose set: detections after NMS 0.95 (only TrackTrack uses it)
  * embeddings for every box (L2-normalised), or None for the no-ReID baseline
  * camera-motion warp (2x3) per frame (Deep OC-SORT and TrackTrack apply it; FastTracker has no CMC)

and their outputs go through the same filter (min box area, max w/h ratio) before evaluation.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np


def _frame_slices(frames: np.ndarray, n_frames: int):
    """Row ranges per frame (rows are sorted by frame)."""
    starts = np.searchsorted(frames, np.arange(1, n_frames + 2))
    return [slice(starts[i], starts[i + 1]) for i in range(n_frames)]


class _Base:
    def __init__(self, cfg: dict, seq, use_reid: bool):
        self.cfg, self.seq, self.use_reid = cfg, seq, use_reid

    def step(self, std, std_emb, loose, loose_emb, warp):  # -> list of (id, x1, y1, w, h, score)
        raise NotImplementedError


class DeepOCSortRunner(_Base):
    def __init__(self, cfg, seq, use_reid):
        super().__init__(cfg, seq, use_reid)
        from .trackers.deepocsort.ocsort import OCSort

        c = dict(cfg)
        self.use_cmc = c.pop("cmc", True)
        self.t = OCSort(embedding_off=not use_reid, cmc_off=not self.use_cmc, grid_off=True,
                        aw_off=False, new_kf_off=False, **c)

    def step(self, std, std_emb, loose, loose_emb, warp):
        out = self.t.update(std, std_emb if self.use_reid else None, warp if self.use_cmc else None)
        return [(int(r[4]), r[0], r[1], r[2] - r[0], r[3] - r[1], 1.0) for r in out]


class TrackTrackRunner(_Base):
    def __init__(self, cfg, seq, use_reid):
        super().__init__(cfg, seq, use_reid)
        if not use_reid:
            raise ValueError("TrackTrack has no appearance-free configuration (cosine distance is part of "
                             "its cost); the no-ReID baseline is not defined for it.")
        from .trackers.tracktrack.tracker import Tracker

        c = dict(cfg)
        self.use_cmc = c.pop("cmc", True)
        args = SimpleNamespace(**c)
        args.max_time_lost = int(round(c.pop("max_time_lost_s", 2.0) * seq.fps))
        args.data_path = "MOT17/"  # TrackTrack switches dataset-specific behaviour on this string
        self.t = Tracker(args, seq.name)

    @staticmethod
    def _pack(dets, emb):
        if len(dets) == 0:
            return np.zeros((0, 6 + emb.shape[1]))
        return np.concatenate([dets[:, :5], np.zeros((len(dets), 1)), emb], axis=1).astype(np.float64)

    def step(self, std, std_emb, loose, loose_emb, warp):
        w = warp if self.use_cmc else None
        if len(std) == 0:
            tracks = self.t.update_without_detections(w)
        else:
            tracks = self.t.update(self._pack(std, std_emb), self._pack(loose, loose_emb), w)
        res = []
        for t in tracks:
            if t.track_id > 0:
                x, y, ww, hh = t.x1y1wh
                res.append((int(t.track_id), x, y, ww, hh, float(t.score)))
        return res


class FastTrackerRunner(_Base):
    def __init__(self, cfg, seq, use_reid):
        super().__init__(cfg, seq, use_reid)
        from .trackers.fasttracker.fasttracker import Fasttracker

        c = dict(cfg)
        c["with_reid"] = use_reid
        c.setdefault("ROIs", {})  # FastTracker's ROI/cone constraints are scene-specific; none for MOT17
        self.t = Fasttracker(None, c, frame_rate=seq.fps)

    def step(self, std, std_emb, loose, loose_emb, warp):
        out = self.t.update(std, std_emb if self.use_reid else None)
        return [(int(t.track_id), *t.tlwh, float(t.score)) for t in out]


RUNNERS = {"deepocsort": DeepOCSortRunner, "tracktrack": TrackTrackRunner, "fasttracker": FastTrackerRunner}


def run_sequence(tracker: str, tcfg: dict, seq, dets: dict, emb: np.ndarray | None, warps: np.ndarray,
                 out_filter: dict):
    """Returns (rows, seconds) where rows are MOT-format result lines."""
    runner = RUNNERS[tracker](tcfg, seq, emb is not None)
    sl = _frame_slices(dets["frame"], seq.length)
    box, score, std_mask = dets["box"].astype(np.float64), dets["score"].astype(np.float64), dets["std"]
    if emb is None:
        emb = np.zeros((len(box), 1), np.float32)
    min_area, max_ar = out_filter["min_box_area"], out_filter["max_aspect_ratio"]
    rows, t_total = [], 0.0
    for fid in range(1, seq.length + 1):
        s = sl[fid - 1]
        loose = np.concatenate([box[s], score[s, None]], axis=1)
        loose_emb = emb[s].astype(np.float64)
        m = std_mask[s]
        t0 = time.perf_counter()
        out = runner.step(loose[m], loose_emb[m], loose, loose_emb, warps[fid - 1])
        t_total += time.perf_counter() - t0
        for tid, x, y, w, h, sc in out:
            if w * h <= min_area or (h > 0 and w / h > max_ar) or h <= 0:
                continue
            rows.append(f"{fid},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{sc:.3f},-1,-1,-1")
    return rows, t_total
