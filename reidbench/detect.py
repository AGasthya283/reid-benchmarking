"""Stage 1 - person detection, cached once and shared by every tracker x ReID combination.

For each sequence we store ``cache/dets/<detector>/<seq>.npz`` with, per frame, the boxes kept by
class-agnostic NMS at IoU 0.95 (the "loose" set TrackTrack needs to recover detections suppressed
by a stricter NMS). The standard detection set (NMS 0.80, TrackTrack's default) is stored as an
index mask into the loose set, so ReID embeddings only have to be computed once per box.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torchvision


def nms_mask(boxes: np.ndarray, scores: np.ndarray, iou: float) -> np.ndarray:
    keep = torchvision.ops.nms(torch.as_tensor(boxes, dtype=torch.float32),
                               torch.as_tensor(scores, dtype=torch.float32), iou).numpy()
    m = np.zeros(len(boxes), dtype=bool)
    m[keep] = True
    return m


def detect_sequence(model, seq, out_path: Path, imgsz=1280, conf=0.1, loose_iou=0.95, std_iou=0.80,
                    min_box_area=0.0, log_every=100):
    frames, boxes, scores, std = [], [], [], []
    t_model = 0.0
    for fid in range(1, seq.length + 1):
        img = seq.image(fid)
        t0 = time.perf_counter()
        r = model.predict(img, imgsz=imgsz, conf=conf, iou=loose_iou, classes=[0], max_det=1000,
                          agnostic_nms=True, verbose=False)[0]
        t_model += time.perf_counter() - t0
        b = r.boxes.xyxy.cpu().numpy().astype(np.float32)
        s = r.boxes.conf.cpu().numpy().astype(np.float32)
        if min_box_area > 0 and len(b):
            a = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
            b, s = b[a >= min_box_area], s[a >= min_box_area]
        order = np.argsort(-s)
        b, s = b[order], s[order]
        m = nms_mask(b, s, std_iou) if len(b) else np.zeros(0, bool)
        frames.append(np.full(len(b), fid, np.int32))
        boxes.append(b)
        scores.append(s)
        std.append(m)
        if log_every and fid % log_every == 0:
            print(f"  {seq.name}: {fid}/{seq.length}  {fid / t_model:.2f} FPS (model)", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, frame=np.concatenate(frames), box=np.concatenate(boxes),
                        score=np.concatenate(scores), std=np.concatenate(std),
                        model_time_s=t_model, n_frames=seq.length)
    return t_model


def load_dets(path: Path) -> dict:
    z = np.load(path)
    return {k: z[k] for k in z.files}
