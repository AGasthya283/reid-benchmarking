"""Drawing + video rendering for tracking results (used by the Streamlit platform and
scripts/render_videos.py)."""
from __future__ import annotations

import colorsys
from pathlib import Path

import cv2
import numpy as np


def id_color(tid: int) -> tuple[int, int, int]:
    """Stable, well-separated BGR colour per track id (golden-ratio hue walk)."""
    h = (tid * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def load_mot(path_or_rows) -> np.ndarray:
    """MOT txt (file path or list of lines) -> (N, 7) [frame, id, x, y, w, h, score]."""
    if isinstance(path_or_rows, (str, Path)):
        p = Path(path_or_rows)
        if not p.exists() or p.stat().st_size == 0:
            return np.zeros((0, 7))
        a = np.loadtxt(p, delimiter=",", ndmin=2)
    else:
        rows = [r for r in path_or_rows if r]
        if not rows:
            return np.zeros((0, 7))
        a = np.array([[float(x) for x in r.split(",")] for r in rows])
    return a[:, :7]


def load_gt(seq_root: Path) -> np.ndarray:
    """Pedestrian GT (class 1, considered) -> (N, 7) [frame, id, x, y, w, h, vis]."""
    g = np.loadtxt(Path(seq_root) / "gt" / "gt.txt", delimiter=",", ndmin=2)
    g = g[(g[:, 6] == 1) & (g[:, 7] == 1)]
    return g[:, [0, 1, 2, 3, 4, 5, 8]]


def by_frame(a: np.ndarray) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    if len(a) == 0:
        return out
    a = a[np.argsort(a[:, 0], kind="stable")]
    frames, starts = np.unique(a[:, 0].astype(int), return_index=True)
    for f, s, e in zip(frames, starts, list(starts[1:]) + [len(a)]):
        out[int(f)] = a[s:e]
    return out


def id_switch_frames(tracks: np.ndarray, gt: np.ndarray, iou_thr: float = 0.5) -> dict[int, set]:
    """Cheap per-frame ID-switch detector for display: greedily match tracks to GT by IoU each frame,
    and flag a track box when the GT identity it covers was last covered by a different track id.
    (Display aid only; official IDSW comes from TrackEval.)"""
    tf, gf = by_frame(tracks), by_frame(gt)
    last: dict[int, int] = {}
    flags: dict[int, set] = {}
    for f in sorted(gf):
        t, g = tf.get(f), gf[f]
        if t is None or len(t) == 0:
            continue
        iou = _iou_xywh(t[:, 2:6], g[:, 2:6])
        while iou.size and iou.max() >= iou_thr:
            i, j = np.unravel_index(iou.argmax(), iou.shape)
            tid, gid = int(t[i, 1]), int(g[j, 1])
            if gid in last and last[gid] != tid:
                flags.setdefault(f, set()).add(tid)
            last[gid] = tid
            iou[i, :] = -1
            iou[:, j] = -1
    return flags


def _iou_xywh(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    return inter / (a[:, 2:3] * a[:, 3:4] + (b[:, 2] * b[:, 3])[None] - inter + 1e-9)


def draw_frame(img: np.ndarray, tracks: np.ndarray | None = None, gt: np.ndarray | None = None,
               dets: np.ndarray | None = None, switched: set | None = None, title: str | None = None,
               trails: dict | None = None, thickness: int = 3) -> np.ndarray:
    """tracks: (N, >=6) [frame, id, x, y, w, h, ...]; gt same layout; dets: (M, 5) xyxy+score."""
    im = img.copy()
    H = im.shape[0]
    fs = max(0.5, H / 1080 * 0.9)
    if gt is not None:
        for r in gt:
            x, y, w, h = r[2:6]
            cv2.rectangle(im, (int(x), int(y)), (int(x + w), int(y + h)), (255, 255, 255), 1, cv2.LINE_AA)
    if dets is not None:
        for x1, y1, x2, y2, s in dets[:, :5]:
            cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (160, 160, 160), 1, cv2.LINE_AA)
    if trails:
        for tid, pts in trails.items():
            if len(pts) > 1:
                cv2.polylines(im, [np.int32(pts)], False, id_color(tid), max(1, thickness - 1), cv2.LINE_AA)
    if tracks is not None:
        for r in tracks:
            tid = int(r[1])
            x, y, w, h = r[2:6]
            c = id_color(tid)
            p1, p2 = (int(x), int(y)), (int(x + w), int(y + h))
            sw = switched is not None and tid in switched
            cv2.rectangle(im, p1, p2, (0, 0, 255) if sw else c, thickness + (2 if sw else 0), cv2.LINE_AA)
            label = f"{tid}" + (" IDSW" if sw else "")
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
            cv2.rectangle(im, (p1[0], p1[1] - th - 6), (p1[0] + tw + 4, p1[1]), (0, 0, 255) if sw else c, -1)
            cv2.putText(im, label, (p1[0] + 2, p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 2, cv2.LINE_AA)
    if title:
        (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, fs * 1.2, 2)
        cv2.rectangle(im, (0, 0), (tw + 20, th + 20), (0, 0, 0), -1)
        cv2.putText(im, title, (10, th + 10), cv2.FONT_HERSHEY_SIMPLEX, fs * 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    return im


class H264Writer:
    """Browser-playable H.264 MP4 via the ffmpeg binary bundled with imageio-ffmpeg."""

    def __init__(self, path, fps, width, height, crf=26):
        import imageio_ffmpeg

        self.w = imageio_ffmpeg.write_frames(str(path), (width, height), fps=fps, codec="libx264",
                                             pix_fmt_in="bgr24", pix_fmt_out="yuv420p", quality=None,
                                             output_params=["-crf", str(crf), "-preset", "veryfast",
                                                            "-movflags", "+faststart"],
                                             macro_block_size=2)
        self.w.send(None)

    def write(self, frame):
        self.w.send(np.ascontiguousarray(frame))

    def close(self):
        self.w.close()


def render_sequence_video(seq, out_path, tracks=None, gt=None, title=None, scale=0.5, show_gt=False,
                          mark_idsw=True, trail_len=0, max_frames=None, progress=None):
    """Render a sequence with tracking overlays to an H.264 MP4. Returns the output path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tf = by_frame(tracks) if tracks is not None else {}
    gf = by_frame(gt) if gt is not None else {}
    sw = id_switch_frames(tracks, gt) if (mark_idsw and tracks is not None and gt is not None) else {}
    W, H = int(seq.width * scale) // 2 * 2, int(seq.height * scale) // 2 * 2
    wr = H264Writer(out_path, seq.fps, W, H)
    n = seq.length if max_frames is None else min(seq.length, max_frames)
    hist: dict[int, list] = {}
    try:
        for f in range(1, n + 1):
            img = seq.image(f)
            t = tf.get(f)
            trails = None
            if trail_len and t is not None:
                for r in t:
                    hist.setdefault(int(r[1]), []).append((r[2] + r[4] / 2, r[3] + r[5]))
                    hist[int(r[1])] = hist[int(r[1])][-trail_len:]
                alive = set(int(i) for i in t[:, 1])
                trails = {k: v for k, v in hist.items() if k in alive}
            im = draw_frame(img, t, gf.get(f) if show_gt else None, switched=sw.get(f),
                            title=f"{title}  #{f}" if title else f"#{f}", trails=trails)
            wr.write(cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA))
            if progress:
                progress(f / n)
    finally:
        wr.close()
    return out_path
