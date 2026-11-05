"""Isolated speed benchmark (run with nothing else on the machine).

Times, on real MOT17 frames / detections:
  * the detector (s/frame)
  * each ReID model at batch sizes 1, 8, 32 (ms/crop) and for the real per-frame batches
Writes results/<run>/speed.json, which the report uses for the speed columns.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reidbench.data import load_sequences  # noqa: E402
from reidbench.detect import load_dets  # noqa: E402
from reidbench.reid.backends import build_embedder  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/benchmark.yaml"))
    ap.add_argument("--seq", default="MOT17-04")
    ap.add_argument("--frames", type=int, default=30)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    threads = cfg.get("threads", 4)
    seq = next(s for s in load_sequences(ROOT / cfg["dataset"]["split_dir"]) if s.name == a.seq)
    d = load_dets(ROOT / "cache" / "dets" / cfg["detector"]["name"] / f"{seq.name}.npz")
    fids = list(range(1, a.frames + 1))
    imgs = {f: seq.image(f) for f in fids}
    out = {"cpu": platform.processor() or platform.machine(), "threads": threads, "sequence": seq.name,
           "frames": a.frames, "reid": {}}

    import torch
    from ultralytics import YOLO

    torch.set_num_threads(threads)
    dc = cfg["detector"]
    det = YOLO(str(ROOT / dc["weights"]))
    kw = dict(imgsz=dc["imgsz"], conf=dc["conf"], iou=dc["nms_loose"], classes=[0], max_det=1000,
              agnostic_nms=True, verbose=False)
    det.predict(imgs[1], **kw)
    t = time.perf_counter()
    for f in fids:
        det.predict(imgs[f], **kw)
    out["detector_s_per_frame"] = (time.perf_counter() - t) / len(fids)
    print(f"detector {dc['name']}: {out['detector_s_per_frame']:.3f} s/frame", flush=True)
    del det

    for name, spec in cfg["reid_models"].items():
        emb = build_embedder(spec, threads=threads, name=name)
        frame_boxes = [d["box"][(d["frame"] == f) & d["std"]] for f in fids]
        pool = np.concatenate(frame_boxes)
        r = {}
        emb(imgs[1], pool[:8])  # warm-up
        for bs in (1, 8, 32):
            x = emb.preprocess(imgs[1], np.resize(pool, (bs, 4)))
            n = max(3, 64 // bs)
            t = time.perf_counter()
            for _ in range(n):
                emb.forward(x)
            r[f"ms_per_crop_bs{bs}"] = 1000 * (time.perf_counter() - t) / (n * bs)
        t = time.perf_counter()
        n_crops = 0
        for f, b in zip(fids, frame_boxes):
            emb(imgs[f], b)
            n_crops += len(b)
        r["ms_per_frame_real"] = 1000 * (time.perf_counter() - t) / len(fids)
        r["crops_per_frame"] = n_crops / len(fids)
        out["reid"][name] = r
        print(name, {k: round(v, 2) for k, v in r.items()}, flush=True)

    p = ROOT / "results" / f"{cfg['dataset']['name']}__{cfg['detector']['name']}" / "speed.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p)


if __name__ == "__main__":
    main()
