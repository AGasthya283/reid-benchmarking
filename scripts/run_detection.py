"""Stage 1: run the person detector once over every sequence and cache the results."""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from reidbench.data import load_sequences  # noqa: E402
from reidbench.detect import detect_sequence  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/benchmark.yaml"))
    ap.add_argument("--seqs", nargs="*")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    dcfg = cfg["detector"]

    import torch
    from ultralytics import YOLO

    torch.set_num_threads(cfg.get("threads", 4))
    model = YOLO(str(ROOT / dcfg["weights"]))
    seqs = load_sequences(ROOT / cfg["dataset"]["split_dir"], a.seqs or cfg["dataset"].get("sequences"))
    out_dir = ROOT / "cache" / "dets" / dcfg["name"]
    for s in seqs:
        out = out_dir / f"{s.name}.npz"
        if out.exists() and not a.force:
            print(f"[skip] {out}")
            continue
        t = time.time()
        tm = detect_sequence(model, s, out, imgsz=dcfg["imgsz"], conf=dcfg["conf"],
                             loose_iou=dcfg["nms_loose"], std_iou=dcfg["nms_std"])
        print(f"[done] {s.name}: {s.length} frames, {s.length / tm:.2f} FPS model, {time.time() - t:.0f}s wall",
              flush=True)


if __name__ == "__main__":
    main()
