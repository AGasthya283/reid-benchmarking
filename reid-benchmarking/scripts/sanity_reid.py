"""Quick ReID sanity check on GT crops: for each sequence, match identities between frame f and
frame f+gap by nearest embedding; prints rank-1 accuracy and mean same-/different-ID cosine."""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reidbench.data import load_sequences  # noqa: E402
from reidbench.reid.backends import build_embedder  # noqa: E402


def gt_boxes(seq, fid):
    g = np.loadtxt(seq.root / "gt" / "gt.txt", delimiter=",")
    g = g[(g[:, 0] == fid) & (g[:, 6] == 1) & (g[:, 7] == 1) & (g[:, 8] >= 0.6)]
    b = g[:, 2:6].copy()
    b[:, 2:] += b[:, :2]
    return g[:, 1].astype(int), b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/benchmark.yaml"))
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--gap", type=int, default=15)
    ap.add_argument("--runtime", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    seqs = load_sequences(ROOT / cfg["dataset"]["split_dir"])
    for name, spec in cfg["reid_models"].items():
        if a.models and name not in a.models:
            continue
        if a.runtime and spec["backend"] != "onnx":
            spec = {**spec, "runtime": a.runtime}
        emb = build_embedder(spec, threads=cfg.get("threads"), name=name)
        hits = tot = 0
        same, diff = [], []
        for s in seqs:
            f1 = 1
            ids1, b1 = gt_boxes(s, f1)
            ids2, b2 = gt_boxes(s, f1 + a.gap)
            common = np.intersect1d(ids1, ids2)
            if len(common) < 2:
                continue
            e1 = emb(s.image(f1), b1[np.isin(ids1, common)])
            e2 = emb(s.image(f1 + a.gap), b2[np.isin(ids2, common)])
            i1, i2 = ids1[np.isin(ids1, common)], ids2[np.isin(ids2, common)]
            sim = e1 @ e2.T
            hits += int((i2[sim.argmax(1)] == i1).sum())
            tot += len(i1)
            m = i1[:, None] == i2[None, :]
            same += list(sim[m])
            diff += list(sim[~m])
        print(f"{name:28s} rank-1 {hits / tot:.3f} ({tot} ids)  cos same {np.mean(same):.3f}  diff {np.mean(diff):.3f}"
              f"  dim {emb.dim}", flush=True)


if __name__ == "__main__":
    main()
