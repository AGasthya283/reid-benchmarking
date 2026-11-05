"""reid-benchmarking pipeline: GMC -> embeddings -> tracking -> TrackEval -> report.

Detections must exist first (scripts/run_detection.py). Every stage is cached, so re-running only
does missing work.

  python scripts/run_benchmark.py                       # everything
  python scripts/run_benchmark.py --stages embed --reid osnet_x1_0_msmt17
  python scripts/run_benchmark.py --stages track eval report --trackers deepocsort
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reidbench.data import load_sequences, write_eval_gt  # noqa: E402
from reidbench.detect import load_dets  # noqa: E402

NO_REID = "none"


def embed_mask(d: dict, cfg: dict) -> np.ndarray:
    """Boxes whose appearance any tracker can consume: the whole standard set (TrackTrack uses
    features of low-score detections too) + loose-set boxes above TrackTrack's det_thr (its
    recovered "deleted" detections). Everything else is never read."""
    return d["std"] | (d["score"] > cfg["trackers"]["tracktrack"]["det_thr"])


def stage_gmc(cfg, seqs):
    from reidbench.gmc import sparse_optflow_gmc

    for s in seqs:
        p = ROOT / "cache" / "gmc" / f"{s.name}.npy"
        if p.exists():
            continue
        t = time.time()
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, sparse_optflow_gmc(s))
        print(f"[gmc] {s.name} {time.time() - t:.0f}s", flush=True)


def stage_embed(cfg, seqs, models):
    from reidbench.reid.backends import build_embedder

    det_name = cfg["detector"]["name"]
    for name in models:
        out_dir = ROOT / "cache" / "emb" / det_name / name
        todo = [s for s in seqs if not (out_dir / f"{s.name}.npy").exists()]
        if not todo:
            continue
        emb = build_embedder(cfg["reid_models"][name], threads=cfg.get("threads"), name=name)
        out_dir.mkdir(parents=True, exist_ok=True)
        for s in todo:
            d = load_dets(ROOT / "cache" / "dets" / det_name / f"{s.name}.npz")
            need = embed_mask(d, cfg)
            feats = None
            emb.time_s, emb.n_crops = 0.0, 0
            t = time.time()
            for fid in range(1, s.length + 1):
                rows = np.where((d["frame"] == fid) & need)[0]
                if len(rows) == 0:
                    continue
                f = emb(s.image(fid), d["box"][rows])
                if feats is None:
                    feats = np.zeros((len(d["box"]), f.shape[1]), np.float16)
                feats[rows] = f.astype(np.float16)
                if fid % 100 == 0:
                    print(f"  [{name}] {s.name} {fid}/{s.length}  {1000 * emb.time_s / max(emb.n_crops, 1):.1f} ms/crop",
                          flush=True)
            np.save(out_dir / f"{s.name}.npy", feats)
            stats = {"crops": emb.n_crops, "model_time_s": emb.time_s, "wall_s": time.time() - t,
                     "dim": int(feats.shape[1]), "runtime": getattr(emb, "runtime", "onnx"),
                     "onnx_parity_min_cos": getattr(emb, "onnx_parity", None)}
            (out_dir / f"{s.name}.json").write_text(json.dumps(stats))
            print(f"[embed] {name} {s.name}: {emb.n_crops} crops, {1000 * emb.time_s / max(emb.n_crops, 1):.1f} ms/crop",
                  flush=True)


def load_emb(cfg, name, seq):
    if name == NO_REID:
        return None
    e = np.load(ROOT / "cache" / "emb" / cfg["detector"]["name"] / name / f"{seq.name}.npy").astype(np.float32)
    n = np.linalg.norm(e, axis=1, keepdims=True)
    return np.divide(e, n, out=np.zeros_like(e), where=n > 0)  # re-normalise after fp16 storage


def run_dir(cfg):
    return ROOT / "results" / f"{cfg['dataset']['name']}__{cfg['detector']['name']}"


def stage_track(cfg, seqs, trackers, models, force=False):
    from reidbench.runners import run_sequence

    det_name = cfg["detector"]["name"]
    base = run_dir(cfg) / "trackers"
    for tr in trackers:
        for name in models:
            if tr == "tracktrack" and name == NO_REID:
                continue
            tag = f"{tr}__{name}"
            data = base / tag / "data"
            timing_p = base / tag / "timing.json"
            if timing_p.exists() and not force:
                continue
            data.mkdir(parents=True, exist_ok=True)
            timing = {}
            for s in seqs:
                d = load_dets(ROOT / "cache" / "dets" / det_name / f"{s.name}.npz")
                warps = np.load(ROOT / "cache" / "gmc" / f"{s.name}.npy")
                rows, secs = run_sequence(tr, cfg["trackers"][tr], s, d, load_emb(cfg, name, s), warps,
                                          cfg["output_filter"])
                (data / f"{s.name}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
                timing[s.name] = {"frames": s.length, "tracker_s": secs}
            timing_p.write_text(json.dumps(timing))
            fps = sum(v["frames"] for v in timing.values()) / max(sum(v["tracker_s"] for v in timing.values()), 1e-9)
            print(f"[track] {tag}: {fps:.1f} FPS (association only)", flush=True)


def stage_eval(cfg, seqs):
    from reidbench.evaluate import evaluate

    rd = run_dir(cfg)
    gt_dir = rd / "gt"
    seqmap = write_eval_gt(seqs, gt_dir)
    names = sorted(p.name for p in (rd / "trackers").iterdir() if (p / "timing.json").exists())
    res = evaluate(gt_dir, seqmap, rd / "trackers", names, threads=cfg.get("threads", 4))
    (rd / "metrics.json").write_text(json.dumps(res, indent=1))
    print(f"[eval] {len(names)} runs -> {rd / 'metrics.json'}")


def stage_report(cfg, seqs):
    from reidbench.report import write_report

    write_report(cfg, run_dir(cfg), seqs, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/benchmark.yaml"))
    ap.add_argument("--stages", nargs="*", default=["gmc", "embed", "track", "eval", "report"])
    ap.add_argument("--reid", nargs="*", help="subset of reid_models (+ 'none' for the no-ReID baseline)")
    ap.add_argument("--trackers", nargs="*")
    ap.add_argument("--seqs", nargs="*")
    ap.add_argument("--force", action="store_true", help="re-run tracking even if cached")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    seqs = load_sequences(ROOT / cfg["dataset"]["split_dir"], a.seqs or cfg["dataset"].get("sequences"))
    models = a.reid or (list(cfg["reid_models"]) + [NO_REID])
    trackers = a.trackers or list(cfg["trackers"])
    if "gmc" in a.stages:
        stage_gmc(cfg, seqs)
    if "embed" in a.stages:
        stage_embed(cfg, seqs, [m for m in models if m != NO_REID])
    if "track" in a.stages:
        stage_track(cfg, seqs, trackers, models, a.force)
    if "eval" in a.stages:
        stage_eval(cfg, seqs)
    if "report" in a.stages:
        stage_report(cfg, seqs)


if __name__ == "__main__":
    main()
