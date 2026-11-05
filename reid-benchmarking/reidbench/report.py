"""Turn metrics.json + timing files into results/<run>/RESULTS.md and CSVs."""
from __future__ import annotations

import csv
import json
import platform
from pathlib import Path

TRACKER_NAMES = {"deepocsort": "Deep OC-SORT", "tracktrack": "TrackTrack", "fasttracker": "FastTracker"}
METRICS = ["HOTA", "AssA", "IDF1", "IDSW", "DetA", "MOTA", "FP", "FN"]


def _sum_json(paths, keys):
    out = {k: 0.0 for k in keys}
    for p in paths:
        d = json.loads(Path(p).read_text())
        for k in keys:
            out[k] += d[k]
    return out


def write_report(cfg: dict, run_dir: Path, seqs, root: Path):
    metrics = json.loads((run_dir / "metrics.json").read_text())
    det_name = cfg["detector"]["name"]
    n_frames = sum(s.length for s in seqs)

    det_s = 0.0
    for s in seqs:
        import numpy as np

        det_s += float(np.load(root / "cache" / "dets" / det_name / f"{s.name}.npz")["model_time_s"])

    reid_stats = {}
    for name in list(cfg["reid_models"]) + ["none"]:
        if name == "none":
            reid_stats[name] = {"crops": 0, "model_time_s": 0.0, "dim": 0}
            continue
        paths = [root / "cache" / "emb" / det_name / name / f"{s.name}.json" for s in seqs]
        if not all(p.exists() for p in paths):
            continue
        st = _sum_json(paths, ["crops", "model_time_s"])
        st["dim"] = json.loads(paths[0].read_text())["dim"]
        reid_stats[name] = st

    # Clean speed numbers (scripts/benchmark_speed.py, nothing else running) override the
    # pipeline's timings, which may have overlapped with other jobs.
    speed = None
    sp = run_dir / "speed.json"
    if sp.exists():
        speed = json.loads(sp.read_text())

    rows = []
    for tag, per_seq in sorted(metrics.items()):
        tracker, reid = tag.split("__", 1)
        timing = json.loads((run_dir / "trackers" / tag / "timing.json").read_text())
        trk_s = sum(v["tracker_s"] for v in timing.values())
        rs = reid_stats.get(reid, {"crops": 0, "model_time_s": 0.0, "dim": 0})
        c = per_seq["COMBINED_SEQ"]
        if speed:
            sr = speed["reid"].get(reid, {})
            ms_crop = sr.get("ms_per_crop_bs32", 0.0)
            e2e = 1.0 / (speed["detector_s_per_frame"] + sr.get("ms_per_frame_real", 0.0) / 1000 + trk_s / n_frames)
        else:
            ms_crop = 1000 * rs["model_time_s"] / rs["crops"] if rs["crops"] else 0.0
            e2e = n_frames / (det_s + rs["model_time_s"] + trk_s)
        rows.append({
            "tracker": tracker, "reid": reid, **{m: c[m] for m in METRICS},
            "reid_ms_per_crop": ms_crop,
            "reid_dim": rs["dim"],
            "assoc_fps": n_frames / trk_s if trk_s else 0.0,
            "e2e_fps": e2e,
            "per_seq": {k: v for k, v in per_seq.items() if k != "COMBINED_SEQ"},
        })

    # CSVs
    flat = ["tracker", "reid", *METRICS, "reid_ms_per_crop", "reid_dim", "assoc_fps", "e2e_fps"]
    with open(run_dir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flat, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k in flat})
    with open(run_dir / "results_per_sequence.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tracker", "reid", "sequence", *METRICS])
        for r in rows:
            for seq, m in r["per_seq"].items():
                w.writerow([r["tracker"], r["reid"], seq, *[round(m[k], 3) if isinstance(m[k], float) else m[k]
                                                             for k in METRICS]])

    # Markdown
    reid_order = list(cfg["reid_models"]) + ["none"]
    desc = {k: v.get("description", k) for k, v in cfg["reid_models"].items()}
    desc["none"] = "no appearance (tracker's motion-only configuration)"
    L = []
    L.append(f"# Results - {cfg['dataset']['name']}, detector `{det_name}`\n")
    L.append(f"{len(seqs)} sequences, {n_frames} frames. Metrics from TrackEval (pedestrian class). "
             f"Hardware: {platform.processor() or platform.machine()}, {cfg.get('threads', 4)} CPU threads, no GPU.\n")
    L.append("## Main table\n")
    L.append("| Tracker | ReID model | HOTA | AssA | IDF1 | IDSW | DetA | MOTA | ReID ms/crop | e2e FPS |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for tr in cfg["trackers"]:
        sub = [r for r in rows if r["tracker"] == tr]
        if not sub:
            continue
        best = max(r["HOTA"] for r in sub)
        for reid in reid_order:
            r = next((x for x in sub if x["reid"] == reid), None)
            if r is None:
                continue
            hota = f"**{r['HOTA']:.2f}**" if r["HOTA"] == best else f"{r['HOTA']:.2f}"
            ms = f"{r['reid_ms_per_crop']:.1f}" if r["reid_ms_per_crop"] else "-"
            L.append(f"| {TRACKER_NAMES.get(tr, tr)} | `{reid}` | {hota} | {r['AssA']:.2f} | {r['IDF1']:.2f} | "
                     f"{r['IDSW']} | {r['DetA']:.2f} | {r['MOTA']:.2f} | {ms} | {r['e2e_fps']:.2f} |")
    L.append("")
    L.append("Bold = best HOTA per tracker. DetA is (almost) constant per tracker because all runs share the "
             "same detections; differences between ReID models show up in AssA / IDF1 / IDSW.\n")
    if speed:
        L.append(f"**Speed columns** (from `speed.json`, measured in isolation on {speed['frames']} frames of "
                 f"{speed['sequence']}, {speed['threads']} CPU threads): `ReID ms/crop` = onnxruntime forward time "
                 "per person crop at batch 32. `e2e FPS` = 1 / (detector + ReID on that frame's detections + "
                 f"association). The detector alone takes {speed['detector_s_per_frame']:.2f} s/frame on this CPU, "
                 "so absolute FPS is CPU-bound; compare the ReID models relative to each other. "
                 "Association-only FPS is in `results.csv`.\n")
        L.append("| ReID model | ms/crop bs1 | ms/crop bs8 | ms/crop bs32 | crops/frame | ReID ms/frame |")
        L.append("|---|---:|---:|---:|---:|---:|")
        for k, v in speed["reid"].items():
            L.append(f"| `{k}` | {v['ms_per_crop_bs1']:.1f} | {v['ms_per_crop_bs8']:.1f} | {v['ms_per_crop_bs32']:.1f} | "
                     f"{v['crops_per_frame']:.1f} | {v['ms_per_frame_real']:.0f} |")
        L.append("")
    else:
        L.append("**Speed columns**: `ReID ms/crop` = model forward time per crop during the pipeline run "
                 "(onnxruntime, CPU; may include contention). `e2e FPS` = frames / (detector + ReID + association). "
                 "Run `scripts/benchmark_speed.py` for clean numbers.\n")
    L.append("## ReID models\n")
    L.append("| key | model | embedding dim | crops embedded |")
    L.append("|---|---|---:|---:|")
    for k in cfg["reid_models"]:
        st = reid_stats.get(k, {})
        L.append(f"| `{k}` | {desc[k]} | {st.get('dim', '?')} | {int(st.get('crops', 0))} |")
    L.append("")
    L.append("## Per-sequence HOTA\n")
    seq_names = [s.name for s in seqs]
    L.append("| Tracker | ReID | " + " | ".join(seq_names) + " |")
    L.append("|---|---|" + "---:|" * len(seq_names))
    for r in rows:
        L.append(f"| {TRACKER_NAMES.get(r['tracker'], r['tracker'])} | `{r['reid']}` | " +
                 " | ".join(f"{r['per_seq'][s]['HOTA']:.1f}" for s in seq_names) + " |")
    L.append("")
    (run_dir / "RESULTS.md").write_text("\n".join(L))
    print(f"[report] {run_dir / 'RESULTS.md'}")
