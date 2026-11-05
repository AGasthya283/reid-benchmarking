"""Test videos.

  --raw       simulated camera feeds: each MOT17 sequence encoded as an H.264 MP4 at its native fps
              (data/videos/<seq>.mp4). These are the "test videos" the Streamlit platform can replay
              or push through the live detector -> ReID -> tracker chain.
  --tracked   annotated result videos for benchmark runs
              (results/<run>/videos/<tracker>__<reid>/<seq>.mp4), with ID colours, trails and
              ID switches (vs GT) highlighted in red.

  python scripts/make_videos.py --raw
  python scripts/make_videos.py --tracked --runs tracktrack__osnet_x1_0_msmt17 --seqs MOT17-04
"""
import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reidbench.data import load_sequences  # noqa: E402
from reidbench.viz import load_gt, load_mot, render_sequence_video  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/benchmark.yaml"))
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--tracked", action="store_true")
    ap.add_argument("--runs", nargs="*", help="tracker__reid tags (default: all evaluated runs)")
    ap.add_argument("--seqs", nargs="*")
    ap.add_argument("--scale", type=float, default=0.5)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    seqs = load_sequences(ROOT / cfg["dataset"]["split_dir"], a.seqs)
    run_dir = ROOT / "results" / f"{cfg['dataset']['name']}__{cfg['detector']['name']}"

    if a.raw:
        for s in seqs:
            out = ROOT / "data" / "videos" / f"{s.name}.mp4"
            if not out.exists():
                render_sequence_video(s, out, scale=1.0)
            print("[raw]", out, flush=True)

    if a.tracked:
        tags = a.runs or sorted(p.name for p in (run_dir / "trackers").iterdir() if (p / "data").exists())
        for tag in tags:
            for s in seqs:
                out = run_dir / "videos" / tag / f"{s.name}.mp4"
                if out.exists():
                    continue
                tr = load_mot(run_dir / "trackers" / tag / "data" / f"{s.name}.txt")
                render_sequence_video(s, out, tracks=tr, gt=load_gt(s.root), title=tag, scale=a.scale,
                                      trail_len=30)
                print("[tracked]", out, flush=True)


if __name__ == "__main__":
    main()
