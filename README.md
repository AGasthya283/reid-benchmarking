# reid-benchmarking

Benchmark of **person re-identification (ReID) models inside multi-object trackers** on MOT17.
The repository root is the benchmark (expected checkout location: `~/RnD/reid-benchmarking`);
all paths are relative to it.

The grid currently benchmarked is **3 trackers x 4 ReID models (+ a no-ReID baseline)**:

| Trackers | ReID models |
|---|---|
| Deep OC-SORT (Maggiolino et al., ICIP 2023) | `yolo26n-reid.onnx` (see [stand-in note](#yolo26n-reidonnx-stand-in)) |
| TrackTrack (Shim et al., CVPR 2025) | deep-person-reid / torchreid: OSNet x1.0, MSMT17 |
| FastTracker (Hashempoor & Hwang, 2025) | fast-reid: SBS(R50-ibn), MSMT17 |
| | LightMBN: LMBN-n, DukeMTMC-reID (lightweight multi-branch person ReID) |

**Results: [`results/MOT17-50__yolo26l_1280/RESULTS.md`](results/MOT17-50__yolo26l_1280/RESULTS.md)**
(also `results.csv`, `results_per_sequence.csv`, `metrics.json`, `speed.json`).

## Protocol

The point is to isolate the effect of the appearance model, so everything except the ReID model is
held fixed:

1. **Detections are computed once** (`scripts/run_detection.py`) and shared by every run:
   Ultralytics **YOLO26-L** (COCO, person class), 1280 px, conf >= 0.10, class-agnostic NMS.
   Two sets are cached per frame: NMS IoU 0.80 (the standard set every tracker receives) and
   NMS IoU 0.95 (the extra "loose" set TrackTrack uses to recover suppressed detections; the 0.80
   set is obtained by re-applying NMS to it).
2. **Camera-motion compensation is computed once** (BoT-SORT sparse-optical-flow GMC, the method
   TrackTrack's released GMC files use) and fed to the trackers that use CMC (Deep OC-SORT, TrackTrack).
   FastTracker has no CMC.
3. **Embeddings are computed once per ReID model** for every box any tracker can consume (the
   whole 0.80 set + loose boxes above TrackTrack's detection threshold), L2-normalised, cached.
4. **Trackers** run from the caches with their published MOT17 hyper-parameters
   (`configs/benchmark.yaml`). The same output filter is applied to all of them (box area > 100,
   w/h <= 1.6, the ByteTrack/TrackTrack MOT17 convention). **No offline post-processing**
   (no interpolation / GBI / AFLink), so these are online results.
5. **Evaluation** with the official TrackEval (HOTA, CLEAR, Identity), pedestrian class, standard
   MOT17 preprocessing (distractor classes removed).

### Dataset: MOT17-50

`MOT17-50` = all 7 MOT17 *train* sequences (FRCNN versions: 02, 04, 05, 09, 10, 11, 13), **first 50%
of the frames of each sequence**, 2,652 frames with ground truth. It is distributed as a GitHub
release asset by BoxMOT and is used here because motchallenge.net, Hugging Face and Google Drive are
not reachable from the machine the benchmark was built on. Caveats:

* It is not the usual "MOT17 half-val" split (that is the *second* half of each sequence), so
  numbers are not directly comparable with papers' ablation tables.
* COCO-trained YOLO26 detections are used instead of the MOT17-trained YOLOX-X the trackers were
  tuned with, so absolute HOTA/MOTA are well below paper numbers (lower DetA). The ReID comparison is
  still apples-to-apples because every run sees identical detections.

To use full MOT17 (or MOT20 / DanceTrack): put it in MOT layout under `data/` and change
`dataset.split_dir` in the config.

## Quick start

```bash
git clone <this repo> ~/RnD/reid-benchmarking && cd ~/RnD/reid-benchmarking
pip install -r requirements.txt
bash scripts/setup.sh                 # pinned third-party ReID code, weights, MOT17-50 (sha256-checked)
python scripts/run_detection.py       # stage 1, cached in cache/dets/      (~30 min on 4 CPU cores)
python scripts/run_benchmark.py       # GMC -> embeddings -> tracking -> TrackEval -> RESULTS.md
python scripts/benchmark_speed.py     # isolated speed numbers -> speed.json (then re-run --stages report)
python scripts/sanity_reid.py         # GT-crop sanity check of every ReID model (rank-1 across 15 frames)
```

Every stage is cached; `run_benchmark.py --stages track eval report --trackers deepocsort --reid none`
etc. re-runs a subset.

## Testing platform (Streamlit)

```bash
python scripts/make_videos.py --raw          # simulated test videos: MOT17 sequences as H.264 camera feeds
streamlit run app/streamlit_app.py           # http://localhost:8501
```

| Page | What it does |
|---|---|
| **Leaderboard** | all tracker x ReID runs (TrackEval HOTA / AssA / IDF1 / IDSW / MOTA), grouped bar chart, per-sequence heat-table, isolated speed numbers |
| **Video viewer** | two runs side by side on a sequence: frame scrubber or rendered MP4, ID colours + trails, optional GT boxes, ID switches vs GT flagged in red |
| **Live replay** | a sequence streamed as a simulated camera feed (real-time, 0.25-4x or max speed) and tracked online frame by frame from the benchmark caches; live counters for active tracks, identities and association latency |
| **Test bench** | pick tracker + ReID + sequences, move the hyper-parameter sliders, *Run & evaluate*: re-tracks from the caches and scores with TrackEval in seconds, with deltas against the benchmark defaults; download the parameters as YAML |
| **Test videos** | the full live chain (detector -> ReID -> GMC -> tracker) on a simulated test video or an uploaded clip, with per-stage latency, and downloadable annotated MP4 + MOT-format tracks. `yolo26n` is available for a ~10x faster CPU detector |

Annotated result videos for the benchmark runs can also be rendered offline:
`python scripts/make_videos.py --tracked [--runs tracktrack__osnet_x1_0_msmt17] [--seqs MOT17-04]`.

## Layout

```
configs/benchmark.yaml        dataset, detector, ReID models, tracker hyper-parameters
reidbench/
  data.py                     MOT sequence loading, TrackEval GT folder
  detect.py                   stage 1 (detections, two NMS sets)
  gmc.py                      sparse-optical-flow camera-motion compensation
  reid/backends.py            ReID backends: onnx | torchreid | fastreid | lmbn (+ ONNX export & parity check)
  runners.py                  feeds cached inputs to each tracker, uniform output filter
  evaluate.py / report.py     TrackEval wrapper, RESULTS.md / CSV generation
  trackers/deepocsort/        vendored Deep OC-SORT   (MIT)  - see "Tracker changes"
  trackers/tracktrack/        vendored TrackTrack     (MIT)
  trackers/fasttracker/       vendored FastTracker    (MIT)
  third_party/trackeval/      vendored TrackEval      (MIT)
  online.py / viz.py          frame-by-frame tracking (live/replay) and drawing / H.264 video rendering
app/streamlit_app.py          testing platform
scripts/                      setup, detection, pipeline, speed, sanity, videos, yolo26n-reid export
results/<dataset>__<detector>/  committed metrics + timings
```

`third_party/` (deep-person-reid, fast-reid, LightMBN), `weights/`, `data/` and `cache/` are
git-ignored and recreated by `scripts/setup.sh` and the pipeline.

## ReID models

| key | code | weights | input | notes |
|---|---|---|---|---|
| `yolo26n_reid` | onnxruntime | `weights/yolo26n-reid.onnx` | 256x128 | stand-in, see below |
| `osnet_x1_0_msmt17` | [KaiyangZhou/deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid) @ f8cd150 | `osnet_x1_0_msmt17.pth` (torchreid model zoo) | 256x128 | official link is Google Drive; setup uses a byte-identical GitHub copy (sha256 checked; found identical in two independent repos; 1041-class MSMT17 head) |
| `fastreid_sbs_r50ibn_msmt17` | [JDAI-CV/fast-reid](https://github.com/JDAI-CV/fast-reid) @ c9bc3ce | `msmt_sbs_R50-ibn.pth` (fast-reid v0.1.1 model zoo) | 384x128 | the zoo checkpoint uses the pre-refactor key `heads.bnneck.*`; fast-reid's own `Checkpointer` **silently skips it** (leaving a random BNNeck and cosine similarities of ~0.98 between everyone). We remap to `heads.bottleneck.0.*` and load strictly |
| `lmbn_n_duke` | [jixunbo/LightMBN](https://github.com/jixunbo/LightMBN) @ 1f4ec0d | `lmbn_n_duke.pt` (BoxMOT release) | 384x128 | keys remapped `backbone.` -> `backone.` (attribute typo in the original code); LightMBN test protocol: per-branch L2 norm, concat of 7 branches |

All PyTorch models are exported to ONNX once and run with onnxruntime (`runtime: onnx`); the export
is checked against PyTorch (cosine >= 0.9999) every time it is loaded. Set `runtime: torch` to run
PyTorch directly. `scripts/sanity_reid.py` gives identical results for both runtimes.

### `yolo26n-reid.onnx` stand-in

There is no public `yolo26n-reid` / `yolov26n-reid.onnx` checkpoint. If you have one, copy it to
`weights/` and set `reid_models.yolo26n_reid.weights` (and `mean`/`std`/`input_hw` if it expects
something else); nothing else changes. Until then `scripts/export_yolo26n_reid_onnx.py` builds the
equivalent of what Ultralytics' own tracker ReID uses: the **ImageNet-pretrained YOLO26n-cls**
backbone + head conv + global-average-pool (1280-d pre-logit embedding), exported to ONNX. It has
never seen person-ReID data, so treat it as the "generic features" baseline.

## Tracker changes

The trackers are vendored from their official repositories (MIT) and changed only where needed to
take pre-computed inputs; association logic and hyper-parameters are the published ones.

* **Deep OC-SORT** ([GerardMaggiolino/Deep-OC-SORT](https://github.com/GerardMaggiolino/Deep-OC-SORT) @ 6bb51d0):
  `update(dets, embs, affine)` instead of computing embeddings / CMC internally; single embedding per box
  (`grid_off`); fixed an upstream crash when embeddings are off (adaptive weighting called on an int).
* **TrackTrack** ([kamkyu94/TrackTrack](https://github.com/kamkyu94/TrackTrack) @ ee7f1c5):
  warp matrix passed to `update()` instead of being read from `GMC-<seq>.txt`; vectorised (identical)
  IoU. TrackTrack's cost always includes cosine distance, so it has no "no-ReID" row.
* **FastTracker** ([Hamidreza-Hashempoor/FastTracker](https://github.com/Hamidreza-Hashempoor/FastTracker) @ d35eeda):
  FastTracker's association is motion/IoU only (its "occlusion-aware re-identification" is
  geometric). To benchmark it *with* ReID we added the BoT-SORT appearance fusion to its first and
  unconfirmed association steps: `cost = min(IoU-cost x score, emb_cost)` where
  `emb_cost = (1 - cos)/2`, set to 1 when `> 0.25` or when IoU distance `> 0.5`; track features are
  an EMA (alpha 0.9). The `none` row is the unmodified FastTracker association. Scene-specific
  ROI/cone constraints (designed for its traffic benchmark) are disabled for MOT17.

## Adding a ReID model or tracker

* **ReID**: add an entry under `reid_models` in the config. Any ONNX model works with
  `backend: onnx`; for a new PyTorch library add a small `Embedder` subclass in
  `reidbench/reid/backends.py` that builds `self.model` (normalised NCHW -> (N, D)).
* **Tracker**: add a runner in `reidbench/runners.py` (`step(std_dets, std_embs, loose_dets, loose_embs, warp)`
  -> list of `(id, x, y, w, h, score)`) and its hyper-parameters under `trackers`.

## Licences

Vendored trackers and TrackEval are MIT (licence files kept next to the code). deep-person-reid (MIT),
fast-reid (Apache-2.0) and LightMBN (MIT) are fetched by `scripts/setup.sh`, not vendored.
Weights keep their original licences. Ultralytics YOLO26 weights are AGPL-3.0.
