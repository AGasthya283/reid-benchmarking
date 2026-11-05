# Results - MOT17-50, detector `yolo26l_1280`

7 sequences, 2652 frames. Metrics from TrackEval (pedestrian class). Hardware: x86_64, 4 CPU threads, no GPU.

## Main table

| Tracker | ReID model | HOTA | AssA | IDF1 | IDSW | DetA | MOTA | ReID ms/crop | e2e FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Deep OC-SORT | `yolo26n_reid` | 43.41 | 55.56 | 49.80 | 117 | 34.10 | 36.04 | 1.0 | 0.86 |
| Deep OC-SORT | `none` | **43.49** | 55.76 | 50.00 | 121 | 34.10 | 36.03 | - | 0.88 |
| TrackTrack | `yolo26n_reid` | **47.21** | 59.10 | 54.56 | 102 | 38.00 | 39.43 | 1.0 | 0.86 |
| FastTracker | `yolo26n_reid` | **46.19** | 58.74 | 53.19 | 143 | 36.65 | 37.29 | 1.0 | 0.86 |
| FastTracker | `none` | 45.29 | 56.67 | 52.01 | 170 | 36.55 | 37.05 | - | 0.88 |

Bold = best HOTA per tracker. DetA is (almost) constant per tracker because all runs share the same detections; differences between ReID models show up in AssA / IDF1 / IDSW.

**Speed columns**: `ReID ms/crop` = model forward time per crop during the pipeline run (onnxruntime, CPU; may include contention). `e2e FPS` = frames / (detector + ReID + association). Run `scripts/benchmark_speed.py` for clean numbers.

## ReID models

| key | model | embedding dim | crops embedded |
|---|---|---:|---:|
| `yolo26n_reid` | yolo26n-reid.onnx (stand-in: YOLO26n-cls ImageNet embedding, see scripts/export_yolo26n_reid_onnx.py) | 1280 | 59938 |
| `osnet_x1_0_msmt17` | deep-person-reid (torchreid) OSNet x1.0, MSMT17 | ? | 0 |
| `fastreid_sbs_r50ibn_msmt17` | fast-reid SBS(R50-ibn), MSMT17 | ? | 0 |
| `lmbn_n_duke` | LightMBN LMBN-n, DukeMTMC-reID | ? | 0 |

## Per-sequence HOTA

| Tracker | ReID | MOT17-02 | MOT17-04 | MOT17-05 | MOT17-09 | MOT17-10 | MOT17-11 | MOT17-13 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Deep OC-SORT | `none` | 34.0 | 43.3 | 41.0 | 51.0 | 45.5 | 50.1 | 48.4 |
| Deep OC-SORT | `yolo26n_reid` | 34.0 | 43.3 | 41.6 | 51.2 | 44.4 | 50.1 | 48.4 |
| FastTracker | `none` | 34.1 | 45.8 | 42.5 | 55.4 | 48.2 | 53.5 | 41.0 |
| FastTracker | `yolo26n_reid` | 37.4 | 45.8 | 43.7 | 56.4 | 48.3 | 54.8 | 43.6 |
| TrackTrack | `yolo26n_reid` | 39.7 | 47.1 | 43.1 | 55.0 | 48.9 | 49.8 | 55.0 |
