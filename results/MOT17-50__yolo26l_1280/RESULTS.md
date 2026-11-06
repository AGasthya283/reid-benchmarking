# Results - MOT17-50, detector `yolo26l_1280`

7 sequences, 2652 frames. Metrics from TrackEval (pedestrian class). Hardware: x86_64, 4 CPU threads, no GPU.

## Main table

| Tracker | ReID model | HOTA | AssA | IDF1 | IDSW | DetA | MOTA | ReID ms/crop | e2e FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Deep OC-SORT | `yolo26n_reid` | 43.41 | 55.56 | 49.80 | 117 | 34.10 | 36.04 | 1.1 | 0.93 |
| Deep OC-SORT | `osnet_x1_0_msmt17` | 43.45 | 55.67 | 49.91 | 113 | 34.10 | 36.06 | 62.1 | 0.30 |
| Deep OC-SORT | `fastreid_sbs_r50ibn_msmt17` | **43.98** | 57.04 | 50.73 | 111 | 34.09 | 36.05 | 243.8 | 0.09 |
| Deep OC-SORT | `lmbn_n_duke` | 43.75 | 56.42 | 50.25 | 111 | 34.10 | 36.05 | 303.6 | 0.08 |
| Deep OC-SORT | `none` | 43.49 | 55.76 | 50.00 | 121 | 34.10 | 36.03 | - | 1.01 |
| TrackTrack | `yolo26n_reid` | 47.21 | 59.10 | 54.56 | 102 | 38.00 | 39.43 | 1.1 | 0.93 |
| TrackTrack | `osnet_x1_0_msmt17` | **48.00** | 60.36 | 55.40 | 91 | 38.41 | 40.20 | 62.1 | 0.30 |
| TrackTrack | `fastreid_sbs_r50ibn_msmt17` | 47.61 | 60.30 | 55.45 | 88 | 37.83 | 39.80 | 243.8 | 0.09 |
| TrackTrack | `lmbn_n_duke` | 47.66 | 60.09 | 55.42 | 84 | 38.04 | 40.02 | 303.6 | 0.08 |
| FastTracker | `yolo26n_reid` | 46.19 | 58.74 | 53.19 | 143 | 36.65 | 37.29 | 1.1 | 0.94 |
| FastTracker | `osnet_x1_0_msmt17` | 46.22 | 58.82 | 53.25 | 143 | 36.66 | 37.21 | 62.1 | 0.30 |
| FastTracker | `fastreid_sbs_r50ibn_msmt17` | 46.20 | 58.75 | 53.23 | 147 | 36.66 | 37.23 | 243.8 | 0.09 |
| FastTracker | `lmbn_n_duke` | **46.26** | 58.91 | 53.33 | 147 | 36.67 | 37.21 | 303.6 | 0.08 |
| FastTracker | `none` | 45.29 | 56.67 | 52.01 | 170 | 36.55 | 37.05 | - | 1.02 |

Bold = best HOTA per tracker. DetA is (almost) constant per tracker because all runs share the same detections; differences between ReID models show up in AssA / IDF1 / IDSW.

**Speed columns** (from `speed.json`, measured in isolation on 30 frames of MOT17-04, 4 CPU threads): `ReID ms/crop` = onnxruntime forward time per person crop at batch 32. `e2e FPS` = 1 / (detector + ReID on that frame's detections + association). The detector alone takes 0.98 s/frame on this CPU, so absolute FPS is CPU-bound; compare the ReID models relative to each other. Association-only FPS is in `results.csv`.

| ReID model | ms/crop bs1 | ms/crop bs8 | ms/crop bs32 | crops/frame | ReID ms/frame |
|---|---:|---:|---:|---:|---:|
| `yolo26n_reid` | 2.3 | 1.2 | 1.1 | 40.3 | 84 |
| `osnet_x1_0_msmt17` | 76.8 | 58.5 | 62.1 | 40.3 | 2323 |
| `fastreid_sbs_r50ibn_msmt17` | 287.4 | 244.9 | 243.8 | 40.3 | 9637 |
| `lmbn_n_duke` | 373.7 | 308.7 | 303.6 | 40.3 | 12094 |

## ReID models

| key | model | embedding dim | crops embedded |
|---|---|---:|---:|
| `yolo26n_reid` | yolo26n-reid.onnx (stand-in: YOLO26n-cls ImageNet embedding, see scripts/export_yolo26n_reid_onnx.py) | 1280 | 59938 |
| `osnet_x1_0_msmt17` | deep-person-reid (torchreid) OSNet x1.0, MSMT17 | 512 | 59938 |
| `fastreid_sbs_r50ibn_msmt17` | fast-reid SBS(R50-ibn), MSMT17 | 2048 | 59938 |
| `lmbn_n_duke` | LightMBN LMBN-n, DukeMTMC-reID | 3584 | 59938 |

## Per-sequence HOTA

| Tracker | ReID | MOT17-02 | MOT17-04 | MOT17-05 | MOT17-09 | MOT17-10 | MOT17-11 | MOT17-13 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Deep OC-SORT | `fastreid_sbs_r50ibn_msmt17` | 35.1 | 43.3 | 43.0 | 54.3 | 45.4 | 50.1 | 48.4 |
| Deep OC-SORT | `lmbn_n_duke` | 34.0 | 43.3 | 41.9 | 54.3 | 45.5 | 50.0 | 48.4 |
| Deep OC-SORT | `none` | 34.0 | 43.3 | 41.0 | 51.0 | 45.5 | 50.1 | 48.4 |
| Deep OC-SORT | `osnet_x1_0_msmt17` | 34.0 | 43.3 | 41.6 | 51.9 | 44.4 | 50.1 | 48.4 |
| Deep OC-SORT | `yolo26n_reid` | 34.0 | 43.3 | 41.6 | 51.2 | 44.4 | 50.1 | 48.4 |
| FastTracker | `fastreid_sbs_r50ibn_msmt17` | 37.4 | 45.8 | 43.4 | 57.3 | 48.3 | 54.8 | 42.9 |
| FastTracker | `lmbn_n_duke` | 37.4 | 45.8 | 43.4 | 57.5 | 48.3 | 55.1 | 43.2 |
| FastTracker | `none` | 34.1 | 45.8 | 42.5 | 55.4 | 48.2 | 53.5 | 41.0 |
| FastTracker | `osnet_x1_0_msmt17` | 37.4 | 45.8 | 43.4 | 57.5 | 48.3 | 54.8 | 43.2 |
| FastTracker | `yolo26n_reid` | 37.4 | 45.8 | 43.7 | 56.4 | 48.3 | 54.8 | 43.6 |
| TrackTrack | `fastreid_sbs_r50ibn_msmt17` | 40.1 | 47.2 | 46.6 | 57.2 | 49.6 | 50.6 | 51.6 |
| TrackTrack | `lmbn_n_duke` | 39.7 | 47.5 | 46.5 | 55.8 | 49.5 | 50.6 | 53.0 |
| TrackTrack | `osnet_x1_0_msmt17` | 39.5 | 48.3 | 46.4 | 55.6 | 49.2 | 50.5 | 54.3 |
| TrackTrack | `yolo26n_reid` | 39.7 | 47.1 | 43.1 | 55.0 | 48.9 | 49.8 | 55.0 |
