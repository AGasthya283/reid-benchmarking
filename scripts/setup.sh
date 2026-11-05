#!/usr/bin/env bash
# Fetch third-party ReID code (pinned), weights and the MOT17-50 dataset, and verify checksums.
# Everything is downloaded from GitHub (code + release assets + raw files), so it also works on
# networks where Google Drive / Hugging Face / motchallenge.net are blocked.
set -euo pipefail
cd "$(dirname "$0")/.."

clone() {  # url dir commit
  if [ ! -d "third_party/$2" ]; then
    git clone -q "$1" "third_party/$2"
    git -C "third_party/$2" checkout -q "$3"
    rm -rf "third_party/$2/.git"
  fi
}
mkdir -p third_party weights data
clone https://github.com/KaiyangZhou/deep-person-reid deep-person-reid f8cd150fdf77e8d9e1ed143b7f308c2c609ded50
clone https://github.com/JDAI-CV/fast-reid           fast-reid        c9bc3ceb2f7a6438b62fb515ea3df6d1e999e95d
clone https://github.com/jixunbo/LightMBN            LightMBN         1f4ec0de6535604f89257195f5c083845491dc59

fetch() {  # url out sha256
  if [ ! -f "$2" ]; then curl -fsSL --retry 3 -o "$2" "$1"; fi
  echo "$3  $2" | sha256sum -c --quiet - || { echo "checksum mismatch: $2"; exit 1; }
}
GH=https://github.com
# detectors (yolo26l: benchmark; yolo26n: fast live mode in the Streamlit app) + YOLO26n-cls (for the yolo26n-reid.onnx stand-in)
fetch $GH/ultralytics/assets/releases/download/v8.4.0/yolo26l.pt      weights/yolo26l.pt      9fe3c544f2b19bebad7ea41e76d7ad3d88b7c2f10d11d24430c5311f6b32db26
fetch $GH/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt      weights/yolo26n.pt      9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef
fetch $GH/ultralytics/assets/releases/download/v8.4.0/yolo26n-cls.pt  weights/yolo26n-cls.pt  0dd6f8dbc448870ac98a3cbb7156f923f7ce21fed3755d4019169ffffd279e81
# fast-reid model zoo: SBS(R50-ibn) MSMT17
fetch $GH/JDAI-CV/fast-reid/releases/download/v0.1.1/msmt_sbs_R50-ibn.pth weights/msmt_sbs_R50-ibn.pth 4e10cfb5dc667df0ff448838df6f9b56c9d10a95b5349bb9afebaf241a16de4a
# LMBN-n DukeMTMC (BoxMOT release asset)
fetch $GH/mikel-brostrom/boxmot/releases/download/v21.0.0/lmbn_n_duke.pt weights/lmbn_n_duke.pt bbc2080e15b1b819a3c3cf6d007963e6ab33ac87eb2781e34196164de10f081f
# torchreid OSNet x1.0 MSMT17. The official link is on Google Drive; this is a byte-identical copy
# (same sha256 found in two independent GitHub repositories; 1041-way MSMT17 classifier).
fetch https://media.githubusercontent.com/media/JongHoonJang/SSS/HEAD/backend/yolo7deep/strong_sort/deep/checkpoint/osnet_x1_0_msmt17.pth \
      weights/osnet_x1_0_msmt17.pth b7d73dc67c016fd044e4027ff856019496392a7aca8fa0ed56d862a1632c1cf2

# MOT17-50: MOT17 train, 7 sequences, first 50% of frames of each (BoxMOT release asset)
if [ ! -d data/MOT17-50 ]; then
  fetch $GH/mikel-brostrom/boxmot/releases/download/v10.0.83/MOT17-50.zip data/MOT17-50.zip \
        6a84f4c5db50b6b8268838fc76bde8eddd2dff3008f7dc744f5cd761997dc614
  unzip -q data/MOT17-50.zip -d data -x '__MACOSX/*' && rm data/MOT17-50.zip
fi

# yolo26n-reid.onnx stand-in (skip if you dropped your own model in weights/)
[ -f weights/yolo26n-reid.onnx ] || python scripts/export_yolo26n_reid_onnx.py
echo "setup done"
