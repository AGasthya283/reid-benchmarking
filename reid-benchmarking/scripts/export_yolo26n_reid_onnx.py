"""Build ``weights/yolo26n-reid.onnx`` - a YOLO26n appearance-embedding model in ONNX.

STAND-IN NOTICE
---------------
No public ``yolo26n-reid`` / ``yolov26n-reid.onnx`` checkpoint exists (as of 2026-11). If you have
your own ``yolov26n-reid.onnx``, drop it at ``weights/yolov26n-reid.onnx`` and point the
``yolo26n_reid`` entry of ``configs/benchmark.yaml`` at it; nothing else needs to change (the ONNX
backend reads the input size from the model).

Until then, this script builds the same thing Ultralytics' own tracker ReID uses
(``trackers/utils/reid`` -> ``YOLO("yolo*-cls.pt")`` with ``embed``): the ImageNet-pretrained
YOLO26n-cls backbone + Classify-head conv + global-average-pool, i.e. the 1280-d pre-logit
embedding, exported to ONNX with a dynamic batch and a person-shaped 256x128 input.
It is *not* trained on person-ReID data, so it is the "generic features" baseline of the table.

Usage:  python scripts/export_yolo26n_reid_onnx.py [--weights weights/yolo26n-cls.pt] [--out ...]
"""
import argparse
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


class YoloClsEmbedder(nn.Module):
    def __init__(self, cls_model):
        super().__init__()
        layers = cls_model.model  # nn.Sequential
        self.body = nn.Sequential(*list(layers)[:-1])
        head = layers[-1]
        self.conv, self.pool = head.conv, head.pool

    def forward(self, x):  # x: (N, 3, H, W) RGB in [0, 1]
        x = self.body(x)
        x = self.pool(self.conv(x)).flatten(1)
        return nn.functional.normalize(x, dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "weights/yolo26n-cls.pt"))
    ap.add_argument("--out", default=str(ROOT / "weights/yolo26n-reid.onnx"))
    ap.add_argument("--imgsz", type=int, nargs=2, default=[256, 128], help="H W")
    a = ap.parse_args()

    from ultralytics import YOLO

    m = YOLO(a.weights).model.float().eval()
    for p in m.parameters():
        p.requires_grad_(False)
    net = YoloClsEmbedder(m).eval()
    dummy = torch.rand(2, 3, *a.imgsz)
    with torch.inference_mode():
        d = net(dummy).shape[1]
    torch.onnx.export(net, dummy, a.out, input_names=["images"], output_names=["embeddings"],
                      dynamic_axes={"images": {0: "batch"}, "embeddings": {0: "batch"}},
                      opset_version=17, dynamo=False)
    print(f"wrote {a.out}  (input 1x3x{a.imgsz[0]}x{a.imgsz[1]}, {d}-d embedding)")


if __name__ == "__main__":
    main()
