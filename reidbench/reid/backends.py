"""Appearance-embedding backends.

Every backend exposes the same interface::

    emb = backend(img_bgr, boxes_xyxy)   # -> (N, D) float32, L2-normalised

so the trackers never know which ReID model produced the features.

Backends
--------
* ``onnx``      - any ONNX person-ReID model (e.g. ``yolov26n-reid.onnx``) via onnxruntime.
* ``torchreid`` - KaiyangZhou/deep-person-reid (OSNet etc.), PyTorch.
* ``fastreid``  - JDAI-CV/fast-reid (BoT / AGW / SBS ResNets), PyTorch.
* ``lmbn``      - jixunbo/LightMBN (LMBN-n), PyTorch.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY = ROOT / "third_party"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _resolve(p: str | os.PathLike) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


class Embedder:
    """Shared crop / resize / normalise / batch logic."""

    def __init__(self, input_hw=(256, 128), mean=IMAGENET_MEAN, std=IMAGENET_STD, rgb=True,
                 batch_size=64, scale=1 / 255.0):
        self.h, self.w = int(input_hw[0]), int(input_hw[1])
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 3, 1, 1)
        self.rgb = rgb
        self.batch_size = batch_size
        self.scale = scale
        self.dim = None
        self.time_s = 0.0  # pure model forward time (no cropping), for throughput reporting
        self.n_crops = 0

    def preprocess(self, img_bgr: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        H, W = img_bgr.shape[:2]
        crops = []
        for x1, y1, x2, y2 in np.asarray(boxes, dtype=np.float64).reshape(-1, 4):
            x1, y1 = int(max(0, np.floor(x1))), int(max(0, np.floor(y1)))
            x2, y2 = int(min(W, np.ceil(x2))), int(min(H, np.ceil(y2)))
            if x2 - x1 < 2 or y2 - y1 < 2:
                crop = np.zeros((self.h, self.w, 3), dtype=np.uint8)
            else:
                crop = cv2.resize(img_bgr[y1:y2, x1:x2], (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            crops.append(crop)
        x = np.stack(crops).astype(np.float32)  # N,H,W,3 (BGR)
        if self.rgb:
            x = x[..., ::-1]
        x = x.transpose(0, 3, 1, 2) * self.scale
        return np.ascontiguousarray((x - self.mean) / self.std, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def __call__(self, img_bgr: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        boxes = np.asarray(boxes).reshape(-1, 4)
        if len(boxes) == 0:
            return np.zeros((0, self.dim or 0), dtype=np.float32)
        x = self.preprocess(img_bgr, boxes)
        out = []
        t0 = time.perf_counter()
        for i in range(0, len(x), self.batch_size):
            out.append(np.asarray(self.forward(x[i:i + self.batch_size]), dtype=np.float32).reshape(
                len(x[i:i + self.batch_size]), -1))
        self.time_s += time.perf_counter() - t0
        self.n_crops += len(x)
        f = np.concatenate(out)
        f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-12
        self.dim = f.shape[1]
        return f


# --------------------------------------------------------------------------------------------
class OnnxEmbedder(Embedder):
    """Generic ONNX ReID model. Input size is read from the model when static."""

    def __init__(self, weights, input_hw=None, threads=None, **kw):
        import onnxruntime as ort

        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = int(threads)
        self.sess = ort.InferenceSession(str(_resolve(weights)), so, providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.inp_name = inp.name
        shape = inp.shape
        if input_hw is None:
            if isinstance(shape[2], int) and isinstance(shape[3], int):
                input_hw = (shape[2], shape[3])
            else:
                input_hw = (256, 128)
        self.static_batch = shape[0] if isinstance(shape[0], int) else None
        super().__init__(input_hw=input_hw, **kw)
        if self.static_batch:
            self.batch_size = self.static_batch

    def forward(self, x):
        n = len(x)
        if self.static_batch and n < self.static_batch:  # pad a static-batch model
            x = np.concatenate([x, np.zeros((self.static_batch - n, *x.shape[1:]), np.float32)])
        return self.sess.run(None, {self.inp_name: x})[0][:n]


# --------------------------------------------------------------------------------------------
class _TorchEmbedder(Embedder):
    """Subclasses build ``self.model`` (an eval-mode nn.Module mapping a normalised NCHW batch to
    (N, D) features) and then call ``super().__init__``.

    runtime="torch": run the PyTorch module directly.
    runtime="onnx" : export the PyTorch module to ONNX once (cached next to the weights), verify
                     parity against PyTorch (cosine >= 0.9999 on a probe batch; the measured value is recorded) and run it with
                     onnxruntime. On this CPU-only box onnxruntime is 1.5-3x faster than PyTorch
                     for these architectures (depthwise / IBN convs), and it puts every model on the
                     same runtime for the speed column.
    """

    def __init__(self, threads=None, runtime="torch", onnx_path=None, **kw):
        import torch

        if threads:
            torch.set_num_threads(int(threads))
        self.torch = torch
        self.runtime = runtime
        super().__init__(**kw)
        if runtime == "onnx":
            self._init_onnx(onnx_path, threads)

    def _torch_forward(self, x):
        with self.torch.inference_mode():
            # copy: fast-reid's preprocess_image normalises its input tensor in place, and
            # torch.from_numpy shares memory with the caller's array
            return self.model(self.torch.from_numpy(x.copy())).float().cpu().numpy()

    def _init_onnx(self, onnx_path, threads):
        import onnxruntime as ort

        torch = self.torch
        onnx_path = _resolve(onnx_path)
        probe = np.random.default_rng(0).standard_normal((4, 3, self.h, self.w)).astype(np.float32)
        if not onnx_path.exists():
            onnx_path.parent.mkdir(parents=True, exist_ok=True)
            with torch.inference_mode():
                torch.onnx.export(self.model, torch.from_numpy(probe), str(onnx_path), input_names=["images"],
                                  output_names=["features"], opset_version=17, dynamo=False,
                                  dynamic_axes={"images": {0: "batch"}, "features": {0: "batch"}})
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = int(threads)
        self.sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
        a = self._torch_forward(probe).reshape(len(probe), -1)
        b = self.sess.run(None, {"images": probe})[0].reshape(len(probe), -1)
        cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
        assert cos.min() >= 0.9999, f"ONNX export parity check failed: min cosine {cos.min()}"
        self.onnx_parity = float(cos.min())

    def forward(self, x):
        if self.runtime == "onnx":
            return self.sess.run(None, {"images": x})[0]
        return self._torch_forward(x)


class TorchreidEmbedder(_TorchEmbedder):
    """KaiyangZhou/deep-person-reid. Only ``torchreid.models`` is imported (training/data
    utilities are not needed for feature extraction)."""

    def __init__(self, arch, weights, num_classes=1000, **kw):
        import importlib.util
        import types

        import torch

        src = THIRD_PARTY / "deep-person-reid" / "torchreid"
        # Import torchreid.models without executing torchreid/__init__.py (which pulls in the
        # training engine, tensorboard, cython eval code, ...).
        pkg = types.ModuleType("torchreid")
        pkg.__path__ = [str(src)]
        sys.modules.setdefault("torchreid", pkg)
        models = importlib.import_module("torchreid.models")
        model = models.build_model(name=arch, num_classes=num_classes, pretrained=False)
        sd = torch.load(_resolve(weights), map_location="cpu", weights_only=True)
        sd = sd.get("state_dict", sd)
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
        sd = {k: v for k, v in sd.items() if not k.startswith("classifier.")}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        missing = [k for k in missing if not k.startswith("classifier.")]
        assert not missing and not unexpected, (missing, unexpected)
        self.model = model.eval()
        super().__init__(**kw)


class FastReIDEmbedder(_TorchEmbedder):
    """JDAI-CV/fast-reid. Uses the repo's own config + build_model + Checkpointer."""

    def __init__(self, config, weights, **kw):
        sys.path.insert(0, str(THIRD_PARTY / "fast-reid"))
        from fastreid.config import get_cfg
        from fastreid.modeling.meta_arch import build_model

        cfg = get_cfg()
        cfg.merge_from_file(str(THIRD_PARTY / "fast-reid" / config))
        cfg.MODEL.BACKBONE.PRETRAIN = False
        cfg.MODEL.DEVICE = "cpu"
        cfg.MODEL.WEIGHTS = str(_resolve(weights))
        cfg.freeze()
        import torch

        model = build_model(cfg)
        # The v0.1.1 model-zoo checkpoints predate a head refactor: ``heads.bnneck`` is now
        # ``heads.bottleneck.0``. fast-reid's
        # Checkpointer silently skips such keys (leaving a freshly initialised BNNeck), so remap them
        # and load strictly.
        sd = torch.load(cfg.MODEL.WEIGHTS, map_location="cpu", weights_only=True)
        sd = sd.get("model", sd)
        remap = {"heads.bnneck.": "heads.bottleneck.0."}
        fixed = {}
        for k, v in sd.items():
            if k.startswith("heads.classifier") or k == "heads.weight":  # ID classifier, unused
                continue
            for old, new in remap.items():
                if k.startswith(old):
                    k = new + k[len(old):]
                    break
            fixed[k] = v
        missing, unexpected = model.load_state_dict(fixed, strict=False)
        unexpected = [k for k in unexpected if k not in ("pixel_mean", "pixel_std")]
        missing = [k for k in missing if k != "heads.weight"]
        assert not missing and not unexpected, (missing[:5], unexpected[:5])
        self.model = model.eval()
        # fast-reid normalises inside the model (PIXEL_MEAN/STD on 0-255 RGB input)
        kw.setdefault("input_hw", tuple(cfg.INPUT.SIZE_TEST))
        kw.update(mean=(0, 0, 0), std=(1, 1, 1), scale=1.0, rgb=True)
        super().__init__(**kw)


class LMBNEmbedder(_TorchEmbedder):
    """jixunbo/LightMBN LMBN-n. Weights published by BoxMOT (``lmbn_n_duke.pt``) use the key
    prefix ``backbone.``; the original code has the attribute typo ``backone`` - we remap."""

    def __init__(self, weights, num_classes=702, feats=512, **kw):
        import importlib
        import types

        import torch

        src = THIRD_PARTY / "LightMBN"
        pkg = types.ModuleType("lmbn_model")
        pkg.__path__ = [str(src / "model")]
        sys.modules.setdefault("lmbn_model", pkg)
        osnet_mod = importlib.import_module("lmbn_model.osnet")
        lmbn_mod = importlib.import_module("lmbn_model.lmbn_n")
        # never download ImageNet weights - the full checkpoint is loaded below
        orig = osnet_mod.osnet_x1_0
        lmbn_mod.osnet_x1_0 = lambda pretrained=True, **k: orig(pretrained=False, **k)
        args = types.SimpleNamespace(num_classes=num_classes, feats=feats, bnneck=True,
                                     activation_map=False, drop_block=True, w_ratio=1.0, h_ratio=0.33)
        model = lmbn_mod.LMBN_n(args)
        sd = torch.load(_resolve(weights), map_location="cpu", weights_only=True)
        sd = {("backone." + k[len("backbone."):]) if k.startswith("backbone.") else k: v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not missing and not unexpected, (missing[:5], unexpected[:5])

        class _Head(torch.nn.Module):
            # LightMBN test protocol: (N, feats, 7) branch features -> L2-normalise each branch
            # -> concatenate (flip TTA omitted, as for every other model here)
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                out = self.m(x)
                out = out / out.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
                return out.flatten(1)

        self.model = _Head(model).eval()
        super().__init__(**kw)


BACKENDS = {
    "onnx": OnnxEmbedder,
    "torchreid": TorchreidEmbedder,
    "fastreid": FastReIDEmbedder,
    "lmbn": LMBNEmbedder,
}


def build_embedder(spec: dict, threads=None, name=None) -> Embedder:
    spec = dict(spec)
    kind = spec.pop("backend")
    if spec.get("runtime") == "onnx" and not spec.get("onnx_path"):
        spec["onnx_path"] = f"weights/onnx/{name or kind}_{spec['input_hw'][0]}x{spec['input_hw'][1]}.onnx"
    if kind == "onnx":
        spec.pop("runtime", None)
    spec.pop("name", None)
    spec.pop("description", None)
    spec.pop("source", None)
    if "input_hw" in spec and spec["input_hw"] is not None:
        spec["input_hw"] = tuple(spec["input_hw"])
    return BACKENDS[kind](threads=threads, **spec)
