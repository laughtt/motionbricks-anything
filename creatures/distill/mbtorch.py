"""PyTorch MotionBricks SOMA (from the released safetensors checkpoint): the teacher for distillation.

Uses NVIDIA's exporter model code (tools/onnx in this repo, Apache 2.0), imported as the package `motionbricks_onnx`.
If the package is not installed (pip install -e tools/onnx), it is loaded from <repo>/tools/onnx.
The NVIDIA checkpoint folder comes from the env var MB_CKPT or the --ckpt flag, for example
checkpoints/MotionBricks/soma (see tools/onnx/README.md, download with `hf download nvidia/MotionBricks`).
"""
from __future__ import annotations

from pathlib import Path

import importlib.util
import os
import sys
import numpy as np
import torch

import mb_common as mb

ONNXDIR = mb.REPO / "tools" / "onnx"
CKPT = Path(os.environ.get("MB_CKPT", mb.REPO / "checkpoints" / "MotionBricks" / "soma"))


INPUT_NAMES = ("global_root_values", "has_global_root_values", "local_root_values", "has_local_root_values",
               "local_poses", "has_local_poses", "allowed_pred_num_tokens")


def _import_exporter():
    if "motionbricks_onnx" in sys.modules:
        return sys.modules["motionbricks_onnx"]
    try:
        import motionbricks_onnx
        return motionbricks_onnx
    except ModuleNotFoundError:
        pass
    spec = importlib.util.spec_from_file_location("motionbricks_onnx", ONNXDIR / "__init__.py",
                                                  submodule_search_locations=[str(ONNXDIR)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["motionbricks_onnx"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_soma(device="cuda", verify=False):
    """BatchedSOMA (eval mode) with the released weights. verify=True checks SHA-256 of every file (slow-ish)."""
    _import_exporter()
    from motionbricks_onnx.checkpoints import checkpoint_root, load_sources
    from motionbricks_onnx.checksums import CHECKPOINTS
    from motionbricks_onnx.models.soma import build_soma
    import motionbricks_onnx.checkpoints as ck
    rel = CHECKPOINTS["soma"]
    if not verify:
        ck.verify_file = lambda p, e: None
    sources = load_sources(checkpoint_root(CKPT, rel), rel)
    model = build_soma(sources)
    del sources
    model = model.to(device)
    inf = model.model
    inf._device = device
    model.capture_offsets = model.capture_offsets.to(device)
    model.standard_offsets = model.standard_offsets.to(device)
    # Stats live in plain tensors on the motion reps; move them
    for rep in {id(r): r for r in _reps(inf)}.values():
        _move_stats(rep, device)
    return model.eval()


def _reps(inf):
    mr = inf.motion_rep
    out = [mr, mr.dual_rep, mr.dual_rep.global_motion_rep, mr.dual_rep.local_motion_rep, inf.global_motion_rep,
           inf.local_motion_rep, inf._vqvae_pose_model.motion_rep]
    return out


def _move_stats(rep, device):
    st = getattr(rep, "stats", None)
    if st is None:
        return
    for k, v in list(vars(st).items()):
        if torch.is_tensor(v):
            setattr(st, k, v.to(device))
    for name, buf in list(st.named_buffers(recurse=False)):
        setattr(st, name, buf.to(device))


class TokenTap:
    """Records pose tokens / logits produced inside predict() (monkeypatches _predict_pose_tokens)."""

    def __init__(self, model):
        self.inf = model.model
        self.orig = self.inf._predict_pose_tokens
        self.tokens = None

        def wrapped(batch, config={}, info={}):
            out = self.orig(batch, config, info)
            self.tokens = out[0].detach()
            return out

        self.inf._predict_pose_tokens = wrapped


def feeds_to_torch(feeds: dict, device):
    out = []
    for k in INPUT_NAMES:
        v = torch.from_numpy(np.ascontiguousarray(feeds[k]))
        out.append(v.to(device))
    return out


@torch.no_grad()
def run(model, feeds: dict, batch=32, device="cuda"):
    """Run the batched graph on numpy feeds (ONNX interface). Returns a dict like quality.run_variant."""
    tap = TokenTap(model)
    N = feeds["global_root_values"].shape[0]
    mot = np.zeros((N, 64, 366), np.float32)
    nt = np.zeros(N, np.int32)
    tok = np.zeros((N, 16, 5), np.int16)
    for i in range(0, N, batch):
        sl = slice(i, min(N, i + batch))
        args = feeds_to_torch({k: feeds[k][sl] for k in INPUT_NAMES}, device)
        m, n = model(*args)
        k = sl.stop - sl.start
        F = m.shape[1]
        mot[sl, :F] = m.float().cpu().numpy()
        nt[sl] = n.cpu().numpy().reshape(-1)
        tok[sl] = tap.tokens.cpu().numpy().astype(np.int16)
    model.model._predict_pose_tokens = tap.orig
    return {"pred_global_motions": mot, "pred_num_tokens": nt, "pose_tokens": tok}


def set_deterministic():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.fp32_precision = "ieee" if hasattr(torch.backends.cuda.matmul, "fp32_precision") else None
