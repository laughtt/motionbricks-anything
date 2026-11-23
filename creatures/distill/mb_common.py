"""Shared helpers for the MotionBricks distillation scripts.

Pure numpy port of the parts of the MotionBricks SOMA engine that the ONNX model
needs around it (BVH loading, input encoding, output decoding). No Warp, no CUDA.

Ported from (Apache-2.0) NVlabs/motionbricks:
  motionbricks/io/bvh.py, motionbricks/model/soma/kernels.py, motionbricks/model/kernels.py

Paths (all can be changed by environment variable or by configure() / the --work, --fp32-onnx, --motions flags):
  MB_REPO          repo root (default: two folders above this file)
  MB_FP32_ONNX     the FP32 teacher ONNX (default: <repo>/motionbricks/assets/onnx/soma/mmm_v6_soma_fp32.onnx)
  MB_MOTIONS       folder with the sample BVH clips (default: <repo>/assets/motions/soma)
  DISTILL_WORK     work folder for data / models / results / runs (default: ./distill_work)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("MB_REPO", Path(__file__).resolve().parents[2]))
ONNX_DIR = REPO / "motionbricks" / "assets" / "onnx" / "soma"
FP32_ONNX = Path(os.environ.get("MB_FP32_ONNX", ONNX_DIR / "mmm_v6_soma_fp32.onnx"))
MOTIONS = Path(os.environ.get("MB_MOTIONS", REPO / "assets" / "motions" / "soma"))
WORK = Path(os.environ.get("DISTILL_WORK", "distill_work"))
DATA = MODELS = RESULTS = RUNS = WORK            # set by configure()


def configure(work=None, fp32_onnx=None, motions=None):
    """Change the global paths at run time (the command line flags call this)."""
    global WORK, DATA, MODELS, RESULTS, RUNS, FP32_ONNX, MOTIONS
    if work:
        WORK = Path(work)
    DATA, MODELS, RESULTS, RUNS = WORK / "data", WORK / "models", WORK / "results", WORK / "runs"
    if fp32_onnx:
        FP32_ONNX = Path(fp32_onnx)
    if motions:
        MOTIONS = Path(motions)


def add_path_args(ap: argparse.ArgumentParser, fp32=True):
    ap.add_argument("--work", default=None, help="work folder for data/models/results/runs (env DISTILL_WORK, default ./distill_work)")
    if fp32:
        ap.add_argument("--fp32-onnx", default=None, help="FP32 ONNX of the model (env MB_FP32_ONNX, default: the NVIDIA SOMA ONNX in the repo)")
    ap.add_argument("--motions", default=None, help="folder with sample BVH clips (env MB_MOTIONS, default <repo>/assets/motions/soma)")


def apply_path_args(a):
    configure(a.work, getattr(a, "fp32_onnx", None), a.motions)


configure()

FPS = 30.0
POSES_PER_TOKEN = 4
BOUNDARY = 3
MIN_TOK, MAX_TOK = 6, 16
INPUT_NAMES = (
    "global_root_values", "has_global_root_values", "local_root_values",
    "has_local_root_values", "local_poses", "has_local_poses", "allowed_pred_num_tokens",
)
# Output feature layout [366]: root(5) | joint positions 29*3 | 6D rotations 30*6 | velocities 30*3 | contacts 4
OFF_POS, OFF_ROT, OFF_VEL, OFF_CONTACT = 5, 92, 272, 362
FOOT_JOINTS = ("LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase")  # contact order: L heel, L toe, R heel, R toe


# The 30 SOMA model joints, same list as "joint_names" in motionbricks/assets/onnx/soma/soma_n1.json
SOMA_JOINT_NAMES = (
    "Hips", "Spine1", "Spine2", "Chest", "Neck1",
    "Neck2", "Head", "Jaw", "LeftEye", "RightEye",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftHandThumbEnd",
    "LeftHandMiddleEnd", "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "RightHandThumbEnd", "RightHandMiddleEnd", "LeftLeg", "LeftShin", "LeftFoot",
    "LeftToeBase", "RightLeg", "RightShin", "RightFoot", "RightToeBase",
)


def joint_names() -> list[str]:
    return list(SOMA_JOINT_NAMES)


# ----------------------------------------------------------------------------- BVH

def _axis_rot(axis: int, deg: np.ndarray) -> np.ndarray:
    a = np.radians(deg.astype(np.float64))
    c, s = np.cos(a), np.sin(a)
    R = np.zeros(a.shape + (3, 3))
    i, j, k = axis, (axis + 1) % 3, (axis + 2) % 3
    R[..., i, i] = 1.0
    R[..., j, j] = c
    R[..., j, k] = -s
    R[..., k, j] = s
    R[..., k, k] = c
    return R


def load_bvh(path: Path):
    """Return (names, parents, offsets_m [J,3], world_pos [F,J,3], world_rot [F,J,3,3], sample_rate).

    Units: metres, Y up (the BVH and SOMA model space). Rotations compose in the
    channel order listed in the file, as in the repo's loader.
    """
    names, parents, offsets, channels = [], [], [], []
    stack, in_end = [], False
    with open(path, encoding="utf-8") as f:
        for line in f:
            t = line.split()
            if not t:
                continue
            if t[0] == "MOTION":
                break
            if t[0] in ("ROOT", "JOINT"):
                names.append(t[1].rsplit(":", 1)[-1])
                parents.append(stack[-1] if stack else -1)
                stack.append(len(names) - 1)
                offsets.append([0.0, 0.0, 0.0])
                channels.append([])
            elif t[:2] == ["End", "Site"]:
                in_end = True
            elif t[0] == "OFFSET" and not in_end:
                offsets[stack[-1]] = [float(v) for v in t[1:4]]
            elif t[0] == "CHANNELS":
                channels[stack[-1]] = t[2:]
            elif t[0] == "}":
                if in_end:
                    in_end = False
                else:
                    stack.pop()
        rate = None
        for line in f:
            t = line.split()
            if t[:2] == ["Frame", "Time:"]:
                rate = round(1.0 / float(t[2]), 2)
                break
        samples = np.loadtxt(f, dtype=np.float64, ndmin=2)
    J, F = len(names), samples.shape[0]
    offsets = np.asarray(offsets) * 0.01
    lpos = np.broadcast_to(offsets, (F, J, 3)).copy()
    lrot = np.broadcast_to(np.eye(3), (F, J, 3, 3)).copy()
    col = 0
    for j, chs in enumerate(channels):
        R = None
        for ch in chs:
            c = ch.lower()
            if c.endswith("position"):
                lpos[:, j, "xyz".index(c[0])] = samples[:, col] * 0.01
            else:
                Ra = _axis_rot("xyz".index(c[0]), samples[:, col])
                R = Ra if R is None else R @ Ra
            col += 1
        if R is not None:
            lrot[:, j] = R
    wpos = np.zeros((F, J, 3))
    wrot = np.zeros((F, J, 3, 3))
    for j in range(J):
        p = parents[j]
        if p < 0:
            wpos[:, j], wrot[:, j] = lpos[:, j], lrot[:, j]
        else:
            wrot[:, j] = wrot[:, p] @ lrot[:, j]
            wpos[:, j] = wpos[:, p] + np.einsum("fij,fj->fi", wrot[:, p], lpos[:, j])
    return names, parents, offsets, wpos, wrot, rate


class Clip:
    """A BVH clip reduced to the 30 model joints at 30 Hz, in world (model) space."""

    def __init__(self, path: Path):
        names, parents, offsets, wpos, wrot, rate = load_bvh(path)
        step = int(round(rate / FPS))
        assert abs(rate / step - FPS) < 1e-3, f"unexpected sample rate {rate}"
        idx = [names.index(n) for n in joint_names()]
        self.name = path.stem
        self.pos = wpos[::step][:, idx]            # [F,30,3]
        self.rot = wrot[::step][:, idx]            # [F,30,3,3]
        self.num_frames = self.pos.shape[0]
        self.bvh_names, self.bvh_parents, self.bvh_offsets = names, parents, offsets


def model_skeleton(clip_path: Path | None = None):
    """Model-joint parents and rest offsets (parent frame) taken from a BVH hierarchy.

    Returns (parents [30], offsets [30,3]). A model joint whose BVH parent is not a
    model joint is attached to its nearest model ancestor (offsets summed, assuming
    identity rotation on the skipped joints).
    """
    if clip_path is None:
        clip_path = sorted(MOTIONS.rglob("*.bvh"))[0]
    names, parents, offsets = [], [], []
    stack, in_end = [], False
    with open(clip_path, encoding="utf-8") as f:
        for line in f:
            t = line.split()
            if not t:
                continue
            if t[0] == "MOTION":
                break
            if t[0] in ("ROOT", "JOINT"):
                names.append(t[1].rsplit(":", 1)[-1]); parents.append(stack[-1] if stack else -1)
                stack.append(len(names) - 1); offsets.append([0.0, 0.0, 0.0])
            elif t[:2] == ["End", "Site"]:
                in_end = True
            elif t[0] == "OFFSET" and not in_end:
                offsets[stack[-1]] = [float(v) * 0.01 for v in t[1:4]]
            elif t[0] == "}":
                if in_end:
                    in_end = False
                else:
                    stack.pop()
    mj = joint_names()
    mp, mo, skipped = [], [], []
    for n in mj:
        j = names.index(n)
        off = np.array(offsets[j])
        p = parents[j]
        while p >= 0 and names[p] not in mj:
            skipped.append((n, names[p]))
            off = off + np.array(offsets[p])
            p = parents[p]
        mp.append(mj.index(names[p]) if p >= 0 else -1)
        mo.append(off)
    return np.array(mp), np.array(mo), skipped


# ----------------------------------------------------------------------------- encode

def _heading(right: np.ndarray, left: np.ndarray) -> np.ndarray:
    ax, az = right[..., 0] - left[..., 0], right[..., 2] - left[..., 2]
    n = np.sqrt(ax * ax + az * az)
    ok = n > 1e-6
    n = np.where(ok, n, 1.0)
    # forward = cross(up, across) = (az, 0, -ax); angle = atan2(fwd.x, fwd.z)
    return np.where(ok, np.arctan2(az / n, -ax / n), 0.0)


def _rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def alignment_of(pos: np.ndarray, desired=(1.0, 0.0)):
    """Alignment (origin [3], R [3,3]) that puts pose `pos` [30,3] at the ground origin facing `desired`."""
    jn = joint_names()
    r, l = jn.index("RightLeg"), jn.index("LeftLeg")
    cur = float(_heading(pos[r], pos[l]))
    des = float(np.arctan2(desired[0], desired[1]))
    return np.array([pos[0, 0], 0.0, pos[0, 2]]), _rot_y(cur - des)


def encode_request(pos: np.ndarray, rot: np.ndarray, num_targets: int, allowed, align=None, has_targets=True):
    """Build one N=1 input dict from 8 world poses (4 context + 4 target).

    pos [8,30,3], rot [8,30,3,3]. Target rows beyond num_targets must already be
    padded with the last given target (as the engine's pad kernel does).
    Returns (inputs, (origin, R)).
    """
    jn = joint_names()
    r, l = jn.index("RightLeg"), jn.index("LeftLeg")
    origin, R = alignment_of(pos[3]) if align is None else align
    ap = (pos - origin) @ R                      # row-vector form of R^T (p - o)
    ar = np.einsum("ij,fkjl->fkil", R.T, rot)
    head = _heading(ap[:, r], ap[:, l])
    nxt = np.array([1, 2, 3, 3, 5, 6, 7, 7])
    dh = head[nxt] - head
    dh = np.arctan2(np.sin(dh), np.cos(dh))
    hips, nh = ap[:, 0], ap[nxt, 0]
    g = np.stack([hips[:, 0], hips[:, 1], hips[:, 2], np.cos(head), np.sin(head)], -1)
    lr = np.stack([dh * FPS, (nh[:, 0] - hips[:, 0]) * FPS, (nh[:, 2] - hips[:, 2]) * FPS, hips[:, 1]], -1)
    rootp = hips * np.array([1.0, 0.0, 1.0])
    cp = (ap[:, 1:] - rootp[:, None]).reshape(8, 87)
    r6 = np.concatenate([ar[..., 0], ar[..., 1]], -1).reshape(8, 180)   # x axis then y axis per joint
    lp = np.concatenate([cp, r6], -1)
    idx = np.arange(8)
    given = (idx < 4 + (num_targets if has_targets else 0)).astype(np.int32)
    hl = given.copy()
    hl[3] = 0
    if has_targets:
        hl[4 + num_targets - 1] = 0
    allowed_vec = np.zeros(11, np.int32)
    for c in allowed:
        allowed_vec[c - MIN_TOK] = 1
    inputs = {
        "global_root_values": g[None].astype(np.float32),
        "has_global_root_values": given[None],
        "local_root_values": lr[None].astype(np.float32),
        "has_local_root_values": hl[None],
        "local_poses": lp[None].astype(np.float32),
        "has_local_poses": given[None].copy(),
        "allowed_pred_num_tokens": allowed_vec[None],
    }
    return inputs, (origin, R)


# ----------------------------------------------------------------------------- decode

def rot_from_6d(r6: np.ndarray) -> np.ndarray:
    """[..., 6] -> [..., 3, 3], same Gram-Schmidt as the engine's _rotation_from_6d."""
    x = r6[..., :3].astype(np.float64)
    y = r6[..., 3:].astype(np.float64)
    x = x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)
    z = np.cross(x, y)
    z = z / np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-12)
    y = np.cross(z, x)
    return np.stack([x, y, z], -1)


_SKEL = None


def skel():
    global _SKEL
    if _SKEL is None:
        _SKEL = model_skeleton()
    return _SKEL


def decode_motion(feat: np.ndarray, num_tokens: int, trim: bool = True):
    """Decode one request's raw output [64,366] like the engine: FK from root position
    and global 6D rotations with the rest bone offsets. Model space (no alignment undo).

    Returns dict(pos [F,30,3], rot [F,30,3,3], contacts [F,4], raw_pos [F,30,3]) where
    raw_pos is the model's own joint-position features (root + root-ground-relative children).
    """
    n = int(num_tokens) * POSES_PER_TOKEN
    lo, hi = (BOUNDARY, n - BOUNDARY) if trim else (0, n)
    f = feat[lo:hi].astype(np.float64)
    F = f.shape[0]
    rot = rot_from_6d(f[:, OFF_ROT:OFF_VEL].reshape(F, 30, 6))
    parents, offsets, _ = skel()
    pos = np.zeros((F, 30, 3))
    pos[:, 0] = f[:, :3]
    for j in range(1, 30):
        p = parents[j]
        pos[:, j] = pos[:, p] + np.einsum("fij,j->fi", rot[:, p], offsets[j])
    raw = np.zeros((F, 30, 3))
    raw[:, 0] = f[:, :3]
    raw[:, 1:] = f[:, OFF_POS:OFF_ROT].reshape(F, 29, 3) + (f[:, :3] * np.array([1.0, 0.0, 1.0]))[:, None]
    return {"pos": pos, "rot": rot, "contacts": np.clip(f[:, OFF_CONTACT:], 0.0, 1.0), "raw_pos": raw}


def to_world(pos: np.ndarray, align) -> np.ndarray:
    origin, R = align
    return pos @ R.T + origin


def to_world_rot(rot: np.ndarray, align) -> np.ndarray:
    return np.einsum("ij,...jk->...ik", align[1], rot)


# ----------------------------------------------------------------------------- ORT

def make_session(path, threads: int = 0, provider: str = "cpu", opt: str = "all", extra=None):
    import onnxruntime as ort

    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = {
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "none": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    }[opt]
    so.log_severity_level = 3
    for k, v in (extra or {}).items():
        so.add_session_config_entry(k, v)
    providers = {
        "cpu": ["CPUExecutionProvider"],
        "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "trt": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
    }[provider]
    return ort.InferenceSession(str(path), so, providers=providers)


def load_golden(path=None):
    z = np.load(path or (DATA / "golden_soma.npz"), allow_pickle=False)
    return {k: z[k] for k in z.files}
