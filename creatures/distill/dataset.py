"""Training samples for the distillation, all in the ONNX request format plus teacher labels.

A sample = one request (8 frames: 4 context + 4 target, ONNX convention) + its labels:
    tokens [16,5] int16  (pose tokens of the teacher; positions >= L are ignored)
    groot  [64,5] float  (normalised, recentred global root trajectory of the teacher; frames >= 4L ignored)
    L      int           (token count 6..16)
Every label comes from the TEACHER (the original NVIDIA model). This file is a library: distill.py gen / genkp / label
call it. A sample file is an npz with the 7 request arrays (mbtorch.INPUT_NAMES) + tokens + groot + L.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import mbtorch as mt
import mb_common as mb

NAMES = mt.INPUT_NAMES
DEFAULT = tuple(range(8, 15))
ALL = tuple(range(6, 17))


def place(pos, rot, ctx_align, dist, bearing, yaw):
    """Same as build_golden.place: re-place a target window relative to the context facing."""
    o_t, R_t = mb.alignment_of(pos[0])
    lp = (pos - o_t) @ R_t
    lr = np.einsum("ij,fkjl->fkil", R_t.T, rot)
    o_c, R_c = ctx_align
    Rw = R_c @ mb._rot_y(-yaw)
    offset = R_c @ np.array([dist * np.cos(bearing), 0.0, dist * np.sin(bearing)])
    return lp @ Rw.T + o_c + offset, np.einsum("ij,fkjl->fkil", Rw, lr)


def pick_allowed(rng):
    u = rng.random()
    if u < 0.6:
        return DEFAULT
    if u < 0.8:
        return ALL
    return (int(rng.integers(6, 17)),)


class Tap:
    def __init__(self, model):
        inf = model.model
        self.inf = inf
        self.o_root, self.o_pose = inf._predict_root_trajectories, inf._predict_pose_tokens
        self.v = {}

        def root(batch, config={}):
            o = self.o_root(batch, config)
            self.v["groot"] = o[1].detach()
            return o

        def pose(batch, config={}, info={}):
            o = self.o_pose(batch, config, info)
            self.v["tokens"] = o[0].detach()
            return o
        inf._predict_root_trajectories, inf._predict_pose_tokens = root, pose

    def close(self):
        self.inf._predict_root_trajectories, self.inf._predict_pose_tokens = self.o_root, self.o_pose


@torch.no_grad()
def run_requests(model, reqs, dev="cuda", batch=256):
    """reqs: list of feeds dicts (N=1 each). Returns stacked feeds + tokens/groot/L + raw motions."""
    feeds = {k: np.concatenate([r[k] for r in reqs], 0) for k in NAMES}
    tap = Tap(model)
    N = len(reqs)
    tok = np.zeros((N, 16, 5), np.int16)
    gro = np.zeros((N, 64, 5), np.float32)
    L = np.zeros(N, np.int16)
    mot = np.zeros((N, 64, 366), np.float32)
    for i in range(0, N, batch):
        sl = slice(i, min(N, i + batch))
        m, n = model(*mt.feeds_to_torch({k: feeds[k][sl] for k in NAMES}, dev))
        mot[sl] = m.cpu().numpy()
        L[sl] = n.cpu().numpy().reshape(-1)
        tok[sl] = tap.v["tokens"].cpu().numpy()
        gro[sl] = tap.v["groot"].cpu().numpy()
    tap.close()
    return feeds, dict(tokens=tok, groot=gro, L=L), mot


def decode_world(mot_i, n, align):
    d = mb.decode_motion(mot_i, int(n), trim=True)
    return mb.to_world(d["pos"], align), mb.to_world_rot(d["rot"], align)


def make_replay(model, rounds, per_round, seed=7, dev="cuda"):
    """Self-rollout pool. Round 0 is seeded from the golden reference OUTPUTS (model-generated motion); only rounds
    >= 1 (whose context/target poses are all model outputs) are kept as training data."""
    rng = np.random.default_rng(seed)
    g = mb.load_golden()
    pool = []
    for i in range(g["pred_num_tokens"].shape[0]):
        d = mb.decode_motion(g["pred_global_motions"][i], int(g["pred_num_tokens"][i]), trim=True)
        pool.append((d["pos"], d["rot"]))
    keep_feeds, keep_t, kinds = [], [], []
    for r in range(rounds + 1):
        reqs, kind = [], []
        n_req = per_round if r > 0 else 2000
        while len(reqs) < n_req:
            u = rng.random()
            a = pool[rng.integers(len(pool))]
            Fa = a[0].shape[0]
            nt = int(rng.integers(1, 5))
            allowed = pick_allowed(rng)
            has_t = True
            if u < 0.35 and Fa >= 20:        # in-clip: context and target from the same generated clip
                gap = int(rng.integers(8, max(9, Fa - 8)))
                t0 = int(rng.integers(0, max(1, Fa - gap - 8)))
                cp, cr = a[0][t0:t0 + 4], a[1][t0:t0 + 4]
                tp, tr = a[0][t0 + 3 + gap:t0 + 7 + gap], a[1][t0 + 3 + gap:t0 + 7 + gap]
                if tp.shape[0] < 4:
                    continue
                k = "in_clip"
            else:
                b = pool[rng.integers(len(pool))]
                ta = int(rng.integers(0, Fa - 4))
                tb = int(rng.integers(0, b[0].shape[0] - 4))
                cp, cr = a[0][ta:ta + 4], a[1][ta:ta + 4]
                tp, tr = b[0][tb:tb + 4], b[1][tb:tb + 4]
                v = rng.random()
                if v < 0.6:
                    dist, yaw, k = rng.uniform(0, 3), rng.uniform(-np.pi / 2, np.pi / 2), "cross"
                elif v < 0.75:
                    dist, yaw, k = rng.uniform(0, 0.1), rng.uniform(-0.2, 0.2), "stationary"
                elif v < 0.9:
                    dist, yaw, k = rng.uniform(0.5, 2.5), rng.uniform(-np.pi, np.pi), "turn"
                else:
                    dist, yaw, k = rng.uniform(4, 9), rng.uniform(-np.pi / 2, np.pi / 2), "far"
                tp, tr = place(tp, tr, mb.alignment_of(cp[3]), dist, rng.uniform(-1, 1), yaw)
                if rng.random() < 0.04:
                    has_t, k = False, "no_target"
            tp, tr = tp.copy(), tr.copy()
            tp[nt:], tr[nt:] = tp[nt - 1], tr[nt - 1]
            inp, align = mb.encode_request(np.concatenate([cp, tp]), np.concatenate([cr, tr]), nt, allowed,
                                           has_targets=has_t)
            inp["_align"] = align
            reqs.append(inp)
            kind.append(k)
        aligns = [q.pop("_align") for q in reqs]
        feeds, tg, mot = run_requests(model, reqs, dev)
        new = [decode_world(mot[i], tg["L"][i], aligns[i]) for i in range(len(reqs))]
        if r == 0:
            pool = new
        else:
            keep_feeds.append(feeds)
            keep_t.append(tg)
            kinds += kind
            pool = pool[-20000:] + new
        print(f"round {r}: {len(reqs)} requests, pool {len(pool)}", flush=True)
    feeds = {k: np.concatenate([f[k] for f in keep_feeds]) for k in NAMES}
    tg = {k: np.concatenate([t[k] for t in keep_t]) for k in ("tokens", "groot", "L")}
    return feeds, tg, np.array(kinds)


def save(path, feeds, tg, **extra):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **feeds, **tg, **extra)


def load(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}
