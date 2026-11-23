"""Build the golden SOMA request set and run the FP32 ONNX Runtime CPU reference.

Usage: python -I creatures/distill/build_golden.py [--work DIR] [--fp32-onnx TEACHER.onnx] [--motions BVH_DIR]
       -> <work>/data/golden_soma.npz + golden_meta.json (244 requests, fixed seed)

Requests are formed from the repo's sample BVH clips the way the inbetweener does:
four consecutive context poses (30 Hz) plus one to four target poses, encoded with a
numpy port of the SOMA engine's input kernels (see mb_common.encode_request).
"""
from __future__ import annotations

import json
import time

import numpy as np

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I drops the script folder
import mb_common as mb
from onnx_utils import expose_tokens

SEED = 20261114
DEFAULT = tuple(range(8, 15))
ALL = tuple(range(6, 17))


def window(clip, t):
    return clip.pos[t:t + 4].copy(), clip.rot[t:t + 4].copy()


def place(pos, rot, ctx_align, dist, bearing, yaw):
    """Re-place a target window: its first pose goes `dist` m from the context root at
    `bearing` (rad, 0 = context facing) and faces `yaw` (rad) relative to the context facing."""
    o_t, R_t = mb.alignment_of(pos[0])
    lp = (pos - o_t) @ R_t                              # target-local, facing +X at origin
    lr = np.einsum("ij,fkjl->fkil", R_t.T, rot)
    o_c, R_c = ctx_align
    Ry = mb._rot_y(-yaw)                                # +yaw turns left (towards -Z... sign is cosmetic)
    Rw = R_c @ Ry
    offset = R_c @ np.array([dist * np.cos(bearing), 0.0, dist * np.sin(bearing)])
    return lp @ Rw.T + o_c + offset, np.einsum("ij,fkjl->fkil", Rw, lr)


def pad_targets(pos, rot, nt):
    pos, rot = pos.copy(), rot.copy()
    pos[nt:], rot[nt:] = pos[nt - 1], rot[nt - 1]
    return pos, rot


def main():
    ap = argparse.ArgumentParser(description="Build the golden request set (244 requests) and run the FP32 ONNX teacher on it.")
    mb.add_path_args(ap)
    mb.apply_path_args(ap.parse_args())
    rng = np.random.default_rng(SEED)
    paths = sorted(mb.MOTIONS.rglob("*.bvh"))
    t0 = time.time()
    clips = [mb.Clip(p) for p in paths]
    print(f"loaded {len(clips)} clips in {time.time() - t0:.1f}s; frames: {sum(c.num_frames for c in clips)}")
    by_name = {c.name: c for c in clips}
    idle = [c for c in clips if "idle" in c.name or "relax" in c.name or "looking" in c.name]
    reqs, meta = [], []

    def add(cat, ca, ta, cp, cr, tp, tr, nt, allowed, has_targets=True, **extra):
        tp, tr = pad_targets(tp, tr, nt)
        inputs, align = mb.encode_request(np.concatenate([cp, tp]), np.concatenate([cr, tr]), nt, allowed,
                                          has_targets=has_targets)
        reqs.append(inputs)
        meta.append(dict(id=len(meta), category=cat, context_clip=ca.name, context_frame=int(ta),
                         num_targets=int(nt) if has_targets else 0, allowed=list(map(int, allowed)),
                         align_origin=align[0].tolist(), align_R=align[1].tolist(), **extra))

    def pick_allowed():
        u = rng.random()
        if u < 0.6:
            return DEFAULT
        if u < 0.8:
            return ALL
        return (int(rng.integers(6, 17)),)

    def in_clip(cat, nt=None, allowed=None, clip_pool=None, gap=None):
        while True:
            c = (clip_pool or clips)[rng.integers(len(clip_pool or clips))]
            g = int(rng.integers(12, 61)) if gap is None else gap
            if c.num_frames > g + 12:
                break
        t = int(rng.integers(0, c.num_frames - g - 8))
        cp, cr = window(c, t)
        tp, tr = window(c, t + 3 + g)
        add(cat, c, t, cp, cr, tp, tr, nt or int(rng.integers(1, 5)), allowed or pick_allowed(),
            target_clip=c.name, target_frame=t + 3 + g, gap_frames=g)

    def cross(cat, dist_range, yaw_range, bearing_range=(-1.0, 1.0), nt=None, allowed=None, pool_a=None, pool_b=None):
        a = (pool_a or clips)[rng.integers(len(pool_a or clips))]
        b = (pool_b or clips)[rng.integers(len(pool_b or clips))]
        ta = int(rng.integers(0, a.num_frames - 4))
        tb = int(rng.integers(0, b.num_frames - 4))
        cp, cr = window(a, ta)
        tp, tr = window(b, tb)
        dist = float(rng.uniform(*dist_range)); yaw = float(rng.uniform(*yaw_range)); bearing = float(rng.uniform(*bearing_range))
        tp, tr = place(tp, tr, mb.alignment_of(cp[3]), dist, bearing, yaw)
        add(cat, a, ta, cp, cr, tp, tr, nt or int(rng.integers(1, 5)), allowed or pick_allowed(),
            target_clip=b.name, target_frame=tb, dist_m=dist, yaw_rad=yaw, bearing_rad=bearing)

    for _ in range(90):
        in_clip("in_clip")
    for _ in range(70):
        cross("cross_clip", (0.0, 3.0), (-np.pi / 2, np.pi / 2))
    for _ in range(12):   # out of the engine's contract (it always gives >= 1 target); the ONNX graph accepts it
        c = clips[rng.integers(len(clips))]
        t = int(rng.integers(0, c.num_frames - 4))
        cp, cr = window(c, t)
        add("edge_no_target", c, t, cp, cr, cp.copy(), cr.copy(), 1, DEFAULT, has_targets=False, target_clip="", target_frame=-1)
    for _ in range(12):
        in_clip("edge_four_targets", nt=4, allowed=DEFAULT)
    for _ in range(12):
        in_clip("edge_one_target", nt=1, allowed=DEFAULT)
    for _ in range(16):
        cross("edge_far_target", (5.0, 12.0), (-np.pi / 2, np.pi / 2), nt=int(rng.integers(1, 5)), allowed=ALL)
    for _ in range(8):
        in_clip("edge_fixed_6_tokens", allowed=(6,), gap=18)
    for _ in range(8):
        in_clip("edge_fixed_16_tokens", allowed=(16,), gap=58)
    for _ in range(8):
        cross("edge_stationary", (0.0, 0.05), (-0.1, 0.1), pool_a=idle, pool_b=idle, allowed=DEFAULT)
    for _ in range(8):
        cross("edge_turn_180", (0.5, 2.0), (np.pi * 0.9, np.pi * 1.1), bearing_range=(np.pi * 0.8, np.pi * 1.2), allowed=DEFAULT)

    N = len(reqs)
    batch = {k: np.concatenate([r[k] for r in reqs], 0) for k in mb.INPUT_NAMES}
    print("requests:", N, {c: sum(m["category"] == c for m in meta) for c in dict.fromkeys(m["category"] for m in meta)})

    # ---- reference: the unmodified FP32 model, ONNX Runtime CPU EP, one request per call
    ref = mb.make_session(mb.FP32_ONNX)
    motions = np.zeros((N, 64, 366), np.float32)
    ntok = np.zeros(N, np.int32)
    for i in range(N):
        m, n = ref.run(None, {k: v[i:i + 1] for k, v in batch.items()})
        motions[i], ntok[i] = m[0], n[0]
    del ref
    assert np.isfinite(motions).all()
    print("pred_num_tokens histogram:", dict(zip(*np.unique(ntok, return_counts=True))))

    # ---- tokens/logits from a copy of the same graph with extra outputs; outputs must match the original
    exposed = mb.MODELS / "fp32_tokens.onnx"
    if not exposed.exists():
        expose_tokens(mb.FP32_ONNX, exposed)
    s = mb.make_session(exposed)
    names = [o.name for o in s.get_outputs()]
    tokens = np.zeros((N, 16, 5), np.int16)
    logits = np.zeros((N, 16, 5, 16), np.float32)
    dur_logits = np.zeros((N, 11), np.float32)
    worst = 0.0
    for i in range(N):
        out = dict(zip(names, s.run(None, {k: v[i:i + 1] for k, v in batch.items()})))
        tokens[i] = out["pose_tokens"][0]
        logits[i] = out["pose_logits"][0].reshape(16, 5, 16)
        dur_logits[i] = out["duration_logits"][0].reshape(-1)[:11]
        worst = max(worst, float(np.abs(out["pred_global_motions"][0] - motions[i]).max()))
        assert out["pred_num_tokens"][0] == ntok[i]
    print("exposed-graph vs original max abs diff:", worst)
    assert worst == 0.0, "token-exposed graph must reproduce the reference bit for bit"
    assert (tokens == logits.argmax(-1)).all()

    # ---- batch invariance of the reference (N=8 calls vs N=1 calls)
    bworst, btok = 0.0, 0
    s8 = mb.make_session(mb.FP32_ONNX)
    for i in range(0, N - N % 8, 8):
        m, n = s8.run(None, {k: v[i:i + 8] for k, v in batch.items()})
        for j in range(8):
            fr = int(ntok[i + j]) * 4
            bworst = max(bworst, float(np.abs(m[j, :fr] - motions[i + j, :fr]).max()))
            btok += int(n[j] != ntok[i + j])
    print(f"N=8 vs N=1: max abs diff on valid frames {bworst:.3e}, pred_num_tokens mismatches {btok}")

    mb.DATA.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(mb.DATA / "golden_soma.npz", **batch, pred_global_motions=motions, pred_num_tokens=ntok,
                        pose_tokens=tokens, pose_logits=logits.astype(np.float16), duration_logits=dur_logits)
    (mb.DATA / "golden_meta.json").write_text(json.dumps(
        dict(seed=SEED, reference=mb.FP32_ONNX.name + ", onnxruntime CPUExecutionProvider, N=1 per call",
             batch8_max_abs_diff=bworst, batch8_num_token_mismatches=btok, requests=meta), indent=0), encoding="utf-8")
    print("saved", mb.DATA / "golden_soma.npz")


if __name__ == "__main__":
    main()
