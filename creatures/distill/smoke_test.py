"""CPU smoke test: no NVIDIA checkpoint, no GPU, no data. Checks the maths of steps 1, 3 and 4 on tiny random inputs.

    python -I creatures/distill/smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I drops the script folder


def test_layer_pick():
    import torch
    import layer_pick as lp
    rng = np.random.default_rng(0)
    L, B, P, D = 16, 4, 6, 32
    # hidden states: layers 3, 4, 8, 11 barely change the state (small updates), the others rotate it a lot
    small = {3, 4, 8, 11}
    h = [rng.normal(size=(B, P, D))]
    for i in range(L):
        step = 0.02 if i in small else 1.0
        h.append(h[-1] + step * rng.normal(size=(B, P, D)) * np.linalg.norm(h[-1], axis=-1, keepdims=True) / np.sqrt(D))
    H = torch.from_numpy(np.stack(h)).float()
    valid = torch.ones(B, P)
    acc, n = lp.accumulate_angular(H, valid)
    Dm = (acc / n).numpy()
    assert Dm.shape == (L + 1, L + 1) and np.allclose(np.diag(Dm), 0, atol=1e-3)
    keep = lp.pick_keep(Dm, 12)
    dropped = sorted(set(range(L)) - set(keep))
    assert dropped == sorted(small), (dropped, sorted(small))
    keep10 = lp.pick_keep(Dm, 10)
    assert len(keep10) == 10 and {0, 1, 15} <= set(keep10)
    print("layer_pick ok: dropped", dropped, "| keep 10 =", keep10)


def test_gptq():
    import gptq
    rng = np.random.default_rng(1)
    out_f, in_f, n = 48, 64, 2000
    A = rng.normal(size=(in_f, in_f)) * np.linspace(0.2, 2.0, in_f)[None]      # correlated, uneven inputs
    X = rng.normal(size=(n, in_f)) @ A
    W = rng.normal(size=(out_f, in_f)) * 0.1
    H = X.T @ X / n
    cal = dict(Hc=H, Ht=H)
    res = {m: gptq.quantise_tensor(m, W, cal) for m in ("rtn", "clip", "gptq")}
    err = {m: res[m]["stats"]["m_cal"] for m in res}
    for m, r in res.items():
        assert r["q"].min() >= -8 and r["q"].max() <= 7, m
    assert err["gptq"] < err["rtn"] * 0.9, err
    print("gptq ok: output MSE rtn %.3e  clip %.3e  gptq %.3e  (gptq/rtn = %.2f)" % (err["rtn"], err["clip"], err["gptq"], err["gptq"] / err["rtn"]))


def test_int4_pack():
    from onnx import numpy_helper
    import export_compact as ec
    rng = np.random.default_rng(2)
    w = rng.normal(size=(5, 7)).astype(np.float32)
    q, s = ec.quantise(w, 4, 0)
    assert q.min() >= -8 and q.max() <= 7 and s.shape == (5,)
    t = ec.int4_tensor("w", q)
    back = numpy_helper.to_array(t)
    assert np.array_equal(back.astype(np.int8), q), "INT4 packing does not round-trip"
    print("int4 pack ok: shape", back.shape, "dtype", back.dtype)


def test_footlock():
    import footlock
    import mb_common as mb
    F = 60
    pos = np.zeros((F, 30, 3))
    jn = mb.joint_names()
    for name, x0 in (("LeftLeg", -0.1), ("RightLeg", 0.1)):
        pos[:, jn.index(name)] = [x0, 0.9, 0.0]
    for name, x0 in (("LeftShin", -0.1), ("RightShin", 0.1)):
        pos[:, jn.index(name)] = [x0, 0.5, 0.05]
    for name, x0 in (("LeftFoot", -0.1), ("RightFoot", 0.1)):
        pos[:, jn.index(name)] = [x0, 0.1, 0.0]
    for name, x0 in (("LeftToeBase", -0.1), ("RightToeBase", 0.1)):
        pos[:, jn.index(name)] = [x0, 0.0, 0.1]
    pos[:, jn.index("LeftFoot"), 2] += np.linspace(0, 0.06, F)             # a sliding left foot
    pos[:, jn.index("LeftToeBase"), 2] += np.linspace(0, 0.06, F)
    contacts = np.ones((F, 4))
    new, _ = footlock.foot_lock(pos, contacts)
    a, b = footlock.motion_metrics(pos, contacts), footlock.motion_metrics(new, contacts)
    assert b["skate"] < a["skate"] * 0.2, (a["skate"], b["skate"])
    print("foot-lock ok: skate %.2f -> %.2f cm/s" % (a["skate"], b["skate"]))


if __name__ == "__main__":
    for t in (test_layer_pick, test_gptq, test_int4_pack, test_footlock):
        t()
    print("all smoke tests passed")
