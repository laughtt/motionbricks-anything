"""Step 3a: GPTQ INT4 for the weights of the root net and the pose net (per-output-channel symmetric, no groups).

Format: one scale per output channel, weights only. The result is a file of integer levels + scales (q_gptq.npz).
export_compact.py then writes them as real INT4 initializers + DequantizeLinear nodes.

Commands (python -I creatures/distill/gptq.py <command> --help):
  calib  --fp32-onnx M.onnx --cache DIR     Hessians H = E[x x^T] of every weight input on the golden requests
                                            (even requests = calibration, odd = held-out error report)
  quant  --fp32-onnx M.onnx --cache DIR     GPTQ (or rtn / clip) on every INT4 weight -> DIR/q_<method>.npz, DIR/stats_<method>.json

Layer objective: output MSE of a layer = mean over calibration rows of |(W-Q) x|^2 = tr(dW H dW^T), H = E[x x^T].
Which weights are INT4: every transformer layer (blocks pose.Lxx, root.shared.Lx, root.token.Lx). The rest
(input / output projections, conv head) stays INT8 (round-to-nearest, made in export_compact.py).
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "8")

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I drops the script folder
import mb_common as mb  # noqa: E402

THREADS = int(os.environ.get("OMP_NUM_THREADS", "8"))
CLIP_GRID = np.round(np.arange(0.80, 1.0001, 0.02), 3)
QMAX = 7


def is_int4_block(block: str) -> bool:
    """INT4 = every transformer layer. Blocks: pose.L00.., root.shared.L0.., root.token.L0.. (INT8: *.io, root.conv_head)."""
    return ".L" in block


def sess_opts():
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = THREADS
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    return so


# ----------------------------------------------------------------------------- graph specs

def _attrs(n):
    return {a.name: (a.i if a.type == 2 else list(a.ints) if a.type == 7 else a.f) for a in n.attribute}


def build_specs(model):
    """One spec per root/pose weight tensor: layout, block, tap tensor (the activation that enters the weight)."""
    from onnx_utils import block_of, weight_initializers
    specs = []
    for t, n, grp, slot in weight_initializers(model):
        if grp not in ("pose", "root"):
            continue
        assert not (n.op_type == "MatMul" and slot != 1), n.name
        sp = dict(name=t.name, node=n.name, op=n.op_type, block=block_of(n.name, grp), shape=list(t.dims))
        if n.op_type == "MatMul":
            sp["layout"], sp["tap"] = "mm", n.input[0]
        elif n.op_type == "Gemm":
            a = _attrs(n)
            sp["layout"], sp["tap"] = ("gemmT" if a.get("transB", 0) else "gemmN"), n.input[0]
        else:
            a = _attrs(n)
            assert a["group"] == 1 and a["strides"] == [1] and a["pads"][0] == a["pads"][1]
            sp["layout"], sp["tap"] = "conv", n.input[0]
            sp["conv"] = (a["kernel_shape"][0], a["dilations"][0], a["pads"][0])
        specs.append(sp)
    return specs


def to_w2(w, sp):
    """Weight -> [out, in*] float64."""
    w = w.astype(np.float64)
    if sp["layout"] in ("mm", "gemmN"):
        return w.T
    if sp["layout"] == "gemmT":
        return w
    return w.reshape(w.shape[0], -1)


def from_w2(w2, sp):
    if sp["layout"] in ("mm", "gemmN"):
        return np.ascontiguousarray(w2.T).astype(np.float32)
    if sp["layout"] == "gemmT":
        return w2.astype(np.float32)
    return w2.reshape(sp["shape"]).astype(np.float32)


def unfold(x, k, dil, pad):
    C, T = x.shape
    xp = np.pad(x, ((0, 0), (pad, pad)))
    P = np.stack([xp[:, j * dil: j * dil + T] for j in range(k)], axis=1)      # [C,k,T]
    return P.reshape(C * k, T).T


def tapkey(sp):
    return sp["tap"] + ("" if sp["layout"] != "conv" else "|%d,%d,%d" % tuple(sp["conv"]))


# ----------------------------------------------------------------------------- calibration

def load_specs(fp32_onnx):
    import onnx
    from onnx_utils import add_token_outputs
    model = onnx.load(str(fp32_onnx))
    add_token_outputs(model)
    return model, build_specs(model)


def cmd_calib(a):
    import onnx
    import onnxruntime as ort
    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    model, specs = load_specs(a.fp32_onnx)
    taps = sorted({s["tap"] for s in specs})
    existing = {o.name for o in model.graph.output}
    for t in taps:
        if t not in existing:
            model.graph.output.append(onnx.helper.make_empty_tensor_value_info(t))
    sess = ort.InferenceSession(model.SerializeToString(), sess_opts(), providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    golden = mb.load_golden()
    N = golden["pred_num_tokens"].shape[0]
    keys = {}
    for s in specs:
        keys.setdefault(tapkey(s), s)
    acc = {}
    t0 = time.time()
    for i in range(N):
        out = dict(zip(names, sess.run(None, {k: golden[k][i:i + 1] for k in mb.INPUT_NAMES})))
        part = "c" if i % 2 == 0 else "t"
        for key, s in keys.items():
            x = out[s["tap"]].astype(np.float64)
            x2 = unfold(x[0], *s["conv"]) if s["layout"] == "conv" else x.reshape(-1, x.shape[-1])
            d = acc.setdefault(key, {})
            if "H" + part not in d:
                d["H" + part] = np.zeros((x2.shape[1],) * 2)
                d["n" + part] = 0
            d["H" + part] += x2.T @ x2
            d["n" + part] += x2.shape[0]
        if i % 40 == 0:
            print(f"calib {i}/{N} [{time.time() - t0:.0f}s]", flush=True)
    arrs = {}
    for j, (key, d) in enumerate(acc.items()):
        for p in "ct":
            arrs[f"H{p}_{j}"] = (d["H" + p] / d["n" + p]).astype(np.float32)
    np.savez(cache / "calib.npz", **arrs)
    (cache / "calib_keys.json").write_text(json.dumps(list(acc.keys())), encoding="utf-8")
    print("calib done", len(acc), "taps", f"{time.time() - t0:.0f}s")


def load_calib(cache):
    z = np.load(Path(cache) / "calib.npz")
    keys = json.loads((Path(cache) / "calib_keys.json").read_text(encoding="utf-8"))
    return {k: dict(Hc=z[f"Hc_{j}"].astype(np.float64), Ht=z[f"Ht_{j}"].astype(np.float64)) for j, k in enumerate(keys)}


# ----------------------------------------------------------------------------- quantisers (all return int levels + per-row scale)

def rtn_levels(W, f):
    s = np.abs(W).max(1) * f / QMAX
    s[s == 0] = 1.0
    return np.clip(np.round(W / s[:, None]), -QMAX - 1, QMAX), s


def row_err(W, q, s, H):
    E = W - q * s[:, None]
    return ((E @ H) * E).sum(1)


def clip_search(W, H, use=True):
    """Per-row clip factor in CLIP_GRID that minimises the row's output MSE under H."""
    if not use:
        q, s = rtn_levels(W, 1.0)
        return q, s, np.ones(W.shape[0])
    best, bf = None, None
    for f in CLIP_GRID:
        q, s = rtn_levels(W, f)
        e = row_err(W, q, s, H)
        if best is None:
            best, bf = e, np.full(W.shape[0], f)
        else:
            m = e < best
            best = np.where(m, e, best)
            bf = np.where(m, f, bf)
    q, s = rtn_levels(W, bf)
    return q, s, bf


def gptq_levels(W, H, f, bs=128, damp=0.01):
    """GPTQ column-by-column error compensation with fixed per-row scales s = max|w| * f / 7."""
    n = W.shape[1]
    s = np.abs(W).max(1) * f / QMAX
    s[s == 0] = 1.0
    H = H.copy()
    dead = np.diag(H) <= 0
    H[dead, dead] = 1.0
    H[np.diag_indices(n)] += damp * np.mean(np.diag(H))
    Hinv = np.linalg.cholesky(np.linalg.inv(H)).T                # upper
    W = W.copy()
    Q = np.zeros_like(W)
    for i1 in range(0, n, bs):
        i2 = min(i1 + bs, n)
        W1 = W[:, i1:i2].copy()
        E1 = np.zeros_like(W1)
        Hi = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            w = W1[:, i]
            q = np.clip(np.round(w / s), -QMAX - 1, QMAX)
            Q[:, i1 + i] = q
            err = (w - q * s) / Hi[i, i]
            W1[:, i:] -= err[:, None] * Hi[i, i:][None, :]
            E1[:, i] = err
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return Q, s


def quantise_tensor(method, W, cal):
    """Returns dict(q int8 levels [out,in*], s scales [out], stats). W [out,in*] float64."""
    Hc, Ht = cal["Hc"], cal["Ht"]
    rq, rs = rtn_levels(W, 1.0)
    stats = dict(rtn_cal=float(row_err(W, rq, rs, Hc).sum()), rtn_test=float(row_err(W, rq, rs, Ht).sum()),
                 energy_cal=float(((W @ Hc) * W).sum()), energy_test=float(((W @ Ht) * W).sum()))
    if method == "rtn":
        q, s = rq, rs
    elif method in ("clip", "gptq"):
        q, s, f = clip_search(W, Hc)
        if method == "gptq":
            best = None
            for ff in (np.ones(W.shape[0]), f):          # try scales without and with the clip search, keep the better
                gq, gs = gptq_levels(W, Hc, ff)
                e = row_err(W, gq, gs, Hc).sum()
                if best is None or e < best[0]:
                    best = (e, gq, gs)
            _, q, s = best
    else:
        raise ValueError(method)
    stats["m_cal"] = float(row_err(W, q, s, Hc).sum())
    stats["m_test"] = float(row_err(W, q, s, Ht).sum())
    return dict(q=q.astype(np.int8), s=s, stats=stats)


def cmd_quant(a):
    from onnx import numpy_helper
    model, specs = load_specs(a.fp32_onnx)
    inits = {t.name: t for t in model.graph.initializer}
    calib = load_calib(a.cache)
    arrs, stats = {}, []
    t0 = time.time()
    todo = [i for i, sp in enumerate(specs) if is_int4_block(sp["block"])]
    for n_done, i in enumerate(todo):
        sp = specs[i]
        W = to_w2(numpy_helper.to_array(inits[sp["name"]]), sp)
        r = quantise_tensor(a.method, W, calib[tapkey(sp)])
        arrs[f"q{i}"], arrs[f"s{i}"] = r["q"], r["s"]
        st = dict(name=sp["name"], node=sp["node"], block=sp["block"], params=int(W.size), **r["stats"])
        stats.append(st)
        print(f"{a.method} {n_done + 1}/{len(todo)} {sp['block']:14s} rel_err cal {st['m_cal'] / st['energy_cal']:.4f} (rtn {st['rtn_cal'] / st['energy_cal']:.4f}) "
              f"test {st['m_test'] / st['energy_test']:.4f} (rtn {st['rtn_test'] / st['energy_test']:.4f}) [{time.time() - t0:.0f}s]", flush=True)
    arrs["names"] = np.array([sp["name"] for sp in specs])      # export_compact.py checks that the tensor order is the same
    Path(a.cache).mkdir(parents=True, exist_ok=True)
    np.savez(Path(a.cache) / f"q_{a.method}.npz", **arrs)
    tot = lambda k: sum(s[k] for s in stats)
    summ = dict(method=a.method, tensors=len(stats), rel_err_cal=tot("m_cal") / tot("energy_cal"), rel_err_test=tot("m_test") / tot("energy_test"),
                rel_err_rtn_cal=tot("rtn_cal") / tot("energy_cal"), rel_err_rtn_test=tot("rtn_test") / tot("energy_test"),
                reduction_cal=1 - tot("m_cal") / tot("rtn_cal"), reduction_test=1 - tot("m_test") / tot("rtn_test"))
    (Path(a.cache) / f"stats_{a.method}.json").write_text(json.dumps(dict(summary=summ, tensors=stats), indent=1), encoding="utf-8")
    print(json.dumps(summ))


def main():
    ap = argparse.ArgumentParser(description="GPTQ INT4 (per-channel) for the root and pose transformer layers of a MotionBricks SOMA ONNX.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("calib", help="Hessians on the golden requests")
    p.add_argument("--fp32-onnx", required=True, help="FP32 ONNX of the model to quantise (for a student: distill.py export)")
    p.add_argument("--cache", required=True, help="cache folder")
    mb.add_path_args(p, fp32=False)
    p = sub.add_parser("quant", help="GPTQ / clip / rtn on the INT4 weights")
    p.add_argument("--fp32-onnx", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--method", default="gptq", choices=("gptq", "clip", "rtn"))
    mb.add_path_args(p, fp32=False)
    a = ap.parse_args()
    mb.configure(work=a.work, motions=a.motions)
    {"calib": cmd_calib, "quant": cmd_quant}[a.cmd](a)


if __name__ == "__main__":
    main()
