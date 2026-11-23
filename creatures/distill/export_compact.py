"""Step 3b: export the compact quantised model: real INT4 / INT8 initializers + DequantizeLinear, FP16 decoder.

Usage: python -I creatures/distill/export_compact.py --fp32-onnx STUDENT_fp32.onnx --q-npz CACHE/q_gptq.npz --out OUT.onnx [--decoder fp16|fp32]

Input : the FP32 ONNX that NVIDIA's exporter made for the (student) model (distill.py export) and the GPTQ levels
        (gptq.py quant). Same tensor inputs and outputs as the NVIDIA graph (plus pose_tokens / pose_logits / duration_logits,
        which the checks use).
Rules : every transformer layer (blocks pose.Lxx, root.shared.Lx, root.token.Lx) = INT4 from the GPTQ levels;
        the other root / pose weights (input / output projections, conv head) = INT8, per-channel symmetric round-to-nearest;
        decoder weights = FP16 + Cast (the decoder output conv stays FP32). Without --q-npz every INT4 weight uses
        round-to-nearest instead of GPTQ.
Every quantised weight is stored as an INT4 / INT8 initializer + a per-channel FP32 scale + a DequantizeLinear node.
Opset is raised to 21 (needed for INT4 DequantizeLinear): load with ONNX Runtime >= 1.20 and graph optimisation BASIC.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I drops the script folder


def quantise(w, bits: int, axis: int):
    """Per-channel symmetric round-to-nearest. Returns int8 levels (same shape as w) and FP32 scales [out]."""
    qmax = 2 ** (bits - 1) - 1
    m = np.moveaxis(w.astype(np.float64), axis, 0)
    m = m.reshape(m.shape[0], -1)
    s = np.abs(m).max(1, keepdims=True) / qmax
    s[s == 0] = 1.0
    q = np.clip(np.round(m / s), -qmax - 1, qmax).astype(np.int8)
    q = np.moveaxis(q.reshape(np.moveaxis(w, axis, 0).shape), 0, axis)
    return np.ascontiguousarray(q), s.reshape(-1).astype(np.float32)


def int4_tensor(name: str, q: np.ndarray):
    """Pack int8 levels (-8..7) into an ONNX INT4 initializer (two values per byte, low nibble first)."""
    import onnx
    from onnx import TensorProto
    flat = q.reshape(-1).astype(np.int8) & 0x0F
    if flat.size % 2:
        flat = np.append(flat, 0).astype(np.int8)
    packed = (flat[0::2] | (flat[1::2] << 4)).astype(np.uint8)
    t = onnx.TensorProto()
    t.name, t.data_type = name, TensorProto.INT4
    t.dims.extend(q.shape)
    t.raw_data = packed.tobytes()
    return t


def export(fp32_onnx, out, q_npz=None, decoder="fp16"):
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    import gptq
    from onnx_utils import add_token_outputs, block_of, weight_initializers

    model = onnx.load(str(fp32_onnx))
    add_token_outputs(model)
    specs = gptq.build_specs(model)
    gq = None
    if q_npz:
        z = np.load(q_npz)
        assert [str(x) for x in z["names"]] == [sp["name"] for sp in specs], "q-npz was made from another ONNX graph"
        gq = {sp["name"]: (z[f"q{i}"], z[f"s{i}"], sp) for i, sp in enumerate(specs) if f"q{i}" in z.files}
    g = model.graph
    last_dec = [n.name for n in g.node if n.name.startswith("/decoder/") and n.op_type == "Conv"][-1]   # output conv stays FP32
    new_nodes, drop = [], set()
    stats = dict(int4=0, int8=0, fp16=0, int4_params=0, int8_params=0, int4_source="gptq" if gq else "rtn")
    for t, n, grp, slot in list(weight_initializers(model)):
        w = numpy_helper.to_array(t)
        if grp in ("pose", "root"):
            if n.op_type == "MatMul":
                ax = 1 if slot == 1 else 0
            elif n.op_type == "Gemm":
                ax = 0 if next((x.i for x in n.attribute if x.name == "transB"), 0) else 1
            else:
                ax = 0
            bits = 4 if gptq.is_int4_block(block_of(n.name, grp)) else 8
            if gq is not None and bits == 4:
                q0, s, sp = gq[t.name]
                s = s.astype(np.float32)
                q = gptq.from_w2(q0.astype(np.float64), sp).astype(np.int8)
                assert q.shape == w.shape and s.shape[0] == w.shape[ax]
            else:
                q, s = quantise(w, bits, ax)
            qn, sn = t.name + "_q", t.name + "_scale"
            g.initializer.append(int4_tensor(qn, q) if bits == 4 else numpy_helper.from_array(q, qn))
            g.initializer.append(numpy_helper.from_array(s, sn))
            new_nodes.append(helper.make_node("DequantizeLinear", [qn, sn], [t.name], name=t.name + "_dq", axis=ax))
            stats[f"int{bits}"] += 1
            stats[f"int{bits}_params"] += int(w.size)
            drop.add(t.name)
        elif grp == "decoder" and decoder == "fp16" and n.name != last_dec:
            h = t.name + "_fp16"
            g.initializer.append(numpy_helper.from_array(w.astype(np.float16), h))
            new_nodes.append(helper.make_node("Cast", [h], [t.name], name=t.name + "_cast", to=TensorProto.FLOAT))
            stats["fp16"] += 1
            drop.add(t.name)
    keep = [t for t in g.initializer if t.name not in drop]
    del g.initializer[:]
    g.initializer.extend(keep)
    old = list(g.node)
    # opset 18 moved the `axes` attribute of the Reduce* ops (except ReduceSum, already done in 13) to an input
    for n in old:
        if n.op_type in ("ReduceL2", "ReduceMax", "ReduceMin", "ReduceMean", "ReduceProd", "ReduceL1", "ReduceLogSum",
                         "ReduceLogSumExp", "ReduceSumSquare"):
            ax = next((x for x in n.attribute if x.name == "axes"), None)
            if ax is not None:
                an = n.name + "_axes"
                g.initializer.append(numpy_helper.from_array(np.array(ax.ints, np.int64), an))
                n.input.append(an)
                n.attribute.remove(ax)
    del g.node[:]
    g.node.extend(new_nodes + old)        # dequant nodes only read initializers, so they sort first
    for op in model.opset_import:
        if op.domain in ("", "ai.onnx"):
            op.version = 21
    model.ir_version = max(model.ir_version, 10)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out))
    return stats


def main():
    ap = argparse.ArgumentParser(description="Export the compact INT4 / INT8 / FP16-decoder ONNX (see the module doc).")
    ap.add_argument("--fp32-onnx", required=True, help="FP32 ONNX from NVIDIA's exporter (distill.py export for a student)")
    ap.add_argument("--q-npz", default=None, help="GPTQ levels from gptq.py quant (q_gptq.npz); without it INT4 uses round-to-nearest")
    ap.add_argument("--decoder", default="fp16", choices=("fp16", "fp32"))
    ap.add_argument("--out", required=True, help="output ONNX file")
    a = ap.parse_args()
    stats = export(a.fp32_onnx, a.out, a.q_npz, a.decoder)
    print(json.dumps(stats), f"{Path(a.out).stat().st_size / 1e6:.1f} MB", a.out)


if __name__ == "__main__":
    main()
