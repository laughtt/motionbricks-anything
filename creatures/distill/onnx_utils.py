"""ONNX graph helpers shared by the golden builder, the GPTQ script and the compact exporter.

* add_token_outputs / expose_tokens: add the outputs pose_tokens, pose_logits, duration_logits to the NVIDIA SOMA graph
  (the graph itself is not changed, only more tensors are marked as outputs).
* classify_nodes / weight_initializers / block_of: find every weight of the root net and the pose net and name its block
  (pose.L00 ..., root.shared.L0 ..., root.token.L0 ..., root.conv_head, pose.io, root.io).
* fake_quant: symmetric round-to-nearest quantise + dequantise (used by tests).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

POSE_ARGMAX = "/ArgMax"            # pose tokens  [N,16,5]  (5 heads x 16 codes)
DURATION_ARGMAX = "/mmm_net/ArgMax"  # token-count choice


def _node(model, name):
    for n in model.graph.node:
        if n.name == name:
            return n
    raise KeyError(name)


def add_token_outputs(model: onnx.ModelProto) -> onnx.ModelProto:
    """Add pose_tokens (int32), pose_logits (float32) and duration_logits (float32) outputs."""
    g = model.graph
    if any(o.name == "pose_tokens" for o in g.output):
        return model
    pa, da = _node(model, POSE_ARGMAX), _node(model, DURATION_ARGMAX)
    g.node.extend([
        helper.make_node("Cast", [pa.output[0]], ["pose_tokens"], name="distill/pose_tokens", to=TensorProto.INT32),
        helper.make_node("Cast", [pa.input[0]], ["pose_logits"], name="distill/pose_logits", to=TensorProto.FLOAT),
        helper.make_node("Cast", [da.input[0]], ["duration_logits"], name="distill/duration_logits", to=TensorProto.FLOAT),
    ])
    g.output.extend([
        helper.make_tensor_value_info("pose_tokens", TensorProto.INT32, None),
        helper.make_tensor_value_info("pose_logits", TensorProto.FLOAT, None),
        helper.make_tensor_value_info("duration_logits", TensorProto.FLOAT, None),
    ])
    return model


def expose_tokens(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(add_token_outputs(onnx.load(str(src))), str(dst))


def weight_initializers(model: onnx.ModelProto, ops=("MatMul", "Gemm", "Conv")):
    """Yield (initializer, consumer node, group) for every >=2-D float weight of the given ops."""
    nodes = list(model.graph.node)
    groups = classify_nodes(nodes)
    inits = {t.name: t for t in model.graph.initializer}
    seen = set()
    for n, grp in zip(nodes, groups):
        if n.op_type not in ops:
            continue
        for slot, name in enumerate(n.input):
            t = inits.get(name)
            if t is None or name in seen or len(t.dims) < 2 or t.data_type != TensorProto.FLOAT:
                continue
            if n.op_type == "Conv" and slot != 1:
                continue
            seen.add(name)
            yield t, n, grp, slot


def classify_nodes(nodes):
    """Sub-network of every node: root / pose / decoder / glue.

    The root and pose predictors both export under '/mmm_net/'. Nodes are in execution
    order and the root predictor runs first, so '/mmm_net/' nodes up to the last node
    of the root-only blocks belong to the root predictor.
    """
    root_marks = ("_shared_transformer_model", "_root_token_transformer_model", "_conv_output")
    last_root = max((i for i, n in enumerate(nodes) if any(m in n.name for m in root_marks)), default=-1)
    out = []
    for i, n in enumerate(nodes):
        name = n.name
        if name.startswith("/decoder/"):
            out.append("decoder")
        elif name.startswith("/mmm_net"):
            if any(m in name for m in root_marks):
                out.append("root")
            elif "/_transformer_model/" in name or name.startswith("/mmm_net_1"):
                out.append("pose")
            else:
                out.append("root" if i <= last_root else "pose")
        else:
            out.append("glue")
    return out


def block_of(node_name: str, grp: str) -> str:
    m = re.search(r"/_transformer_model/layers\.(\d+)/", node_name)
    if m:
        return f"pose.L{int(m.group(1)):02d}"
    m = re.search(r"/_shared_transformer_model/layers\.(\d+)/", node_name)
    if m:
        return f"root.shared.L{m.group(1)}"
    m = re.search(r"/_root_token_transformer_model/layers\.(\d+)/", node_name)
    if m:
        return f"root.token.L{m.group(1)}"
    if "/_conv_output/" in node_name:
        return "root.conv_head"
    return f"{grp}.io"          # input / output projections and embeddings


# ----------------------------------------------------------------------------- fake-quant (quality proxies)

def fake_quant(w: np.ndarray, bits: int, mode: str, out_axis: int, group: int = 32, in_axis: int | None = None):
    """Symmetric round-to-nearest quantise + dequantise. Returns float32 with the same shape.

    mode: 'tensor' (one scale), 'channel' (one scale per output channel, what QNN HTP offers),
          'group' (one scale per `group` input elements of each output channel, what ORT MatMulNBits
          / llama.cpp-style kernels use; NOT available on mobile NPUs).
    """
    qmax = 2 ** (bits - 1) - 1
    w64 = w.astype(np.float64)
    if mode == "tensor":
        s = np.abs(w64).max() / qmax
        s = s if s > 0 else 1.0
        return (np.clip(np.round(w64 / s), -qmax - 1, qmax) * s).astype(np.float32)
    m = np.moveaxis(w64, out_axis, 0)
    shp = m.shape
    m = m.reshape(shp[0], -1)
    if mode == "channel":
        s = np.abs(m).max(1, keepdims=True) / qmax
        s[s == 0] = 1.0
        q = np.clip(np.round(m / s), -qmax - 1, qmax) * s
    elif mode == "outlier":
        # per-channel INT<bits> for the bulk, the largest 1% of each channel kept at per-channel INT8
        # (PowerInfer-2 style outlier handling; costs ~1% extra INT8 values + an index).
        thr = np.quantile(np.abs(m), 0.99, axis=1, keepdims=True)
        thr[thr == 0] = 1.0
        s = thr / qmax
        q = np.clip(np.round(m / s), -qmax - 1, qmax) * s
        s8 = np.abs(m).max(1, keepdims=True) / 127.0
        s8[s8 == 0] = 1.0
        q8 = np.clip(np.round(m / s8), -128, 127) * s8
        q = np.where(np.abs(m) > thr, q8, q)
    elif mode == "group":
        k = m.shape[1]
        pad = (-k) % group
        mp = np.pad(m, ((0, 0), (0, pad))).reshape(shp[0], -1, group)
        s = np.abs(mp).max(2, keepdims=True) / qmax
        s[s == 0] = 1.0
        q = (np.clip(np.round(mp / s), -qmax - 1, qmax) * s).reshape(shp[0], -1)[:, :k]
    else:
        raise ValueError(mode)
    return np.moveaxis(q.reshape(shp), 0, out_axis).astype(np.float32)
