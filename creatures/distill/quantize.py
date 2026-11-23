"""Step 3: quantise a distilled student: GPTQ INT4 on every transformer layer, INT8 for the input / output projections
and the conv head, FP16 decoder. One command that runs gptq.py calib, gptq.py quant and export_compact.py, then scores the file.

    python -I creatures/distill/quantize.py --name S10 [--work DIR]
    python -I creatures/distill/quantize.py --fp32-onnx student_fp32.onnx --out model_compact.onnx --cache DIR

The input is the FP32 ONNX that 'distill.py export' made (default <work>/runs/<name>/<name>_fp32.onnx).
Outputs: <work>/runs/<name>/q/ (Hessians and GPTQ levels, cached), <work>/runs/<name>/<name>_compact_fp16dec.onnx, qeval.json.
Needs the golden set (build_golden.py). The score (--eval, default on) compares the compact file with the FP32 teacher outputs
on the golden requests, and runs the 7 rollouts with foot-lock. It needs the sample BVH clips (--motions).
ONNX Runtime >= 1.20 is needed to load the INT4 file.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I drops the script folder
import mb_common as mb  # noqa: E402


def qeval(comp, out_json):
    """ORT CPU: golden outputs of the FP32 teacher vs the compact file, then the 7 rollouts (skate with foot-lock, jitter)."""
    import numpy as np
    import footlock
    import quality
    import rollout
    golden = mb.load_golden()
    meta = json.loads((mb.DATA / "golden_meta.json").read_text(encoding="utf-8"))["requests"]
    o = quality.run_variant(comp, golden, threads=8, opt="basic")
    s = quality.score(o, golden, meta)["summary"]
    gen = rollout.Generator(comp)
    clips = {c.stem: mb.Clip(c) for c in mb.MOTIONS.rglob("*.bvh")
             if c.stem in (rollout.WALK, rollout.JOG, rollout.IDLE, rollout.CROUCH, rollout.SIT, rollout.ZOMBIE)}
    sk, skf, jit = [], [], []
    for sc, spec in rollout.SCENARIOS.items():
        fn = rollout.scenario_chain if spec["kind"] == "chain" else rollout.scenario_replan
        rr = fn(gen, clips, spec, tuple(range(8, 15)))
        pos, c = rr["pos"].astype(np.float64), rr["contacts"].astype(np.float64)
        x, y = footlock.motion_metrics(pos, c), footlock.motion_metrics(footlock.foot_lock(pos, c)[0], c)
        sk.append(x["skate"]); skf.append(y["skate"]); jit.append(x["jitter"])
    r = dict(bytes=Path(comp).stat().st_size, token_match=s["token_match"], num_tokens_agree=s["num_tokens_agree"],
             joint_err_mean_cm=s["joint_err_mean_cm"], joint_err_p95_cm=s["joint_err_p95_cm"],
             joint_err_max_cm=s["joint_err_max_cm"], rollout_skate=float(np.mean(sk)),
             rollout_skate_fl=float(np.mean(skf)), rollout_jitter=float(np.mean(jit)))
    print(json.dumps(r), flush=True)
    Path(out_json).write_text(json.dumps(r, indent=1))
    return r


def main():
    ap = argparse.ArgumentParser(description="GPTQ INT4 + INT8 + FP16 decoder export of a distilled student (see the module doc).")
    ap.add_argument("--name", default=None, help="distill run name (input <work>/runs/<name>/<name>_fp32.onnx)")
    ap.add_argument("--fp32-onnx", default=None, help="FP32 ONNX of the model to quantise (instead of --name)")
    ap.add_argument("--out", default=None, help="output ONNX (default <work>/runs/<name>/<name>_compact_fp16dec.onnx)")
    ap.add_argument("--cache", default=None, help="cache folder for Hessians + GPTQ levels (default <work>/runs/<name>/q)")
    ap.add_argument("--method", default="gptq", choices=("gptq", "clip", "rtn"))
    ap.add_argument("--no-eval", action="store_true", help="skip the golden + rollout score")
    ap.add_argument("--work", default=None, help="work folder (env DISTILL_WORK, default ./distill_work)")
    ap.add_argument("--motions", default=None, help="folder with sample BVH clips (env MB_MOTIONS)")
    a = ap.parse_args()
    mb.configure(work=a.work, motions=a.motions)
    if not a.name and not (a.fp32_onnx and a.out and a.cache):
        ap.error("give --name, or all of --fp32-onnx, --out and --cache")
    d = mb.RUNS / (a.name or "")
    src = Path(a.fp32_onnx) if a.fp32_onnx else d / f"{a.name}_fp32.onnx"
    out = Path(a.out) if a.out else d / f"{a.name}_compact_fp16dec.onnx"
    cache = Path(a.cache) if a.cache else d / "q"
    if not src.exists():
        raise SystemExit(f"missing {src}: run 'distill.py export --name {a.name}' first")
    import export_compact
    import gptq
    t0 = time.time()
    ns = argparse.Namespace(fp32_onnx=str(src), cache=str(cache), method=a.method)
    if not (cache / "calib.npz").exists():
        gptq.cmd_calib(ns)
    if not (cache / f"q_{a.method}.npz").exists():
        gptq.cmd_quant(ns)
    stats = export_compact.export(src, out, cache / f"q_{a.method}.npz", "fp16")
    print(json.dumps(stats))
    print(f"quantised in {time.time() - t0:.0f}s -> {out} ({out.stat().st_size / 1e6:.1f} MB)", flush=True)
    if not a.no_eval:
        qeval(out, out.parent / "qeval.json")


if __name__ == "__main__":
    main()
