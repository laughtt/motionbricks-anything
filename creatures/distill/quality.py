"""Run a model variant on the golden requests and score it against the FP32 reference.

Usage: python -I creatures/distill/quality.py MODEL.onnx [MODEL.onnx ...] [--batch N] [--provider cpu|cuda|trt] [--work DIR]
       python -I creatures/distill/quality.py fp32_original      (the teacher ONNX, --fp32-onnx or MB_FP32_ONNX)
       python -I creatures/distill/quality.py --npz out_x.npz --name x      (score saved outputs)

Needs the golden set from build_golden.py. Writes <work>/results/out_<name>.npz and quality_<name>.json.
All distances are in the model's own space (metres -> reported in cm).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I drops the script folder
import mb_common as mb

JN = mb.joint_names()
FEET = [JN.index(n) for n in mb.FOOT_JOINTS]


def run_variant(path, golden, batch=1, provider="cpu", threads=0, opt="all"):
    s = mb.make_session(path, threads=threads, provider=provider, opt=opt)
    names = [o.name for o in s.get_outputs()]
    N = golden["pred_num_tokens"].shape[0]
    out = {"pred_global_motions": np.zeros((N, 64, 366), np.float32), "pred_num_tokens": np.zeros(N, np.int32)}
    if "pose_tokens" in names:
        out["pose_tokens"] = np.zeros((N, 16, 5), np.int16)
        out["pose_logits"] = np.zeros((N, 16, 5, 16), np.float32)
    t0 = time.perf_counter()
    for i in range(0, N, batch):
        sl = slice(i, min(i + batch, N))
        feeds = {k: golden[k][sl] for k in mb.INPUT_NAMES}
        if sl.stop - sl.start < batch:      # static-batch models: pad the last call
            pad = batch - (sl.stop - sl.start)
            feeds = {k: np.concatenate([v, np.repeat(v[-1:], pad, 0)]) for k, v in feeds.items()}
        res = dict(zip(names, s.run(None, feeds)))
        k = sl.stop - sl.start
        out["pred_global_motions"][sl] = res["pred_global_motions"][:k]
        out["pred_num_tokens"][sl] = res["pred_num_tokens"][:k]
        if "pose_tokens" in res:
            out["pose_tokens"][sl] = res["pose_tokens"][:k]
            out["pose_logits"][sl] = res["pose_logits"][:k].reshape(k, 16, 5, 16)
    out["seconds"] = time.perf_counter() - t0
    out["providers"] = s.get_providers()
    return out


def motion_stats(feat, ntok):
    """Per-request intrinsic motion quality on the valid (trimmed) frames."""
    d = mb.decode_motion(feat, ntok, trim=True)
    pos, c = d["pos"], d["contacts"]
    vel = np.diff(pos, axis=0) * mb.FPS                                    # m/s  [F-1,30,3]
    foot_speed = np.linalg.norm(vel[:, FEET][..., [0, 2]], axis=-1)        # horizontal, [F-1,4]
    planted = (c[:-1] > 0.5) & (c[1:] > 0.5)
    skate = float(foot_speed[planted].mean()) if planted.any() else 0.0
    acc = np.diff(pos, n=2, axis=0) * mb.FPS ** 2                          # m/s^2
    jitter = float(np.linalg.norm(acc, axis=-1).mean())
    jerk = np.diff(pos, n=3, axis=0) * mb.FPS ** 3
    return dict(skate=skate, planted_frac=float(planted.mean()), jitter=jitter,
                jerk=float(np.linalg.norm(jerk, axis=-1).mean()), decoded=d)


def endpoint_errors(feat, ntok, golden, i):
    """Raw (pre-correction) distance between the generated endpoints and the given keyframes, cm.

    Uses the model's own joint-position features, so no FK approximation is involved.
    start: raw frame 3 vs last context pose.  end: raw frame 4*tokens-4+k vs target pose k.
    """
    n = int(ntok) * 4
    raw = mb.decode_motion(feat, ntok, trim=False)["raw_pos"]
    g, lp = golden["global_root_values"][i], golden["local_poses"][i]
    kp = np.zeros((8, 30, 3))
    kp[:, 0] = g[:, :3]
    kp[:, 1:] = lp[:, :87].reshape(8, 29, 3) + (g[:, :3] * np.array([1.0, 0.0, 1.0]))[:, None]
    start = float(np.linalg.norm(raw[3] - kp[3], axis=-1).mean() * 100)
    given = golden["has_local_poses"][i, 4:].astype(bool)
    if given.any():
        k = np.nonzero(given)[0]
        end = float(np.linalg.norm(raw[n - 4 + k] - kp[4 + k], axis=-1).mean() * 100)
        end_root = float(np.linalg.norm(raw[n - 4, 0] - kp[4, 0]) * 100)
    else:
        end, end_root = float("nan"), float("nan")
    return start, end, end_root


def score(out, golden, meta):
    N = golden["pred_num_tokens"].shape[0]
    ref_m, ref_n = golden["pred_global_motions"], golden["pred_num_tokens"]
    var_m, var_n = out["pred_global_motions"], out["pred_num_tokens"]
    rows = []
    for i in range(N):
        r = dict(id=i, category=meta[i]["category"], ref_tokens=int(ref_n[i]), var_tokens=int(var_n[i]))
        finite = bool(np.isfinite(var_m[i, : int(var_n[i]) * 4]).all()) and mb.MIN_TOK <= int(var_n[i]) <= mb.MAX_TOK
        r["finite"] = finite
        if not finite:
            rows.append(r)
            continue
        rs, vs = motion_stats(ref_m[i], ref_n[i]), motion_stats(var_m[i], var_n[i])
        rd, vd = rs.pop("decoded"), vs.pop("decoded")
        F = min(rd["pos"].shape[0], vd["pos"].shape[0])       # equal when the token counts agree
        e = np.linalg.norm(vd["pos"][:F] - rd["pos"][:F], axis=-1) * 100        # cm [F,30]
        r["joint_err_mean_cm"], r["joint_err_max_cm"] = float(e.mean()), float(e.max())
        r["root_err_mean_cm"], r["root_err_max_cm"] = float(e[:, 0].mean()), float(e[:, 0].max())
        hr = np.arctan2(ref_m[i, 3:3 + F, 4], ref_m[i, 3:3 + F, 3])
        hv = np.arctan2(var_m[i, 3:3 + F, 4], var_m[i, 3:3 + F, 3])
        dh = np.degrees(np.abs(np.arctan2(np.sin(hv - hr), np.cos(hv - hr))))
        r["heading_err_mean_deg"], r["heading_err_max_deg"] = float(dh.mean()), float(dh.max())
        r["contact_flip_frac"] = float(((vd["contacts"][:F] > 0.5) != (rd["contacts"][:F] > 0.5)).mean())
        for k in ("skate", "jitter", "jerk"):
            r[f"ref_{k}"], r[f"var_{k}"] = rs[k], vs[k]
        r["ref_start_cm"], r["ref_end_cm"], r["ref_end_root_cm"] = endpoint_errors(ref_m[i], ref_n[i], golden, i)
        r["var_start_cm"], r["var_end_cm"], r["var_end_root_cm"] = endpoint_errors(var_m[i], var_n[i], golden, i)
        if "pose_tokens" in out and "pose_tokens" in golden:
            k = min(int(ref_n[i]), int(var_n[i]))
            r["token_match"] = float((out["pose_tokens"][i, :k] == golden["pose_tokens"][i, :k]).mean())
            # generated tokens only (first = context token, last = target token are conditioned)
            r["token_match_inner"] = float((out["pose_tokens"][i, 1:k - 1] == golden["pose_tokens"][i, 1:k - 1]).mean())
        rows.append(r)

    ok = [r for r in rows if r["finite"]]
    same = [r for r in ok if r["ref_tokens"] == r["var_tokens"]]

    def agg(key, rs=None, fn=np.nanmean):
        vals = [r[key] for r in (ok if rs is None else rs) if key in r]
        return float(fn(vals)) if vals else None

    def p95(v):
        return np.nanpercentile(v, 95)

    summary = dict(
        requests=N, non_finite=N - len(ok),
        num_tokens_agree=len(same) / N,
        token_match=agg("token_match", same), token_match_inner=agg("token_match_inner", same),
        token_exact_requests=(float(np.mean([r["token_match"] == 1.0 for r in same if "token_match" in r]))
                              if same and "token_match" in same[0] else None),
        joint_err_mean_cm=agg("joint_err_mean_cm"), joint_err_p95_cm=agg("joint_err_mean_cm", fn=p95),
        joint_err_max_cm=agg("joint_err_max_cm", fn=np.nanmax),
        joint_err_mean_cm_same_len=agg("joint_err_mean_cm", same),
        root_err_mean_cm=agg("root_err_mean_cm"), root_err_max_cm=agg("root_err_max_cm", fn=np.nanmax),
        heading_err_mean_deg=agg("heading_err_mean_deg"), heading_err_max_deg=agg("heading_err_max_deg", fn=np.nanmax),
        contact_flip_frac=agg("contact_flip_frac"),
        ref_skate_cm_s=agg("ref_skate") * 100, var_skate_cm_s=agg("var_skate") * 100,
        ref_jitter_m_s2=agg("ref_jitter"), var_jitter_m_s2=agg("var_jitter"),
        ref_jerk=agg("ref_jerk"), var_jerk=agg("var_jerk"),
        ref_start_cm=agg("ref_start_cm"), var_start_cm=agg("var_start_cm"),
        ref_end_cm=agg("ref_end_cm"), var_end_cm=agg("var_end_cm"),
        ref_end_root_cm=agg("ref_end_root_cm"), var_end_root_cm=agg("var_end_root_cm"),
        frac_requests_joint_err_gt_1cm=float(np.mean([r["joint_err_mean_cm"] > 1.0 for r in ok])) if ok else None,
        frac_requests_joint_max_gt_5cm=float(np.mean([r["joint_err_max_cm"] > 5.0 for r in ok])) if ok else None,
    )
    summary["skate_delta_pct"] = 100 * (summary["var_skate_cm_s"] / summary["ref_skate_cm_s"] - 1)
    summary["jitter_delta_pct"] = 100 * (summary["var_jitter_m_s2"] / summary["ref_jitter_m_s2"] - 1)
    by_cat = {}
    for c in dict.fromkeys(r["category"] for r in rows):
        rs = [r for r in ok if r["category"] == c]
        by_cat[c] = dict(n=len(rs), joint_err_mean_cm=agg("joint_err_mean_cm", rs),
                         joint_err_max_cm=agg("joint_err_max_cm", rs, np.nanmax),
                         num_tokens_agree=float(np.mean([r["ref_tokens"] == r["var_tokens"] for r in rs])) if rs else None,
                         token_match=agg("token_match", [r for r in rs if r["ref_tokens"] == r["var_tokens"]]))
    return dict(summary=summary, by_category=by_cat, rows=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="*", help="ONNX files, or fp32_original")
    mb.add_path_args(ap)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--provider", default="cpu")
    ap.add_argument("--npz")
    ap.add_argument("--name")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--opt", default="all", choices=("all", "basic", "none"),
                    help="ORT graph optimisation level. Use basic for QDQ models: level-2 QDQ fusion changes their numerics.")
    a = ap.parse_args()
    mb.apply_path_args(a)
    golden = mb.load_golden()
    meta = json.loads((mb.DATA / "golden_meta.json").read_text(encoding="utf-8"))["requests"]
    mb.RESULTS.mkdir(parents=True, exist_ok=True)
    jobs = [(a.name, None)] if a.npz else [((v if v == "fp32_original" else Path(v).stem) + a.suffix, v) for v in a.variants]
    for name, v in jobs:
        try:
            if a.npz:
                z = np.load(a.npz)
                out = {k: z[k] for k in z.files}
                info = {}
            else:
                path = mb.FP32_ONNX if v == "fp32_original" else Path(v)
                out = run_variant(path, golden, a.batch, a.provider, opt=a.opt)
                info = dict(run_seconds=out.pop("seconds"), providers=out.pop("providers"), batch=a.batch, ort_opt=a.opt)
                np.savez_compressed(mb.RESULTS / f"out_{name}.npz", **{k: (v_.astype(np.float16) if k == "pose_logits" else v_) for k, v_ in out.items()})
            res = score(out, golden, meta)
            res["info"] = info
        except Exception as e:
            res = dict(error=f"{type(e).__name__}: {e}"[:800])
        (mb.RESULTS / f"quality_{name}.json").write_text(json.dumps(res), encoding="utf-8")
        if "error" in res:
            print(name, "ERROR", res["error"])
            continue
        s = res["summary"]
        print(f"{name}: ntok_agree={s['num_tokens_agree']:.3f} tok_match={s['token_match'] if s['token_match'] is None else round(s['token_match'], 4)} "
              f"joint mean={s['joint_err_mean_cm']:.3f}cm p95={s['joint_err_p95_cm']:.3f} max={s['joint_err_max_cm']:.2f} "
              f"root mean={s['root_err_mean_cm']:.3f} skate {s['ref_skate_cm_s']:.2f}->{s['var_skate_cm_s']:.2f}cm/s "
              f"jitter {s['ref_jitter_m_s2']:.3f}->{s['var_jitter_m_s2']:.3f} end {s['ref_end_cm']:.2f}->{s['var_end_cm']:.2f}cm "
              f"start {s['ref_start_cm']:.2f}->{s['var_start_cm']:.2f}cm nonfinite={s['non_finite']}")


if __name__ == "__main__":
    main()
