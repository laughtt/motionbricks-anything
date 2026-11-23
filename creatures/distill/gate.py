"""Step 4: the "do no harm" gate. The 7 rollout scenarios of rollout.py, each with N randomised variants
(distance / bearing / yaw +-20%, clip frames +-15, replan lead / turn rate +-20%): 7 x 8 = 56 runs per model with --n 8.
One deterministic run per scenario would be too noisy for students: walk_to_sit skate swings +-1 cm/s with tiny numeric changes.
Metrics: foot skate (after foot-lock, cm/s) and jitter (m/s^2), mean over the scenarios.

    python -I creatures/distill/gate.py --n 8 --ref base --tol 0.05 --out gate.json base MODEL.onnx

A model = 'base' (the teacher, PyTorch, GPU, needs the NVIDIA checkpoint), a distill run name (PyTorch, GPU)
or an .onnx path (ORT CPU, BASIC optimisation). With --ref, every other model gets PASS / FAIL:
skate and jitter must be <= (1 + tol) x the reference. (The original scripts only printed the numbers;
the PASS / FAIL line is added here so that the +5 % rule of the README is checked by code.)
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I drops the script folder
import numpy as np  # noqa: E402
import mb_common as mb  # noqa: E402
import rollout  # noqa: E402
import footlock  # noqa: E402


def perturb(spec, rng):
    s = json.loads(json.dumps(spec))
    j = lambda x, f=0.2: float(x) * float(rng.uniform(1 - f, 1 + f))
    if s["kind"] == "chain":
        s["start"] = (s["start"][0], max(0, int(s["start"][1] + rng.integers(-15, 16))))
        s["targets"] = [(c, max(0, int(f + rng.integers(-15, 16))), j(d), float(b + rng.uniform(-0.2, 0.2)),
                         float(y + rng.uniform(-0.2, 0.2)), nt) for (c, f, d, b, y, nt) in s["targets"]]
    else:
        s["start"] = (s["start"][0], max(0, int(s["start"][1] + rng.integers(-15, 16))))
        s["goal"] = (s["goal"][0], max(0, int(s["goal"][1] + rng.integers(-15, 16))))
        s["lead"], s["turn_rate"] = j(s["lead"]), j(s["turn_rate"])
    return s


def session_for(name):
    if name.endswith(".onnx"):
        return mb.make_session(name, threads=8, opt="basic")
    import distill as dl             # torch + the NVIDIA checkpoint are needed only for torch models
    return dl.TorchSession(dl.load_student_model(name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="'base', a distill run name, or an .onnx path")
    ap.add_argument("--n", type=int, default=8, help="variants per scenario")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default="gate.json", help="file name under <work>/results (or an absolute path)")
    ap.add_argument("--ref", default=None, help="reference model (one of the models) for PASS / FAIL, normally base")
    ap.add_argument("--tol", type=float, default=0.05, help="allowed increase over the reference (0.05 = +5 %%)")
    ap.add_argument("--ckpt", default=None, help="NVIDIA SOMA checkpoint folder (env MB_CKPT)")
    mb.add_path_args(ap)
    a = ap.parse_args()
    mb.apply_path_args(a)
    if a.ckpt:
        import mbtorch as mt
        mt.CKPT = Path(a.ckpt)
    names = (rollout.WALK, rollout.JOG, rollout.IDLE, rollout.CROUCH, rollout.SIT, rollout.ZOMBIE)
    clips = {p.stem: mb.Clip(p) for p in mb.MOTIONS.rglob("*.bvh") if p.stem in names}
    rng = np.random.default_rng(a.seed)
    specs = {sc: [sp] + [perturb(sp, rng) for _ in range(a.n - 1)] for sc, sp in rollout.SCENARIOS.items()}
    res = {}
    for m in a.models:
        t0 = time.time()
        gen = rollout.Generator(session=session_for(m))
        per = {}
        for sc, vs in specs.items():
            fn = rollout.scenario_chain if vs[0]["kind"] == "chain" else rollout.scenario_replan
            sk, jit = [], []
            for spec in vs:
                r = fn(gen, clips, spec, tuple(range(8, 15)))
                pos, c = r["pos"].astype(np.float64), r["contacts"].astype(np.float64)
                sk.append(footlock.motion_metrics(footlock.foot_lock(pos, c)[0], c)["skate"])
                jit.append(footlock.motion_metrics(pos, c)["jitter"])
            per[sc] = dict(skate_fl=float(np.mean(sk)), jitter=float(np.mean(jit)), skate_fl_all=np.round(sk, 3).tolist())
        res[m] = dict(skate_fl=float(np.mean([v["skate_fl"] for v in per.values()])),
                      jitter=float(np.mean([v["jitter"] for v in per.values()])), per=per, sec=round(time.time() - t0))
        print(m, f"skate+FL {res[m]['skate_fl']:.3f} jitter {res[m]['jitter']:.3f}  ({res[m]['sec']}s)  " +
              " ".join(f"{k.split('_', 1)[1][:14]}={v['skate_fl']:.2f}" for k, v in per.items()), flush=True)
    if a.ref:
        r0 = res[a.ref]
        for m, r in res.items():
            if m == a.ref:
                continue
            ds_, dj = r["skate_fl"] / r0["skate_fl"] - 1, r["jitter"] / r0["jitter"] - 1
            ok = ds_ <= a.tol and dj <= a.tol
            res[m]["gate"] = dict(skate_delta=ds_, jitter_delta=dj, tol=a.tol, passed=bool(ok))
            print(f"gate {m}: skate {ds_:+.1%}  jitter {dj:+.1%}  (limit +{a.tol:.0%})  -> {'PASS' if ok else 'FAIL'}", flush=True)
    out = Path(a.out) if Path(a.out).is_absolute() else mb.RESULTS / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
