"""Leg overlap report (capsule metric, creature/collision.py).
python -I eval_collision.py --out <report.json> --data <dataset.npz> [--roll <rollout.npz> ...] [--collision spider_collision.json]
Rows: dataset parts (split / kind), every real clip (the un-augmented ones), the given rollouts (npz with "local" [F,J,3], e.g. from creature/eval_rollout.py --npz)."""
import json, os, sys, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True); ap.add_argument("--data", required=True); ap.add_argument("--roll", action="append", default=[])
    ap.add_argument("--collision", default=HERE + "/spider_collision.json")
    a = ap.parse_args()
    import torch
    from creature.collision import Collision, fmt_row, HEADER
    out, ds = a.out, a.data
    C = Collision(a.collision, "cuda" if torch.cuda.is_available() else "cpu")
    z = np.load(ds, allow_pickle=True); L = z["local"]; rep = {}
    starts, lens, names, split, kind = z["starts"], z["lens"], z["names"], z["split"], z["kind"]
    for sp, kd in sorted(set(zip(split.tolist(), kind.tolist()))):
        idx = np.concatenate([np.arange(s, s + l) for s, l, a, b in zip(starts, lens, split, kind) if a == sp and b == kd])
        rep[f"data {sp}/{kd}"] = C.measure(L[idx])
    for s, l, n, kd in zip(starts, lens, names, kind):
        if kd == "real": rep[f"clip {n}"] = C.measure(L[s:s + l])
    for p in a.roll:
        r = np.load(p, allow_pickle=True); rep[f"rollout {os.path.basename(p)[:-4]}"] = C.measure(r["local"])
    json.dump(rep, open(out, "w"), indent=1)
    print(HEADER)
    for k, v in rep.items(): print(fmt_row(k, v))
    for k, v in rep.items():
        if not k.startswith("clip"): print(k, "top pairs", v["top_pairs_pct"])


if __name__ == "__main__":
    main()
