"""Leg-lane search for the synthetic gait.
python -I tune_lanes.py --raw <clips folder> --configs '<json list of configs>' [--clips 8]
config = {"fan_deg": [a1, a2, a3, a4], "k_ankle": .., "k_lift": .., "stride_cap": ..} or null (the generator without lanes). Prints the overlap of the
generated clips before / after the capsule repair (creature/collision.resolve), per speed band."""
import sys, os, json, argparse
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=os.environ.get("SPIDER_RAW", ""), help="folder with the extracted clips (extract_clips.py); env SPIDER_RAW")
    ap.add_argument("--configs", required=True, help="JSON list of lane configs"); ap.add_argument("--clips", type=int, default=8)
    a = ap.parse_args()
    import torch
    from bodyplans.spider import spider_lib
    from bodyplans.spider.spider_data import real_canonical, gait_of, idle_of, make_synth_clip
    from creature.collision import Collision, resolve
    spider_lib.set_raw_dir(a.raw)
    cfgs = json.loads(a.configs); NCL = a.clips
    Cc = Collision(str(HERE / "spider_collision.json"), "cuda" if torch.cuda.is_available() else "cpu")
    for cfg in cfgs:
        cfg = dict(cfg) if cfg else None; rs = (cfg or {}).pop("resolve", {})
        C, clips = real_canonical(cfg)
        G = gait_of(clips["bga_spider__Spider_Walk"]["pos"], C, 2); idle = idle_of(clips["bga_spider__Spider_Idle_long"]["pos"], C)
        walk_h = float(clips["bga_spider__Spider_Walk"]["pos"][:, 0, 1].mean())
        Ps, keys = zip(*[make_synth_clip(C, G, idle, walk_h, 1000 + i, 20.0) for i in range(NCL)])
        P = np.concatenate(Ps).astype(np.float32); key = np.concatenate(keys); st = np.cumsum([0] + [len(p) for p in Ps])[:-1]
        Q = resolve(Cc, P, st, **rs)
        spd = np.hypot(key[:, 0], key[:, 1]); out = dict(cfg=cfg, resolve=rs)
        for nm, X in (("gen", P), ("fix", Q)):
            fd, g = Cc.frame_depth(X); row = {}
            for b, lo, hi in (("all", -1, 99), ("stand", -1, 0.05), ("<2", 0.05, 2), ("2-4", 2, 4), (">4", 4, 99)):
                m = (spd > lo) & (spd <= hi); h = fd[m, 0] > 0
                row[b] = "%.1f%% %.1fcm" % (h.mean() * 100, fd[m, 0][h].mean() * 100 if h.any() else 0)
            row["body"] = "%.1f%%" % ((fd[:, 1] > 0).mean() * 100); row["max_cm"] = round(float(fd.max() * 100), 1); out[nm] = row
        out["top_gen"] = Cc.measure(P)["top_pairs_pct"]; out["top_fix"] = Cc.measure(Q)["top_pairs_pct"]
        mv = np.linalg.norm(Q - P, axis=-1); out["repair_move_cm_mean_p99"] = [round(float(mv.mean() * 100), 2), round(float(np.percentile(mv, 99) * 100), 1)]
        real = {n: c["pos"].astype(np.float32) for n, c in clips.items() if "eath" not in n and "dead" not in n}
        out["real_gen"] = {n.split("__")[1]: round(float((Cc.frame_depth(p)[0].max(1) > 0).mean() * 100), 1) for n, p in real.items()}
        out["real_fix"] = {n.split("__")[1]: round(float((Cc.frame_depth(resolve(Cc, p, None, **rs))[0].max(1) > 0).mean() * 100), 1) for n, p in real.items()}
        print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
