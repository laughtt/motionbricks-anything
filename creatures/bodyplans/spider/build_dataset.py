"""Spider dataset: real clips (retargeted to the canonical spider) + augmentations + SYNTHETIC procedural locomotion.
python -I build_dataset.py --raw <folder of extract_clips.py> --out <data dir> [--synth_minutes 26] [--lanes spider_lanes.json | none]
--lanes (the leg-overlap fix): leg lanes (Canon.set_lanes) for every clip + capsule repair of all clips (creature/collision.resolve);
train clips that still overlap on more than 20 % of their frames are dropped. The overlap before / after is in data_report.json.
Out: dataset.npz, spider_spec.json, data_report.json, synth_gait.npz, synth_example.npz, real_walk_unrolled.npz"""
import sys, os, json, argparse
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from bodyplans.spider.spider_lib import *
from bodyplans.spider.spider_lib import set_raw_dir
from bodyplans.spider.spider_data import *
from creature.features import features


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=os.environ.get("SPIDER_RAW", ""), help="folder with the clips extracted from your copies of the two CC-BY packs (extract_clips.py); env SPIDER_RAW")
    ap.add_argument("--out", required=True, help="output folder for dataset.npz and the reports")
    ap.add_argument("--synth_minutes", type=float, default=26.0, help="minutes of synthetic gait")
    ap.add_argument("--lanes", default=str(HERE / "spider_lanes.json"), help="leg-lane JSON; 'none' = the old generator without lanes and without the capsule repair")
    a = ap.parse_args()
    if not a.raw:
        ap.error("--raw (or env SPIDER_RAW) is needed")
    set_raw_dir(a.raw)
    OUT = a.out; os.makedirs(OUT, exist_ok=True)
    LANES = None if a.lanes.lower() == "none" else json.load(open(a.lanes))
    SYN_MIN = a.synth_minutes
    rng = np.random.default_rng(0)
    C, clips = real_canonical(LANES)
    spec = dict(name="SpiderSOMA", version=1, joints=NAMES, parents=PAR, rest_metres=np.round(C.rest, 5).tolist(), feet=FEET, legs=LEGS,
                root=0, front=1, mirror=MIRROR, contact_speed=0.6, contact_height=0.05, fps=30,
                axes="Y up, forward +Z, left +X, metres", canonical_from="bga_spider bind pose x 0.025, symmetrized (hip-knee = two source bones merged)",
                orient=dict(left="L2_hip", right="R2_hip", front="head", back="abdomen"),
                sources=["Spider animated character (Bestgameassets, CC-BY)", "Spiders - characters with animations (Mixall, CC-BY; Argiope)"])
    json.dump(spec, open(f"{OUT}/spider_spec.json", "w"), indent=1)

    G = gait_of(clips["bga_spider__Spider_Walk"]["pos"], C, 2)
    idle = idle_of(clips["bga_spider__Spider_Idle_long"]["pos"], C)
    walk_h = float(clips["bga_spider__Spider_Walk"]["pos"][:, 0, 1].mean())
    np.savez(f"{OUT}/synth_gait.npz", **{k: v for k, v in G.items() if k != "extra"}, walk_h=walk_h)


    def warp(P, k):
        """time-warp: k > 1 = faster."""
        F = len(P); t = np.arange(0, F - 1, k); i = np.floor(t).astype(int); w = (t - i)[:, None, None]
        return P[i] * (1 - w) + P[np.minimum(i + 1, F - 1)] * w


    def loop_to(P, n):
        """repeat a short in-place clip to at least n frames: cyclic when the ends match, else ping-pong."""
        if len(P) >= n: return P
        cyc = np.linalg.norm(P[0] - P[-1], axis=-1).mean() < 0.03
        unit_ = P[:-1] if cyc else np.concatenate([P[:-1], P[::-1][:-1]])
        return np.concatenate([unit_] * int(np.ceil(n / len(unit_))) + [unit_[:1]])


    entries = []   # (name, pos, split, kind, key command stream [F,3] = side, forward, yaw rate)
    def add(name, P, split, kind, key=None):
        entries.append((name, P, split, kind, np.zeros((len(P), 3)) if key is None else key))

    VAL_REAL = {"bga_spider__Spider_Walk", "bga_spider__Spider_Idle", "bga_spider__Spider_Attack_2", "mixall_argiope__idle_2"}
    EXCLUDE = {"bga_spider__Spider_Death": "death: ends on the back, heading undefined",
               "mixall_argiope__dead": "death",
               "mixall_argiope__walk": "legs do not hold the ground (duty 0.25-0.3, feet never rest); not usable as locomotion data"}
    if LANES is not None:
        # v3: with the lane stance this crouched idle (body 0.41 m; standing 0.53 m) is close to the walking poses and became an absorbing state in
        # keyboard rollouts (the model stays crouched and slides for the rest of the run). No command triggers it, so it is not training data.
        EXCLUDE["mixall_argiope__idle"] = "crouched idle (body 0.41 m): absorbing state in keyboard rollouts with the lane stance; no trigger"
    raw_seconds = {}
    for name, c in clips.items():
        P = c["pos"]; raw_seconds[name] = (len(P) - 1) / FPS
        if name in EXCLUDE: continue
        if name == "bga_spider__Spider_Walk":
            U = unroll(P, C, 6); add(name, U, "val", "real", np.tile([0.0, G["speed"], 0.0], (len(U), 1))); continue
        if name in VAL_REAL:
            add(name, loop_to(P, 100), "val", "real"); continue
        for mir in (0, 1):
            Q = mirror(P) if mir else P
            for k in (1.0, 0.7, 0.85, 1.2, 1.5):
                kind = "real" if (not mir and k == 1.0) else "real_aug"
                add(f"{name}{'_m' if mir else ''}_w{k:g}", loop_to(warp(Q, k), 100), "train", kind)

    # synthetic locomotion from the real gait
    VMAX = 6.5
    def synth_clip(seed, seconds, style="mix"):
        return make_synth_clip(C, G, idle, walk_h, seed, seconds, style, VMAX)

    n = 0; tot = 0
    while tot < SYN_MIN * 60:
        sec = float(rng.uniform(14, 26)); Pk = synth_clip(1000 + n, sec); add(f"synth_{n:03d}", Pk[0], "train", "synth", Pk[1]); tot += sec; n += 1
    for i in range(8):
        Pk = synth_clip(9000 + i, 20.0); add(f"synthval_{i:02d}", Pk[0], "valsyn", "synth", Pk[1])
    # straight walks at the real speed (comparison with the real walk)
    Pk = synth_clip(9100, 8.0, "straight"); add("synthval_real_speed", Pk[0], "valsyn", "synth", Pk[1])

    col_rep = None
    if LANES is not None:
        import torch
        from creature.collision import Collision, resolve
        Cc = Collision(str(HERE / "spider_collision.json"), "cuda" if torch.cuda.is_available() else "cpu")
        allP = np.concatenate([e[1] for e in entries]).astype(np.float32); st = np.cumsum([0] + [len(e[1]) for e in entries])
        fixed = resolve(Cc, allP, st[:-1])
        groups = {}
        for i, e in enumerate(entries): groups.setdefault(f"{e[2]}/{e[3]}", []).append(np.arange(st[i], st[i + 1]))
        col_rep = dict(lanes=LANES, before={k: Cc.measure(allP[np.concatenate(v)]) for k, v in groups.items()}, after={k: Cc.measure(fixed[np.concatenate(v)]) for k, v in groups.items()},
                       repair_move_cm_mean=float(np.linalg.norm(fixed - allP, axis=-1).mean() * 100), per_real_clip_after_pct={}, dropped=[])
        new = []
        for i, e in enumerate(entries):
            Q = fixed[st[i]:st[i + 1]].astype(np.float64); fd, _ = Cc.frame_depth(Q); pct = float((fd.max(1) > 0).mean() * 100)
            if e[3] != "synth": col_rep["per_real_clip_after_pct"][e[0]] = round(pct, 1)
            if e[2] == "train" and pct > 20.0: col_rep["dropped"].append(e[0]); continue
            new.append((e[0], Q, e[2], e[3], e[4]))
        entries = new
        print("COLLISION before", {k: round(v["any"]["pct_frames"], 2) for k, v in col_rep["before"].items()}, "after", {k: round(v["any"]["pct_frames"], 2) for k, v in col_rep["after"].items()}, "dropped", col_rep["dropped"])

    pose, root, local, starts, lens, names, split, kind, cmds = [], [], [], [], [], [], [], [], []
    o = 0
    for name, P, sp, kd, key in entries:
        f = features(P, spec, floor=0.0); cmds.append(key[:len(f["pose"])].astype(np.float32))
        pose.append(f["pose"]); root.append(f["root"]); local.append(f["local"])
        starts.append(o); lens.append(len(f["pose"])); names.append(name); split.append(sp); kind.append(kd); o += len(f["pose"])
    pose = np.concatenate(pose); root = np.concatenate(root); local = np.concatenate(local)
    tr = np.concatenate([np.arange(s, s + l) for s, l, sp in zip(starts, lens, split) if sp == "train"])
    pm, ps = pose[tr].mean(0), pose[tr].std(0) + 1e-4
    nc = len(FEET); pm[-nc:] = 0; ps[-nc:] = 1                      # contacts stay 0 / 1
    rm, rs = root[tr].mean(0), root[tr].std(0) + 1e-4
    meta = dict(J=J, joints=NAMES, parents=PAR, feet=FEET, n_contacts=nc, fps=30, spec=spec)
    np.savez(f"{OUT}/dataset.npz", pose=pose, root=root, local=local, starts=np.array(starts), lens=np.array(lens), names=np.array(names),
             split=np.array(split), kind=np.array(kind), cmd=np.concatenate(cmds), pose_mean=pm, pose_std=ps, root_mean=rm, root_std=rs, meta=json.dumps(meta))
    mins = {}
    for l, sp, kd in zip(lens, split, kind):
        mins[f"{sp}/{kd}"] = mins.get(f"{sp}/{kd}", 0) + l / FPS / 60
    spd = np.hypot(root[tr, 0], root[tr, 1])
    rep = dict(raw_seconds=raw_seconds, raw_total_seconds=sum(raw_seconds.values()), excluded=EXCLUDE, val_real=sorted(VAL_REAL),
               minutes={k: round(v, 3) for k, v in mins.items()}, frames=int(len(pose)), D=int(pose.shape[1]),
               gait=dict(period_s=G["T"], duty=float(np.mean(G["duty"])), stride_m=float(np.mean(G["stride"])), speed_mps=G["speed"],
                         step_height_m=float(np.mean(G["step_height"])), swing_centre=G["swing_centre"].tolist(), walk_body_h=walk_h, stand_h=C.stand_h,
                         source_walk_yaw_sway_deg=C.walk_yaw_sway_deg),
               scale_A=C.scale_A, train_speed_hist=np.histogram(spd, bins=[0, 0.05, 0.5, 1, 2, 3, 4, 5, 7])[0].tolist(),
               contact_rate_train=float(pose[tr][:, -nc:].mean()), collision=col_rep)
    json.dump(rep, open(f"{OUT}/data_report.json", "w"), indent=1)
    # example clips for viewing
    np.savez(OUT + "/synth_example.npz", world=entries[[e[0] for e in entries].index("synth_000")][1].astype(np.float32), fps=30)
    np.savez(OUT + "/real_walk_unrolled.npz", world=entries[[e[0] for e in entries].index("bga_spider__Spider_Walk")][1].astype(np.float32), fps=30)
    print("DATASET_DONE", json.dumps(rep["minutes"]), "D", pose.shape[1], json.dumps(rep["gait"]), rep["train_speed_hist"], rep["contact_rate_train"])


if __name__ == "__main__":
    main()
