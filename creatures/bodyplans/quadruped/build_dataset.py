"""MANN dog BVH files -> dataset.npz for creature/ (30 fps, metres).
python -I build_dataset.py --bvh_dir <folder of download_mann.py> --out <data dir>/dataset.npz
Every 8th file is held out ("val"). The joint order is the BVH order (27 joints with the end sites; see creature.json)."""
import sys, glob, json, argparse
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from bodyplans.quadruped import bvh
from creature import features as feat

VAL_EVERY = 8


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bvh_dir", required=True); ap.add_argument("--out", required=True, help="path of the dataset npz")
    ap.add_argument("--cfg", default=str(HERE / "creature.json"), help="creature config (skeleton: root, front, contacts)")
    ap.add_argument("--unit", type=float, default=0.01, help="BVH unit in metres (MANN: centimetres)")
    a = ap.parse_args()
    sk = json.load(open(a.cfg))["skeleton"]
    spec = dict(root=sk["root"], front=sk["front"], feet=sk["contacts"])     # Hips, Neck, paws
    fs = sorted(glob.glob(str(Path(a.bvh_dir) / "**" / "*.bvh"), recursive=True))
    if not fs:
        sys.exit(f"no .bvh files in {a.bvh_dir}")
    pose, root, local, contact, starts, lens, names, split = [], [], [], [], [], [], [], []
    n = 0; jn = None
    for k, f in enumerate(fs):
        d = bvh.read(f)
        jn = d["names"]
        if jn != sk["joints"]:
            sys.exit(f"{f}: joints do not match the skeleton of {a.cfg}")
        p = feat.resample(d["pos"] * a.unit, d["fps"])
        if len(p) < 70:
            continue
        ft = feat.features(p, spec)
        F = len(ft["pose"])
        pose.append(ft["pose"]); root.append(ft["root"]); local.append(ft["local"]); contact.append(ft["contact"])
        starts.append(n); lens.append(F); names.append(Path(f).stem); split.append("val" if k % VAL_EVERY == 3 else "train"); n += F
    pose = np.concatenate(pose); root = np.concatenate(root); local = np.concatenate(local)
    tr = np.concatenate([np.arange(s, s + l) for s, l, sp in zip(starts, lens, split) if sp == "train"])
    pm, ps = pose[tr].mean(0), pose[tr].std(0) + 1e-3
    rm, rs = root[tr].mean(0), root[tr].std(0) + 1e-3
    J = local.shape[1]
    meta = dict(J=J, joints=jn, spec=spec, fps=30, source="MANN dog mocap (Zhang et al. 2018), CC BY-NC 4.0, research only")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, pose=pose, root=root, local=local, contact=np.concatenate(contact),
             starts=np.array(starts), lens=np.array(lens), names=np.array(names), split=np.array(split),
             pose_mean=pm.astype(np.float32), pose_std=ps.astype(np.float32), root_mean=rm.astype(np.float32), root_std=rs.astype(np.float32),
             meta=json.dumps(meta))
    trf = sum(l for l, s in zip(lens, split) if s == "train"); vaf = sum(l for l, s in zip(lens, split) if s == "val")
    print("DATASET_DONE", dict(clips=len(lens), train_frames=trf, val_frames=vaf, minutes=round(n / 30 / 60, 1), D=pose.shape[1], J=J,
                               root_h=[round(float(x), 3) for x in np.percentile(root[:, 3], [5, 50, 95])],
                               speed=[round(float(x), 2) for x in np.percentile(np.hypot(root[:, 0], root[:, 1]), [5, 50, 95])],
                               contact_rate=np.concatenate(contact).mean(0).round(2).tolist()))


if __name__ == "__main__":
    main()
