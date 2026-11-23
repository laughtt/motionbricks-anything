"""Spider collision spec (capsules) from spider_spec.json + the spider SOMA skin radii (blender_make_spider_soma.py: radius()).
python -I make_collision_spec.py [--spec spider_spec.json] [--out spider_collision.json] (defaults: the files next to this script). The first leg segment starts at the fraction 'root_t0' where the
rest pose has no leg-leg overlap (the hips of an 8-leg body sit closer together than the leg thickness; that part is inside the body)."""
import json, os, sys, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--spec", default=HERE + "/spider_spec.json"); ap.add_argument("--out", default=HERE + "/spider_collision.json")
ARGS = ap.parse_args() if __name__ == "__main__" else ap.parse_args([])
from creature.collision import Collision
spec = json.load(open(ARGS.spec)); N = spec["joints"]; rest = np.array(spec["rest_metres"], np.float32)
# SOMA skin radii (m) at the joints; subdivision shrinks the skin a little, so the capsules are slightly conservative
# measured on the SOMA mesh (vertex distance to the bone line): femur 0.072 -> 0.066, tibia 0.055 -> 0.049, tarsus 0.033 -> 0.024
# = 0.92 x the skin radii (subdivision shrinks the skin)
RAD = {k: round(v * 0.92, 4) for k, v in dict(hip=0.085, knee=0.07, ankle=0.05, tip=0.022).items()}
ix = N.index


def build(root_t0):
    segs = []
    for li, leg in enumerate(spec["legs"]):
        nm = N[leg[0]].split("_")[0]; side = nm[0]; order = int(nm[1])
        for k in range(3):
            a, b = leg[k], leg[k + 1]; ra = RAD[N[a].split("_")[1]]; rb = RAD[N[b].split("_")[1]]; t0 = root_t0 if k == 0 else 0.0
            segs.append(dict(name=f"{nm}_s{k}", a=a, b=b, t0=t0, t1=1.0, r0=round(ra + (rb - ra) * t0, 4), r1=rb, kind="leg", part=nm, side=side, order=order,
                             skip=["cephalothorax"] if k == 0 else []))
    segs.append(dict(name="cephalothorax", a=ix("body"), b=ix("head"), t0=0.0, t1=1.0, r0=0.26, r1=0.19, kind="body", part="cephalothorax"))
    egg = [(0.0, 0.2), (0.3, 0.42), (0.6, 0.5), (0.85, 0.4), (1.0, 0.2)]       # the abdomen egg of the SOMA (fraction junction -> end, radius)
    for i in range(4):
        segs.append(dict(name=f"abdomen_{i}", a=ix("abdomen"), b=ix("abdomen_end"), t0=egg[i][0], t1=egg[i + 1][0], r0=egg[i][1], r1=egg[i + 1][1], kind="body", part="abdomen"))
    for p in ("palp_L", "palp_R"):
        segs.append(dict(name=p, a=ix("head"), b=ix(p), t0=0.45, t1=1.0, r0=0.06, r1=0.035, kind="body", part="palps"))
    gj = [j for leg in spec["legs"] for j in leg[1:]]
    return dict(creature="spider", note="tapered capsules from the spider SOMA skin radii", root_t0=root_t0, margin=0.0, segments=segs, chains=spec["legs"],
                ground=dict(joints=gj, radius=[0.0 if N[j].endswith("tip") else RAD[N[j].split("_")[1]] * 0.5 for j in gj]))



def main():
    for t0 in np.arange(0.0, 0.9, 0.05):
        c = build(float(round(t0, 2))); rep = Collision(c).measure(rest[None] - np.array([0, rest[spec["feet"], 1].min(), 0], np.float32))
        if rep["any"]["pct_frames"] == 0 and Collision(c).depth(__import__("torch").tensor(rest)[None])[0].max() < -0.03: break
    json.dump(c, open(ARGS.out, "w"), indent=1)
    C = Collision(c)
    print("COLLISION_SPEC root_t0", c["root_t0"], "segments", len(c["segments"]), "pairs", len(C.pi), "rest max depth", float(C.depth(__import__("torch").tensor(rest)[None])[0].max()))


if __name__ == "__main__":
    main()
