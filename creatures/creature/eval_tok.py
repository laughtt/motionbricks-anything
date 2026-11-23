"""Tokenizer round trip check: joint error, foot-skate numbers, contact channel, planted % of the round trip vs mocap (held-out clips).
python -I eval_tok.py --cfg <creature.json> --tok tok_a [--tok tok_b ...]"""
import sys, json, argparse
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from creature.core import Cfg, Data, load_tok, DEV
from creature.losses import skate_losses, local_pos
from creature import rollout as RO


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cfg", required=True); ap.add_argument("--tok", action="append", required=True); ap.add_argument("--out", default="")
    a = ap.parse_args()
    cfg = Cfg(a.cfg); va = Data(cfg.data, cfg.val_split); pm, ps, rm, rs = cfg.tensors(va)
    e = dict(walk=1.0, trot=2.2); e.update(cfg.raw.get("eval", {}))
    res = {}
    for tag in a.tok:
        tokz = load_tok(cfg, tag); T = 128
        offs = va.windows(T, T)
        hs_r, hs_g, fw, acc, S, n, E = [], [], [], [0, 0, 0, 0], {}, 0, 0.0
        with torch.no_grad():
            for i in range(0, len(offs), 64):
                P, R, L = va.take(offs[i:i + 64], T)
                rec, _ = tokz(P); Rr = R * rs + rm
                sl = skate_losses(rec, P, Rr, cfg, pm, ps)
                for k, v in sl.items():
                    S[k] = S.get(k, 0.0) + float(v) * len(P)
                n += len(P); E += float((local_pos(rec * ps + pm, cfg.J) - L).norm(dim=-1).mean()) * 100 * len(P)
                cg = (P * ps + pm)[..., -cfg.C:] > 0.5; cp = (rec * ps + pm)[..., -cfg.C:] > 0.5
                acc[0] += (cg & cp).sum().item(); acc[1] += cg.sum().item(); acc[2] += cp.sum().item(); acc[3] += (cg == cp).float().mean().item() * len(P)
                lr = local_pos(rec * ps + pm, cfg.J).cpu().numpy().astype(np.float64); Rn = Rr.cpu().numpy().astype(np.float64); Ln = L.cpu().numpy().astype(np.float64)
                for b in range(len(P)):
                    wr = RO.integrate(lr[b], Rn[b], cfg.fps)[:, cfg.contacts]; wg = RO.integrate(Ln[b], Rn[b], cfg.fps)[:, cfg.contacts]
                    hs_r.append(np.linalg.norm((wr[1:] - wr[:-1])[..., [0, 2]], axis=-1) * cfg.fps); hs_g.append(np.linalg.norm((wg[1:] - wg[:-1])[..., [0, 2]], axis=-1) * cfg.fps)
                    k = 15; fw.append(np.convolve(np.pad(Rn[b, :, 1], (k, k), mode="edge"), np.ones(31) / 31, mode="valid")[:-1])
        hs_r, hs_g, fw = np.concatenate(hs_r), np.concatenate(hs_g), np.concatenate(fw)
        r = dict(val_cm=E / n, contact_acc=acc[3] / n, contact_f1=2 * acc[0] / max(acc[1] + acc[2], 1), **{k: v / n for k, v in S.items()})
        for name, c in (("walk", e["walk"]), ("trot", e["trot"])):
            m = (fw >= c * 0.8) & (fw < c * 1.2)
            r[name] = dict(frames=int(m.sum()), planted_roundtrip=float((hs_r[m] < 0.2).mean() * 100), planted_mocap=float((hs_g[m] < 0.2).mean() * 100))
        m = np.ones(len(fw), bool); r["all"] = dict(planted_roundtrip=float((hs_r < 0.2).mean() * 100), planted_mocap=float((hs_g < 0.2).mean() * 100))
        res[tag] = r; print(tag, json.dumps(r), flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
