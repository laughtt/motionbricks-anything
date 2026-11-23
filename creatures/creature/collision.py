"""Capsule collision metric + differentiable penalty for articulated creatures (spec-driven, skeleton-agnostic).

Collision spec JSON (see bodyplans/spider/spider_collision.json, made by bodyplans/spider/make_collision_spec.py):
  {"segments": [{"name": "L1_s0", "a": 4, "b": 5, "t0": 0.45, "t1": 1.0, "r0": 0.08, "r1": 0.07, "kind": "leg", "part": "L1", "side": "L", "order": 0,
                 "skip": ["cephalothorax"]}, ...
                {"name": "ceph", "a": 0, "b": 1, "t0": 0, "t1": 1, "r0": 0.26, "r1": 0.19, "kind": "body", "part": "cephalothorax"}, ...],
   "ground": {"joints": [...], "radius": [...]},      # joint spheres that must stay above y = 0
   "margin": 0.0}
A segment is a tapered capsule on the line joint a -> joint b, from fraction t0 to t1, radius r0 -> r1.
Pairs: leg segment vs leg segment of a DIFFERENT part ("same_side_neighbour": same side and |order difference| = 1; "left_right": other side;
"same_side_other"), leg segment vs body segment ("leg_body"; a leg segment skips the body parts in its "skip" list).
Penetration depth of a pair = r1(s) + r2(t) - d at the closest points of the two axes (0 = touch).
Use: C = Collision(spec); C.measure(pos [F,J,3] numpy) -> report dict; C.penalty(pos [...,J,3] torch, margin) -> differentiable loss terms.
CLI: python -I collision.py <collision.json> <file.npz> [array key=local]"""
from __future__ import annotations
import json, sys
import numpy as np
import torch

TYPES = ("same_side_neighbour", "left_right", "same_side_other", "leg_body")


def seg_seg(p1, q1, p2, q2, eps=1e-9):
    """closest points of segments p1-q1 and p2-q2 (any batch shape, last dim 3). Returns distance, s, t (fractions on each segment)."""
    d1 = q1 - p1; d2 = q2 - p2; r = p1 - p2
    a = (d1 * d1).sum(-1).clamp(min=eps); e = (d2 * d2).sum(-1).clamp(min=eps)
    f = (d2 * r).sum(-1); c = (d1 * r).sum(-1); b = (d1 * d2).sum(-1)
    den = a * e - b * b
    s = torch.where(den > eps, (b * f - c * e) / den.clamp(min=eps), torch.zeros_like(den)).clamp(0, 1)
    t = (b * s + f) / e
    tc = t.clamp(0, 1)
    s = torch.where((t < 0) | (t > 1), ((b * tc - c) / a).clamp(0, 1), s)
    diff = (p1 + d1 * s[..., None]) - (p2 + d2 * tc[..., None])
    return torch.sqrt((diff * diff).sum(-1) + 1e-10), s, tc


class Collision:
    def __init__(self, spec, device="cpu"):
        if isinstance(spec, str): spec = json.load(open(spec))
        self.spec = spec; self.dev = device; S = spec["segments"]; self.names = [s["name"] for s in S]
        t = lambda k, dt=torch.float32: torch.tensor([s[k] for s in S], dtype=dt, device=device)
        self.a, self.b = t("a", torch.long), t("b", torch.long); self.t0, self.t1, self.r0, self.r1 = t("t0"), t("t1"), t("r0"), t("r1")
        I, K, ty = [], [], []
        for i, si in enumerate(S):
            for k, sk in enumerate(S):
                if k <= i: continue
                kinds = (si["kind"], sk["kind"])
                if kinds == ("leg", "leg"):
                    if si["part"] == sk["part"]: continue
                    if si["side"] != sk["side"]: typ = 1
                    else: typ = 0 if abs(si["order"] - sk["order"]) == 1 else 2
                elif "leg" in kinds and "body" in kinds:
                    lg, bd = (si, sk) if si["kind"] == "leg" else (sk, si)
                    if bd["part"] in lg.get("skip", []): continue
                    typ = 3
                else: continue
                I.append(i); K.append(k); ty.append(typ)
        self.pi = torch.tensor(I, device=device); self.pk = torch.tensor(K, device=device); self.ptype = torch.tensor(ty, device=device)
        self.pair_names = [f"{self.names[i]}|{self.names[k]}" for i, k in zip(I, K)]
        g = spec.get("ground", {"joints": [], "radius": []})
        self.gj = torch.tensor(g["joints"], dtype=torch.long, device=device); self.gr = torch.tensor(g["radius"], dtype=torch.float32, device=device)
        self.margin = float(spec.get("margin", 0.0))

    def ends(self, pos):
        A = pos[..., self.a, :]; B = pos[..., self.b, :]; d = B - A
        return A + d * self.t0[:, None], A + d * self.t1[:, None]

    def depth(self, pos):
        """pos [...,J,3] -> penetration depth per pair [...,P] (negative = free space), ground depth per ground joint [...,G]."""
        P0, P1 = self.ends(pos)
        d, s, t = seg_seg(P0[..., self.pi, :], P1[..., self.pi, :], P0[..., self.pk, :], P1[..., self.pk, :])
        ri = self.r0[self.pi] + (self.r1[self.pi] - self.r0[self.pi]) * s; rk = self.r0[self.pk] + (self.r1[self.pk] - self.r0[self.pk]) * t
        gd = self.gr - pos[..., self.gj, 1] if len(self.gj) else pos.new_zeros(pos.shape[:-2] + (0,))
        return ri + rk - d, gd

    def penalty(self, pos, margin=None, ground_margin=0.0):
        """differentiable: mean over frames of sum over pairs relu(r1 + r2 + margin - d)^2, split in legs / body / ground terms."""
        m = self.margin if margin is None else margin
        pen, gd = self.depth(pos)
        q = torch.relu(pen + m) ** 2; leg = self.ptype < 3
        return dict(legs=q[..., leg].sum(-1).mean(), body=q[..., ~leg].sum(-1).mean(), ground=(torch.relu(gd + ground_margin) ** 2).sum(-1).mean())

    @torch.no_grad()
    def measure(self, pos, chunk=4096, top=6, ground_tol=0.01):
        """pos [F,J,3] (numpy or torch, floor at y = 0) -> report: per pair type the % of frames with an overlap and the depth
        (deepest pair of the frame, over the frames that overlap): mean / p95 / max, metres."""
        pos = torch.as_tensor(np.asarray(pos), dtype=torch.float32)
        F = len(pos); nt = len(TYPES); mx = np.zeros((F, nt), np.float32); pair_hit = np.zeros(len(self.pi)); gmin = np.zeros(F, np.float32)
        for o in range(0, F, chunk):
            pen, gd = self.depth(pos[o:o + chunk].to(self.dev))
            for k in range(nt):
                m = self.ptype == k
                if m.any(): mx[o:o + chunk, k] = pen[:, m].max(-1).values.cpu().numpy()
            pair_hit += (pen > 0).sum(0).cpu().numpy()
            if gd.shape[-1]: gmin[o:o + chunk] = gd.max(-1).values.cpu().numpy()
        def stat(v):
            h = v > 0
            return dict(pct_frames=float(h.mean() * 100), mean_cm=float(v[h].mean() * 100) if h.any() else 0.0,
                        p95_cm=float(np.percentile(v[h], 95) * 100) if h.any() else 0.0, max_cm=float(v.max() * 100) if h.any() else 0.0)
        rep = {TYPES[k]: stat(mx[:, k]) for k in range(nt)}
        rep["legs_any"] = stat(mx[:, :3].max(1)); rep["any"] = stat(mx.max(1))
        g = gmin - ground_tol; rep["ground"] = stat(g); rep["ground"]["tol_cm"] = ground_tol * 100
        rep["frames"] = int(F)
        order = np.argsort(-pair_hit)[:top]
        rep["top_pairs_pct"] = {self.pair_names[i]: round(float(pair_hit[i] / F * 100), 2) for i in order if pair_hit[i] > 0}
        return rep

    @torch.no_grad()
    def frame_depth(self, pos):
        """numpy [F,J,3] -> deepest leg-leg / leg-body penetration per frame [F,2] and ground depth [F] (metres; <= 0 = free)."""
        pos = torch.as_tensor(np.asarray(pos), dtype=torch.float32, device=self.dev)
        pen, gd = self.depth(pos); leg = self.ptype < 3
        z = pen.new_full((len(pos),), -1.0)
        return (torch.stack([pen[:, leg].max(-1).values if leg.any() else z, pen[:, ~leg].max(-1).values if (~leg).any() else z], 1).cpu().numpy(),
                (gd.max(-1).values if gd.shape[-1] else z).cpu().numpy())


def resolve(C, pos, clip_starts=None, chains=None, margin=0.015, iters=300, lr=0.004, w_len=30.0, w_reg=0.02, w_smooth=2.0, chunk=20000, fabrik=12):
    """Repair: push colliding capsules apart by moving the INNER joints of each chain (knee, ankle); chain ends (hip, tip) stay fixed, so planted
    feet stay planted and the body does not move. Gradient descent on penalty(margin) + bone length + small change + smooth change in time,
    then a FABRIK projection to the exact bone lengths. pos [F,J,3] numpy (floor y = 0); clip_starts: frame indices where a new clip starts
    (no smoothing across them). Returns the repaired positions (numpy)."""
    chains = chains if chains is not None else C.spec["chains"]
    pos = np.asarray(pos, np.float32); F = len(pos); out = pos.copy()
    brk = np.zeros(F, bool); brk[0] = True
    if clip_starts is not None: brk[np.asarray(clip_starts, int)] = True
    free = sorted({j for ch in chains for j in ch[1:-1]}); bones = [(ch[i], ch[i + 1]) for ch in chains for i in range(len(ch) - 1)]
    ba = torch.tensor([b[0] for b in bones], device=C.dev); bb = torch.tensor([b[1] for b in bones], device=C.dev); fr = torch.tensor(free, device=C.dev)
    for o in range(0, F, chunk):
        P0 = torch.tensor(pos[o:o + chunk], device=C.dev); n = len(P0)
        L0 = (P0[:, bb] - P0[:, ba]).norm(dim=-1)
        ok = torch.tensor(~brk[o:o + chunk], device=C.dev).float()[:, None, None]                # frame t continues frame t-1
        D = torch.zeros(n, len(free), 3, device=C.dev, requires_grad=True)
        opt = torch.optim.Adam([D], lr)
        for it in range(iters):
            P = P0.clone(); P[:, fr] = P0[:, fr] + D
            pen = C.penalty(P, margin, ground_margin=0.0)
            ln = (((P[:, bb] - P[:, ba]).norm(dim=-1) - L0) ** 2).sum(-1).mean()
            d1 = (D[1:] - D[:-1]) * ok[1:]; d2 = (D[2:] - 2 * D[1:-1] + D[:-2]) * ok[2:] * ok[1:-1] if n > 2 else D[:0]
            loss = pen["legs"] + pen["body"] + pen["ground"] + w_len * ln + w_reg * (D ** 2).sum(-1).mean() + w_smooth * ((d1 ** 2).sum(-1).mean() + (d2 ** 2).sum(-1).mean())
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            P = P0.clone(); P[:, fr] = P0[:, fr] + D
            for ch in chains:                                    # exact bone lengths, ends fixed (FABRIK)
                Ls = [(P0[:, ch[i + 1]] - P0[:, ch[i]]).norm(dim=-1, keepdim=True) for i in range(len(ch) - 1)]
                u = lambda v: v / v.norm(dim=-1, keepdim=True).clamp(min=1e-9)
                for _ in range(fabrik):
                    for i in range(len(ch) - 2, 0, -1): P[:, ch[i]] = P[:, ch[i + 1]] + Ls[i] * u(P[:, ch[i]] - P[:, ch[i + 1]])
                    for i in range(1, len(ch) - 1): P[:, ch[i]] = P[:, ch[i - 1]] + Ls[i - 1] * u(P[:, ch[i]] - P[:, ch[i - 1]])
            out[o:o + chunk] = P.cpu().numpy()
    return out


def fmt_row(name, rep):
    c = lambda k: f"{rep[k]['pct_frames']:.2f} % / {rep[k]['mean_cm']:.1f} / {rep[k]['p95_cm']:.1f} / {rep[k]['max_cm']:.1f}"
    return f"| {name} | {rep['frames']} | {c('legs_any')} | {c('same_side_neighbour')} | {c('left_right')} | {c('same_side_other')} | {c('leg_body')} | {c('ground')} |"


HEADER = ("| Motion | Frames | Any leg-leg: % frames / mean / p95 / max cm | Same-side neighbours | Left-right | Same side, other | Leg-body | Foot below ground (> tol) |\n"
          "|---|---|---|---|---|---|---|---|")

if __name__ == "__main__":
    C = Collision(sys.argv[1]); z = np.load(sys.argv[2], allow_pickle=True)
    print(json.dumps(C.measure(z[sys.argv[3] if len(sys.argv) > 3 else "local"]), indent=1))
