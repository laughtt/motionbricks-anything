"""Train the creature pose net (masked pose tokens; 4 context frames + root path [+ target frames]) with seam-aware and foot-skate-aware options.
python -I train_pose.py --cfg <creature.json> --tok tok_v2 --tag pose_v2 [--steps 6000] [--rollout 0.5] [--aux 1.0] [--aux_mode st|soft] [--ce_contact 0]
--rollout p : seam-aware training. With probability p (after a warm-up) the 4 context frames are the model's OWN output: it first generates
              1-3 chained plans (replan every 12 frames, as the runtime) from mocap context with the true root path, and is then trained to
              predict the true tokens of the next overlapping segment from that generated context.
--aux w     : foot-skate loss through the frozen tokenizer decoder: logits -> expected code vector per head (st = straight-through: the forward
              pass uses the argmax code, the gradient the soft expectation) -> decoded features -> feature L1 + paw slide on contact frames
              + paw world-velocity error.
--carry p   : in a rollout batch, with probability p, token 1 (the first 4 new frames) is given = the previous plan's token for those frames
              (the runtime "seam contract": a regular replan keeps that token).
--ce_contact w : token cross-entropy weight 1 + w * (share of contact paw-frames in the token's 4 frames)."""
import sys, json, time, math, argparse
from pathlib import Path
import numpy as np, torch
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from creature.core import Cfg, Data, PoseNet, DEV, FPT, CTX, MINT, MAXT, load_tok, decode
from creature.losses import skate_losses

RP = 12     # replan interval of the runtime (frames)


def make_cond(P, rng, p_target, train=True):
    B, T, D = P.shape
    has = torch.zeros(B, T, device=DEV); has[:, :CTX] = 1
    for b in range(B):
        if (not train) or rng.random() < p_target:
            k = 4 if not train else int(rng.integers(1, 5)); has[b, T - k:] = 1
            if train and rng.random() < 0.3:
                has[b, int(rng.integers(4, T - 4))] = 1
    return P * has[..., None], has


@torch.no_grad()
def evaluate(cfg, net, tokz, va, evals, rng):
    pm, ps, _, _ = cfg.tensors(va); res = {}; J = cfg.J
    for L, (P, R, Lc, gt) in evals.items():
        T = L * FPT; g = slice(4, T - 4)
        for task in ("inb", "pred"):
            E, A = [], []
            for i in range(0, len(P), 256):
                sl = slice(i, i + 256)
                cond, has = make_cond(P[sl], rng, 1.0, train=False)
                if task == "pred":
                    has[:, CTX:] = 0; cond = cond * has[..., None]
                tok = decode(net, R[sl], cond, has, L)
                loc = (tokz.decode_idx(tok) * ps + pm)[..., :J * 3].view(len(tok), T, J, 3)
                E.append((loc[:, g] - Lc[sl, g]).norm(dim=-1).mean((1, 2))); A.append((tok == gt[sl]).float().mean((1, 2)))
            res[f"L{L}_{task}"] = dict(cm=float(torch.cat(E).mean() * 100), tok_acc=float(torch.cat(A).mean()))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True); ap.add_argument("--tok", required=True); ap.add_argument("--tag", default="pose_v2")
    ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--balance", type=float, default=1.0, help="1 = speed-balanced window sampling (Data.set_balance), 0 = uniform over frames"); ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--rollout", type=float, default=0.5); ap.add_argument("--aux", type=float, default=1.0); ap.add_argument("--aux_mode", default="st")
    ap.add_argument("--ce_contact", type=float, default=0.0); ap.add_argument("--p_target", type=float, default=0.4); ap.add_argument("--p_l8", type=float, default=0.6)
    ap.add_argument("--carry", type=float, default=0.5, help="share of rollout batches where token 1 is carried over from the previous plan")
    ap.add_argument("--init", default="", help="fine-tune: start from this checkpoint (.pt)"); ap.add_argument("--lr_scale", type=float, default=1.0)
    ap.add_argument("--no_mirror", action="store_true"); ap.add_argument("--root_noise", type=float, default=0.0)
    a = ap.parse_args()
    cfg = Cfg(a.cfg); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    tr, va = Data(cfg.data, "train", mirror=None if a.no_mirror else cfg.mirror), Data(cfg.data, cfg.val_split)
    if a.balance > 0:
        tr.set_balance()
    pm, ps, rm, rs = cfg.tensors(tr)
    tokz = load_tok(cfg, a.tok)
    NC, H, D = cfg.NC, cfg.heads, cfg.D
    table = tokz.code_table(NC, DEV)                       # [NC, d]
    if a.init:
        sd0 = {k: v.float() for k, v in torch.load(a.init, map_location=DEV).items()}
        net = PoseNet(D, H, NC, n_pos=sd0["pos_emb"].shape[0]).to(DEV); net.load_state_dict(sd0); print("INIT", a.init, flush=True)
    else:
        net = PoseNet(D, H, NC).to(DEV)
    params = list(net.parameters())
    opt = torch.optim.AdamW(params, 1e-3 * a.lr_scale, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3 * a.lr_scale, total_steps=a.steps, pct_start=0.05)
    evals = {}
    for L in (8, 12):
        T = L * FPT
        P, R, Lc = va.take(va.windows(T, T // 2), T)
        with torch.no_grad():
            evals[L] = (P, R, Lc, tokz.encode(P)[1])
    log, best, best_sd, t0 = [], None, None, time.time()
    Cn = cfg.C
    for it in range(a.steps + 1):
        if it % 500 == 0 or it == a.steps:
            net.eval(); res = evaluate(cfg, net, tokz, va, evals, rng); net.train()
            rec_ = dict(it=it, s=round(time.time() - t0), score_cm=res["L8_pred"]["cm"], **res)
            log.append(rec_); print(json.dumps(rec_), flush=True)
            if it >= a.steps // 2 and (best is None or rec_["score_cm"] < best["score_cm"]):
                best = rec_; best_sd = {k: v.clone() for k, v in net.state_dict().items()}
        if it == a.steps:
            break
        L = 8 if rng.random() < a.p_l8 else int(rng.integers(MINT, MAXT + 1)); T = L * FPT
        roll = L == 8 and it > 0.25 * a.steps and rng.random() < a.rollout
        k = int(rng.integers(1, 4)) if roll else 0
        P, R, _ = tr.sample(a.batch, T + RP * k, rng)
        if roll:
            net.eval()
            with torch.no_grad():
                ctx = P[:, :CTX]
                for i in range(k):
                    s = RP * i; cond = torch.zeros(len(P), T, D, device=DEV); cond[:, :CTX] = ctx
                    has = torch.zeros(len(P), T, device=DEV); has[:, :CTX] = 1
                    ptok = decode(net, R[:, s:s + T], cond, has, L)
                    ctx = tokz.decode_idx(ptok)[:, RP:RP + CTX]
            net.train()
            P, R = P[:, RP * k:], R[:, RP * k:]
            cond = torch.zeros_like(P); cond[:, :CTX] = ctx; has = torch.zeros(len(P), T, device=DEV); has[:, :CTX] = 1
        else:
            cond, has = make_cond(P, rng, a.p_target)
        with torch.no_grad():
            _, gt = tokz.encode(P)
        Rin = R + torch.randn_like(R) * a.root_noise if a.root_noise > 0 else R
        B = len(P)
        pf = torch.cos(math.pi / 2 * torch.rand(B, 1, 1, device=DEV))
        focus = torch.rand(B, L, H, device=DEV) < pf.clamp(min=1.0 / (L * H))
        r = torch.rand(B, L, H, device=DEV)
        inp = torch.where(focus & (r < 0.8), torch.full_like(gt, NC), gt)
        inp = torch.where(focus & (r >= 0.8) & (r < 0.9), torch.randint_like(gt, 0, NC), inp)
        if roll and a.carry > 0 and rng.random() < a.carry:      # seam contract: token 1 = the previous plan's token of the same frames (given, no loss)
            inp[:, 1] = ptok[:, RP // FPT + 1]; focus[:, 1] = False
        logits = net(inp, Rin, cond, has, L)
        ce = nn.functional.cross_entropy(logits.reshape(-1, NC), gt.reshape(-1), reduction="none").view(B, L, H)
        w = focus.float()
        if a.ce_contact > 0:
            con = ((P * ps + pm)[..., -Cn:] > 0.5).float().mean(-1).view(B, L, FPT).mean(-1)      # [B,L]
            w = w * (1 + a.ce_contact * con[..., None])
        loss = (ce * w).sum() / w.sum().clamp(min=1)
        if a.aux > 0:
            prob = logits.softmax(-1)
            q = prob @ table                                                                   # [B,L,H,d] expected code vector
            if a.aux_mode == "st":
                q = q + (table[prob.argmax(-1)] - q).detach()
            rec = tokz.decode_q(q.reshape(B, L, -1))
            sl = skate_losses(rec, P, R * rs + rm, cfg, pm, ps)
            loss = loss + a.aux * ((rec - P).abs().mean() + sl["slide"] + 0.5 * sl["pawvel"])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sch.step()
    torch.save(best_sd, f"{cfg.out}/{a.tag}.pt")
    json.dump(dict(args=vars(a), train_frames=tr.frames, params=sum(p.numel() for p in params), best=best, log=log), open(f"{cfg.out}/{a.tag}.json", "w"), indent=1)
    print("POSE_DONE", a.tag, json.dumps(best))


if __name__ == "__main__":
    main()
