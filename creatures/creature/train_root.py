"""Train the root net: command + root history + pose context -> root values [vx, vz, yaw rate, height] of the next 28 frames.
python -I train_root.py --cfg <creature.json> --tag root_v2 [--tok tok_v2] [--pose pose_v2] [--steps 8000]
Commands are synthetic, derived from the mocap root track: the mean of the TRUE [vx, vz, yaw rate] over a random window ahead of the plan start (start 4-12 frames, length 10-22 frames: the command leads the motion by about 0.5 s),
then made keyboard-like (noise, quantized steps, dead zone, side speed dropped). So command != trajectory: the net learns how the real animal
accelerates, turns and bobs (step rhythm from the pose context) toward a command.
Context variety (so it works on the model's own output): mocap frames, tokenizer round trip, pose-net generation; root history noise;
3 plans unrolled on the net's own root history (differentiable)."""
import sys, json, time, argparse
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from creature.core import Cfg, Data, RootNet, DEV, FPT, CTX, load_tok, load_pose, decode


def synth_cmd(Rraw, rng, lo=(4, 12), ln=(10, 22)):
    """Rraw [B,T>=64,4] raw root values from the plan start. Returns the keyboard-like command [B,3] raw = [vx side, vz forward, yaw rate]."""
    B = Rraw.shape[0]
    cs = torch.cat([torch.zeros(B, 1, 3, device=Rraw.device), Rraw[:, :64, :3].cumsum(1)], 1)
    s = torch.tensor(rng.integers(lo[0], lo[1] + 1, B), device=Rraw.device); e = s + torch.tensor(rng.integers(ln[0], ln[1] + 1, B), device=Rraw.device)
    ar = torch.arange(B, device=Rraw.device)
    out = (cs[ar, e] - cs[ar, s]) / (e - s)[:, None].float()
    g = torch.Generator(device="cpu"); g.manual_seed(int(rng.integers(1 << 30)))
    r = lambda *sh: torch.rand(*sh, generator=g).to(Rraw.device)
    n = lambda *sh: torch.randn(*sh, generator=g).to(Rraw.device)
    out = out + n(B, 3) * (0.05 * out.abs() + torch.tensor([0.02, 0.03, 0.05], device=Rraw.device))
    step = torch.tensor([0.1, 0.25, 0.3], device=Rraw.device)
    q = torch.round(out / step) * step
    out = torch.where(r(B, 1) < 0.4, q, out)
    dead = out.abs() < torch.tensor([0.06, 0.1, 0.12], device=Rraw.device)
    out = torch.where(dead & (r(B, 1) < 0.7), torch.zeros_like(out), out)
    out[:, 0] = torch.where(r(B) < 0.5, torch.zeros(B, device=Rraw.device), out[:, 0])       # a keyboard has no side command most of the time
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True); ap.add_argument("--tag", default="root_v2"); ap.add_argument("--tok", default=""); ap.add_argument("--pose", default="")
    ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--balance", type=float, default=1.0, help="1 = speed-balanced window sampling (Data.set_balance), 0 = uniform over frames"); ap.add_argument("--unroll", type=int, default=3)
    ap.add_argument("--track", type=float, default=0.3, help="weight of the command tracking term (mean of the late predicted forward speed / yaw rate = command)")
    ap.add_argument("--width", type=int, default=512); ap.add_argument("--drop", type=float, default=0.2)
    ap.add_argument("--init", default="", help="fine-tune: start from this checkpoint (.pt)"); ap.add_argument("--lr_scale", type=float, default=1.0)
    a = ap.parse_args()
    cfg = Cfg(a.cfg); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    tr, va = Data(cfg.data, "train", mirror=cfg.mirror), Data(cfg.data, cfg.val_split)
    if a.balance > 0:
        tr.set_balance()
    pm, ps, rm, rs = cfg.tensors(tr)
    tokz = load_tok(cfg, a.tok) if a.tok else None
    pose = load_pose(cfg, a.pose) if a.pose else None
    L = 8; T = L * FPT; fut = T - CTX; RP = 12
    net = RootNet(cfg.D, fut, a.width, a.drop).to(DEV)
    if a.init:
        net.load_state_dict(torch.load(a.init, map_location=DEV)); print("INIT", a.init, flush=True)
    opt = torch.optim.AdamW(net.parameters(), 1e-3 * a.lr_scale, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3 * a.lr_scale, total_steps=a.steps, pct_start=0.05)
    need = RP * (a.unroll - 1) + 64 + CTX

    def contexts(P, R):
        """pose context per unrolled plan [K][B,CTX,D]: mocap / tokenizer round trip / pose-net generation (GT root)."""
        mode = rng.random()
        if tokz is None or mode < 0.34:
            src = P
        elif pose is None or mode < 0.67:
            with torch.no_grad():
                n4 = (P.shape[1] // 4) * 4; src = torch.cat([tokz(P[:, :n4])[0], P[:, n4:]], 1)
        else:
            with torch.no_grad():
                ctx = P[:, :CTX]; outs = [ctx]
                for k in range(a.unroll - 1):
                    s = k * RP; cond = torch.zeros(len(P), T, cfg.D, device=DEV); cond[:, :CTX] = ctx
                    has = torch.zeros(len(P), T, device=DEV); has[:, :CTX] = 1
                    rec = tokz.decode_idx(decode(pose, R[:, s:s + T], cond, has, L)); ctx = rec[:, RP:RP + CTX]; outs.append(ctx)
            return outs
        return [src[:, k * RP:k * RP + CTX] for k in range(a.unroll)]

    def run(P, R, train=True):
        ctxs = contexts(P, R) if train else [P[:, k * RP:k * RP + CTX] for k in range(a.unroll)]
        rh = R[:, :CTX]
        if train:
            rh = rh + torch.randn_like(rh) * 0.05 * float(rng.random() < 0.5)
        loss, maes = 0.0, []
        Rraw = R * rs + rm
        for k in range(a.unroll):
            s = k * RP
            cmd = synth_cmd(Rraw[:, s:], rng) if train else Rraw[:, s + 8:s + 24, :3].mean(1)
            cn = (cmd - rm[:3]) / rs[:3]
            pred = net(cn, rh, ctxs[k])
            gt = R[:, s + CTX:s + T]
            loss = loss + (pred - gt).abs().mean() + (pred[:, 1:] - pred[:, :-1] - gt[:, 1:] + gt[:, :-1]).abs().mean()
            if train and a.track > 0:
                loss = loss + a.track * (pred[:, 10:, 1:3].mean(1) - cn[:, 1:3]).abs().mean()
            maes.append(((pred - gt) * rs).abs().mean((0, 1)))
            rh = pred[:, RP - CTX:RP]              # the next plan starts from the net's own root values
        return loss / a.unroll, torch.stack(maes).mean(0)

    vo = va.windows(need, 30); Pv, Rv, _ = va.take(vo, need)
    log, t0, best = [], time.time(), None
    for it in range(a.steps + 1):
        if it % 500 == 0 or it == a.steps:
            net.eval()
            with torch.no_grad():
                vl, vm = run(Pv, Rv, train=False)
            net.train()
            log.append(dict(it=it, s=round(time.time() - t0), val_loss=float(vl), val_mae_vx=float(vm[0]), val_mae_vz=float(vm[1]), val_mae_yaw=float(vm[2]), val_mae_h_cm=float(vm[3]) * 100))
            print(json.dumps(log[-1]), flush=True)
            if it >= a.steps // 3 and (best is None or float(vl) < best[0]):
                best = (float(vl), it, {k: v.clone() for k, v in net.state_dict().items()})
        if it == a.steps:
            break
        P, R, _ = tr.sample(256, need, rng)
        loss, _ = run(P, R)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step()
    torch.save(best[2], f"{cfg.out}/{a.tag}.pt")
    json.dump(dict(args=vars(a), log=log, best_it=best[1], best_val=best[0], fut=fut, params=sum(p.numel() for p in net.parameters())), open(f"{cfg.out}/{a.tag}.json", "w"), indent=1)
    print("ROOT_DONE", a.tag, json.dumps(log[-1]))


if __name__ == "__main__":
    main()
