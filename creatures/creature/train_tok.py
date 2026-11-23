"""Train the creature FSQ tokenizer with foot-skate aware losses.
python -I train_tok.py --cfg <creature.json> --tag tok_v2 [--steps 8000] [--skate 1.0]
Loss = feature L1 + 0.5 * frame-difference L1 (the old loss)
     + skate * ( paw slide on GT contact frames + 0.5 * paw world-velocity error + 5 * paw position error + contact channel error ).
The paw world velocity is computed from the DECODED positions and the true root motion (what the engine will show), not from the velocity channels."""
import sys, json, time, argparse
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from creature.core import Cfg, Data, Tokenizer, DEV
from creature.losses import skate_losses, local_pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True); ap.add_argument("--tag", default="tok_v2"); ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--skate", type=float, default=1.0); ap.add_argument("--no_mirror", action="store_true"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--init", default="", help="fine-tune: start from this checkpoint (.pt)"); ap.add_argument("--lr_scale", type=float, default=1.0); ap.add_argument("--balance", type=float, default=0.0, help="1 = speed-balanced window sampling (Data.set_balance), 0 = uniform over frames")
    a = ap.parse_args()
    cfg = Cfg(a.cfg); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    tr, va = Data(cfg.data, "train", mirror=None if a.no_mirror else cfg.mirror), Data(cfg.data, cfg.val_split)
    if a.balance > 0:
        tr.set_balance()
    pm, ps, rm, rs = cfg.tensors(tr)
    net = Tokenizer(cfg.D, cfg.levels, cfg.heads).to(DEV)
    if a.init:
        net.load_state_dict(torch.load(a.init, map_location=DEV)); print("INIT", a.init, flush=True)
    opt = torch.optim.AdamW(net.parameters(), 3e-4 * a.lr_scale, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3 * a.lr_scale, total_steps=a.steps, pct_start=0.05)
    vo = va.windows(64, 32); t0 = time.time(); log = []
    for it in range(a.steps + 1):
        if it % 1000 == 0 or it == a.steps:
            net.eval()
            with torch.no_grad():
                E, S, n, used = 0.0, {}, 0, None
                for i in range(0, len(vo), 256):
                    Pv, Rv, Lv = va.take(vo[i:i + 256], 64)
                    rv, idx = net(Pv)
                    E += float((local_pos(rv * ps + pm, cfg.J) - Lv).norm(dim=-1).mean()) * 100 * len(Pv); n += len(Pv)
                    sl = skate_losses(rv, Pv, Rv * rs + rm, cfg, pm, ps)
                    for k, v in sl.items():
                        S[k] = S.get(k, 0.0) + float(v) * len(Pv)
                    u = [set(idx[..., h].unique().tolist()) for h in range(cfg.heads)]
                    used = u if used is None else [x | y for x, y in zip(used, u)]
            net.train()
            log.append(dict(it=it, s=round(time.time() - t0), val_cm=E / n, codes_used=[len(x) for x in used], **{"val_" + k: v / n for k, v in S.items()}))
            print(json.dumps(log[-1]), flush=True)
        if it == a.steps:
            break
        P, R, _ = tr.sample(128, 64, rng)
        rec, _ = net(P)
        loss = (rec - P).abs().mean() + 0.5 * ((rec[:, 1:] - rec[:, :-1]) - (P[:, 1:] - P[:, :-1])).abs().mean()
        if a.skate > 0:
            sl = skate_losses(rec, P, R * rs + rm, cfg, pm, ps)
            loss = loss + a.skate * (sl["slide"] + 0.5 * sl["pawvel"] + 5 * sl["pawpos"] + sl["contact"])
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    Path(cfg.out).mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), f"{cfg.out}/{a.tag}.pt")
    json.dump(dict(args=vars(a), log=log, D=cfg.D, train_frames=tr.frames, val_frames=va.frames), open(f"{cfg.out}/{a.tag}.json", "w"), indent=1)
    print("TOK_DONE", a.tag, json.dumps(log[-1]))


if __name__ == "__main__":
    main()
