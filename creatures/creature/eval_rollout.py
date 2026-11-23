"""Long chained rollouts driven by keyboard-like command scripts + held-out mocap checks, for one or more model sets.
python -I eval_rollout.py --cfg <creature.json> --model old=tok_l8x8,pose_small_f1_tok_l8x8,table --model v2=tok_v2,pose_v2,root_v2 [--out result.json] [--npz dir]
A model set is name=<tokenizer tag>,<pose tag>,<root net tag | table>[,seam]. "table" = the old runtime root (acceleration-limited command + height table).
"seam" = run that model with the v2 seam contract (first 2 tokens carried over on a regular replan, 2 tokens of decoder left context)."""
import sys, json, argparse
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from creature.core import Cfg, Data, DEV, FPT, CTX, load_tok, load_pose, load_root, decode
from creature import rollout as RO


def scripts(cfg):
    e = dict(walk=1.0, trot=2.2, back=-0.3, slow=0.5, turn=1.2, turn_trot=0.9, turn_stand=1.0); e.update(cfg.raw.get("eval", {}))
    W, T_ = e["walk"], e["trot"]
    keys = [(3, 0, 0, "stand"), (6, W, 0, "walk"), (4, W, e["turn"], "walk_turn"), (2, W, 0, "walk"), (6, T_, 0, "trot"), (4, T_, -e["turn_trot"], "trot_turn"),
            (2, T_, 0, "trot"), (3, W, 0, "walk"), (3, 0, 0, "stop"), (4, 0, e["turn_stand"], "turn_spot"), (2, 0, 0, "stand"), (4, e["back"], 0, "back"), (2, 0, 0, "stand"),
            (1, W, 0, "quick"), (1, 0, 0, "quick"), (1, T_, 0, "quick"), (1, W, 0, "quick"), (1, W, e["turn"], "quick"), (1, W, -e["turn"], "quick"), (1, 0, 0, "quick"),
            (4, e["slow"], 0, "slow"), (4, 0, 0, "stand")]
    steady = [(2, 0, 0, "stand"), (28, W, 0, "walk"), (28, T_, 0, "trot"), (2, 0, 0, "stand")]
    steady_b = [(1.3, 0, 0, "stand"), (14, W, 0, "walk"), (14, T_, 0, "trot"), (1.7, 0, 0, "stand"), (14, T_, 0, "trot"), (14, W, 0, "walk"), (1, 0, 0, "stand")]
    out = dict(keys60=keys, steady60=steady, steady60b=steady_b)
    if "gallop" in e:   # optional high-speed script : only when the creature cfg has eval.gallop
        G = e["gallop"]
        out["gallop60"] = [(2, 0, 0, "stand"), (5, W, 0, "walk"), (5, T_, 0, "trot"), (16, G, 0, "gallop"), (6, G, 0.5 * e["turn_trot"], "gallop_turn"),
                           (4, G, 0, "gallop"), (8, T_, 0, "trot"), (8, W, 0, "walk"), (2, 0, 0, "stop"), (4, 0, 0, "stand")]
    return out, e


def response(rt, cfg, e):
    """reaction to command steps: seconds to reach 80 % of the walk speed from a stand, seconds to stop (< 0.15 m/s), yaw change in 1.5 s after a turn reversal."""
    fps = cfg.fps; W, tr_ = e["walk"], e["turn"]
    sm = lambda x: np.convolve(np.pad(x, (7, 7), mode="edge"), np.ones(15) / 15, mode="valid")
    cmds, _ = RO.script_from([(2, 0, 0), (4, W, 0), (3, 0, 0), (3, W, tr_), (1.5, W, -tr_)], fps)
    run = rt.run(cmds); v = sm(run["root"][:, 1]); a, b, c = 2 * fps, 6 * fps, 12 * fps
    up = np.where(v[a:b] >= 0.8 * W)[0]; dn = np.where(np.abs(v[b:b + 3 * fps]) < 0.15)[0]
    yaw = np.cumsum(run["root"][:, 2]) / fps
    return dict(start_s=float(up[0] / fps) if len(up) else None, stop_s=float(dn[0] / fps) if len(dn) else None,
                reversal_yaw_change_rad=float(yaw[c + int(1.5 * fps) - 1] - yaw[c - 1]), reversal_ideal_rad=float(-1.5 * tr_))


def seg_mask(F, segs, labels, skip):
    m = np.zeros(F, bool)
    for a, b, lab, _ in segs:
        if lab in labels and b - a > skip:
            m[a + skip:b] = True
    return m


def eval_script(run, segs, cfg, low_thr):
    F = len(run["cmd"]); fps = cfg.fps; Cn = cfg.C
    local = run["feat"][:, :cfg.J * 3].reshape(F, cfg.J, 3)
    mc = run["feat"][:, -Cn:]
    res = {}
    for lab in sorted(set(s[2] for s in segs)):
        m = seg_mask(F, segs, {lab}, fps)[:-1]
        if m.any():
            res[lab] = RO.paw_stats(run["world"], cfg.contacts, fps, m, low_thr, mc)
    res["all"] = RO.paw_stats(run["world"], cfg.contacts, fps, None, low_thr, mc)
    res["seam"] = RO.seam_stats(local, run["seam"], fps)
    res["seam"]["gap_cm"] = float(run["seam_gap"].mean() * 100); res["seam"]["gap_cm_p95"] = float(np.percentile(run["seam_gap"], 95) * 100)
    ef, ey, rows = [], [], []
    for a, b, lab, c in segs:
        if b - a >= 2 * fps:
            r = run["root"][a + int(1.5 * fps):b]
            ef.append(abs(r[:, 1].mean() - c[0])); ey.append(abs(r[:, 2].mean() - c[2]))
            rows.append(dict(label=lab, cmd_fwd=c[0], cmd_yaw=c[2], fwd=float(r[:, 1].mean()), yaw=float(r[:, 2].mean()), side=float(r[:, 0].mean()), height=float(r[:, 3].mean())))
    res["track"] = dict(fwd_err=float(np.mean(ef)), yaw_err=float(np.mean(ey)), segments=rows)
    # response: frames until the forward speed first reaches 80 % of the first walk command
    return res


@torch.no_grad()
def heldout(cfg, va, tokz, pose, rootnet, low_thr, e):
    pm, ps, rm, rs = cfg.tensors(va)
    L = 8; T = L * FPT; J = cfg.J
    P, R, Lc = va.take(va.windows(T, T // 2), T)
    E, FL = [], []
    for i in range(0, len(P), 256):
        sl = slice(i, i + 256)
        cond = torch.zeros_like(P[sl]); cond[:, :CTX] = P[sl, :CTX]; has = torch.zeros(len(cond), T, device=DEV); has[:, :CTX] = 1
        rec = tokz.decode_idx(decode(pose, R[sl], cond, has, L)); rt_, _ = tokz(P[sl])
        g = slice(4, T - 4)
        loc = (rec * ps + pm)[..., :J * 3].view(len(rec), T, J, 3); locf = (rt_ * ps + pm)[..., :J * 3].view(len(rec), T, J, 3)
        E.append((loc[:, g] - Lc[sl, g]).norm(dim=-1).mean((1, 2))); FL.append((locf[:, g] - Lc[sl, g]).norm(dim=-1).mean((1, 2)))
    out = dict(L8_pred_cm=float(torch.cat(E).mean() * 100), tok_floor_cm=float(torch.cat(FL).mean() * 100), windows=len(P))
    # tokenizer contact channel reliability
    rt_all = torch.cat([tokz(P[i:i + 256])[0] for i in range(0, len(P), 256)])
    cg = (P * ps + pm)[..., -cfg.C:] > 0.5; cp = (rt_all * ps + pm)[..., -cfg.C:] > 0.5
    tp = (cg & cp).sum().item(); out["tok_contact_acc"] = float((cg == cp).float().mean()); out["tok_contact_f1"] = float(2 * tp / max(cg.sum().item() + cp.sum().item(), 1))
    # closed loop on held-out clips: GT root, and (when there is a root net) the root net driven by the clip's own command
    plans = 10; need = CTX + plans * 12 + 64
    offs = va.windows(need, 60)
    for name, rn in (("chain_gt_root", None),) + ((("chain_root_net", rootnet),) if rootnet is not None else ()):
        feats, roots, gl = [], [], []
        for i in range(0, len(offs), 128):
            f_, r_, g_ = RO.chain_gt(cfg, va, tokz, pose, offs[i:i + 128], plans=plans, rootnet=rn, cmd_fn=RO.clean_cmd)
            feats.append(f_); roots.append(r_); gl.append(g_)
        f_, r_, g_ = torch.cat(feats), torch.cat(roots), torch.cat(gl)
        loc = (f_ * ps + pm)[..., :J * 3].view(len(f_), -1, J, 3)
        err = (loc - g_).norm(dim=-1).mean(-1) * 100                       # [B, n]
        Pg, Rg, _ = va.take(offs, need); Rg = (Rg * rs + rm)[:, CTX:CTX + plans * 12]
        d = dict(joint_cm=float(err.mean()), joint_cm_last_second=float(err[:, -30:].mean()), windows=len(offs))
        if rn is not None:
            d.update(root_fwd_mae=float((r_[..., 1] - Rg[..., 1]).abs().mean()), root_side_mae=float((r_[..., 0] - Rg[..., 0]).abs().mean()),
                     root_yaw_mae=float((r_[..., 2] - Rg[..., 2]).abs().mean()), root_height_mae_cm=float((r_[..., 3] - Rg[..., 3]).abs().mean() * 100))
        # paws: world from the played local positions + the root used
        ln, rn_ = loc.cpu().numpy().astype(np.float64), r_.cpu().numpy().astype(np.float64); gn, Rn = g_.cpu().numpy().astype(np.float64), Rg.cpu().numpy().astype(np.float64)
        for band, (lo, hi) in dict(walk=(e["walk"] * 0.8, e["walk"] * 1.2), trot=(e["trot"] * 0.87, e["trot"] * 1.13)).items():
            hs_m, hs_g, low_m = [], [], []
            for b in range(len(ln)):
                fw = Rn[b, :, 1].mean()
                if lo <= fw < hi and np.abs(Rn[b, :, 2]).mean() < 0.4:
                    wm = RO.integrate(ln[b], rn_[b], cfg.fps)[:, cfg.contacts]; wg = RO.integrate(gn[b], Rn[b], cfg.fps)[:, cfg.contacts]
                    hs_m.append(np.linalg.norm((wm[1:] - wm[:-1])[..., [0, 2]], axis=-1) * cfg.fps); low_m.append(wm[:-1, :, 1] < low_thr)
                    hs_g.append(np.linalg.norm((wg[1:] - wg[:-1])[..., [0, 2]], axis=-1) * cfg.fps)
            if hs_m:
                hm, hg, lw = np.concatenate(hs_m), np.concatenate(hs_g), np.concatenate(low_m)
                d[band] = dict(windows=len(hs_m), planted_pct=float((hm < 0.2).mean() * 100), planted_pct_mocap_same_windows=float((hg < 0.2).mean() * 100),
                               slide_low_cm_s=float(hm[lw].mean() * 100) if lw.any() else None)
        out[name] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True); ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--out", default=""); ap.add_argument("--npz", default=""); ap.add_argument("--no_heldout", action="store_true")
    ap.add_argument("--rt", default="{}", help="JSON overrides of the runtime settings (replan_frames, crossfade_frames, ...)")
    a = ap.parse_args()
    cfg = Cfg(a.cfg); tr, va = Data(cfg.data, "train"), Data(cfg.data, cfg.val_split)
    con = tr.pose[tr.frame_ids()][:, -cfg.C:] > 0.5
    ph = tr.local[tr.frame_ids()][:, cfg.contacts, 1]
    low_thr = float(np.percentile(ph[con], 90))
    scr, e = scripts(cfg)
    bands = dict(walk=(e["walk"] * 0.8, e["walk"] * 1.2), trot=(e["trot"] * 0.87, e["trot"] * 1.13), slow=(e["slow"] * 0.6, e["slow"] * 1.4), stand=(-0.05, 0.05))
    if "gallop" in e: bands["gallop"] = (e["gallop"] * 0.75, e["gallop"] * 1.3)
    full = Data(cfg.data, "train"); full.clips = full.clips + va.clips
    result = dict(cfg=a.cfg, low_thr_m=low_thr, mocap=RO.mocap_reference(full, cfg, bands, low_thr), mocap_val=RO.mocap_reference(va, cfg, bands, low_thr), models={})
    print("low paw threshold %.3f m | mocap reference" % low_thr, json.dumps(result["mocap"]), flush=True)
    for spec in a.model:
        name, rest = spec.split("="); parts = rest.split(","); tok_t, pose_t, root_t = parts[:3]
        rt_kw = json.loads(a.rt)
        if len(parts) > 3 and parts[3] == "seam":          # model v2 seam contract (see rollout.RT_DEFAULT)
            rt_kw = dict(carry_tokens=1, left_ctx=2, **rt_kw)
        tokz, pose = load_tok(cfg, tok_t), load_pose(cfg, pose_t)
        rootnet = None if root_t == "table" else load_root(cfg, root_t)
        rt = RO.Runtime(cfg, tr, tokz, pose, rootnet, **rt_kw)
        res = dict(tok=tok_t, pose=pose_t, root=root_t, runtime=rt.rt, scripts={})
        for sname, segs in scr.items():
            cmds, sg = RO.script_from(segs, cfg.fps)
            run = rt.run(cmds)
            res["scripts"][sname] = eval_script(run, sg, cfg, low_thr)
            if a.npz:
                Path(a.npz).mkdir(parents=True, exist_ok=True)
                F = len(cmds); local = run["feat"][:, :cfg.J * 3].reshape(F, cfg.J, 3)
                np.savez(f"{a.npz}/{name}_{sname}.npz", world=run["world"].astype(np.float32), local=local.astype(np.float32), root_cmd=run["root"].astype(np.float32),
                         cmd=cmds, seam=run["seam"], contact=run["feat"][:, -cfg.C:].astype(np.float32), fps=cfg.fps)
        res["response"] = response(rt, cfg, e)
        if not a.no_heldout:
            res["heldout"] = heldout(cfg, va, tokz, pose, rootnet, low_thr, e)
        result["models"][name] = res
        s, k = res["scripts"]["steady60"], res["scripts"]["keys60"]
        print(f"== {name}: steady walk planted {s['walk']['planted_pct']:.1f}% slide {s['walk']['slide_low_cm_s']:.1f} cm/s | trot planted {s['trot']['planted_pct']:.1f}% slide {s['trot']['slide_low_cm_s']:.1f} cm/s"
              f" | keys walk planted {k['walk']['planted_pct']:.1f}% | seam acc {k['seam']['seam_acc']:.2f} vs {k['seam']['other_acc']:.2f} (x{k['seam']['ratio']:.2f}) gap {k['seam']['gap_cm']:.2f} cm"
              f" | track fwd {k['track']['fwd_err']:.3f} m/s yaw {k['track']['yaw_err']:.3f} rad/s", flush=True)
        if "gallop60" in res["scripts"]:
            gs = res["scripts"]["gallop60"]; gt = [r for r in gs["track"]["segments"] if r["label"] == "gallop"]
            print(f"   gallop60: gallop planted {gs['gallop']['planted_pct']:.1f}% slide {gs['gallop']['slide_low_cm_s']:.1f} cm/s | speed {gt[0]['fwd'] if gt else float('nan'):.2f} m/s for cmd {e['gallop']} | seam gap {gs['seam']['gap_cm']:.2f} cm", flush=True)
        print("   response", json.dumps(res["response"]), flush=True)
        if "heldout" in res:
            print("   heldout", json.dumps(res["heldout"]), flush=True)
    if a.out:
        clean = lambda o: {k: clean(v) for k, v in o.items()} if isinstance(o, dict) else [clean(v) for v in o] if isinstance(o, (list, tuple)) else (None if isinstance(o, float) and o != o else o)
        json.dump(clean(result), open(a.out, "w"), indent=1)
        print("EVAL_DONE", a.out)


if __name__ == "__main__":
    main()
