"""Runtime simulator (Python reference of the engine runtime contract) + rollout metrics. Skeleton-agnostic.
The simulator mirrors the C++ runtime (runtime/, dog_brain_core.cpp): 30 fps, a plan = L tokens = 4 context frames + new frames, replan every N frames
(sooner on a command change), crossfade old plan -> new plan, root integrated from the plan's root values."""
from __future__ import annotations
import numpy as np
import torch
from .core import DEV, FPT, CTX, decode, height_table, start_context

RT_DEFAULT = dict(replan_frames=12, crossfade_frames=4, idle_replan_frames=28, min_replan_gap=4, replan_delta_speed=0.25, replan_delta_yaw=0.35,
                  accel_forward=2.5, accel_side=1.5, accel_yaw=5.0, tokens=8, root_idle_eps=0.08, carry_tokens=-1, left_ctx=0)
# carry_tokens: -1 = off (every plan starts from all-mask tokens, decoded alone = model v1 contract).
#   n >= 0 (model v2 "seam contract"), on a regular (token-aligned, command unchanged) replan: token 0 (the context frames) and the next n tokens
#   are copied from the previous plan, and the tokenizer decoder gets left_ctx previous tokens in front (its output for them is dropped).


def script_from(segs, fps=30):
    """[(seconds, forward m/s, yaw rate rad/s[, side m/s][, label])] -> cmds [F,3] (forward, side, yaw), segments [(start, end, label, cmd)]."""
    cmds, out, f = [], [], 0
    for s in segs:
        dur, fwd, yaw = s[0], s[1], s[2]
        side = s[3] if len(s) > 3 and not isinstance(s[3], str) else 0.0
        label = s[-1] if isinstance(s[-1], str) else ""
        n = int(round(dur * fps)); cmds += [(fwd, side, yaw)] * n
        out.append((f, f + n, label, (fwd, side, yaw))); f += n
    return np.array(cmds, np.float32), out


class Runtime:
    def __init__(self, cfg, data, tokz, pose, rootnet=None, **rt):
        self.cfg, self.tokz, self.pose, self.rootnet = cfg, tokz, pose, rootnet
        self.rt = dict(RT_DEFAULT); self.rt.update(rt)
        self.pm, self.ps, self.rm, self.rs = data.pm.astype(np.float64), data.ps.astype(np.float64), data.rm.astype(np.float64), data.rs.astype(np.float64)
        self.hs, self.hv = height_table(data.root[data.frame_ids()])
        self.ctx0, self.root0 = start_context(data)
        self.L = self.rt["tokens"]; self.T = self.L * FPT

    def _plan(self, hist, hist_root, tgt, init=None, left=None):
        """hist [CTX,D] normalized, hist_root [CTX,4] raw, tgt (forward, side, yaw) -> plan [T,D] normalized, root [T,4] raw."""
        rt, T = self.rt, self.T
        root = np.zeros((T, 4)); root[:CTX] = hist_root
        t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=DEV)
        if self.rootnet is None:
            side, fwd, yw = hist_root[-1, 0], hist_root[-1, 1], hist_root[-1, 2]; dt = 1.0 / self.cfg.fps
            tw = lambda c, g, d: c + np.clip(g - c, -d, d)
            for i in range(CTX, T):
                fwd = tw(fwd, tgt[0], rt["accel_forward"] * dt); side = tw(side, tgt[1], rt["accel_side"] * dt); yw = tw(yw, tgt[2], rt["accel_yaw"] * dt)
                root[i] = (side, fwd, yw, np.interp(np.hypot(fwd, side), self.hs, self.hv))
        else:
            cmd = (np.array([tgt[1], tgt[0], tgt[2]]) - self.rm[:3]) / self.rs[:3]
            with torch.no_grad():
                fut = self.rootnet(t(cmd)[None], t((hist_root - self.rm) / self.rs)[None], t(hist)[None])[0].cpu().numpy().astype(np.float64)
            root[CTX:] = fut * self.rs + self.rm
        cond = torch.zeros(1, T, self.cfg.D, device=DEV); cond[0, :CTX] = t(hist)
        has = torch.zeros(1, T, device=DEV); has[0, :CTX] = 1
        tok = decode(self.pose, t((root - self.rm) / self.rs)[None], cond, has, self.L, init=init)
        with torch.no_grad():
            if left is not None:
                plan = self.tokz.decode_idx(torch.cat([left, tok], 1))[0, left.shape[1] * FPT:].cpu().numpy().astype(np.float64)
            else:
                plan = self.tokz.decode_idx(tok)[0].cpu().numpy().astype(np.float64)
        return plan, root, tok

    def run(self, cmds):
        cfg, rt, T = self.cfg, self.rt, self.T
        D, J = cfg.D, cfg.J
        hist = self.ctx0.astype(np.float64).copy()
        hist_root = np.zeros((CTX, 4)); hist_root[:, 3] = np.interp(0, self.hs, self.hv)
        if self.rootnet is not None:
            hist_root = self.root0.astype(np.float64).copy()
        plan = old = proot = ptok = None
        cursor = old_cursor = xi = xn = since = 0
        plan_tgt = np.array(cmds[0], np.float64)
        x = z = yaw = 0.0
        F = len(cmds)
        feat_o, root_o, world_o, seam = np.zeros((F, D)), np.zeros((F, 4)), np.zeros((F, J, 3)), np.zeros(F, bool)
        gaps = []        # per seam: mean joint distance (m) between the old plan's next frame and the new plan's first new frame (what the crossfade hides)
        for f in range(F):
            tgt = np.asarray(cmds[f], np.float64)
            lr = hist_root[-1]
            eps = 1e-4 if self.rootnet is None else rt["root_idle_eps"]
            idle = (not tgt.any()) and (abs(lr[0]) + abs(lr[1]) + abs(lr[2]) < eps)
            every = int(np.clip(max(rt["idle_replan_frames"], rt["replan_frames"]) if idle else rt["replan_frames"], 1, T - CTX))
            changed = abs(tgt[0] - plan_tgt[0]) > rt["replan_delta_speed"] or abs(tgt[1] - plan_tgt[1]) > rt["replan_delta_speed"] or abs(tgt[2] - plan_tgt[2]) > rt["replan_delta_yaw"]
            if plan is None or cursor >= T or since >= every or (changed and since >= max(1, rt["min_replan_gap"])):
                init = left = None
                nc, lc = int(rt["carry_tokens"]), int(rt["left_ctx"])
                a0 = cursor // FPT - 1                           # old token of the context frames
                if nc >= 0 and ptok is not None and cursor % FPT == 0 and a0 + 1 + nc <= self.L and a0 >= lc and not changed:
                    init = ptok[:, a0:a0 + 1 + nc]
                    left = ptok[:, a0 - lc:a0] if lc > 0 else None
                fresh, root, ptok = self._plan(hist, hist_root, tgt, init, left)
                if plan is not None and cursor < T:
                    d = ((plan[cursor] - fresh[CTX]) * self.ps)[:J * 3].reshape(J, 3)
                    gaps.append(float(np.linalg.norm(d, axis=-1).mean()))
                if plan is not None and rt["crossfade_frames"] > 0:
                    old, old_cursor, xi, xn = plan, cursor, 0, rt["crossfade_frames"]
                plan, proot, cursor, since, plan_tgt = fresh, root, CTX, 0, tgt.copy()
                seam[f] = True
            ft = plan[cursor].copy()
            if old is not None and xi < xn and old_cursor + xi < T:
                w = (xi + 1) / (xn + 1); o = old[old_cursor + xi]
                ft = o + (ft - o) * w
            xi += 1
            r = proot[cursor]
            hist = np.concatenate([hist[1:], ft[None]]); hist_root = np.concatenate([hist_root[1:], r[None]])
            raw = ft * self.ps + self.pm
            loc = raw[:J * 3].reshape(J, 3)
            c, s = np.cos(yaw), np.sin(yaw)
            Rm = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            feat_o[f], root_o[f], world_o[f] = raw, r, loc @ Rm.T + np.array([x, 0, z])
            v = Rm @ np.array([r[0], 0, r[1]]) / cfg.fps
            x += v[0]; z += v[2]; yaw += r[2] / cfg.fps
            cursor += 1; since += 1
        return dict(feat=feat_o, root=root_o, world=world_o, seam=seam, cmd=np.asarray(cmds, np.float64), seam_gap=np.array(gaps))


def integrate(local, root, fps):
    """local [F,J,3] + root [F,4] -> world [F,J,3] (same integration as the runtime)."""
    yaw = np.concatenate([[0], np.cumsum(root[:-1, 2] / fps)])
    c, s = np.cos(yaw), np.sin(yaw)
    vx = (c * root[:, 0] + s * root[:, 1]) / fps; vz = (-s * root[:, 0] + c * root[:, 1]) / fps
    x = np.concatenate([[0], np.cumsum(vx[:-1])]); z = np.concatenate([[0], np.cumsum(vz[:-1])])
    w = np.empty_like(local)
    w[..., 0] = c[:, None] * local[..., 0] + s[:, None] * local[..., 2] + x[:, None]
    w[..., 1] = local[..., 1]
    w[..., 2] = -s[:, None] * local[..., 0] + c[:, None] * local[..., 2] + z[:, None]
    return w


def paw_stats(world, contacts, fps, mask=None, low_thr=0.04, model_contact=None):
    """world [F,J,3]. mask [F-1] bool = frames to count. Returns planted % (paw horizontal speed < 0.2 m/s), slide (cm/s) of low paws, slide on model contact."""
    p = world[:, contacts]
    hs = np.linalg.norm((p[1:] - p[:-1])[..., [0, 2]], axis=-1) * fps          # [F-1,C]
    low = p[:-1, :, 1] < low_thr
    m = np.ones(len(hs), bool) if mask is None else mask[:len(hs)]
    hs, low = hs[m], low[m]
    out = dict(n=int(hs.size), planted_pct=float((hs < 0.2).mean() * 100) if hs.size else float("nan"),
               low_pct=float(low.mean() * 100) if hs.size else float("nan"),
               slide_low_cm_s=float(hs[low].mean() * 100) if low.any() else float("nan"), paw_speed=float(hs.mean()) if hs.size else float("nan"))
    if model_contact is not None:
        mc = model_contact[:-1][m] > 0.5
        out["contact_pct"] = float(mc.mean() * 100) if mc.size else float("nan")
        out["slide_contact_cm_s"] = float(hs[mc].mean() * 100) if mc.any() else float("nan")
    return out


def seam_stats(local, seam, fps, xfade=4):
    """Joint acceleration (m/s^2, mean over joints of |x[t+1]-2x[t]+x[t-1]| * fps^2) at plan seams vs elsewhere."""
    acc = np.linalg.norm(local[2:] - 2 * local[1:-1] + local[:-2], axis=-1).mean(-1) * fps * fps      # frame t = 1..F-2
    F = len(local); near = np.zeros(F, bool)
    for f in np.where(seam)[0]:
        near[max(f - 1, 0):f + xfade + 1] = True
    near = near[1:-1]
    a_s, a_o = acc[near], acc[~near]
    return dict(seam_acc=float(a_s.mean()), other_acc=float(a_o.mean()), ratio=float(a_s.mean() / max(a_o.mean(), 1e-9)), seam_p95=float(np.percentile(a_s, 95)),
                other_p95=float(np.percentile(a_o, 95)), seams=int(seam.sum()))


def mocap_reference(data, cfg, bands, low_thr):
    """planted % / slide for mocap frames whose (1 s smoothed) forward speed is in a band. bands {name: (lo, hi)}."""
    acc = {k: dict(hs=[], low=[], accj=[], con=[]) for k in bands}
    for s, l in data.clips:
        if l < 40:
            continue
        loc, root = data.local[s:s + l].astype(np.float64), data.root[s:s + l].astype(np.float64)
        w = integrate(loc, root, cfg.fps); p = w[:, cfg.contacts]
        hs = np.linalg.norm((p[1:] - p[:-1])[..., [0, 2]], axis=-1) * cfg.fps
        k = 15; sm = lambda a: np.convolve(np.pad(a, (k, k), mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), mode="valid")
        fw, yw = sm(root[:, 1])[:-1], sm(np.abs(root[:, 2]))[:-1]
        a = np.linalg.norm(loc[2:] - 2 * loc[1:-1] + loc[:-2], axis=-1).mean(-1) * cfg.fps ** 2
        for name, (lo, hi) in bands.items():
            m = (fw >= lo) & (fw < hi) & (yw < 0.4)
            acc[name]["hs"].append(hs[m]); acc[name]["low"].append(p[:-1, :, 1][m] < low_thr); acc[name]["accj"].append(a[m[1:]])
            acc[name]["con"].append(data.pose[s:s + l - 1, -cfg.C:][m] > 0.5)
    out = {}
    for name, d in acc.items():
        hs, low, a, con = np.concatenate(d["hs"]), np.concatenate(d["low"]), np.concatenate(d["accj"]), np.concatenate(d["con"])
        out[name] = dict(frames=int(len(hs)), planted_pct=float((hs < 0.2).mean() * 100) if hs.size else None,
                         slide_low_cm_s=float(hs[low].mean() * 100) if low.any() else None, low_pct=float(low.mean() * 100) if hs.size else None,
                         joint_acc=float(a.mean()) if a.size else None, contact_pct=float(con.mean() * 100) if con.size else None,
                         slide_contact_cm_s=float(hs[con].mean() * 100) if con.any() else None)
    return out


@torch.no_grad()
def chain_gt(cfg, data, tokz, pose, offs, plans=8, replan=12, xfade=4, L=8, rootnet=None, cmd_fn=None):
    """Closed-loop chained generation that follows held-out clips: each plan gets the GT root path (or the root net's path when rootnet is set;
    cmd_fn(Rraw window [B,64,4]) -> clean command), context = the model's own last 4 played frames. Returns played features [B,plans*replan,D]
    normalized, the root used [B,plans*replan,4] raw, GT local [B,plans*replan,J,3]. Windows start at offs and need CTX + plans*replan + 64 frames."""
    T = L * FPT; n = plans * replan
    P, R, Lc = data.take(offs, CTX + n + 64)
    pm, ps, rm, rs = cfg.tensors(data)
    B = len(offs)
    ctx = P[:, :CTX].clone(); rh = R[:, :CTX].clone()
    out, rout, old = [], [], None
    for k in range(plans):
        a = k * replan
        if rootnet is None:
            root = R[:, a:a + T]
        else:
            cmd = cmd_fn((R[:, a:a + 64] * rs + rm))
            root = torch.cat([rh, rootnet((cmd - rm[:3]) / rs[:3], rh, ctx)], 1)
        cond = torch.zeros(B, T, cfg.D, device=DEV); cond[:, :CTX] = ctx
        has = torch.zeros(B, T, device=DEV); has[:, :CTX] = 1
        plan = tokz.decode_idx(decode(pose, root, cond, has, L))
        play = plan[:, CTX:CTX + replan].clone()
        if old is not None and xfade > 0:
            w = ((torch.arange(xfade, device=DEV) + 1) / (xfade + 1))[None, :, None]
            play[:, :xfade] = old[:, CTX + replan:CTX + replan + xfade] * (1 - w) + play[:, :xfade] * w
        out.append(play); rout.append(root[:, CTX:CTX + replan] * rs + rm)
        ctx = play[:, -CTX:]; rh = root[:, replan:replan + CTX]; old = plan
    return torch.cat(out, 1), torch.cat(rout, 1), Lc[:, CTX:CTX + n]


def clean_cmd(Rraw, a=8, b=24):
    """the command a root window stands for: mean [vx, vz, yaw rate] over frames a..b after the plan start."""
    return Rraw[:, a:b, :3].mean(1)
