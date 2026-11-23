"""Creature MotionBricks core: config, data, FSQ tokenizer, masked pose net, root net, MaskGIT decode. Skeleton-agnostic."""
from __future__ import annotations
import json, math, os
import numpy as np
import torch
from torch import nn

DEV = os.environ.get("CREATURE_DEV") or ("cuda" if torch.cuda.is_available() else "cpu")
FPT = 4            # frames per token
CTX = 4            # context frames of a plan
MINT, MAXT = 6, 16  # token counts the pose net is trained for


class Cfg:
    """Creature config JSON -> sizes and paths. skeleton may be inline or a path to a skeleton spec JSON.
    Relative paths ("data", "out", "skeleton", "export.rig_spec") are relative to the folder of the config file.
    Env CREATURE_DATA / CREATURE_OUT replace "data" / "out"."""

    def __init__(self, path):
        c = json.load(open(path, encoding="utf-8")); self.raw, self.path = c, path
        base = os.path.dirname(os.path.abspath(path))
        self.rel = lambda p: p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))
        sk = c["skeleton"]
        if isinstance(sk, str):
            sk = json.load(open(self.rel(sk), encoding="utf-8"))
        self.skeleton = sk
        self.joints, self.parents = list(sk["joints"]), [int(x) for x in sk["parents"]]
        self.contacts = [int(x) for x in sk.get("contacts", sk.get("feet"))]
        self.root, self.front = int(sk.get("root", 0)), int(sk.get("front", 1))
        self.J, self.C = len(self.joints), len(self.contacts)
        self.D = 6 * self.J + self.C
        self.name = c["name"]
        self.data = os.environ.get("CREATURE_DATA") or self.rel(c["data"])
        self.out = os.environ.get("CREATURE_OUT") or self.rel(c["out"])
        self.val_split = c.get("val_split", "val")     # name of the held-out split in the dataset
        self.fps = int(c.get("fps", 30)); self.heads = int(c.get("heads", 5))
        self.levels = tuple(int(x) for x in c.get("levels", (8, 8))); self.NC = int(np.prod(self.levels))
        self.MASK = self.NC
        self.mirror = self._mirror(sk)

    def _mirror(self, sk):
        """left/right mirror as index maps for the raw features: (feature permutation [D], sign [D]) or None. skeleton "mirror": joint permutation,
        or found from the joint names (Left*/Right*, L1_*/R1_*, *_L/*_R)."""
        m = sk.get("mirror")
        if m is None:
            idx = {n: i for i, n in enumerate(self.joints)}; m = list(range(self.J))
            for n, i in idx.items():
                cands = []
                for a, b in (("Left", "Right"), ("Right", "Left"), ("L", "R"), ("R", "L"), ("left", "right"), ("right", "left")):
                    if n.startswith(a) and (len(a) > 1 or (len(n) > 1 and (n[1].isdigit() or n[1] == "_"))):
                        cands.append(b + n[len(a):])
                for a, b in (("_L", "_R"), ("_R", "_L"), ("_l", "_r"), ("_r", "_l")):
                    if n.endswith(a):
                        cands.append(n[:-len(a)] + b)
                for c in cands:
                    if c in idx:
                        m[i] = idx[c]; break
            if m == list(range(self.J)):
                return None
        J, C = self.J, self.C
        perm = np.arange(self.D); sign = np.ones(self.D, np.float32)
        for blk in (0, 3 * J):
            for j in range(J):
                for k in range(3):
                    perm[blk + j * 3 + k] = blk + m[j] * 3 + k
                sign[blk + j * 3] = -1
        for k, j in enumerate(self.contacts):
            perm[6 * J + k] = 6 * J + self.contacts.index(m[j])
        return perm, sign, np.array(m)

    def tensors(self, data):
        f = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=DEV)
        return f(data.pm), f(data.ps), f(data.rm), f(data.rs)


class Data:
    def __init__(self, path, split="train", frac=1.0, seed=0, mirror=None):
        """mirror: Cfg.mirror -> sample() flips half of the windows left/right (x -> -x, L/R joints swapped, side speed and yaw rate negated)."""
        self.mirror = mirror
        z = np.load(path, allow_pickle=True)
        starts, lens, sp = z["starts"], z["lens"], z["split"]
        keep = [i for i in range(len(starts)) if sp[i] == split]
        if frac < 1.0:
            rng = np.random.default_rng(seed); order = rng.permutation(keep)
            tot = sum(lens[i] for i in keep); acc, sel = 0, []
            for i in order:
                if acc >= frac * tot:
                    break
                sel.append(i); acc += lens[i]
            keep = sorted(sel)
        self.clips = [(int(starts[i]), int(lens[i])) for i in keep]
        self.names = [str(z["names"][i]) for i in keep]
        self.pose, self.root, self.local = z["pose"], z["root"], z["local"]
        self.pm, self.ps, self.rm, self.rs = z["pose_mean"], z["pose_std"], z["root_mean"], z["root_std"]
        self.J = self.local.shape[1]
        self.frames = sum(l for _, l in self.clips)

    def _t(self, offs, T, flip=None):
        P = np.stack([self.pose[o:o + T] for o in offs]); R = np.stack([self.root[o:o + T] for o in offs]); L = np.stack([self.local[o:o + T] for o in offs])
        if flip is not None and flip.any():
            perm, sign, jm = self.mirror
            P[flip] = P[flip][:, :, perm] * sign
            R[flip] = R[flip] * np.array([-1, 1, -1, 1], np.float32)
            L[flip] = L[flip][:, :, jm] * np.array([-1, 1, 1], np.float32)
        P = (P - self.pm) / self.ps; R = (R - self.rm) / self.rs
        f = lambda a: torch.tensor(a, dtype=torch.float32, device=DEV)
        return f(P), f(R), f(L)

    def set_balance(self, base=0.25, vmax=3.0, yaw=0.3):
        """speed-balanced window sampling: a window's weight = mean over its frames of (base + min(root speed, vmax) + yaw * |yaw rate|).
        Mocap is mostly standing; this shows the fast gaits and turns more often (same data, no new frames)."""
        w = base + np.minimum(np.hypot(self.root[:, 0], self.root[:, 1]), vmax) + yaw * np.minimum(np.abs(self.root[:, 2]), 3.0)
        self._bal = np.concatenate([[0.0], np.cumsum(w.astype(np.float64))]); self._bal_cache = {}

    def sample(self, B, T, rng):
        flip = rng.random(B) < 0.5 if self.mirror is not None else None
        if getattr(self, "_bal", None) is not None:
            if T not in self._bal_cache:
                st = np.concatenate([np.arange(s, s + l - T + 1) for s, l in self.clips if l >= T])
                p = self._bal[st + T] - self._bal[st]
                self._bal_cache[T] = (st, np.cumsum(p / p.sum()))
            st, cdf = self._bal_cache[T]
            return self._t(st[np.minimum(np.searchsorted(cdf, rng.random(B)), len(st) - 1)], T, flip)
        ok = [c for c in self.clips if c[1] >= T]
        w = np.array([c[1] - T + 1 for c in ok], np.float64); w /= w.sum()
        ids = rng.choice(len(ok), B, p=w)
        return self._t([ok[k][0] + int(rng.integers(0, ok[k][1] - T + 1)) for k in ids], T, flip)

    def windows(self, T, stride):
        return [s + o for s, l in self.clips for o in range(0, l - T + 1, stride)]

    def take(self, offs, T):
        return self._t(offs, T)

    def frame_ids(self):
        return np.concatenate([np.arange(s, s + l) for s, l in self.clips])


class Res(nn.Module):
    def __init__(self, c, d=1):
        super().__init__()
        self.b = nn.Sequential(nn.ReLU(), nn.Conv1d(c, c, 3, padding=d, dilation=d), nn.ReLU(), nn.Conv1d(c, c, 1))

    def forward(self, x):
        return x + self.b(x)


class FSQ(nn.Module):
    def __init__(self, levels):
        super().__init__()
        self.L = torch.tensor(levels, dtype=torch.float32); self.d = len(levels)
        self.basis = torch.tensor(np.cumprod([1] + list(levels[:-1])), dtype=torch.long)

    def forward(self, z):
        L = self.L.to(z.device)
        u = (torch.tanh(z) + 1) / 2 * (L - 1)
        q = u + (torch.round(u) - u).detach()
        idx = (torch.round(u).long() * self.basis.to(z.device)).sum(-1)
        return q / (L - 1) * 2 - 1, idx

    def from_idx(self, idx):
        L = self.L.to(idx.device).long(); b = self.basis.to(idx.device)
        digits = (idx[..., None] // b) % L
        return digits.float() / (L - 1).float() * 2 - 1


class Tokenizer(nn.Module):
    """[B,T,D] normalized pose features -> [B,T/4,heads] token ids -> [B,T,D]. """

    def __init__(self, D, levels=(8, 8), heads=5, width=256):
        super().__init__()
        self.fsq = FSQ(levels); d = self.fsq.d; self.heads = heads
        self.enc = nn.Sequential(nn.Conv1d(D, width, 3, padding=1), Res(width), Res(width, 3),
                                 nn.Conv1d(width, width, 4, 2, 1), Res(width), Res(width, 3),
                                 nn.Conv1d(width, width, 4, 2, 1), Res(width), Res(width, 3),
                                 nn.Conv1d(width, heads * d, 3, padding=1))
        self.dec = nn.Sequential(nn.Conv1d(heads * d, width, 3, padding=1), Res(width), Res(width, 3),
                                 nn.Upsample(scale_factor=2), nn.Conv1d(width, width, 3, padding=1), Res(width), Res(width, 3),
                                 nn.Upsample(scale_factor=2), nn.Conv1d(width, width, 3, padding=1), Res(width), Res(width, 3),
                                 nn.ReLU(), nn.Conv1d(width, D, 3, padding=1))

    def encode(self, x):
        z = self.enc(x.transpose(1, 2)).transpose(1, 2)
        B, N, _ = z.shape
        q, idx = self.fsq(z.view(B, N, self.heads, self.fsq.d))
        return q.view(B, N, -1), idx

    def decode_q(self, q):
        return self.dec(q.transpose(1, 2)).transpose(1, 2)

    def decode_idx(self, idx):
        B, N, _ = idx.shape
        return self.decode_q(self.fsq.from_idx(idx).view(B, N, -1))

    def code_table(self, NC, dev):
        return self.fsq.from_idx(torch.arange(NC, device=dev))          # [NC, d]

    def forward(self, x):
        q, idx = self.encode(x)
        return self.decode_q(q), idx


class PoseNet(nn.Module):
    """Masked pose-token transformer (small, from scratch)."""

    def __init__(self, D, heads=5, NC=64, W=256, NH=4, NL=4, FF=512, n_pos=64):
        super().__init__()
        self.H, self.NC = heads, NC
        self.tok_emb = nn.Embedding(heads * (NC + 1), 32)
        self.proj_tok = nn.Sequential(nn.Linear(heads * 32, 640), nn.ReLU(), nn.Linear(640, 640))
        self.proj_pose = nn.Linear(D, 640 // FPT)
        self.proj_root = nn.Linear(4 * FPT, 256)
        self.ntok = nn.Embedding(MAXT - MINT + 1, 128)
        self.proj_in = nn.Sequential(nn.Linear(640 + 256 + 128, W), nn.ReLU())
        layer = nn.TransformerEncoderLayer(W, NH, FF, batch_first=True)
        self.tf = nn.TransformerEncoder(layer, NL, enable_nested_tensor=False)
        self.out = nn.Linear(W, heads * NC)
        self.pos_emb = nn.Parameter(torch.randn(n_pos, W) * 0.02)

    def forward(self, tokens, root, cond, has_cond, L):
        B, N, H = tokens.shape
        off = torch.arange(H, device=tokens.device) * (self.NC + 1)
        te = self.proj_tok(self.tok_emb(tokens + off).view(B, N, -1)).view(B, N * FPT, -1)
        ce = self.proj_pose(cond)
        hc = has_cond[..., None]
        pe = (ce * hc + te * (1 - hc)).reshape(B, N, 640)
        re = self.proj_root(root.reshape(B, N, -1))
        ne = self.ntok(torch.full((B, N), L - MINT, device=tokens.device, dtype=torch.long))
        x = self.proj_in(torch.cat([pe, re, ne], -1)) + self.pos_emb[:N]
        return self.out(self.tf(x)).view(B, N, H, self.NC)


class RootNet(nn.Module):
    """Command + root history + pose context -> root values of the next segment.
    cmd [B,3] normalized like root[:, :3] = [vx side, vz forward, yaw rate]; root_hist [B,CTX,4] normalized; ctx [B,CTX,D] normalized pose features.
    Returns [B,fut,4] normalized root values [vx, vz, yaw rate, height] for the frames after the context."""

    def __init__(self, D, fut=28, width=512, drop=0.2):
        super().__init__()
        self.fut = fut
        self.pose = nn.Sequential(nn.Dropout(drop), nn.Linear(CTX * D, width), nn.ReLU(), nn.Dropout(drop))
        self.net = nn.Sequential(nn.Linear(width + 3 + CTX * 4, width), nn.ReLU(), nn.Dropout(drop), nn.Linear(width, width), nn.ReLU(), nn.Dropout(drop),
                                 nn.Linear(width, fut * 4))

    def forward(self, cmd, root_hist, ctx):
        B = cmd.shape[0]
        x = torch.cat([self.pose(ctx.reshape(B, -1)), cmd, root_hist.reshape(B, -1)], -1)
        return root_hist[:, -1:, :] + self.net(x).view(B, self.fut, 4)


def keep_schedule(n, steps=6, known=0):
    """number of known token slots after each MaskGIT step (n slots, 'known' of them given at the start)."""
    return [known + int(math.ceil((n - known) * (1 - math.cos(math.pi / 2 * (s + 1) / steps)))) for s in range(steps)]


@torch.no_grad()
def decode(net, root, cond, has, L, steps=6, init=None):
    """MaskGIT confidence decoding (argmax). init [B,n,H] (optional): the first n tokens are given (carried over from the previous plan)."""
    B = root.shape[0]; H, NC = net.H, net.NC
    tok = torch.full((B, L, H), NC, device=root.device, dtype=torch.long)
    known = torch.zeros(B, L, H, dtype=torch.bool, device=root.device)
    k0 = 0
    if init is not None:
        k0 = init.shape[1]; tok[:, :k0] = init; known[:, :k0] = True
    ks = keep_schedule(L * H, steps, k0 * H)
    for s in range(steps):
        logits = net(tok, root, cond, has, L)
        conf, pred = logits.softmax(-1).max(-1)
        conf = torch.where(known, torch.full_like(conf, 2.0), conf)
        thr = conf.view(B, -1).sort(-1, descending=True).values[:, ks[s] - 1:ks[s]]
        new = (conf.view(B, -1) >= thr).view(B, L, H) & ~known
        tok = torch.where(new, pred, tok); known |= new
    return tok


def load_tok(cfg, tag):
    t = Tokenizer(cfg.D, cfg.levels, cfg.heads).to(DEV)
    t.load_state_dict(torch.load(f"{cfg.out}/{tag}.pt", map_location=DEV)); t.eval()
    for p in t.parameters():
        p.requires_grad_(False)
    return t


def load_pose(cfg, tag):
    sd = {k: v.float() for k, v in torch.load(f"{cfg.out}/{tag}.pt", map_location=DEV).items()}
    net = PoseNet(cfg.D, cfg.heads, cfg.NC, n_pos=sd["pos_emb"].shape[0]).to(DEV)
    net.load_state_dict(sd); net.eval()
    return net


def load_root(cfg, tag):
    sd = torch.load(f"{cfg.out}/{tag}.pt", map_location=DEV)
    net = RootNet(cfg.D, fut=sd["net.6.bias"].shape[0] // 4, width=sd["net.6.weight"].shape[1]).to(DEV)
    net.load_state_dict(sd); net.eval()
    return net


def height_table(root_raw, nb=12, vmax=3.0):
    """median root height per speed bin (the old runtime's root height)."""
    spd = np.hypot(root_raw[:, 0], root_raw[:, 1]); h = root_raw[:, 3]
    bins = np.linspace(0, vmax, nb + 1)
    hb = np.array([np.median(h[(spd >= a) & (spd < b)]) if ((spd >= a) & (spd < b)).sum() > 50 else np.nan for a, b in zip(bins[:-1], bins[1:])])
    hb = np.where(np.isnan(hb), np.nanmedian(hb), hb)
    return (bins[:-1] + bins[1:]) / 2, hb


def start_context(data):
    """4 standing frames from the data (low speed, normal height), normalized."""
    spd = np.hypot(data.root[:, 0], data.root[:, 1]); h = data.root[:, 3]
    ids = data.frame_ids()
    ok = ids[(spd[ids] < 0.05) & (np.abs(h[ids] - np.median(h[ids])) < 0.02)]
    ends = set(s + l for s, l in data.clips)
    ok = [int(i) for i in ok if not any(i + k in ends for k in range(1, CTX))]
    s0 = ok[len(ok) // 2]
    return ((data.pose[s0:s0 + CTX] - data.pm) / data.ps).astype(np.float32), data.root[s0:s0 + CTX].astype(np.float32)
