"""Step 2: distil the MotionBricks SOMA pose transformer (16 layers, 136.5M params) into a shallower student.

Recipe:
  * layer pick by angular distance between hidden states (layer_pick.py): drop the layers that change the hidden state
    least; the student starts from the teacher's own weights for the kept layers (it is "healed" by the fine-tune);
  * loss = KL(teacher || student) on the pose-token logits (teacher = the ORIGINAL model, soft labels, not argmax)
    + a relative L2 term on hidden states (weight --hid-w, default 0.5);
  * on-policy rounds: the STUDENT runs self-rollout chains, the frozen teacher labels the student's own requests
    (the runtime replans from the student's own motion, so the student must see its own mistakes);
  * input corruption = the MaskGIT-style training corruption, with a large share of fully masked samples
    (the runtime does ONE fully masked pass).
Root model, tokenizer and decoder stay the teacher's (frozen); only the pose transformer is distilled.

Needs: a CUDA GPU, the NVIDIA checkpoint (--ckpt or env MB_CKPT), the teacher FP32 ONNX and the golden set (build_golden.py).
Commands (run with python -I creatures/distill/distill.py <command> --help for the flags):
  gen      teacher self-rollout requests, labelled by the teacher     -> <work>/runs/teacher_pool_s<seed>.npz
  genkp    requests with key-pose targets, labelled by the teacher    -> <work>/runs/keypose_pool_s<seed>.npz
  label    label your own captured requests (npz, optional)           -> your --out npz
  select   angular-distance table of the 16 layers                    -> <work>/runs/angular.json
  train    distil (use --keep auto10, or --init RUN to fine-tune)     -> <work>/runs/<name>/student.pt, curve.json, config.json
  eval     golden set + 7 rollouts + foot-lock for 'base' (teacher) or run names
  export   student -> FP32 ONNX with NVIDIA's exporter                -> <work>/runs/<name>/<name>_fp32.onnx
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))      # python -I drops the script folder
import mb_common as mb  # noqa: E402
import mbtorch as mt  # noqa: E402
import layer_pick as lpk  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402
from torch import nn  # noqa: E402

NAMES = mt.INPUT_NAMES


# ------------------------------------------------------------------ request preprocessing (= predict(), runtime path)
class Prep:
    def __init__(self, model):
        from features import Featurizer
        self.inf = model.model
        self.fz = Featurizer(model)
        self.hidx = self.inf.global_motion_rep.indices["global_root_pos"][[1]]

    def __call__(self, b, L):
        g, hg, lroot, hl, lp, hp = (b[k] for k in NAMES[:6])
        inf = self.inf
        lp = self.fz.retarget(lp)
        _, _, gre = inf._extract_initial_root_info(g)
        x = torch.cat([gre[:, :, self.hidx], lp], -1)
        lpn = (x - self.fz.mean) / torch.sqrt(self.fz.std ** 2 + inf.EPS)
        B = g.shape[0]
        pose_cond = torch.zeros(B, 64, lpn.shape[-1], device=g.device)
        has = torch.zeros(B, 64, dtype=torch.bool, device=g.device)
        pose_cond[:, :4] = lpn[:, :4]
        has[:, :4] = hp[:, :4].bool()
        ar = torch.arange(B, device=g.device)
        for i in range(4):
            fi = (L.long() - 1) * 4 + i
            pose_cond[ar, fi] = pose_cond[ar, fi] + lpn[:, 4 + i]
            has[ar, fi] = has[ar, fi] | hp[:, 4 + i].bool()
        root_cond = self.fz._x(b["groot"], self.fz.grep, "root_without_hip_height")
        return dict(pose_cond=pose_cond, has_pose_cond=has, root_cond=root_cond)


def batch_from(d, idx, dev):
    b = {k: torch.from_numpy(d[k][idx]).to(dev) for k in NAMES}
    b["tokens"] = torch.from_numpy(d["tokens"][idx].astype(np.int64)).to(dev)
    b["groot"] = torch.from_numpy(d["groot"][idx]).to(dev)
    b["L"] = torch.from_numpy(d["L"][idx].astype(np.int64)).to(dev)
    return b


def mask_tokens(gt, L, MASK, full_prob, gen):
    """Original training corruption (cosine focus, 80% MASK, part random); positions >= L are MASK as at inference."""
    B, P, H = gt.shape
    dev = gt.device
    valid = (torch.arange(P, device=dev)[None] < L[:, None])[:, :, None].expand(B, P, H).reshape(B, -1)
    n_all = (L * H).float()
    r = torch.cos(math.pi * 0.5 * torch.rand(B, device=dev, generator=gen))
    full = torch.rand(B, device=dev, generator=gen) < full_prob
    r = torch.where(full, torch.ones_like(r), r)
    n_focus = (n_all * r).int().clamp(min=1)
    n_mask = torch.where(full, n_focus, torch.floor(n_focus * 0.8).int())
    n_inc = torch.floor((n_focus - n_mask) * torch.rand(B, device=dev, generator=gen)).int()
    score = torch.rand(B, P * H, device=dev, generator=gen) + (~valid).float() * 10
    order = score.argsort(-1).argsort(-1)
    masked = order < n_mask[:, None]
    inc = (order >= n_mask[:, None]) & (order < (n_mask + n_inc)[:, None])
    flat = gt.reshape(B, -1).clone()
    flat = torch.where(inc, torch.randint(0, 16, flat.shape, device=dev, generator=gen), flat)
    flat = torch.where(masked | ~valid, torch.full_like(flat, MASK), flat)
    return flat.reshape(B, P, H), valid.reshape(B, P, H)


# ------------------------------------------------------------------ hidden-state capture
class Hidden:
    """Records the input of layer 0 and the output of every layer of a TransformerEncoder."""

    def __init__(self, enc):
        self.h = []
        self.hs = [enc.layers[0].register_forward_pre_hook(self._pre, with_kwargs=True)]
        self.hs += [l.register_forward_hook(self._post) for l in enc.layers]

    def _pre(self, mod, args, kwargs):
        self.h = [args[0] if args else kwargs["src"]]

    def _post(self, mod, args, out):
        self.h.append(out)

    def close(self):
        for h in self.hs:
            h.remove()


def no_nested(teacher):
    """Padded (dense) hidden states for the hooks; same valid-position math as the nested fast path."""
    enc = teacher.model._mmm_pose_model.mmm_net._transformer_model
    enc.enable_nested_tensor = False
    enc.use_nested_tensor = False


def pose_logits(net, pc, b, tin):
    return net(tin, pc["root_cond"], pc["pose_cond"], pc["has_pose_cond"], b["L"][:, None].int(), None, None)["pose_logits"]


def make_student(teacher_pose, keep):
    st = copy.deepcopy(teacher_pose)
    enc = st._transformer_model
    enc.layers = nn.ModuleList([enc.layers[i] for i in keep])
    enc.num_layers = len(keep)
    enc.enable_nested_tensor = False
    enc.use_nested_tensor = False
    for p in st.parameters():
        p.requires_grad_(True)
    return st


# ------------------------------------------------------------------ data
def cap_gpu(frac):
    """Optional cap of this process's share of the GPU memory (0 = no cap). On a shared card, over-subscribing VRAM can make
    the driver spill into system RAM and the run crawls. A cap turns that into a clear out-of-memory error instead."""
    if frac > 0:
        torch.cuda.set_per_process_memory_fraction(frac)


def expand_paths(spec):
    """Comma list of npz files and / or folders (a folder = all *.npz inside, sorted) -> list of Path."""
    out = []
    for s in [x for x in (spec or "").split(",") if x]:
        p = Path(s)
        out += sorted(p.glob("*.npz")) if p.is_dir() else [p]
    return out


def load_data(pools):
    """Concatenate teacher-labelled npz files (distill.py gen / genkp / label). The held-out set (agreement checks) is the
    last --holdout requests of the FIRST file."""
    import dataset as ds
    if not pools:
        raise SystemExit("no training data: pass --pool FILE_OR_FOLDER (make it with 'distill.py gen', 'genkp' or 'label')")
    data = None
    for p in pools:
        z = ds.load(p)
        z.pop("kind", None)
        data = z if data is None else concat(data, z)
    return data


def keypose_bank(path=""):
    """Target key poses: 4 frames each, world space, [4,30,3] positions + [4,30,3,3] rotations.
    path = an npz with arrays <name>__pos and <name>__rot (your own key poses: poses your game puts into the target slots).
    Without it, the bank is made from the sample BVH clips (--motions): a 4-frame window every 30 frames of every clip."""
    bank, names = [], []
    if path:
        z = np.load(path)
        for k in z.files:
            if k.endswith("__pos"):
                base = k[: -len("__pos")]
                bank.append((z[k].astype(np.float64), z[base + "__rot"].astype(np.float64)))
                names.append(base)
        return bank, names
    for p in sorted(mb.MOTIONS.rglob("*.bvh")):
        c = mb.Clip(p)
        for f in range(0, c.num_frames - 4, 30):
            bank.append((c.pos[f:f + 4].copy(), c.rot[f:f + 4].copy()))
            names.append(f"{c.name}@{f}")
    if not bank:
        raise SystemExit(f"no key poses: no BVH clips under {mb.MOTIONS}; pass --keypose-npz or --motions")
    return bank, names


def make_keypose_pool(teacher, rounds, per_round, seed, dev="cuda", keypose_npz=""):
    """Requests whose targets are key poses, placed relative to a teacher context; later rounds start from the
    teacher's own outputs (so transitions OUT of the key poses are covered too). Labels = the teacher."""
    import dataset as ds
    rng = np.random.default_rng(seed)
    bank, names = keypose_bank(keypose_npz)
    g = mb.load_golden()
    pool = []
    for i in range(g["pred_num_tokens"].shape[0]):
        d = mb.decode_motion(g["pred_global_motions"][i], int(g["pred_num_tokens"][i]), trim=True)
        pool.append((d["pos"], d["rot"]))
    keep_f, keep_t, used = [], [], []
    for r in range(rounds):
        reqs, aligns = [], []
        while len(reqs) < per_round:
            a = pool[rng.integers(len(pool))]
            ta = int(rng.integers(0, a[0].shape[0] - 4))
            cp, cr = a[0][ta:ta + 4], a[1][ta:ta + 4]
            ki = int(rng.integers(len(bank)))
            tp, tr = bank[ki]
            nt = int(rng.integers(1, 5))
            u = rng.random()
            dist = rng.uniform(0, 0.3) if u < 0.3 else rng.uniform(0.3, 3.0) if u < 0.9 else rng.uniform(3, 6)
            tp, tr = ds.place(tp, tr, mb.alignment_of(cp[3]), dist, rng.uniform(-1.2, 1.2), rng.uniform(-np.pi, np.pi) * 0.6)
            tp, tr = tp.copy(), tr.copy()
            tp[nt:], tr[nt:] = tp[nt - 1], tr[nt - 1]
            inp, al = mb.encode_request(np.concatenate([cp, tp]), np.concatenate([cr, tr]), nt, ds.pick_allowed(rng))
            reqs.append(inp)
            aligns.append(al)
            used.append(ki)
        feeds, tg, mot = ds.run_requests(teacher, reqs, dev)
        new = [ds.decode_world(mot[i], tg["L"][i], aligns[i]) for i in range(len(reqs))]
        pool = pool[-20000:] + new
        keep_f.append(feeds)
        keep_t.append(tg)
        print(f"keypose round {r}: {len(reqs)} requests, pool {len(pool)}", flush=True)
    feeds = {k: np.concatenate([f[k] for f in keep_f]) for k in NAMES}
    tg = {k: np.concatenate([t[k] for t in keep_t]) for k in ("tokens", "groot", "L")}
    hist = np.bincount(np.array(used), minlength=len(bank))
    return feeds, tg, {names[i]: int(hist[i]) for i in range(len(bank))}


def cmd_genkp(a):
    cap_gpu(a.mem_frac)
    mt.set_deterministic()
    teacher = mt.load_soma("cuda")
    t0 = time.time()
    feeds, tg, hist = make_keypose_pool(teacher, a.rounds, a.per_round, a.seed, keypose_npz=a.keypose_npz)
    out = mb.RUNS / f"keypose_pool_s{a.seed}.npz"
    mb.RUNS.mkdir(parents=True, exist_ok=True)
    np.savez(out, **feeds, **tg)
    (mb.RUNS / f"keypose_pool_s{a.seed}.json").write_text(json.dumps(hist, indent=0))
    print(f"{len(tg['L'])} key-pose requests ({len(hist)} key poses) -> {out}, L hist "
          f"{np.bincount(tg['L'], minlength=17)[6:].tolist()}, {time.time() - t0:.0f}s", flush=True)


def cmd_gen(a):
    """Fresh teacher self-rollout requests (the teacher's own tokens / root / L as labels), saved for training."""
    import dataset as ds
    cap_gpu(a.mem_frac)
    mt.set_deterministic()
    teacher = mt.load_soma("cuda")
    t0 = time.time()
    feeds, tg, kinds = ds.make_replay(teacher, a.rounds, a.per_round, seed=a.seed)
    out = mb.RUNS / f"teacher_pool_s{a.seed}.npz"
    mb.RUNS.mkdir(parents=True, exist_ok=True)
    np.savez(out, **feeds, **tg)
    print(f"{len(kinds)} teacher requests -> {out} ({out.stat().st_size / 1e9:.2f} GB), {time.time() - t0:.0f}s", flush=True)


def cmd_label(a):
    """Your own captured requests (optional). Input: npz files (or a folder of npz files) that hold the 7 request arrays
    (mbtorch.INPUT_NAMES: global_root_values, has_global_root_values, local_root_values, has_local_root_values,
    local_poses, has_local_poses, allowed_pred_num_tokens), N requests per file. The teacher labels them."""
    cap_gpu(a.mem_frac)
    mt.set_deterministic()
    teacher = mt.load_soma("cuda")
    files = expand_paths(a.requests)
    if not files:
        raise SystemExit("no npz files found in --requests")
    feeds = {k: [] for k in NAMES}
    for f in files:
        z = np.load(f, allow_pickle=False)
        missing = [k for k in NAMES if k not in z.files]
        if missing:
            raise SystemExit(f"{f}: missing arrays {missing}")
        for k in NAMES:
            feeds[k].append(z[k])
    feeds = {k: np.concatenate(v) for k, v in feeds.items()}
    out = teacher_label(teacher, feeds, "cuda")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **out)
    print(f"{len(out['L'])} requests from {len(files)} files, labelled by the teacher -> {a.out}", flush=True)


def concat(a, b):
    return {k: np.concatenate([a[k], b[k]]) for k in a if k in b}


@torch.no_grad()
def teacher_label(teacher, feeds, dev):
    """Teacher tokens / root trajectory / token count for given requests (labels the student's on-policy requests)."""
    import dataset as ds
    tap = ds.Tap(teacher)
    N = feeds["global_root_values"].shape[0]
    tok, gro, L = np.zeros((N, 16, 5), np.int16), np.zeros((N, 64, 5), np.float32), np.zeros(N, np.int16)
    for i in range(0, N, 256):
        sl = slice(i, min(N, i + 256))
        _, n = teacher(*mt.feeds_to_torch({k: feeds[k][sl] for k in NAMES}, dev))
        L[sl] = n.cpu().numpy().reshape(-1)
        tok[sl] = tap.v["tokens"].cpu().numpy()
        gro[sl] = tap.v["groot"].cpu().numpy()
    tap.close()
    return dict(feeds, tokens=tok, groot=gro, L=L)


def onpolicy_round(teacher, student_pose, rounds, per_round, seed, dev):
    """Student self-rollouts (dataset.make_replay with the student pose net swapped in), labelled by the teacher."""
    import dataset as ds
    inf = teacher.model
    orig = inf._mmm_pose_model.mmm_net
    student_pose.eval()
    torch.cuda.empty_cache()
    inf._mmm_pose_model.mmm_net = student_pose
    try:
        mt.set_deterministic()
        feeds, _, _ = ds.make_replay(teacher, rounds, per_round, seed=seed, dev=dev)
    finally:
        inf._mmm_pose_model.mmm_net = orig
    out = teacher_label(teacher, feeds, dev)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.empty_cache()
    return out


# ------------------------------------------------------------------ select
@torch.no_grad()
def angular_table(teacher, prep, data, dev, n=2048):
    pose = teacher.model._mmm_pose_model.mmm_net
    hid = Hidden(pose._transformer_model)
    idx = np.arange(len(data["L"]) - n, len(data["L"]))
    acc = None
    cnt = 0
    for j in range(0, n, 256):
        b = batch_from(data, idx[j:j + 256], dev)
        pc = prep(b, b["L"])
        tin = torch.full_like(b["tokens"], pose.POSE_MASK_ID)
        pose_logits(pose, pc, b, tin)
        H = torch.stack(hid.h)                                     # [17, B, P, D]
        valid = (torch.arange(H.shape[2], device=dev)[None] < b["L"][:, None]).float()
        a, n_valid = lpk.accumulate_angular(H, valid)
        acc = a if acc is None else acc + a
        cnt += n_valid
    hid.close()
    return (acc / cnt).cpu().numpy()                               # [17,17]: d(h_i, h_j)


def cmd_select(a):
    dev = "cuda"
    mt.set_deterministic()
    teacher = mt.load_soma(dev)
    no_nested(teacher)
    prep = Prep(teacher)
    D = angular_table(teacher, prep, load_data(expand_paths(a.pool)), dev)
    L = D.shape[0] - 1
    per_layer = [float(D[i, i + 1]) for i in range(L)]
    blocks = {nrem: lpk.best_block(D, nrem) for nrem in (4, 6, 8, 10, 12)}
    out = dict(per_layer=per_layer, blocks={k: dict(dist=v[0], start=v[1], removes=list(range(v[1], v[1] + k)))
                                            for k, v in blocks.items()},
               greedy={k: lpk.pick_keep(D, k) for k in (12, 10, 8, 6, 4)}, D=D.tolist())
    mb.RUNS.mkdir(parents=True, exist_ok=True)
    (mb.RUNS / "angular.json").write_text(json.dumps(out, indent=1))
    print("per-layer angular distance d(h_i, h_i+1):", " ".join(f"{i}:{v:.3f}" for i, v in enumerate(per_layer)))
    for k, v in out["blocks"].items():
        print(f"best contiguous block of {k}: remove {v['removes']} dist {v['dist']:.3f}")
    for k, v in out["greedy"].items():
        print(f"greedy keep {k}: {v}")


def resolve_keep(spec):
    if spec.startswith("auto"):
        k = int(spec[4:])
        f = mb.RUNS / "angular.json"
        if not f.exists():
            raise SystemExit("run 'distill.py select' first")
        return json.loads(f.read_text())["greedy"][str(k)]
    if spec.startswith("block"):                                    # block8 = drop the best contiguous block of 8
        k = int(spec[5:])
        rem = set(json.loads((mb.RUNS / "angular.json").read_text())["blocks"][str(k)]["removes"])
        return [i for i in range(16) if i not in rem]
    if spec.startswith("every"):                                    # every2 = keep 0,2,4,... + last
        s = int(spec[5:])
        keep = list(range(0, 16, s))
        return keep if 15 in keep else keep + [15]
    return [int(x) for x in spec.split(",")]


# ------------------------------------------------------------------ train
@torch.no_grad()
def agree(teacher_pose, student, prep, data, idx, dev):
    """Fully masked one pass (the runtime case): student argmax == teacher argmax, and KL."""
    student.eval()
    hit, n, kl = 0.0, 0, 0.0
    for j in range(0, len(idx), 256):
        b = batch_from(data, idx[j:j + 256], dev)
        pc = prep(b, b["L"])
        tin = torch.full_like(b["tokens"], teacher_pose.POSE_MASK_ID)
        tl, sl = pose_logits(teacher_pose, pc, b, tin), pose_logits(student, pc, b, tin)
        valid = (torch.arange(16, device=dev)[None] < b["L"][:, None])[:, :, None].expand(tl.shape[:3])
        hit += ((tl.argmax(-1) == sl.argmax(-1)) & valid).sum().item()
        n += valid.sum().item()
        k = (Fn.softmax(tl, -1) * (Fn.log_softmax(tl, -1) - Fn.log_softmax(sl, -1))).sum(-1)
        kl += (k * valid).sum().item()
    return dict(tok_agree=hit / max(n, 1), kl=kl / max(n, 1))


def cmd_train(a):
    dev = "cuda"
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    run = mb.RUNS / a.name
    run.mkdir(parents=True, exist_ok=True)
    log = open(run / "train.log", "a", buffering=1)

    def say(*x):
        s = " ".join(str(v) for v in x)
        print(s, flush=True)
        log.write(s + "\n")

    cap_gpu(a.mem_frac)
    mt.set_deterministic()
    teacher = mt.load_soma(dev)
    for p in teacher.parameters():
        p.requires_grad_(False)
    no_nested(teacher)
    torch.backends.cuda.matmul.allow_tf32 = a.tf32 == "on"
    prep = Prep(teacher)
    tpose = teacher.model._mmm_pose_model.mmm_net
    if a.init:
        ck = torch.load(mb.RUNS / a.init / "student.pt", map_location=dev)
        keep = ck["keep"]
        student = make_student(tpose, keep)
        student.load_state_dict(ck["state"])
    else:
        keep = resolve_keep(a.keep)
        student = make_student(tpose, keep)
    npar = sum(p.numel() for p in student.parameters()) / 1e6
    say(f"[{a.name}] keep {keep} ({len(keep)} layers), student pose params {npar:.1f}M (teacher 136.5M)")
    pools = expand_paths(a.pool)
    data = load_data(pools)
    N = len(data["L"])
    n_first = len(np.load(pools[0])["L"])                           # the held-out set = the last --holdout of the first file
    va = np.arange(n_first - a.holdout, n_first)
    tr = np.concatenate([np.arange(n_first - a.holdout), np.arange(n_first, N)])
    say(f"training requests {len(tr)} from {len(pools)} files, held-out {len(va)}")
    MASK = tpose.POSE_MASK_ID
    th = Hidden(tpose._transformer_model)
    sh = Hidden(student._transformer_model)
    # student hidden j+1 (after its layer j = teacher layer keep[j]) should equal the teacher hidden the NEXT kept layer reads
    tgt_h = [(keep[j + 1] if j + 1 < len(keep) else 16) for j in range(len(keep))]
    opt = torch.optim.AdamW(student.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / a.warmup) * (0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)) * 0.95 + 0.05))
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    onp_steps = sorted({int(a.steps * float(f)) for f in a.onpolicy.split(",") if f}) if a.onpolicy else []
    onp = None
    curve = [dict(step=0, **agree(tpose, student, prep, data, va, dev))]
    say("step 0", json.dumps(curve[-1]))
    t0, run_l = time.time(), []
    best = (-1, None)
    for step in range(1, a.steps + 1):
        if step in onp_steps:
            tt = time.time()
            new = onpolicy_round(teacher, student, a.onp_rounds, a.onp_per_round, seed=1000 + step, dev=dev)
            ag = agree(tpose, student, prep, new, np.arange(min(1500, len(new["L"]))), dev)
            onp = new if onp is None else concat(onp, new)
            say(f"on-policy round at step {step}: +{len(new['L'])} student requests (pool {len(onp['L'])}), "
                f"{time.time() - tt:.0f}s; agree on student requests {json.dumps(ag)}")
        student.train() if a.dropout == "on" else student.eval()
        n_on = int(a.bs * a.onp_ratio) if onp is not None else 0
        parts = [(data, rng.choice(tr, a.bs - n_on))] + ([(onp, rng.integers(0, len(onp["L"]), n_on))] if n_on else [])
        total, terms = 0.0, {"kl": 0.0, "hid": 0.0}
        for d, idx in parts:
            b = batch_from(d, idx, dev)
            pc = prep(b, b["L"])
            tin, valid = mask_tokens(b["tokens"], b["L"], MASK, a.full_mask, gen)
            with torch.no_grad():
                tl = pose_logits(tpose, pc, b, tin)
                T_h = [h.detach() for h in th.h]
            sl = pose_logits(student, pc, b, tin)
            T = a.temp
            kl = (Fn.softmax(tl / T, -1) * (Fn.log_softmax(tl / T, -1) - Fn.log_softmax(sl / T, -1))).sum(-1) * T * T
            kl = (kl * valid).sum() / valid.sum()
            vpos = valid[:, :, 0].float()[:, :, None]
            hl = torch.zeros((), device=dev)
            if a.hid_w > 0:
                for j, ti in enumerate(tgt_h):
                    s_, t_ = sh.h[j + 1], T_h[ti]
                    hl = hl + (((s_ - t_) ** 2) * vpos).sum() / ((t_ ** 2) * vpos).sum().clamp(min=1e-6)
                hl = hl / len(tgt_h)
            w = len(idx) / a.bs
            total = total + (kl + a.hid_w * hl) * w
            terms["kl"] += kl.item() * w
            terms["hid"] += hl.item() * w
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        sched.step()
        run_l.append(terms)
        if step % a.eval_every == 0 or step == a.steps:
            ev = agree(tpose, student, prep, data, va, dev)
            row = dict(step=step, sec=round(time.time() - t0, 1), lr=sched.get_last_lr()[0],
                       train_kl=float(np.mean([x["kl"] for x in run_l])), train_hid=float(np.mean([x["hid"] for x in run_l])), **ev)
            run_l = []
            curve.append(row)
            say(json.dumps(row))
            (run / "curve.json").write_text(json.dumps(curve, indent=0))
            if ev["tok_agree"] > best[0]:
                best = (ev["tok_agree"], step)
                torch.save(dict(keep=keep, state=student.state_dict()), run / "student.pt")
        if a.max_steps_smoke and step >= a.max_steps_smoke:
            break
    cfg = vars(a) | dict(keep=keep, student_params_M=npar, train_sec=round(time.time() - t0, 1), best_step=best[1],
                         best_tok_agree=best[0])
    (run / "config.json").write_text(json.dumps(cfg, indent=1))
    th.close(); sh.close()
    say("saved", run / "student.pt", "best step", best[1], "agree", round(best[0], 4))


# ------------------------------------------------------------------ eval (golden set + rollouts + foot-lock)
def load_student_model(name, dev="cuda"):
    mt.set_deterministic()
    model = mt.load_soma(dev)
    if name == "base":
        return model.eval()
    ck = torch.load(mb.RUNS / name / "student.pt", map_location=dev)
    pose = model.model._mmm_pose_model.mmm_net
    st = make_student(pose, ck["keep"])
    st.load_state_dict(ck["state"])
    model.model._mmm_pose_model.mmm_net = st.eval()
    return model.eval()


class TorchSession:
    def __init__(self, model, dev="cuda"):
        self.m, self.dev = model, dev

    @torch.no_grad()
    def run(self, names, inputs):
        m, n = self.m(*mt.feeds_to_torch(inputs, self.dev))
        return [m.cpu().numpy(), n.cpu().numpy().astype(np.int32).reshape(-1)]


def cmd_eval(a):
    import quality
    import rollout
    import footlock
    for name in a.name.split(","):
        model = load_student_model(name)
        golden = mb.load_golden()
        meta = json.loads((mb.DATA / "golden_meta.json").read_text(encoding="utf-8"))["requests"]
        t0 = time.time()
        out = mt.run(model, {k: golden[k] for k in NAMES}, batch=32)
        q = quality.score(out, golden, meta)
        gen = rollout.Generator(session=TorchSession(model))
        clips = {p.stem: mb.Clip(p) for p in mb.MOTIONS.rglob("*.bvh")
                 if p.stem in (rollout.WALK, rollout.JOG, rollout.IDLE, rollout.CROUCH, rollout.SIT, rollout.ZOMBIE)}
        roll = {}
        for sc, spec in rollout.SCENARIOS.items():
            fn = rollout.scenario_chain if spec["kind"] == "chain" else rollout.scenario_replan
            r = fn(gen, clips, spec, tuple(range(8, 15)))
            pos, c = r["pos"].astype(np.float64), r["contacts"].astype(np.float64)
            new, _ = footlock.foot_lock(pos, c)
            x, y = footlock.motion_metrics(pos, c), footlock.motion_metrics(new, c)
            roll[sc] = dict(skate=x["skate"], skate_fl=y["skate"], jitter=x["jitter"])
        s = q["summary"]
        summ = dict(name=name, token_match=s["token_match"], num_tokens_agree=s["num_tokens_agree"],
                    joint_err_mean_cm=s["joint_err_mean_cm"], joint_err_p95_cm=s["joint_err_p95_cm"],
                    joint_err_max_cm=s["joint_err_max_cm"],
                    rollout_skate=float(np.mean([v["skate"] for v in roll.values()])),
                    rollout_skate_fl=float(np.mean([v["skate_fl"] for v in roll.values()])),
                    rollout_jitter=float(np.mean([v["jitter"] for v in roll.values()])), sec=round(time.time() - t0, 1))
        d = mb.RUNS / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "eval.json").write_text(json.dumps(dict(summary=summ, by_category=q["by_category"], rollouts=roll), indent=1))
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summ.items()}), flush=True)


def cmd_export(a):
    """Student -> FP32 ONNX with NVIDIA's exporter (same I/O as the shipped graph), checked by ORT CPU vs PyTorch."""
    import quality
    d = mb.RUNS / a.name
    out = d / f"{a.name}_fp32.onnx"
    torch.set_num_threads(8)
    model = load_student_model(a.name, dev="cpu")         # also registers the motionbricks_onnx package
    from motionbricks_onnx.export import export_fp32, validate_output
    t0 = time.time()
    export_fp32("soma", model, out)
    validate_output(out, "soma")
    golden = mb.load_golden()
    meta = json.loads((mb.DATA / "golden_meta.json").read_text(encoding="utf-8"))["requests"]
    ref = mt.run(model, {k: golden[k] for k in NAMES}, batch=32, device="cpu")
    ort_out = quality.run_variant(out, golden, threads=8)
    s = quality.score(ort_out, dict(golden, **ref), meta)["summary"]
    res = dict(file=str(out), bytes=out.stat().st_size, vs="pytorch student", num_tokens_agree=s["num_tokens_agree"],
               token_match=s["token_match"], joint_err_mean_cm=s["joint_err_mean_cm"], joint_err_max_cm=s["joint_err_max_cm"],
               sec=round(time.time() - t0, 1))
    (d / "export.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)


def main():
    ap = argparse.ArgumentParser(description="Distil the MotionBricks SOMA pose transformer into a shallower student (see the module doc).")
    sp = ap.add_subparsers(dest="cmd", required=True)

    def sub(name, help_):
        p = sp.add_parser(name, help=help_, description=help_)
        mb.add_path_args(p)
        p.add_argument("--ckpt", default=None, help="NVIDIA SOMA checkpoint folder (env MB_CKPT, default <repo>/checkpoints/MotionBricks/soma)")
        p.add_argument("--mem-frac", type=float, default=0.0, help="cap of this process's share of GPU memory, 0 = no cap")
        return p

    s = sub("select", "angular-distance table of the 16 layers -> <work>/runs/angular.json")
    s.add_argument("--pool", required=True, help="comma list of teacher-labelled npz files or folders (from gen / genkp / label)")
    t = sub("train", "distil the student")
    t.add_argument("--name", required=True)
    t.add_argument("--keep", default="auto10", help="auto10 (greedy angular, needs 'select') | block8 | every2 | 0,1,3,...")
    t.add_argument("--steps", type=int, default=8000)
    t.add_argument("--bs", type=int, default=256)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--warmup", type=int, default=200)
    t.add_argument("--temp", type=float, default=1.0)
    t.add_argument("--hid-w", type=float, default=0.5)
    t.add_argument("--full-mask", type=float, default=0.6, help="share of fully masked samples (= the runtime pass)")
    t.add_argument("--onpolicy", default="0.35,0.7", help="fractions of the run where a student rollout round is added")
    t.add_argument("--onp-rounds", type=int, default=2)
    t.add_argument("--onp-per-round", type=int, default=4000)
    t.add_argument("--onp-ratio", type=float, default=0.4)
    t.add_argument("--eval-every", type=int, default=500)
    t.add_argument("--dropout", choices=["on", "off"], default="off")
    t.add_argument("--tf32", choices=["on", "off"], default="on")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--max-steps-smoke", type=int, default=0, help="stop early (smoke test)")
    t.add_argument("--init", default="", help="start from this run's student.pt (fine-tune); --keep is then ignored")
    t.add_argument("--pool", required=True, help="comma list of teacher-labelled npz files or folders (from gen / genkp / label)")
    t.add_argument("--holdout", type=int, default=1500, help="last N requests of the first pool file = held-out set")
    k = sub("genkp", "requests with key-pose targets, labelled by the teacher")
    k.add_argument("--rounds", type=int, default=4)
    k.add_argument("--per-round", type=int, default=20000)
    k.add_argument("--seed", type=int, default=31)
    k.add_argument("--keypose-npz", default="", help="optional: your own key poses (npz with <name>__pos, <name>__rot); default = windows of the sample BVH clips")
    g = sub("gen", "teacher self-rollout requests, labelled by the teacher")
    g.add_argument("--rounds", type=int, default=10)
    g.add_argument("--per-round", type=int, default=10000)
    g.add_argument("--seed", type=int, default=21)
    lb = sub("label", "label your own captured requests with the teacher (optional)")
    lb.add_argument("--requests", required=True, help="comma list of npz files or folders; each has the 7 request arrays")
    lb.add_argument("--out", required=True, help="output npz (use it as a --pool file)")
    e = sub("eval", "golden set + 7 rollouts + foot-lock")
    e.add_argument("--name", required=True, help="comma list; 'base' = the teacher")
    x = sub("export", "student -> FP32 ONNX with NVIDIA's exporter")
    x.add_argument("--name", required=True)
    a = ap.parse_args()
    mb.apply_path_args(a)
    if a.ckpt:
        mt.CKPT = Path(a.ckpt)
    {"select": cmd_select, "train": cmd_train, "eval": cmd_eval, "export": cmd_export, "gen": cmd_gen, "genkp": cmd_genkp,
     "label": cmd_label}[a.cmd](a)


if __name__ == "__main__":
    main()
