"""Export a creature model set to ONNX + JSON spec (engine runtime; no Python at runtime). The runtime contract is MODEL_CONTRACT.md (written next to the models).
python -I export_onnx.py --cfg <creature.json> --tok tok --pose pose --root root --out <dir> [--spec_name dog_brain.json]
The C++ runtime loads the spec by the name dog_brain.json (a historic name; it is used for every creature).
Writes: pose_step.onnx (tokens [1,L,H] int64, ctx [1,4,D], root [1,T,4] -> logits [1,L,H,NC])   one MaskGIT decode step
        tok_dec.onnx   (tokens [1,L,H] int64 -> pose features [1,T,D] normalized; the last C channels are the contact head)
        root_net.onnx  (cmd [1,3], root_hist [1,4,4], ctx [1,4,D] -> root [1,T-4,4] normalized)
        <spec>.json    (sizes, normalization, skeleton, start context, height table (old root path), root net block, contact block, PyTorch test vectors)
cfg["export"] (optional): {"rig_spec": <json with rest_metres (Blender axes), joints, parents, feet, glb_scale>, "spec_name": "..."}.
skeleton "orient" (needed by the C++ runtime for every skeleton that is not the 27-joint dog): {"left": joint, "right": joint, "front": joint, "back": joint}
= four joints that are rigid with the body (joint names or ids); they give the body frame for the root bone rotation."""
import sys, json, time, math, argparse
from pathlib import Path
import numpy as np, torch
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import creature.core as core
core.DEV = "cpu"
from creature.core import Cfg, Data, Tokenizer, PoseNet, RootNet, FPT, CTX, decode, keep_schedule, height_table, start_context, load_tok, load_pose, load_root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True); ap.add_argument("--tok", required=True); ap.add_argument("--pose", required=True); ap.add_argument("--root", default="")
    ap.add_argument("--out", required=True); ap.add_argument("--spec_name", default=""); ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--carry", type=int, default=1, help="seam contract: tokens carried over after the context token (-1 = no seam block)")
    ap.add_argument("--left_ctx", type=int, default=2, help="seam contract: previous tokens given to the decoder as left context")
    a = ap.parse_args()
    cfg = Cfg(a.cfg); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ex = cfg.raw.get("export", {}); spec_name = a.spec_name or ex.get("spec_name", "dog_brain.json")
    tr, va = Data(cfg.data, "train"), Data(cfg.data, cfg.val_split)
    D, H, NC, L = cfg.D, cfg.heads, cfg.NC, a.tokens; T = L * FPT
    tokz, net = load_tok(cfg, a.tok), load_pose(cfg, a.pose)
    rootnet = load_root(cfg, a.root) if a.root else None
    for p in list(net.parameters()) + list(tokz.parameters()):
        p.requires_grad_(True)          # grad enabled: nn.TransformerEncoderLayer takes the normal (exportable) path

    class Step(nn.Module):
        def __init__(s):
            super().__init__(); s.net = net

        def forward(s, tokens, ctx, root):
            cond = torch.cat([ctx, torch.zeros(1, T - CTX, D)], 1)
            has = torch.cat([torch.ones(1, CTX), torch.zeros(1, T - CTX)], 1)
            return s.net(tokens, root, cond, has, L)

    class Dec(nn.Module):
        def __init__(s):
            super().__init__(); s.t = tokz

        def forward(s, tokens):
            return s.t.decode_idx(tokens)

    tok0 = torch.full((1, L, H), NC, dtype=torch.long)
    P, R, _ = va.take(va.windows(T, T)[:64], T)
    # wrappers in eval mode: torch.onnx.export restores the wrapper's mode recursively (a train-mode wrapper turns dropout back on)
    torch.onnx.export(Step().eval(), (tok0, P[:1, :CTX], R[:1]), str(out / "pose_step.onnx"), input_names=["tokens", "ctx", "root"], output_names=["logits"], opset_version=17, dynamo=False)
    torch.onnx.export(Dec().eval(), (tok0.clamp(max=NC - 1),), str(out / "tok_dec.onnx"), input_names=["tokens"], output_names=["pose"], opset_version=17, dynamo=False,
                      dynamic_axes={"tokens": {1: "n_tokens"}, "pose": {1: "n_frames"}})      # any token count: the seam contract decodes [left context + plan]
    if rootnet is not None:
        rootnet.eval()
        torch.onnx.export(rootnet, (R[:1, 0, :3], R[:1, :CTX], P[:1, :CTX]), str(out / "root_net.onnx"), input_names=["cmd", "root_hist", "ctx"], output_names=["root"], opset_version=17, dynamo=False)
    net.eval(); tokz.eval(); assert not any(m.training for m in net.modules())

    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 4
    s1 = ort.InferenceSession(str(out / "pose_step.onnx"), so, providers=["CPUExecutionProvider"])
    s2 = ort.InferenceSession(str(out / "tok_dec.onnx"), so, providers=["CPUExecutionProvider"])
    s3 = ort.InferenceSession(str(out / "root_net.onnx"), so, providers=["CPUExecutionProvider"]) if rootnet is not None else None
    STEPS = 6; KEEP = keep_schedule(L * H, STEPS)

    def ort_plan(ctx, root):
        tok = np.full((1, L, H), NC, np.int64); known = np.zeros(L * H, bool)
        for s in range(STEPS):
            lg = s1.run(None, {"tokens": tok, "ctx": ctx, "root": root})[0].reshape(L * H, NC).astype(np.float64)
            e = np.exp(lg - lg.max(-1, keepdims=True)); pr = e / e.sum(-1, keepdims=True)
            conf = pr.max(-1); pred = pr.argmax(-1); conf[known] = 2.0
            thr = np.sort(conf)[::-1][KEEP[s] - 1]
            new = (conf >= thr) & ~known
            tok.reshape(-1)[new] = pred[new]; known |= new
        return tok, s2.run(None, {"tokens": tok})[0]

    d_log = d_pose = d_root = 0.0; tok_eq = n = 0
    with torch.no_grad():
        for i in range(len(P)):
            c, r = P[i:i + 1, :CTX], R[i:i + 1]
            lt = Step().eval()(tok0, c, r).numpy(); lo = s1.run(None, {"tokens": tok0.numpy(), "ctx": c.numpy(), "root": r.numpy()})[0]
            d_log = max(d_log, float(np.abs(lt - lo).max()))
            cond = torch.zeros(1, T, D); cond[0, :CTX] = c[0]; has = torch.zeros(1, T); has[0, :CTX] = 1
            tk = decode(net, r, cond, has, L); rec = tokz.decode_idx(tk).numpy()
            to, ro = ort_plan(c.numpy(), r.numpy())
            same = bool((tk.numpy() == to).all()); tok_eq += same; n += 1
            if same:
                d_pose = max(d_pose, float(np.abs(rec - ro).max()))
            if rootnet is not None:
                cmd = r[:, 10:, :3].mean(1)
                d_root = max(d_root, float(np.abs(rootnet(cmd, r[:, :CTX], c).numpy() - s3.run(None, {"cmd": cmd.numpy(), "root_hist": r[:, :CTX].numpy(), "ctx": c.numpy()})[0]).max()))
    t0 = time.perf_counter()
    for i in range(50):
        k = i % len(P)
        if s3 is not None:
            s3.run(None, {"cmd": R[k:k + 1, 0, :3].numpy(), "root_hist": R[k:k + 1, :CTX].numpy(), "ctx": P[k:k + 1, :CTX].numpy()})
        ort_plan(P[k:k + 1, :CTX].numpy(), R[k:k + 1].numpy())
    ms = (time.perf_counter() - t0) / 50 * 1000
    print("VERIFY pose logits max abs diff %.2e | plans with identical tokens %d/%d | decoded pose max abs diff (normalized) %.2e | root net max abs diff (normalized) %.2e | ORT CPU python %.2f ms per plan"
          % (d_log, tok_eq, n, d_pose, d_root, ms))

    f = lambda x: [round(float(v), 6) for v in np.asarray(x, np.float64).ravel()]
    hs, hv = height_table(tr.root[tr.frame_ids()])
    ctx_start, root_start = start_context(tr)
    rr = tr.root[tr.frame_ids()]; spd = np.hypot(rr[:, 0], rr[:, 1]); pct = lambda x, q: [round(float(v), 3) for v in np.percentile(x, q)]
    # PyTorch test vector: start context + a walk command (old root path), and the root net on the start context
    rc = np.tile(np.array([0, 1.0, 0.3, float(np.interp(1.0, hs, hv))], np.float32), (T, 1)); rn = ((rc - tr.rm) / tr.rs).astype(np.float32)
    with torch.no_grad():
        cond = torch.zeros(1, T, D); cond[0, :CTX] = torch.tensor(ctx_start); has = torch.zeros(1, T); has[0, :CTX] = 1
        tk = decode(net, torch.tensor(rn)[None], cond, has, L); rec = tokz.decode_idx(tk)[0].numpy()
    J, Cn = cfg.J, cfg.C
    spec = dict(name=f"{cfg.name}_motionbricks", version=2, creature=cfg.name, contract="MODEL_CONTRACT.md", contacts_n=Cn, pose_tag=a.pose, tok_tag=a.tok, root_tag=a.root, fps=cfg.fps, tokens=L, frames_per_token=FPT, frames=T, context_frames=CTX,
                heads=H, codes=NC, mask_id=NC, levels=list(cfg.levels), decode_steps=STEPS, keep_schedule=KEEP, pose_dim=D, joints_n=J,
                layout=f"pose = {J}*3 local joint positions (heading frame, root xz = 0, metres) + {J}*3 world velocities in the heading frame (m/s) + {Cn} contacts; "
                       "root = [vx side, vz forward, yaw rate, root height]",
                axes="Y up, forward +Z, left +X",
                pose_step="pose_step.onnx", tok_dec="tok_dec.onnx",
                joints=cfg.joints, parents=cfg.parents, feet=cfg.contacts,
                pose_mean=f(tr.pm), pose_std=f(tr.ps), root_mean=f(tr.rm), root_std=f(tr.rs),
                height_speed=f(hs), height_value=f(hv), ctx_start=f(ctx_start),
                contact=dict(offset=6 * J, count=Cn, joints=cfg.contacts, threshold=0.5,
                             note="channels offset..offset+count of the DENORMALIZED decoder output; > threshold = the foot is planted"),
                data_stats=dict(speed_pct_50_90_99_max=pct(spd, [50, 90, 99, 100]), fwd_pct_1_50_99=pct(rr[:, 1], [1, 50, 99]), side_abs_pct_90_99=pct(np.abs(rr[:, 0]), [90, 99]),
                                yaw_abs_pct_90_99_max=pct(np.abs(rr[:, 2]), [90, 99, 100]), frames=int(len(spd))),
                test=dict(root=f(rn), tokens=[int(x) for x in tk.numpy().ravel()], pose=f(rec)),
                verify=dict(logits_max_abs_diff=d_log, plans_identical_tokens=f"{tok_eq}/{n}", pose_max_abs_diff_norm=d_pose, root_net_max_abs_diff_norm=d_root, ort_cpu_python_ms_per_plan=round(ms, 2)))
    if rootnet is not None:
        cmd = ((np.array([0.0, 1.0, 0.3]) - tr.rm[:3]) / tr.rs[:3]).astype(np.float32); rh = ((root_start - tr.rm) / tr.rs).astype(np.float32)
        with torch.no_grad():
            ro = rootnet(torch.tensor(cmd)[None], torch.tensor(rh)[None], torch.tensor(ctx_start)[None])[0].numpy()
        spec["root_net"] = dict(file="root_net.onnx", enabled=True, frames=T - CTX, idle_eps=0.08,
                                inputs=dict(cmd="[1,3] (command [side m/s, forward m/s, yaw rate rad/s] - root_mean[0:3]) / root_std[0:3]",
                                            root_hist=f"[1,{CTX},4] (root values of the last {CTX} played frames - root_mean) / root_std",
                                            ctx=f"[1,{CTX},{D}] normalized pose features of the last {CTX} played frames (the same tensor as pose_step ctx)"),
                                output=f"root [1,{T - CTX},4] normalized root values of the plan frames {CTX}..{T - 1}: value * root_std + root_mean",
                                root_start=f(root_start), test=dict(cmd=f(cmd), root_hist=f(rh), ctx=f(ctx_start), root=f(ro)))
    if a.carry >= 0:
        a0 = 3; init = tk[:, a0:a0 + 1 + a.carry]; left = tk[:, a0 - a.left_ctx:a0]
        with torch.no_grad():
            tk2 = decode(net, torch.tensor(rn)[None], cond, has, L, init=init); full = tokz.decode_idx(torch.cat([left, tk2], 1))
            rec2 = full[0, a.left_ctx * FPT:].numpy()
        d_seam = float(np.abs(s2.run(None, {"tokens": torch.cat([left, tk2], 1).numpy()})[0] - full.numpy()).max())
        print("VERIFY decoder with left context (%d tokens) max abs diff %.2e" % (a.left_ctx + L, d_seam))
        spec["verify"]["dec_left_ctx_max_abs_diff_norm"] = d_seam
        spec["seam"] = dict(enabled=True, carry_tokens=a.carry, left_ctx=a.left_ctx, keep_schedule=keep_schedule(L * H, STEPS, (1 + a.carry) * H),
                            note="regular replan (cursor on a token border, command unchanged): tokens[0..carry] = previous plan tokens [c/4-1 .. c/4-1+carry] (given, known); "
                                 "decoder input = previous plan tokens [c/4-1-left_ctx .. c/4-2] + the new tokens; drop the first left_ctx*4 output frames",
                            test=dict(init=[int(x) for x in init.numpy().ravel()], left=[int(x) for x in left.numpy().ravel()], tokens=[int(x) for x in tk2.numpy().ravel()], pose=f(rec2)))
    if ex.get("rig_spec"):
        soma = json.load(open(cfg.rel(ex["rig_spec"])))
        rest = np.array(soma["rest_metres"]); rest = np.stack([rest[:, 0], rest[:, 2], -rest[:, 1]], 1)       # Blender (Z up, forward -Y) -> model (Y up, forward +Z)
        spec.update(joints=soma["joints"], parents=soma["parents"], feet=soma["feet"], rest_metres=[f(r) for r in rest], rig_scale=soma["glb_scale"])
    else:
        ids = tr.frame_ids(); st = ids[(spd < 0.05)]
        spec.update(rest_metres=[f(r) for r in np.median(tr.local[st], 0)], rig_scale=1.0)
    if cfg.skeleton.get("orient"):
        jid = lambda v: spec["joints"].index(v) if isinstance(v, str) else int(v)
        spec["orient"] = {k: jid(cfg.skeleton["orient"][k]) for k in ("left", "right", "front", "back")}
    json.dump(spec, open(out / spec_name, "w"))
    print("EXPORT_DONE", out, {p.name: p.stat().st_size for p in out.iterdir()})


if __name__ == "__main__":
    main()
