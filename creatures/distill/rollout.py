"""Chained generation ("rollouts"): feed generated motion back as context, like a game loop.

Usage: python -I creatures/distill/rollout.py MODEL.onnx [MODEL.onnx ...] [--work DIR] [--motions DIR]
Writes <work>/results/rollout_<name>.npz and rollout_<name>.json (metrics). Needs the sample BVH clips (--motions).

This is a numpy port of what MotionBricksSOMAEngine does around the network
(decode -> endpoint correction -> next context). It is the test for error build-up and
for pose pops at replan boundaries; single-request metrics cannot show those.

Scenarios (deterministic):
  chain_*  : play each generation to its end, then target the next keyframe (inbetweener style).
  replanR_*: locomotion style. Every R frames, re-plan from the last 4 played frames towards
             a walk/jog pose placed ahead on a slowly turning path (30/R requests per second).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I drops the script folder
import mb_common as mb
from build_golden import place

JN = mb.joint_names()
FEET = [JN.index(n) for n in mb.FOOT_JOINTS]
PARENTS = OFFSETS = None      # read from the sample BVH hierarchy on first use (see _ensure)


def _ensure():
    global PARENTS, OFFSETS
    if PARENTS is None:
        PARENTS, OFFSETS, _ = mb.skel()


def local_rots(wrot):
    """World rotations [...,30,3,3] -> local (parent-relative; root stays world)."""
    _ensure()
    out = wrot.copy()
    for j in range(1, 30):
        out[..., j, :, :] = np.swapaxes(wrot[..., PARENTS[j], :, :], -1, -2) @ wrot[..., j, :, :]
    return out


def fk(root_pos, lrot):
    """root_pos [F,3], local rots [F,30,3,3] -> world pos [F,30,3], world rot."""
    _ensure()
    F = lrot.shape[0]
    wrot = np.zeros_like(lrot)
    pos = np.zeros((F, 30, 3))
    wrot[:, 0], pos[:, 0] = lrot[:, 0], root_pos
    for j in range(1, 30):
        p = PARENTS[j]
        wrot[:, j] = wrot[:, p] @ lrot[:, j]
        pos[:, j] = pos[:, p] + np.einsum("fij,j->fi", wrot[:, p], OFFSETS[j])
    return pos, wrot


def scaled(delta, w):
    """slerp(identity, delta, w) for rotation matrices [30,3,3] and weights [F] -> [F,30,3,3]."""
    rv = Rot.from_matrix(delta).as_rotvec()                    # [30,3]
    return Rot.from_rotvec((w[:, None, None] * rv[None]).reshape(-1, 3)).as_matrix().reshape(len(w), 30, 3, 3)


class Generator:
    def __init__(self, path=None, threads=0, session=None):
        """path: an ONNX file (ORT CPU, basic optimisation) or session: an object with run(names, inputs)."""
        if session is not None:          # an already built onnxruntime session or a torch wrapper
            self.s = session
        else:
            self.s = mb.make_session(path, threads=threads, opt="basic")
        self.calls = 0

    def infer(self, inputs):
        """One network call. Subclasses replace this (split graph, token reuse)."""
        return self.s.run(["pred_global_motions", "pred_num_tokens"], inputs)

    def generate(self, cpos, crot, tpos, trot, nt, allowed, correct=True):
        """World context [4] and targets [4] -> corrected world motion (pos, rot, contacts) + diagnostics."""
        tpos, trot = tpos.copy(), trot.copy()
        tpos[nt:], trot[nt:] = tpos[nt - 1], trot[nt - 1]
        inputs, align = mb.encode_request(np.concatenate([cpos, tpos]), np.concatenate([crot, trot]), nt, allowed)
        m, n = self.infer(inputs)
        self.calls += 1
        d = mb.decode_motion(m[0], n[0], trim=True)
        wrot = mb.to_world_rot(d["rot"], align)
        root = mb.to_world(d["pos"][:, 0], align)
        L = local_rots(wrot)
        F = L.shape[0]
        raw_pos, _ = fk(root, L)
        Lc, Lt = local_rots(crot[3]), local_rots(trot[0])
        start_gap = float(np.linalg.norm(raw_pos[0] - fk(cpos[3:4, 0], Lc[None])[0][0], axis=-1).mean() * 100)
        end_gap = float(np.linalg.norm(raw_pos[-1] - fk(tpos[0:1, 0], Lt[None])[0][0], axis=-1).mean() * 100)
        if correct and F >= 2:
            w = np.arange(F) / (F - 1)
            ds = np.swapaxes(L[0], -1, -2) @ Lc
            de = np.swapaxes(L[-1], -1, -2) @ Lt
            L = L @ scaled(ds, 1 - w) @ scaled(de, w)
            root = root + (cpos[3, 0] - root[0])[None] * (1 - w)[:, None] + (tpos[0, 0] - root[-1])[None] * w[:, None]
        pos, wrot = fk(root, L)
        return dict(pos=pos, rot=wrot, contacts=d["contacts"], tokens=int(n[0]), start_gap_cm=start_gap, end_gap_cm=end_gap)


def scenario_chain(gen, clips, spec, allowed):
    """spec: start (clip, frame) then a list of (clip, frame, dist, bearing, yaw, num_targets)."""
    c0 = clips[spec["start"][0]]
    t = min(spec["start"][1], c0.num_frames - 4)
    cpos, crot = c0.pos[t:t + 4].copy(), c0.rot[t:t + 4].copy()
    frames, rots, contacts, bounds, gaps, targets = [cpos], [crot], [np.zeros((4, 4))], [], [], []
    for (cn, tf, dist, bearing, yaw, nt) in spec["targets"]:
        c = clips[cn]
        tf = min(tf, c.num_frames - 4)      # clip frames are at 30 Hz; keep the window inside the clip
        tp, tr = place(c.pos[tf:tf + 4], c.rot[tf:tf + 4], mb.alignment_of(cpos[3]), dist, bearing, yaw)
        g = gen.generate(cpos, crot, tp, tr, nt, allowed)
        bounds.append(sum(len(f) for f in frames) - 1)
        gaps.append((g["start_gap_cm"], g["end_gap_cm"], g["tokens"]))
        targets.append(tp[0])
        frames.append(g["pos"][1:]); rots.append(g["rot"][1:]); contacts.append(g["contacts"][1:])
        cpos, crot = g["pos"][-4:], g["rot"][-4:]
    return dict(pos=np.concatenate(frames), contacts=np.concatenate(contacts), bounds=bounds, gaps=gaps, targets=np.array(targets))


def scenario_replan(gen, clips, spec, allowed):
    """Locomotion loop: re-plan every R frames towards a pose `lead` m ahead on a turning path."""
    c0, goal = clips[spec["start"][0]], clips[spec["goal"][0]]
    t, R, total = spec["start"][1], spec["interval"], spec["frames"]
    cpos, crot = c0.pos[t:t + 4].copy(), c0.rot[t:t + 4].copy()
    frames, contacts, bounds, gaps, targets = [cpos], [np.zeros((4, 4))], [], [], []
    played = 0
    k = 0
    while played < total:
        yaw = spec["turn_rate"] * np.sin(2 * np.pi * played / spec["turn_period"])     # rad, relative to current facing
        gf = min(spec["goal"][1] + (k * 7) % 40, goal.num_frames - 4)
        tp, tr = place(goal.pos[gf:gf + 4], goal.rot[gf:gf + 4], mb.alignment_of(cpos[3]), spec["lead"], yaw, yaw)
        g = gen.generate(cpos, crot, tp, tr, spec.get("num_targets", 4), allowed)
        bounds.append(sum(len(f) for f in frames) - 1)
        gaps.append((g["start_gap_cm"], g["end_gap_cm"], g["tokens"]))
        targets.append(tp[0])
        take = min(R, len(g["pos"]) - 1)
        frames.append(g["pos"][1:1 + take]); contacts.append(g["contacts"][1:1 + take])
        # the next context is the last four *played* frames
        hist_p = np.concatenate([cpos, g["pos"][1:1 + take]])[-4:]
        hist_r = np.concatenate([crot, g["rot"][1:1 + take]])[-4:]
        cpos, crot = hist_p, hist_r
        played += take
        k += 1
    return dict(pos=np.concatenate(frames), contacts=np.concatenate(contacts), bounds=bounds, gaps=gaps, targets=np.array(targets))


WALK = "Neutral_walk_forward_002__A057"
JOG = "neutral_jog_ff_180_R_001__A534"
IDLE = "idle_loop_001__A268"
CROUCH = "crouch_ff_loop_180_R_001__A196"
SIT = "sit_on_chair_start_R_001__A047"
ZOMBIE = "zombie_walk_180_R_003__A330"

SCENARIOS = {
    "chain_idle_walk_jog_idle": dict(kind="chain", start=(IDLE, 30), targets=[
        (WALK, 60, 1.5, 0.0, 0.0, 4), (WALK, 100, 1.8, 0.3, 0.4, 4), (JOG, 60, 2.5, -0.3, -0.5, 4),
        (JOG, 90, 2.8, 0.0, 0.0, 4), (WALK, 140, 1.6, 0.4, 0.6, 4), (IDLE, 200, 0.8, 0.0, 0.0, 1)]),
    "chain_styles_crouch_zombie": dict(kind="chain", start=(WALK, 40), targets=[
        (CROUCH, 60, 1.2, 0.0, 0.0, 4), (CROUCH, 120, 1.2, 0.5, 0.8, 4), (ZOMBIE, 200, 1.0, 0.0, 0.0, 4),
        (ZOMBIE, 260, 1.0, -0.4, -0.6, 4), (IDLE, 100, 0.6, 0.0, 0.0, 1)]),
    "chain_walk_to_sit": dict(kind="chain", start=(IDLE, 400), targets=[
        (WALK, 80, 1.4, 0.0, 0.0, 4), (SIT, 20, 1.2, 0.2, 0.3, 4), (SIT, 150, 0.6, 0.0, 0.0, 1)]),
    "replan6_walk_turning": dict(kind="replan", start=(WALK, 40), goal=(WALK, 60), interval=6, frames=240,
                                 lead=1.6, turn_rate=0.5, turn_period=180),
    "replan3_walk_turning": dict(kind="replan", start=(WALK, 40), goal=(WALK, 60), interval=3, frames=240,
                                 lead=1.6, turn_rate=0.5, turn_period=180),
    "replan9_walk_turning": dict(kind="replan", start=(WALK, 40), goal=(WALK, 60), interval=9, frames=240,
                                 lead=1.6, turn_rate=0.5, turn_period=180),
    "replan6_jog_turning": dict(kind="replan", start=(JOG, 40), goal=(JOG, 60), interval=6, frames=240,
                                lead=3.0, turn_rate=0.4, turn_period=150),
}


def rollout_metrics(r):
    pos, c = r["pos"], r["contacts"]
    vel = np.diff(pos, axis=0) * mb.FPS
    fs = np.linalg.norm(vel[:, FEET][..., [0, 2]], axis=-1)
    planted = (c[:-1] > 0.5) & (c[1:] > 0.5)
    acc = np.linalg.norm(np.diff(pos, n=2, axis=0) * mb.FPS ** 2, axis=-1).mean(1)        # per frame
    b = np.array([x for x in r["bounds"] if 1 <= x < len(acc) + 1])
    at_b = np.concatenate([acc[np.clip(b - 1, 0, len(acc) - 1)], acc[np.clip(b, 0, len(acc) - 1)]]) if len(b) else np.array([0.0])
    gaps = np.array(r["gaps"], dtype=float)
    return dict(frames=int(len(pos)), requests=len(r["bounds"]),
                skate_cm_s=float(fs[planted].mean() * 100) if planted.any() else 0.0,
                planted_frac=float(planted.mean()),
                jitter_m_s2=float(acc.mean()), jitter_at_boundaries_m_s2=float(at_b.mean()),
                boundary_ratio=float(at_b.mean() / max(acc.mean(), 1e-9)),
                start_gap_cm_mean=float(gaps[:, 0].mean()), start_gap_cm_max=float(gaps[:, 0].max()),
                end_gap_cm_mean=float(gaps[:, 1].mean()), end_gap_cm_max=float(gaps[:, 1].max()),
                tokens=[int(x) for x in gaps[:, 2]],
                final_root=[float(x) for x in pos[-1, 0]], path_length_m=float(np.linalg.norm(np.diff(pos[:, 0, [0, 2]], axis=0), axis=-1).sum()))


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Run the 7 rollout scenarios with one or more ONNX models and write metrics.")
    ap.add_argument("models", nargs="+", help="ONNX files")
    mb.add_path_args(ap)
    a = ap.parse_args()
    mb.apply_path_args(a)
    mb.RESULTS.mkdir(parents=True, exist_ok=True)
    clips = {p.stem: mb.Clip(p) for p in mb.MOTIONS.rglob("*.bvh")
             if p.stem in (WALK, JOG, IDLE, CROUCH, SIT, ZOMBIE)}
    for path in a.models:
        v = Path(path).stem
        gen = Generator(path)
        out, met = {}, {}
        for name, spec in SCENARIOS.items():
            fn = scenario_chain if spec["kind"] == "chain" else scenario_replan
            r = fn(gen, clips, spec, tuple(range(8, 15)))
            met[name] = rollout_metrics(r)
            out[name + "/pos"] = r["pos"].astype(np.float32)
            out[name + "/contacts"] = r["contacts"].astype(np.float32)
            out[name + "/bounds"] = np.array(r["bounds"], np.int32)
            out[name + "/targets"] = r["targets"].astype(np.float32)
        np.savez_compressed(mb.RESULTS / f"rollout_{v}.npz", **out)
        (mb.RESULTS / f"rollout_{v}.json").write_text(json.dumps(met, indent=1), encoding="utf-8")
        for name, m in met.items():
            print(f"{v:20s} {name:28s} fr={m['frames']:4d} req={m['requests']:3d} skate={m['skate_cm_s']:.2f}cm/s jitter={m['jitter_m_s2']:.2f} "
                  f"bound_ratio={m['boundary_ratio']:.2f} start_gap={m['start_gap_cm_mean']:.2f}/{m['start_gap_cm_max']:.2f}cm end_gap={m['end_gap_cm_mean']:.2f}cm path={m['path_length_m']:.2f}m")


if __name__ == "__main__":
    main()
