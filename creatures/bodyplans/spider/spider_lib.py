"""Spider canonical skeleton tools (numpy only): canonical rest, body frame, leg IK, retarget to canonical proportions, gait extraction.
World axes: Y up, forward +Z, left +X, metres, 30 fps."""
from __future__ import annotations
import json, os
import numpy as np
from .spider_joints import NAMES, PAR, LEGS, FEET, MIRROR, IDX, J

FPS = 30
HIPS = [l[0] for l in LEGS]
RIGID = [0, 1, 2] + HIPS                     # joints rigid with the cephalothorax
RAW = os.environ.get("SPIDER_RAW", "")      # folder with the extracted clips (extract_clips.py); set_raw_dir() or env SPIDER_RAW
CANON_SCALE = 0.025                          # bga_spider units -> metres (body height ~0.48 m, leg span ~3.5 m: a mount)


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def set_raw_dir(path):
    global RAW
    RAW = str(path)


def load_raw(rig, clip):
    if not RAW or not os.path.isfile(f"{RAW}/{rig}__{clip}.npz"):
        raise FileNotFoundError(f"spider clip {rig}__{clip}.npz not found in '{RAW}': run extract_clips.py first and give --raw (or env SPIDER_RAW)")
    z = np.load(f"{RAW}/{rig}__{clip}.npz")
    return z["pos"][:, :J].astype(np.float64), z["rest"][:J].astype(np.float64)


def mirror(pos):
    """left/right mirror of [F,J,3] (x -> -x, swap L/R joints)."""
    return pos[:, MIRROR] * np.array([-1.0, 1, 1])


def canonical_rest():
    """bga_spider bind pose, symmetrized, scaled; feet on y=0; body xz at 0."""
    _, rest = load_raw("bga_spider", "Spider_Walk")
    r = rest * CANON_SCALE
    r = (r + r[MIRROR] * np.array([-1.0, 1, 1])) / 2
    r[:, [0, 2]] -= r[0, [0, 2]]
    r[:, 1] -= r[FEET, 1].min()
    for j in (0, 1, 2, 3):
        r[j, 0] = 0
    return r


def kabsch(A, B):
    """rotation R [F,3,3] with R @ A_i ~= B_i. A [N,3] (centred), B [F,N,3] (centred)."""
    H = np.einsum("fni,nj->fij", B, A)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.zeros((len(B), 3, 3)); D[:, 0, 0] = 1; D[:, 1, 1] = 1; D[:, 2, 2] = d
    return U @ D @ Vt


def body_rot(pos, rest):
    """cephalothorax rotation per frame relative to the rest pose (rigid joints: body, head, abdomen junction, 8 hips)."""
    A = rest[RIGID] - rest[RIGID].mean(0)
    B = pos[:, RIGID] - pos[:, RIGID].mean(1, keepdims=True)
    return kabsch(A, B)


def rot_between(a, b):
    """minimal rotation matrices taking unit vectors a -> b ([F,3])."""
    a = unit(a); b = unit(b)
    v = np.cross(a, b); c = (a * b).sum(-1); s2 = (v * v).sum(-1)
    K = np.zeros(a.shape[:-1] + (3, 3))
    K[..., 0, 1] = -v[..., 2]; K[..., 0, 2] = v[..., 1]; K[..., 1, 0] = v[..., 2]; K[..., 1, 2] = -v[..., 0]; K[..., 2, 0] = -v[..., 1]; K[..., 2, 1] = v[..., 0]
    k = np.where(s2 > 1e-12, (1 - c) / np.maximum(s2, 1e-12), 0.5)
    return np.eye(3) + K + K @ K * k[..., None, None]


def leg_ik(hip, tip, d3, pole, l1, l2, l3):
    """3-segment leg. ankle = tip - l3*d3 (d3 = ankle->tip direction hint), knee from a 2-bone solve bent toward 'pole'.
    All [F,3]. Returns knee, ankle, tip (tip is moved only when the chain cannot reach)."""
    ankle = tip - l3 * unit(d3)
    for _ in range(3):
        v = ankle - hip; d = np.linalg.norm(v, axis=-1, keepdims=True)
        dc = np.clip(d, abs(l1 - l2) * 1.02 + 1e-4, (l1 + l2) * 0.995)
        ankle = hip + v / np.maximum(d, 1e-9) * dc
        ankle = tip + unit(ankle - tip) * l3
    v = ankle - hip; d = np.linalg.norm(v, axis=-1, keepdims=True)
    dc = np.clip(d, abs(l1 - l2) * 1.02 + 1e-4, (l1 + l2) * 0.995)
    ankle = hip + v / np.maximum(d, 1e-9) * dc
    n = unit(ankle - hip)
    a = (l1 * l1 - l2 * l2 + dc * dc) / (2 * dc)
    h = np.sqrt(np.maximum(l1 * l1 - a * a, 0))
    p = pole - (pole * n).sum(-1, keepdims=True) * n
    knee = hip + a * n + h * unit(p)
    tip2 = ankle + unit(tip - ankle) * l3
    return knee, ankle, tip2


class Canon:
    def __init__(self):
        self.rest = canonical_rest()
        self.len = np.array([np.linalg.norm(self.rest[j] - self.rest[PAR[j]]) if PAR[j] >= 0 else 0 for j in range(J)])
        self.stand_h = float(self.rest[0, 1])
        self.d3_fit = None
        self.lane = None; self.stride_cap = None

    def set_lanes(self, fan_deg, k_ankle=0.6, k_lift=0.5, stride_cap=None, knee_fan_deg=None):
        """Leg lanes (leg-overlap fix, v3). The neutral stance = the rest leg turned about the vertical axis at the hip by fan_deg (per leg pair,
        + = toward the front), so every leg owns an azimuth sector. Lane IK (pose()): the ankle follows only k_ankle of the foot's offset from
        its neutral place (the tarsus leans for the rest) and the knee stays on the side of its neutral place, so a femur swings less than its
        foot and does not sweep through the neighbour leg. knee_fan_deg: own turn for the neutral knee (a rear femur that follows its foot lane
        goes into the abdomen). stride_cap (m): longest stride of the synthetic gait."""
        self.nleg = np.zeros((8, 4, 3))                       # neutral hip, knee, ankle, tip relative to the body joint (body frame)
        for li, leg in enumerate(LEGS):
            a = np.radians(fan_deg[li % 4]) * (-1 if li < 4 else 1); c, s = np.cos(a), np.sin(a)
            Rm = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            for q, j in enumerate(leg): self.nleg[li, q] = self.rest[leg[0]] + Rm @ (self.rest[j] - self.rest[leg[0]]) - self.rest[0]
            if knee_fan_deg is not None:
                a = np.radians(knee_fan_deg[li % 4]) * (-1 if li < 4 else 1); c, s = np.cos(a), np.sin(a)
                self.nleg[li, 1] = self.rest[leg[0]] + np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]) @ (self.rest[leg[1]] - self.rest[leg[0]]) - self.rest[0]
        self.lane = dict(fan_deg=[float(x) for x in fan_deg], k_ankle=float(k_ankle), k_lift=float(k_lift), knee_fan_deg=knee_fan_deg); self.stride_cap = stride_cap

    def foot_neutral(self):
        """neutral foot places relative to the body, on the ground [8,3]."""
        if self.lane is None: return (self.rest[FEET] - self.rest[0]) * np.array([1.0, 0, 1])
        return self.nleg[:, 3] * np.array([1.0, 0, 1])

    def fit_d3(self, clips):
        """Linear model of the last-segment direction (body frame) from the foot position relative to the hip (body frame),
        per leg pair (left legs + mirrored right legs). clips: list of canonical [F,J,3]."""
        X = [[] for _ in range(4)]; Y = [[] for _ in range(4)]
        for P in clips:
            for PP in (P, mirror(P)):
                R = body_rot(PP, self.rest)
                for i in range(4):
                    h, k, a, t = LEGS[i]
                    rel = np.einsum("fji,fj->fi", R, PP[:, t] - PP[:, h])
                    d3 = np.einsum("fji,fj->fi", R, unit(PP[:, t] - PP[:, a]))
                    X[i].append(np.concatenate([rel, np.ones((len(rel), 1))], 1)); Y[i].append(d3)
        self.d3_fit = [np.linalg.lstsq(np.concatenate(X[i]), np.concatenate(Y[i]), rcond=1e-3)[0] for i in range(4)]

    def d3(self, leg, R, hip, tip):
        """ankle->tip direction hint (world) for leg index 0..7."""
        i = leg % 4; sgn = np.array([-1.0, 1, 1]) if leg >= 4 else np.ones(3)
        if self.d3_fit is None:
            d = self.rest[LEGS[leg][3]] - self.rest[LEGS[leg][2]]
            return np.einsum("fij,j->fi", R, unit(d))
        rel = np.einsum("fji,fj->fi", R, tip - hip) * sgn
        d = np.concatenate([rel, np.ones((len(rel), 1))], 1) @ self.d3_fit[i]
        return np.einsum("fij,fj->fi", R, unit(d) * sgn)

    def pose(self, body, R, tips, extra=None, pole=None, d3=None):
        """Full canonical pose from body position [F,3], body rotation [F,3,3], 8 tip targets [F,8,3].
        extra: dict joint name -> body-frame offset [F,3] from its parent (overrides rest) for head / abdomen / abdomen_end / palps."""
        F = len(body); P = np.zeros((F, J, 3)); P[:, 0] = body
        extra = extra or {}
        for j in (1, 2, 3, IDX["palp_L"], IDX["palp_R"]):
            off = extra.get(NAMES[j])
            off = np.broadcast_to(self.rest[j] - self.rest[PAR[j]], (F, 3)) if off is None else unit(off) * self.len[j]
            P[:, j] = P[:, PAR[j]] + np.einsum("fij,fj->fi", R, off)
        up = np.broadcast_to(np.array([0.0, 1, 0]), (F, 3))
        for li, (h, k, a, t) in enumerate(LEGS):
            P[:, h] = body + np.einsum("fij,j->fi", R, self.rest[h] - self.rest[0])
            if self.lane is not None and d3 is None:
                nk, na, nt = (body + np.einsum("fij,j->fi", R, self.nleg[li, q]) for q in (1, 2, 3))
                off = tips[:, li] - nt
                at = na + off * np.array([self.lane["k_ankle"], self.lane["k_lift"], self.lane["k_ankle"]])
                dd = tips[:, li] - at; pp = nk - P[:, h]
            else:
                dd = d3[li] if d3 is not None else self.d3(li, R, P[:, h], tips[:, li])
                pp = pole[li] if pole is not None else up
            P[:, k], P[:, a], P[:, t] = leg_ik(P[:, h], tips[:, li], dd, pp, self.len[k], self.len[a], self.len[t])
        return P


def retarget(pos, src_rest, canon: Canon, scale, src_stand, keep_leg_shape, yaw_fix=0.0):
    """Source clip [F,J,3] (canonical axes, source units) -> canonical proportions.
    Body: position scaled, height = canonical standing height + scaled source height change; rotation = source cephalothorax rotation.
    Head / abdomen / palps: rest-pose delta (the canonical offset turned by the source bone's rotation from its own rest direction).
    Legs: foot target = canonical neutral foot + scaled source foot offset from the source's neutral stance (body heading frame) -> leg IK
    (keep_leg_shape: last-segment direction and knee side from the source; else the canonical leg-shape model).
    src_stand: dict(tips_rel [8,3] neutral foot positions relative to the body in the heading frame, floor [8], body_h)."""
    F = len(pos); R = body_rot(pos, src_rest); R0 = R
    if isinstance(yaw_fix, str):             # "travel": per frame, turn the cephalothorax about the vertical so that it faces +Z (the feet stay)
        fw = R[:, :, 2]; a = -np.arctan2(fw[:, 0], fw[:, 2]); c, s = np.cos(a), np.sin(a)
        Rf = np.zeros((F, 3, 3)); Rf[:, 0, 0] = c; Rf[:, 0, 2] = s; Rf[:, 1, 1] = 1; Rf[:, 2, 0] = -s; Rf[:, 2, 2] = c
        R = Rf @ R
    fwd = np.einsum("fij,j->fi", R, np.array([0.0, 0, 1])); yaw = np.arctan2(fwd[:, 0], fwd[:, 2])
    c, s = np.cos(yaw), np.sin(yaw)
    Ry = np.zeros((F, 3, 3)); Ry[:, 0, 0] = c; Ry[:, 0, 2] = s; Ry[:, 1, 1] = 1; Ry[:, 2, 0] = -s; Ry[:, 2, 2] = c
    body = pos[:, 0] * scale
    body[:, 1] = canon.stand_h + (pos[:, 0, 1] - src_stand["body_h"]) * scale
    extra = {}
    for j in (1, 2, 3, IDX["palp_L"], IDX["palp_R"]):
        p = PAR[j]
        d_rest = np.broadcast_to(unit(src_rest[j] - src_rest[p]), (F, 3))
        d_now = np.einsum("fji,fj->fi", R0, unit(pos[:, j] - pos[:, p]))         # source direction in the source body frame
        extra[NAMES[j]] = np.einsum("fij,j->fi", rot_between(d_rest, d_now), canon.rest[j] - canon.rest[p])
    tips = np.zeros((F, 8, 3)); pole = []; d3 = []
    for li, (h, k, a, t) in enumerate(LEGS):
        rel = np.einsum("fji,fj->fi", Ry, pos[:, t] - pos[:, 0]); rel[:, 1] = pos[:, t, 1] - src_stand["floor"][li]
        dev = (rel - np.array([src_stand["tips_rel"][li][0], 0, src_stand["tips_rel"][li][2]])) * scale
        neutral = canon.foot_neutral()[li]
        tips[:, li] = body * np.array([1.0, 0, 1]) + np.einsum("fij,fj->fi", Ry, neutral + dev)
        tips[:, li, 1] = np.maximum(tips[:, li, 1], 0)
        pole.append(pos[:, k] - pos[:, h]); d3.append(pos[:, t] - pos[:, a])
    if keep_leg_shape and canon.lane is None:
        return canon.pose(body, R, tips, extra, pole=pole, d3=d3)
    return canon.pose(body, R, tips, extra)


def stand_ref(pos_list, src_rest):
    """Neutral stance of a source rig from its clips (median over frames): foot positions relative to the body (heading frame), per-foot floor."""
    rel = []; low = []; bh = []
    for pos in pos_list:
        R = body_rot(pos, src_rest); fwd = np.einsum("fij,j->fi", R, np.array([0.0, 0, 1])); yaw = np.arctan2(fwd[:, 0], fwd[:, 2])
        c, s = np.cos(yaw), np.sin(yaw)
        for li, l in enumerate(LEGS):
            d = pos[:, l[3]] - pos[:, 0]
            rel.append(np.stack([c * d[:, 0] - s * d[:, 2], d[:, 1], s * d[:, 0] + c * d[:, 2]], 1))
        low.append(pos[:, FEET, 1]); bh.append(pos[:, 0, 1])
    rel = np.stack([np.concatenate(rel[i::8]) for i in range(8)], 0)          # [8, N, 3]
    floor = np.percentile(np.concatenate(low), 3, axis=0)
    return dict(tips_rel=np.median(rel, 1), floor=floor, body_h=float(np.median(np.concatenate(bh))))
