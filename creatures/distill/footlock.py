"""Foot-lock post-process: pin planted feet with the predicted contacts, fix the leg with 2-bone IK.

This file is a library (foot_lock, FootLocker, motion_metrics); gate.py and quantize.py use it.

CPU only, runs after the network on decoded joint positions. Per leg and frame:
  1. anchor = ankle if the heel contact is on, else toe if the toe contact is on, else none;
  2. while anchored, the anchor joint keeps the ground position it had when the anchor started
     (taken from the already corrected pose, so locking never jumps = implicit blend in);
  3. after release the offset fades out linearly over BLEND frames (blend out);
  4. a lock that drifts more than MAX_OFFSET from the generated foot is released (no stuck feet);
  5. ankle and toe move by the offset, the knee is re-solved analytically (2-bone IK, original
     bend direction as pole), the reach is clamped to 98.5% of the leg length (or the generated extension if larger).
Position-level implementation for measurement and the viewer; an engine applies the same
offsets and solves the thigh/shin rotations instead.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I drops the script folder
import mb_common as mb

JN = mb.joint_names()
LEGS = (  # hip, knee, ankle, toe, heel-contact column, toe-contact column
    (JN.index("LeftLeg"), JN.index("LeftShin"), JN.index("LeftFoot"), JN.index("LeftToeBase"), 0, 1),
    (JN.index("RightLeg"), JN.index("RightShin"), JN.index("RightFoot"), JN.index("RightToeBase"), 2, 3),
)
FEET = [JN.index(n) for n in mb.FOOT_JOINTS]
BLEND = 5            # frames to fade the offset out after release (1/6 s)
MAX_OFFSET = 0.20    # m; a larger lock error means the plan moved on: release
THRESH = 0.5
REACH = 0.985       # share of the leg length the lock may stretch the leg to


class FootLocker:
    """Streaming foot-lock: feed one frame at a time (game loop). Same rule as foot_lock()."""

    def __init__(self, blend: int = BLEND, max_offset: float = MAX_OFFSET):
        self.blend, self.max_offset = blend, max_offset
        self.stats = dict(clamped=0, forced_release=0, max_offset_cm=0.0, locked_frames=0)
        self.f = 0
        self.legs = [dict(off=np.zeros(3), lock=None, prev=-1, rel_off=np.zeros(3), rel_f=-10 ** 6, blocked=False) for _ in LEGS]

    def step(self, pos: np.ndarray, contacts: np.ndarray) -> np.ndarray:
        """pos [30,3] (Y up), contacts [4] -> corrected [30,3]."""
        out = pos.copy()
        f, stats = self.f, self.stats
        for (hip, knee, ankle, toe, ch, ct), L in zip(LEGS, self.legs):
            anchor = ankle if contacts[ch] > THRESH else (toe if contacts[ct] > THRESH else -1)
            if anchor < 0:
                L["blocked"] = False               # after a forced release, wait for the contact to end
            if anchor >= 0 and not L["blocked"]:
                if anchor != L["prev"] or L["lock"] is None:
                    L["lock"] = pos[anchor] + L["off"]       # where the anchor is right now (corrected pose)
                new = L["lock"] - pos[anchor]
                new[1] = 0.0                              # ground plane only; height stays the model's
                if np.hypot(new[0], new[2]) > self.max_offset:
                    stats["forced_release"] += 1
                    L["blocked"], L["lock"], anchor = True, None, -1
                    L["rel_off"], L["rel_f"] = L["off"].copy(), f
                else:
                    L["off"] = new
                    stats["locked_frames"] += 1
            if anchor < 0 or L["blocked"]:
                if L["prev"] >= 0 and not L["blocked"]:
                    L["rel_off"], L["rel_f"] = L["off"].copy(), f
                    L["lock"] = None
                L["off"] = L["rel_off"] * max(0.0, 1.0 - (f - L["rel_f"] + 1) / self.blend)
            L["prev"] = anchor if not L["blocked"] else -1
            off = L["off"]
            if not off.any():
                continue
            stats["max_offset_cm"] = max(stats["max_offset_cm"], float(np.hypot(off[0], off[2]) * 100))
            h, k, a = pos[hip], pos[knee], pos[ankle]
            l1, l2 = np.linalg.norm(k - h), np.linalg.norm(a - k)
            tgt = a + off
            v = tgt - h
            d = np.linalg.norm(v)
            # never straighten the leg more than the generated pose did (or 98.5%): a nearly straight
            # leg makes the knee position ill-conditioned and shows as a knee pop
            dmax = max((l1 + l2) * REACH, np.linalg.norm(a - h))
            if d > dmax:
                stats["clamped"] += 1
                v *= dmax / d
                d = dmax
                tgt = h + v
            along = v / max(d, 1e-9)
            x = (l1 * l1 - l2 * l2 + d * d) / (2 * d)
            r = np.sqrt(max(l1 * l1 - x * x, 0.0))
            pole = (k - h) - along * np.dot(k - h, along)
            pn = np.linalg.norm(pole)
            pole = pole / pn if pn > 1e-6 else np.array([0.0, 0.0, 0.0])
            out[knee] = h + along * x + pole * r
            out[ankle] = tgt
            out[toe] = pos[toe] + (tgt - a)
        self.f += 1
        return out


def foot_lock(pos: np.ndarray, contacts: np.ndarray, blend: int = BLEND, max_offset: float = MAX_OFFSET):
    """pos [F,30,3] (Y up), contacts [F,4] -> (new pos, stats). Offline wrapper of FootLocker."""
    fl = FootLocker(blend, max_offset)
    out = np.stack([fl.step(pos[f], contacts[f]) for f in range(pos.shape[0])]) if len(pos) else pos.copy()
    return out, fl.stats


def motion_metrics(pos, contacts):
    vel = np.diff(pos, axis=0) * mb.FPS
    fs = np.linalg.norm(vel[:, FEET][..., [0, 2]], axis=-1)
    planted = (contacts[:-1] > THRESH) & (contacts[1:] > THRESH)
    acc = np.linalg.norm(np.diff(pos, n=2, axis=0) * mb.FPS ** 2, axis=-1)
    knees = [LEGS[0][1], LEGS[1][1]]
    return dict(skate=float(fs[planted].mean() * 100) if planted.any() else 0.0,
                skate_p95=float(np.percentile(fs[planted], 95) * 100) if planted.any() else 0.0,
                jitter=float(acc.mean()), knee_acc_max=float(acc[:, knees].max()) if len(acc) else 0.0,
                knee_acc_mean=float(acc[:, knees].mean()) if len(acc) else 0.0)
