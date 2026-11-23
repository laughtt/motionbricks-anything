"""Clip (world joint positions [F,J,3], metres, Y up, forward +Z, left +X) -> MotionBricks-style per-frame features, driven by a skeleton spec.
pose feature per frame = local joint positions (heading frame, root xz at 0) J*3 + joint velocities J*3 (world velocity in the heading frame) + C foot contacts
root values per frame  = [vx side, vz forward (heading frame), yaw rate, root height]
spec: {"root": joint id, "front": joint id (heading = root -> front), "feet" or "contacts": [joint ids],
       "contact_speed": m/s (default 0.35), "contact_height": m (default 0.06)}"""
from __future__ import annotations
import numpy as np

FPS = 30


def resample(pos, fps):
    """[F,J,3] at any frame rate -> 30 fps (linear)."""
    if abs(fps - FPS) < 1e-3:
        return pos
    F = len(pos); T = (F - 1) / fps
    t_new = np.arange(0, T, 1.0 / FPS)
    x = np.arange(F) / fps
    i = np.clip(np.searchsorted(x, t_new) - 1, 0, F - 2)
    w = ((t_new - x[i]) * fps)[:, None, None]
    return pos[i] * (1 - w) + pos[i + 1] * w


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    R = np.zeros(a.shape + (3, 3)); R[..., 0, 0] = c; R[..., 0, 2] = s; R[..., 1, 1] = 1; R[..., 2, 0] = -s; R[..., 2, 2] = c
    return R


def features(pos, spec, floor=None):
    """pos [F,J,3] at 30 fps -> dict(pose [F-1,D], root [F-1,4], local [F-1,J,3], contact [F-1,C]).
    floor: height of the ground in the clip (None = the 5th percentile of the lowest foot)."""
    pos = pos.astype(np.float64).copy(); feet = spec.get("feet", spec.get("contacts"))
    if floor is None:
        floor = np.percentile(pos[:, feet, 1].min(1), 5)
    pos[..., 1] -= floor
    f = pos[:, spec["front"]] - pos[:, spec["root"]]
    yaw = np.unwrap(np.arctan2(f[:, 0], f[:, 2]))
    Rinv = rot_y(-yaw)                       # world -> heading frame
    r = pos[:, spec["root"]].copy(); r[:, 1] = 0
    loc = np.einsum("fij,fkj->fki", Rinv, pos - r[:, None])
    vel = np.einsum("fij,fkj->fki", Rinv[:-1], pos[1:] - pos[:-1]) * FPS
    rv = np.einsum("fij,fj->fi", Rinv[:-1], r[1:] - r[:-1]) * FPS
    dyaw = (yaw[1:] - yaw[:-1]) * FPS
    h = pos[:-1, spec["root"], 1]
    fv = np.linalg.norm(pos[1:, feet] - pos[:-1, feet], axis=-1) * FPS
    fh = pos[:-1, feet, 1]
    contact = ((fv < spec.get("contact_speed", 0.35)) & (fh < spec.get("contact_height", 0.06))).astype(np.float64)
    F = len(pos) - 1
    pose = np.concatenate([loc[:-1].reshape(F, -1), vel.reshape(F, -1), contact], 1)
    root = np.stack([rv[:, 0], rv[:, 2], dyaw, h], 1)
    return dict(pose=pose.astype(np.float32), root=root.astype(np.float32), local=loc[:-1].astype(np.float32), contact=contact.astype(np.float32))


def integrate(local, root):
    """heading-frame joints [F,J,3] + root values [F,4] -> world joints (start at the origin, facing +Z)."""
    yaw, x, z = 0.0, 0.0, 0.0; world = np.zeros_like(local)
    for f in range(len(local)):
        c, s = np.cos(yaw), np.sin(yaw); Rm = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        world[f] = local[f] @ Rm.T + np.array([x, 0, z])
        v = Rm @ np.array([root[f, 0], 0, root[f, 1]]) / FPS; x += v[0]; z += v[2]; yaw += root[f, 2] / FPS
    return world
