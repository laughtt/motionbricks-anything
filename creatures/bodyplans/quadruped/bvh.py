"""Minimal BVH reader: hierarchy + motion -> world joint positions [F,J,3] (BVH units), names, parents, fps."""
from __future__ import annotations
import numpy as np


def _rotmat(axis, deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    if axis == "X":
        m = [[o, z, z], [z, c, -s], [z, s, c]]
    elif axis == "Y":
        m = [[c, z, s], [z, o, z], [-s, z, c]]
    else:
        m = [[c, -s, z], [s, c, z], [z, z, o]]
    return np.stack([np.stack(r, -1) for r in m], -2)


def read(path):
    txt = open(path, encoding="utf-8", errors="ignore").read().split()
    i = 0
    names, parents, offsets, chans = [], [], [], []
    stack = []
    while txt[i] != "MOTION":
        t = txt[i]
        if t in ("ROOT", "JOINT"):
            names.append(txt[i + 1]); parents.append(stack[-1] if stack else -1); offsets.append(None); chans.append([])
            i += 2
        elif t == "End":
            names.append(names[stack[-1]] + "_end"); parents.append(stack[-1]); offsets.append(None); chans.append([])
            i += 2
        elif t == "{":
            stack.append(len(names) - 1); i += 1
        elif t == "}":
            stack.pop(); i += 1
        elif t == "OFFSET":
            offsets[stack[-1]] = [float(x) for x in txt[i + 1:i + 4]]; i += 4
        elif t == "CHANNELS":
            n = int(txt[i + 1]); chans[stack[-1]] = txt[i + 2:i + 2 + n]; i += 2 + n
        else:
            i += 1
    nf = int(txt[i + 2]); ft = float(txt[i + 5]); i += 6
    data = np.array(txt[i:], dtype=np.float64)
    nc = sum(len(c) for c in chans)
    data = data[: nf * nc].reshape(nf, nc)
    J = len(names)
    off = np.array(offsets)
    pos = np.zeros((nf, J, 3)); R = np.zeros((nf, J, 3, 3))
    col = 0
    for j in range(J):
        lr = np.broadcast_to(np.eye(3), (nf, 3, 3)).copy(); lt = np.broadcast_to(off[j], (nf, 3)).copy()
        for c in chans[j]:
            v = data[:, col]; col += 1
            if c.endswith("position"):
                lt[:, "XYZ".index(c[0])] = v
            else:
                lr = lr @ _rotmat(c[0], v)
        p = parents[j]
        if p < 0:
            R[:, j] = lr; pos[:, j] = lt
        else:
            R[:, j] = R[:, p] @ lr; pos[:, j] = pos[:, p] + np.einsum("fij,fj->fi", R[:, p], lt)
    return dict(pos=pos, names=names, parents=np.array(parents), fps=1.0 / ft, offsets=off)
