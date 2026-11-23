"""Real spider clips -> canonical proportions; gait extraction from the real walk; procedural (synthetic) locomotion with planted feet."""
from __future__ import annotations
import numpy as np
from .spider_lib import *
from .spider_lib import load_raw

B_CLIPS = ["Spider_Walk", "Spider_Idle", "Spider_Idle_long", "Spider_Attack_1", "Spider_Attack_2", "Spider_Damage", "Spider_Death"]
A_CLIPS = ["walk", "idle", "idle_2", "attack_1", "attack_2", "dead"]


def real_canonical(lanes=None):
    """All real clips retargeted to the canonical spider. Returns (Canon, {name: dict(pos, rig, clip)}). Walks are still in place.
    lanes: dict for Canon.set_lanes (leg-overlap fix v3: fanned neutral stance + lane IK for every clip) or None (v2 behaviour)."""
    C = Canon(); out = {}
    if lanes: C.set_lanes(**lanes)
    rawB = {c: load_raw("bga_spider", c) for c in B_CLIPS}
    restB = rawB["Spider_Walk"][1]
    stB = stand_ref([rawB["Spider_Idle"][0], rawB["Spider_Idle_long"][0]], restB)
    for c in B_CLIPS:
        if c == "Spider_Walk": continue
        out[f"bga_spider__{c}"] = dict(pos=retarget(rawB[c][0], restB, C, CANON_SCALE, stB, True), rig="bga_spider", clip=c)
    C.fit_d3([out[f"bga_spider__{c}"]["pos"] for c in B_CLIPS if c not in ("Spider_Death", "Spider_Walk")])
    # the source walk travels along world Z while the cephalothorax yaws about +-20 deg once per 0.8 s (waddle): the yaw is removed
    # (body faces the travel direction; feet unchanged; legs re-solved with the leg-shape model) so the root path stays smooth
    Rw = body_rot(rawB["Spider_Walk"][0], restB); fw = Rw[:, :, 2]; yw = np.degrees(np.arctan2(fw[:, 0], fw[:, 2]))
    C.walk_yaw_sway_deg = [float(yw.min()), float(yw.max())]
    out["bga_spider__Spider_Walk"] = dict(pos=retarget(rawB["Spider_Walk"][0], restB, C, CANON_SCALE, stB, False, "travel"), rig="bga_spider", clip="Spider_Walk")
    rawA = {c: load_raw("mixall_argiope", c) for c in A_CLIPS}
    restA = rawA["walk"][1]
    stA = stand_ref([rawA["idle"][0], rawA["idle_2"][0]], restA)
    rc = np.linalg.norm((C.rest[FEET] - C.rest[0])[:, [0, 2]], axis=1).mean()
    ra = np.linalg.norm(stA["tips_rel"][:, [0, 2]], axis=1).mean()
    for c in A_CLIPS:
        out[f"mixall_argiope__{c}"] = dict(pos=retarget(rawA[c][0], restA, C, rc / ra, stA, False), rig="mixall_argiope", clip=c)
    C.scale_A = float(rc / ra)
    return C, out


def heading(pos, C):
    R = body_rot(pos, C.rest); fwd = np.einsum("fij,j->fi", R, np.array([0.0, 0, 1]))
    return np.unwrap(np.arctan2(fwd[:, 0], fwd[:, 2])), R


def roty(a):
    a = np.asarray(a, np.float64); c, s = np.cos(a), np.sin(a)
    R = np.zeros(a.shape + (3, 3)); R[..., 0, 0] = c; R[..., 0, 2] = s; R[..., 1, 1] = 1; R[..., 2, 0] = -s; R[..., 2, 2] = c
    return R


SWING_Y = 0.04          # a foot higher than this (m) is in swing


def gait_of(pos, C, cycles=1):
    """Gait of an in-place walk clip that holds 'cycles' whole cycles (last frame = first frame). Phase 0..1 over one cycle
    (the cycles are averaged)."""
    P = pos[:-1]; T = len(P) // cycles
    P = np.mean([P[c * T:(c + 1) * T] for c in range(cycles)], 0)
    yaw, R = heading(P, C); Ry = roty(np.full(T, np.mean(yaw)))
    rel = np.einsum("fji,fkj->fki", Ry, P[:, FEET] - P[:, :1] * np.array([1.0, 0, 1]))        # [T,8,3] mean-heading frame
    vz = (np.roll(rel[..., 2], -1, 0) - np.roll(rel[..., 2], 1, 0)) * FPS / 2
    swing = rel[..., 1] > SWING_Y
    ph = np.arange(T) / T
    duty = 1 - swing.mean(0)
    ang = np.exp(2j * np.pi * ph)[:, None]
    sw_c = (np.angle((swing * ang).sum(0)) / (2 * np.pi)) % 1.0                                 # swing centre phase per leg
    stride = np.array([np.ptp(rel[~swing[:, i], i, 2]) if (~swing[:, i]).any() else 0 for i in range(8)])
    inner = ~swing & ~np.roll(swing, 1, 0) & ~np.roll(swing, -1, 0)
    speed = float(np.median([-vz[inner[:, i], i].mean() for i in range(8)]))
    height = np.array([rel[swing[:, i], i, 1].max() - rel[~swing[:, i], i, 1].mean() for i in range(8)])
    # swing height profile (16 samples, normalized time), averaged over the legs
    prof = np.zeros(16); zprof = np.zeros(16)
    for i in range(8):
        s0 = (sw_c[i] - (1 - duty[i]) / 2) % 1.0
        t = (s0 + np.linspace(0, 1, 16) * (1 - duty[i])) % 1.0
        y = np.interp(t, np.append(ph, 1.0), np.append(rel[:, i, 1], rel[0, i, 1])) - rel[~swing[:, i], i, 1].mean()
        zz = np.interp(t, np.append(ph, 1.0), np.append(rel[:, i, 2], rel[0, i, 2]))
        prof += np.maximum(y, 0) / max(y.max(), 1e-6) / 8; zprof += (zz - zz[0]) / (zz[-1] - zz[0] + 1e-9) / 8
    # body: height, sideways sway, rotation relative to the mean heading; extras (body-frame directions)
    Rrel = np.einsum("fji,fjk->fik", Ry, R)
    extra = {}
    for j in (1, 2, 3, IDX["palp_L"], IDX["palp_R"]):
        extra[NAMES[j]] = np.einsum("fji,fj->fi", R, P[:, j] - P[:, PAR[j]])
    bxz = np.einsum("fji,fj->fi", Ry, (P[:, 0] - P[:, 0].mean(0)))
    return dict(T=T / FPS, duty=duty, swing_centre=sw_c, stride=stride, speed=speed, step_height=height, swing_y=prof, swing_z=zprof,
                body_y=P[:, 0, 1] - P[:, 0, 1].mean(), body_xz=bxz, Rrel=Rrel, extra=extra, yaw_dev=yaw - np.mean(yaw), swing=swing)


def unroll(pos, C, loops=1):
    """In-place walk cycle -> travelling walk: the body moves so that the stance feet stay still (per-frame mean stance-foot velocity)."""
    P = pos[:-1]; T = len(P)
    sw_mask = P[:, FEET, 1] > SWING_Y
    per_leg = []
    for i in range(8):                       # constant body velocity = median over the legs of the mean stance-foot velocity (inner stance frames)
        st = np.array([~sw_mask[f, i] & ~sw_mask[(f + 1) % T, i] & ~sw_mask[(f - 1) % T, i] for f in range(T)])
        d = (np.roll(P[:, FEET[i]], -1, 0) - np.roll(P[:, FEET[i]], 1, 0))[st] * FPS / 2
        per_leg.append(-d.mean(0))
    v = np.median(np.array(per_leg), 0) * np.array([1.0, 0, 1])
    off = np.arange(T * loops)[:, None] * v[None] / FPS
    Q = np.tile(P, (loops, 1, 1)) + off[:, None]
    return Q


def periodic(tab, ph):
    """sample a periodic table [T,...] at phases ph [F] (linear)."""
    T = len(tab); x = (ph % 1.0) * T; i = np.floor(x).astype(int) % T; w = (x - np.floor(x)).reshape((-1,) + (1,) * (tab.ndim - 1))
    return tab[i] * (1 - w) + tab[(i + 1) % T] * w


def random_commands(rng, seconds, vmax, style="mix"):
    """Piecewise commands [F,3] = (side m/s, forward m/s, yaw rate rad/s), smoothed (acceleration limit)."""
    F = int(seconds * FPS); cmd = np.zeros((F, 3)); f = 0
    while f < F:
        n = int(rng.uniform(1.0, 5.0) * FPS); r = rng.random(); c = np.zeros(3)
        if style == "straight":
            c[1] = rng.uniform(0.2, vmax)
        elif r < 0.14: pass                                             # stand
        elif r < 0.50: c[1] = rng.uniform(0.2, vmax)                    # straight
        elif r < 0.82: c[1] = rng.uniform(0.2, vmax); c[2] = rng.uniform(-1.3, 1.3) * min(1.0, 1.2 / max(c[1], 0.3))   # curve
        elif r < 0.90: c[2] = rng.uniform(0.3, 1.5) * rng.choice([-1, 1])                                              # turn on the spot
        elif r < 0.95: c[1] = -rng.uniform(0.15, 0.6)                   # back
        else: c[0] = rng.uniform(0.15, 0.5) * rng.choice([-1, 1]); c[1] = rng.uniform(0, 0.5)                          # side step
        cmd[f:f + n] = c; f += n
    k = int(rng.integers(6, 16))
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    sm = np.stack([np.convolve(np.pad(cmd[:, i], (k, k), mode="edge"), ker, mode="valid") for i in range(3)], 1)
    key = np.concatenate([cmd[k:], np.repeat(cmd[-1:], k, 0)])      # keyboard-like stream: the key changes k frames before the centre of the response (causal)
    return sm, key


def synth_walk(C, G, idle, cmd, rng, stride_scale=1.0, height_scale=1.0, step_scale=1.0, body_h=None, phase0=0.0, preroll=60):
    """Procedural spider locomotion from the real gait G (gait_of) along the commanded root path.
    Stance feet are fixed in the world; swing feet follow the real swing arc; body bob / sway and abdomen / head motion come from the real
    walk cycle by phase (scaled with the speed) and from a real idle clip when standing. Returns world [F,J,3]."""
    cmd = np.concatenate([np.repeat(cmd[:1], preroll, 0), cmd]); F = len(cmd)
    H = C.stand_h * height_scale if body_h is None else body_h
    yaw = np.concatenate([[0], np.cumsum(cmd[:-1, 2]) / FPS])
    Ry = roty(yaw)
    vel = np.einsum("fij,fj->fi", Ry, np.stack([cmd[:, 0], np.zeros(F), cmd[:, 1]], 1))
    root = np.concatenate([np.zeros((1, 3)), np.cumsum(vel[:-1], 0) / FPS])
    neutral = C.foot_neutral()                                                          # [8,3]
    # foot ground speed in the body frame -> stride, frequency
    w = cmd[:, 2]
    fv = np.stack([cmd[:, 0][:, None] + w[:, None] * neutral[None, :, 2], cmd[:, 1][:, None] - w[:, None] * neutral[None, :, 0]], -1)   # [F,8,2]
    u = np.linalg.norm(fv, axis=-1).mean(1)
    duty = float(np.clip(np.mean(G["duty"]), 0.5, 0.75)); S0 = float(np.mean(G["stride"])); u0 = G["speed"]; f0 = 1.0 / G["T"]
    S = S0 * np.clip((u / u0) ** 0.5, 0.3, 1.5) * stride_scale
    if C.stride_cap: S = np.minimum(S, C.stride_cap)                                      # lanes: a foot must not reach into the next leg's lane
    freq = np.where(u > 0.04, np.maximum(u * duty / np.maximum(S, 1e-6), 0.45 * f0), 0.0)
    moving = u > 0.04
    start = (G["swing_centre"] - (1 - duty) / 2) % 1.0                                    # swing start phase per leg
    sw = 1 - duty
    # phase clock; when the command stops, keep the clock running until every swinging leg has landed (legs do not lift while stopped)
    phase = np.zeros(F); in_sw = np.zeros((F, 8), bool); sprog = np.zeros((F, 8)); ph = phase0
    lifted = np.zeros(8, bool)
    for f in range(F):
        psi = (ph - start) % 1.0
        want = psi < sw
        for i in range(8):
            if lifted[i]:
                if not want[i]: lifted[i] = False
            elif want[i] and moving[f] and psi[i] < sw * 0.34:
                lifted[i] = True
        phase[f] = ph; in_sw[f] = lifted; sprog[f] = np.where(lifted, psi / sw, 0)
        ph += (freq[f] if moving[f] else (f0 if lifted.any() else 0.0)) / FPS
    amp = np.clip(u / u0, 0, 1.2)
    k = 8; amp = np.convolve(np.pad(amp, (k, k), mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), mode="valid")
    # feet
    tips = np.zeros((F, 8, 3)); hmax = float(np.mean(G["step_height"])) * step_scale
    for i in range(8):
        psi = (phase - start[i]) % 1.0
        cur = root[0] + Ry[0] @ (neutral[i] + np.array([0, 0, (0.5 - ((psi[0] - sw) / duty if psi[0] >= sw else 0.5)) * S[0] * (1 if moving[0] else 0)]))
        f = 0
        while f < F:
            if not in_sw[f, i]:
                tips[f, i] = cur; f += 1; continue
            fs = f; fe = fs
            while fe < F and in_sw[fe, i]: fe += 1
            fm = fe; target_psi = sw + duty / 2                                         # mid-stance frame of the next stance
            while fm < F - 1 and not in_sw[fm, i] and ((phase[fm] - start[i]) % 1.0) < target_psi and (moving[fm] or in_sw[fm].any()): fm += 1
            fm = min(fm, F - 1)
            jit = rng.normal(0, 0.012, 3) * np.array([1, 0, 1])
            tgt = root[fm] + Ry[fm] @ (neutral[i] + jit)
            n = fe - fs
            for g in range(fs, fe):
                s = (g - fs + 1) / (n + 1)
                zz = np.interp(s, np.linspace(0, 1, 16), G["swing_z"]); zz = float(np.clip(zz, 0, 1))
                tips[g, i] = cur + (tgt - cur) * zz
                tips[g, i, 1] = hmax * np.interp(s, np.linspace(0, 1, 16), G["swing_y"]) * min(1.0, 0.35 + np.linalg.norm(tgt - cur) / max(S0, 1e-6))
            cur = tgt; f = fe
    # body
    body = root.copy(); body[:, 1] = H + periodic(G["body_y"], phase) * amp
    sway = periodic(G["body_xz"], phase) * amp[:, None]
    body += np.einsum("fij,fj->fi", Ry, np.stack([sway[:, 0], np.zeros(F), sway[:, 2] * 0], 1)) * 0.0     # root xz stays on the commanded path
    Rw = periodic(G["Rrel"], phase)
    # remove the yaw part of the real sway (the root yaw must follow the command), keep roll / pitch, scale by amp
    fw = Rw[:, :, 2]; yw = np.arctan2(fw[:, 0], fw[:, 2]); Rw = np.einsum("fij,fjk->fik", roty(-yw), Rw)
    I = np.eye(3); Rs = I + (Rw - I) * amp[:, None, None]
    U, _, Vt = np.linalg.svd(Rs); Rs = U @ Vt
    # idle life (real idle clip, looped by reflection): body height + tilt + extras, weight = 1 - amp
    wi = np.clip(1 - amp / 0.5, 0, 1); nI = len(idle["body_y"]); o = int(rng.integers(0, nI))
    ii = np.abs(((np.arange(F) + o) % (2 * nI - 2)) - (nI - 1)); ii = (nI - 1) - ii
    body[:, 1] += idle["body_y"][ii] * wi
    Ri = I + (idle["Rrel"][ii] - I) * wi[:, None, None]; U, _, Vt = np.linalg.svd(Ri); Ri = U @ Vt
    R = Ry @ Rs @ Ri
    extra = {}
    for n_ in G["extra"]:
        j = IDX[n_]; rest_off = C.rest[j] - C.rest[PAR[j]]
        e = rest_off + (periodic(G["extra"][n_], phase) - rest_off) * amp[:, None] + (idle["extra"][n_][ii] - rest_off) * wi[:, None]
        extra[n_] = e
    P = C.pose(body, R, tips, extra)
    return P[preroll:], dict(phase=phase[preroll:], in_swing=in_sw[preroll:], cmd=cmd[preroll:], tips=tips[preroll:])


def idle_of(pos, C):
    """body height change, tilt and extras of a real standing clip (for the synthetic standing phases)."""
    yaw, R = heading(pos, C); Ry = roty(np.full(len(pos), np.mean(yaw)))
    Rrel = np.einsum("fji,fjk->fik", Ry, R)
    fw = Rrel[:, :, 2]; yw = np.arctan2(fw[:, 0], fw[:, 2]); Rrel = np.einsum("fij,fjk->fik", roty(-yw), Rrel)
    Rm = Rrel.mean(0); U, _, Vt = np.linalg.svd(Rm); Rm = U @ Vt
    Rrel = np.einsum("ji,fjk->fik", Rm, Rrel)                         # relative to the clip's mean tilt
    extra = {}
    for j in (1, 2, 3, IDX["palp_L"], IDX["palp_R"]):
        e = np.einsum("fji,fj->fi", R, pos[:, j] - pos[:, PAR[j]])
        extra[NAMES[j]] = e - e.mean(0) + (C.rest[j] - C.rest[PAR[j]])
    return dict(body_y=pos[:, 0, 1] - pos[:, 0, 1].mean(), Rrel=Rrel, extra=extra)


def make_synth_clip(C, G, idle, walk_h, seed, seconds, style="mix", vmax=6.5):
    """One synthetic locomotion clip (random keyboard-like commands). Returns world [F,J,3], key stream [F,3]."""
    r = np.random.default_rng(seed)
    cmd, key = random_commands(r, seconds, vmax, style)
    if style == "mix":
        key[:, 1] = np.sign(key[:, 1]) * vmax * (np.abs(key[:, 1]) / vmax) ** 1.5       # more slow / medium walking than running
        cmd[:, 1] = np.sign(cmd[:, 1]) * vmax * (np.abs(cmd[:, 1]) / vmax) ** 1.5
    u = np.hypot(cmd[:, 0], cmd[:, 1])
    hs = r.uniform(0.94, 1.08)
    bh = C.stand_h * hs + (walk_h - C.stand_h) * np.clip(u / G["speed"], 0, 1)
    k = 10; bh = np.convolve(np.pad(bh, (k, k), mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), mode="valid")
    bh = np.concatenate([np.repeat(bh[:1], 60), bh])
    P, info = synth_walk(C, G, idle, cmd, r, stride_scale=r.uniform(0.85, 1.15), step_scale=r.uniform(0.8, 1.2), body_h=None, phase0=r.random())
    P = P.copy(); d = (bh[60:] - C.stand_h)
    # body height offset: re-pose with the offset (keep feet): shift the non-leg joints and re-solve the legs
    body = P[:, 0].copy(); body[:, 1] += d
    R = body_rot(P, C.rest)
    extra = {NAMES[j]: np.einsum("fji,fj->fi", R, P[:, j] - P[:, PAR[j]]) for j in (1, 2, 3, IDX["palp_L"], IDX["palp_R"])}
    P = C.pose(body, R, info["tips"], extra)
    if r.random() < 0.5: P = mirror(P); key = key * np.array([-1.0, 1, -1])
    return P, key
