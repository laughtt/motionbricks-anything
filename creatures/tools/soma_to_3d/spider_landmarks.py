"""Semi-automatic landmarks: place the 38 spider SOMA joints inside a generated mesh (same pose as the SOMA template).
python -I spider_landmarks.py <gen_mesh.npz> <soma_mesh.npz> <spider_soma_spec.json> <landmarks.json> <out_prefix> [init]
(npz = blender_mesh_dump.py output.)  'init' = (re)compute yaw / scale / offset by the best chamfer over 4 yaws and start from the SOMA rest joints;
without it the json is read and only the views are drawn (after hand edits: json key "hand": {joint: [x, y, z]} overrides).
Canonical fit frame (Blender axes, metres): V' = Rz(yaw_deg) V * scale + offset; forward -Y, left +X, ground z=0.
Refinement: leg joints snap to the centre of the leg tube around them (mesh vertices nearer to this leg than to any other leg or the body),
tips to the lowest point of the last segment; abdomen end to the rear cap."""
import sys, json, math, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
gen, soma, specf, LMF, OUT = sys.argv[1:6]; INIT = len(sys.argv) > 6 and sys.argv[6] == "init"
spec = json.load(open(specf)); NAMES, PAR, LEGS = spec["joints"], spec["parents"], spec["legs"]; J = len(NAMES)
Js = np.array(spec["rest_blender"], float)
z = np.load(gen); V0 = z["V"].astype(float); col = z["col"]; tri = z["tri"]
S = np.load(soma)["V"].astype(float)


def canon(V, rot, scale, off):
    return V @ np.array(rot, float).T * scale + np.array(off)


def umeyama(A, B):
    """similarity (R, s, t) with s R A + t ~= B."""
    ma, mb = A.mean(0), B.mean(0); H = (B - mb).T @ (A - ma) / len(A)
    U, D, Vt = np.linalg.svd(H); d = np.sign(np.linalg.det(U @ Vt)); R = U @ np.diag([1, 1, d]) @ Vt
    s = (D * [1, 1, d]).sum() / ((A - ma) ** 2).sum(1).mean()
    return R, s, mb - s * R @ ma


def seg_dist(P, a, b):
    ab = b - a; t = np.clip(((P - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1), t


if INIT or not os.path.exists(LMF):
    # The image-to-3D mesh comes in the CAMERA frame of the template render (x = image right, z = image up, y = depth).
    # Start from the known template camera (yaw / elevation of render_spider_template.py), both depth signs, then similarity ICP to the SOMA mesh.
    YAW, EL = math.radians(float(os.environ.get("TPL_YAW", "35"))), math.radians(float(os.environ.get("TPL_ELEV", "32")))
    d = np.array([math.cos(YAW) * math.cos(EL), -math.sin(YAW) * math.cos(EL), math.sin(EL)]); f = -d
    rgt = np.cross(f, [0, 0, 1.0]); rgt /= np.linalg.norm(rgt); up = np.cross(rgt, f)
    best = None; ts = cKDTree(S); Ssub = S[::3]
    for sgn in (1, -1):
        for flipx in (1,):
            R = np.stack([rgt * (1 if sgn == 1 else -1), f * sgn, up], 1)          # columns: where the mesh x, y, z axes go (proper rotation for both signs)
            if np.linalg.det(R) < 0: R[:, 0] *= -1
            V = V0 @ R.T
            sc = float(np.sqrt(np.ptp(S[:, 0]) * np.ptp(S[:, 1]) / (np.ptp(V[:, 0]) * np.ptp(V[:, 1])))); t = S.mean(0) - (V * sc).mean(0)
            for it in range(40):                                   # similarity ICP (both directions)
                Vc = V0 @ R.T * sc + t; sub = Vc[::5]
                i1 = ts.query(sub)[1]; tv = cKDTree(Vc); i2 = tv.query(Ssub)[1]
                A = np.concatenate([V0[::5], V0[i2]]); B = np.concatenate([S[i1], Ssub])
                R, sc, t = umeyama(A, B)
            Vc = V0 @ R.T * sc + t
            ch = ts.query(Vc[::5])[0].mean() + cKDTree(Vc).query(Ssub)[0].mean()
            print("depth sign %2d scale %.3f chamfer %.4f" % (sgn, sc, ch))
            if best is None or ch < best[0]: best = (ch, R.tolist(), float(sc), t.tolist())
    lm = dict(rot=best[1], scale=best[2], offset=best[3], chamfer=best[0], names=NAMES, hand={})
    V = canon(V0, lm["rot"], lm["scale"], lm["offset"])
    lm["offset"][2] -= float(np.percentile(V[:, 2], 0.2)); V = canon(V0, lm["rot"], lm["scale"], lm["offset"])     # feet on z = 0
    Jf = Js.copy()
    # ---- legs from the mesh itself: remove the body, connected components = legs, geodesic centre line, best 3-segment chain
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components, dijkstra
    key = np.round(V / 1e-5).astype(np.int64); _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True); inv = inv.ravel()
    U = V[first]; tw = inv[tri]
    e = np.concatenate([tw[:, [0, 1]], tw[:, [1, 2]], tw[:, [2, 0]]]); e = e[e[:, 0] != e[:, 1]]
    BODY_R = float(os.environ.get("BODY_R", "0.40")); ABD_R = float(os.environ.get("ABD_R", "0.66"))
    d_c = seg_dist(U, Jf[1], Jf[2])[0]; d_a = seg_dist(U, Jf[2] + (Jf[3] - Jf[2]) * 0.25, Jf[3])[0]
    leg = (d_c > BODY_R) & (d_a > ABD_R)
    ee = e[leg[e[:, 0]] & leg[e[:, 1]]]; w = np.linalg.norm(U[ee[:, 0]] - U[ee[:, 1]], axis=1)
    Gm = coo_matrix((np.concatenate([w, w]), (np.concatenate([ee[:, 0], ee[:, 1]]), np.concatenate([ee[:, 1], ee[:, 0]]))), shape=(len(U), len(U))).tocsr()
    nc, lab = connected_components(Gm, directed=False)
    comps = []
    for c_ in range(nc):
        m = (lab == c_) & leg
        if m.sum() < 150: continue
        P = U[m]; ext = np.linalg.norm(P - Jf[0], axis=1)
        if ext.max() - ext.min() < 0.6: continue
        comps.append(np.nonzero(m)[0])
    print("LEG components:", [(len(c_), np.round(U[c_][np.argmax(np.linalg.norm(U[c_] - Jf[0], axis=1))], 2).tolist()) for c_ in comps])
    lines = []
    for c_ in comps:
        P = U[c_]; dr = np.linalg.norm(P[:, :2] - Jf[0, :2], axis=1)
        src = c_[dr <= np.percentile(dr, 4)]
        gd = dijkstra(Gm, directed=False, indices=src, min_only=True)[c_]
        ok = np.isfinite(gd); P = P[ok]; gd = gd[ok]
        qs = np.quantile(gd, np.linspace(0, 1, 17)); cen = np.array([P[(gd >= qs[k]) & (gd <= qs[k + 1])].mean(0) for k in range(16)])
        far = P[gd >= np.quantile(gd, 0.97)]; tip = far[far[:, 2] <= np.percentile(far[:, 2], 30)].mean(0)
        lines.append((cen, tip))
    left = sorted([l for l in lines if l[1][0] > 0], key=lambda l: l[1][1]); right = sorted([l for l in lines if l[1][0] <= 0], key=lambda l: l[1][1])
    print("legs found: left(+x) %d right(-x) %d" % (len(left), len(right)))
    def chain_err(pts, chain):
        return sum(min(seg_dist(p[None], chain[k], chain[k + 1])[0][0] for k in range(3)) for p in pts)
    for side, found in (("L", left), ("R", right)):
        if len(found) != 4:
            # fused legs on this side: find the 4 feet (blobs of low vertices), split the leg vertices by the nearest hip->foot line in the top view,
            # centre line = centroids of bins along that line
            sg = 1 if side == "L" else -1
            sidem = leg & (U[:, 0] * sg > 0.5)
            low = U[sidem & (U[:, 2] < np.percentile(U[sidem, 2], 1) + float(os.environ.get("FOOT_Z", "0.22")))]
            tk = cKDTree(low); seen = np.zeros(len(low), bool); blobs = []
            for i0 in range(len(low)):
                if seen[i0]: continue
                st = [i0]; seen[i0] = True; cur = []
                while st:
                    a_ = st.pop(); cur.append(a_)
                    for b_ in tk.query_ball_point(low[a_], 0.09):
                        if not seen[b_]: seen[b_] = True; st.append(b_)
                if len(cur) >= 12: blobs.append(low[cur])
            blobs = sorted(blobs, key=len, reverse=True)[:4]; blobs = sorted(blobs, key=lambda b: b[:, 1].mean())
            print("  side", side, ": legs fused; feet blobs:", [(len(b), np.round(b.mean(0), 2).tolist()) for b in blobs])
            if len(blobs) != 4: print("  -> SOMA rest joints kept (hand placement needed)"); continue
            tips_ = [b[b[:, 2] <= np.percentile(b[:, 2], 25)].mean(0) for b in blobs]
            hips_ = [Jf[LEGS[i + (0 if side == "L" else 4)][0]] for i in range(4)]
            Pm = U[leg & (U[:, 0] * sg > 0.02)]
            d2 = np.stack([seg_dist(Pm[:, :2], hips_[i][:2], tips_[i][:2])[0] for i in range(4)], 1); own = d2.argmin(1)
            found = []
            for i in range(4):
                P = Pm[(own == i) & (d2.min(1) < 0.2)]; t_ = seg_dist(P[:, :2], hips_[i][:2], tips_[i][:2])[1]
                qs = np.linspace(0.12, 1.0, 17); cen = []
                for k_ in range(16):
                    q = P[(t_ >= qs[k_]) & (t_ <= qs[k_ + 1])]
                    cen.append(q.mean(0) if len(q) >= 4 else (cen[-1] if cen else hips_[i]))
                found.append((np.array(cen), tips_[i]))
        for i, (cen, tip) in enumerate(found):
            l = LEGS[i + (0 if side == "L" else 4)]; hip = Jf[l[0]]; best_ = None
            zmax = cen[2:12, 2].max()
            for a in range(2, 12):
                if cen[a, 2] < zmax - 0.10: continue                  # the knee is the apex of the leg
                for b in range(a + 2, 15):
                    ch_ = [hip, cen[a], cen[b], tip]
                    er = chain_err(cen, ch_) + float(os.environ.get("LEN_W", "0.35")) * sum(abs(math.log(max(np.linalg.norm(ch_[k + 1] - ch_[k]), 1e-3) / np.linalg.norm(Js[l[k + 1]] - Js[l[k]]))) for k in range(3))
                    if best_ is None or er < best_[0]: best_ = (er, a, b)
            Jf[l[1]] = cen[best_[1]]; Jf[l[2]] = cen[best_[2]]; Jf[l[3]] = tip
    # abdomen end: centre of the rear cap; palps: the two lowest front points next to the head
    rear = V[V[:, 1] > V[:, 1].max() - 0.10]; Jf[3] = rear.mean(0) - np.array([0, 0.03, 0])
    lm["joints"] = Jf.round(4).tolist()
    json.dump(lm, open(LMF, "w"), indent=1)
lm = json.load(open(LMF)); Jf = np.array(lm["joints"], float)
for n, p in lm.get("hand", {}).items(): Jf[NAMES.index(n)] = p
lm["joints"] = Jf.round(4).tolist(); json.dump(lm, open(LMF, "w"), indent=1)
V = canon(V0, lm["rot"], lm["scale"], lm["offset"])
print("BBOX gen", V.min(0).round(3), V.max(0).round(3), "soma", S.min(0).round(3), S.max(0).round(3))
bl = lambda A: np.array([np.linalg.norm(A[j] - A[PAR[j]]) for j in range(1, J)])
r = bl(Jf) / bl(Js); print("bone length ratio landmark / SOMA: min %.2f max %.2f ; worst:" % (r.min(), r.max()), [(NAMES[j + 1], round(float(r[j]), 2)) for j in np.argsort(np.abs(np.log(r)))[-6:]])
sub = np.random.default_rng(0).choice(len(V), min(len(V), 60000), replace=False)
for nm, h, v, hs in (("top", 0, 1, 1), ("side", 1, 2, 1), ("front", 0, 2, 1)):
    o = np.argsort(V[sub, 3 - h - v] * (1 if nm != "front" else -1))
    fig, a = plt.subplots(figsize=(16, 16 * (np.ptp(V[:, v]) + 0.3) / (np.ptp(V[:, h]) + 0.3)), dpi=70)
    a.scatter(V[sub][o, h], V[sub][o, v], s=3, c=np.clip(col[sub][o] * 1.6 + 0.1, 0, 1))
    for j in range(1, J):
        a.plot([Jf[PAR[j], h], Jf[j, h]], [Jf[PAR[j], v], Jf[j, v]], "-", color="yellow", lw=2)
        a.plot([Js[PAR[j], h], Js[j, h]], [Js[PAR[j], v], Js[j, v]], "-", color="cyan", lw=0.8, alpha=0.7)
    a.scatter(Jf[:, h], Jf[:, v], c="red", s=30, zorder=5)
    for j in range(J):
        if NAMES[j].endswith(("tip", "end")) or j < 3: a.annotate(NAMES[j], (Jf[j, h], Jf[j, v]), color="magenta", fontsize=9)
    a.set_aspect("equal"); a.grid(True, color="#ff00ff", lw=0.4, alpha=0.5); a.set_facecolor("#9fb4c8"); a.set_title(nm + " (yellow = landmarks, cyan = SOMA rest)")
    a.set_xticks(np.arange(math.floor(V[:, h].min() * 5) / 5, V[:, h].max() + 0.2, 0.2)); a.set_yticks(np.arange(math.floor(V[:, v].min() * 5) / 5, V[:, v].max() + 0.2, 0.2)); a.tick_params(labelsize=7)
    fig.tight_layout(); fig.savefig(f"{OUT}_{nm}.png"); plt.close(fig)
print("LANDMARKS_DONE", LMF)
