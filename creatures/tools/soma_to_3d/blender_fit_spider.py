"""Fit an image-to-3D spider mesh onto the spider SOMA with LANDMARKS and skin it (spec-driven).
blender -b --factory-startup --python blender_fit_spider.py -- <spider_soma.blend> <gen.glb> <landmarks.json> <out.blend>
1. canonical frame from the landmarks json (rot 3x3, scale, offset)          2. weld (UV seams), bone heat with a temporary armature at the landmarks
3. the cephalothorax is ONE rigid piece: weights of head / abdomen-junction / hip bones go to 'root'
4. rules: saddle and gear boxes rigid (json "locks"), long Laplacian smoothing on the body + leg roots, leg segments sharpened (rigid segments)
5. bake landmark pose -> SOMA rest pose (swing + stretch per bone)        6. bind to the SOMA armature, Preserve Volume, no hard influence limit
Env: BODY_PASSES (150), ROOT_R0/ROOT_R1 (0.22/0.5 m from a hip joint = the leg-root blend zone), LEG_POW (3), DQ (1), SOFT (0.001)"""
import bpy, sys, os, json, math
import numpy as np
from mathutils import Vector
from mathutils import kdtree
argv = sys.argv[sys.argv.index("--") + 1:]
SOMA, GEN, LMF, OUT = argv[:4]; ENV = os.environ.get
bpy.ops.wm.open_mainfile(filepath=SOMA)
arm = [o for o in bpy.data.objects if o.type == "ARMATURE"][0]
for o in [o for o in bpy.data.objects if o.type == "MESH"]: bpy.data.objects.remove(o, do_unlink=True)
spec = json.load(open(SOMA.replace(".blend", "_spec.json")))
NAMES, PAR, LEGS = spec["joints"], spec["parents"], spec["legs"]; J = len(NAMES)
Js = np.array(spec["rest_blender"], float)
BJ = list(range(1, J)); BN = [NAMES[j] for j in BJ]; BI = {n: i for i, n in enumerate(BN)}; NB = len(BN)
before = set(bpy.data.objects); bpy.ops.import_scene.gltf(filepath=GEN)
new = [o for o in bpy.data.objects if o not in before]; gm = [o for o in new if o.type == "MESH"]
bpy.ops.object.select_all(action="DESELECT")
for o in gm: o.select_set(True)
bpy.context.view_layer.objects.active = gm[0]
if len(gm) > 1: bpy.ops.object.join()
g = bpy.context.view_layer.objects.active
for o in new:
    if o is not g and o.name in bpy.data.objects: bpy.data.objects.remove(o, do_unlink=True)
mw = g.matrix_world.copy(); g.parent = None; g.matrix_world = mw
bpy.ops.object.select_all(action="DESELECT"); g.select_set(True); bpy.context.view_layer.objects.active = g
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
me = g.data; nv = len(me.vertices)
V = np.zeros(nv * 3); me.vertices.foreach_get("co", V); V = V.reshape(-1, 3)
lm = json.load(open(LMF))
V = V @ np.array(lm["rot"], float).T * lm["scale"] + np.array(lm["offset"])
Jf = np.array(lm["joints"], float)
blen = lambda A: np.array([np.linalg.norm(A[j] - A[PAR[j]]) for j in BJ])
legb = [i for i, j in enumerate(BJ) if NAMES[j].endswith(("knee", "ankle", "tip"))]
sg = float(np.median((blen(Js) / blen(Jf))[legb])) if ENV("GLOBAL_SCALE", "1") == "1" else 1.0      # one global scale (median leg-bone ratio) about the body joint
V = Js[0] + (V - Jf[0]) * sg; Jf = Js[0] + (Jf - Jf[0]) * sg
print("FIT global scale %.3f" % sg)
kb = blen(Js) / blen(Jf)
print("FIT per-bone stretch along the bone: min %.2f max %.2f" % (kb.min(), kb.max()))
# welded topology
key = np.round(V / 1e-5).astype(np.int64)
_, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True); inv = inv.ravel()
U = V[first]; nu = len(U)
tri = []
for p in me.polygons:
    vs = list(p.vertices)
    for i in range(1, len(vs) - 1): tri.append((vs[0], vs[i], vs[i + 1]))
tri = np.array(tri); tw = inv[tri]
ed = np.sort(np.concatenate([tw[:, [0, 1]], tw[:, [1, 2]], tw[:, [2, 0]]]), 1); ed = np.unique(ed[ed[:, 0] != ed[:, 1]], axis=0)
deg = np.zeros(nu); np.add.at(deg, ed[:, 0], 1); np.add.at(deg, ed[:, 1], 1)
comp = np.arange(nu)
for _ in range(10000):
    m = np.minimum(comp[ed[:, 0]], comp[ed[:, 1]]); c2 = comp.copy(); np.minimum.at(c2, ed[:, 0], m); np.minimum.at(c2, ed[:, 1], m)
    if (c2 == comp).all(): break
    comp = c2
labs, cnt = np.unique(comp, return_counts=True); main = comp == labs[np.argmax(cnt)]
print("MESH verts %d welded %d edges %d components %d (main %d)" % (nv, nu, len(ed), len(labs), main.sum()))


def lap_smooth(W, passes, fac=0.5, mask=None):
    f = fac if mask is None else (fac * mask)[:, None]
    for _ in range(passes):
        A = np.zeros_like(W); np.add.at(A, ed[:, 0], W[ed[:, 1]]); np.add.at(A, ed[:, 1], W[ed[:, 0]])
        A /= np.maximum(deg, 1)[:, None]; W = W + f * (A - W)
    return W


def seg_dist(Pts, a, b):
    ab = b - a; t = np.clip(((Pts - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
    return np.linalg.norm(Pts - (a + t[:, None] * ab), axis=1), t


def sstep(x, a, b):
    t = np.clip((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t)


# temporary armature at the landmarks, bone heat on the welded mesh
ad = bpy.data.armatures.new("FitArm"); fit = bpy.data.objects.new("FitArm", ad); bpy.context.scene.collection.objects.link(fit)
bpy.ops.object.select_all(action="DESELECT"); bpy.context.view_layer.objects.active = fit; bpy.ops.object.mode_set(mode="EDIT")
r = ad.edit_bones.new("root"); r.head = Vector(Jf[0]); r.tail = Vector(Jf[0]) + Vector((0, 0, 0.3))
for j in BJ:
    b = ad.edit_bones.new(NAMES[j]); b.head = Vector(Jf[PAR[j]]); b.tail = Vector(Jf[j]); b.parent = ad.edit_bones["root" if PAR[j] == 0 else NAMES[PAR[j]]]; b.use_connect = False
bpy.ops.object.mode_set(mode="OBJECT"); ad.bones["root"].use_deform = False
wm = bpy.data.meshes.new("weld"); wm.from_pydata([tuple(p) for p in U], [], [tuple(t) for t in tw if len(set(t)) == 3]); wm.update()
wo = bpy.data.objects.new("weld", wm); bpy.context.scene.collection.objects.link(wo)
bpy.ops.object.select_all(action="DESELECT"); wo.select_set(True); fit.select_set(True); bpy.context.view_layer.objects.active = fit
bpy.ops.object.parent_set(type="ARMATURE_AUTO")
gn = [vg.name for vg in wo.vertex_groups]; W = np.zeros((nu, NB))
for v in wo.data.vertices:
    for ge in v.groups:
        if gn[ge.group] in BI: W[v.index, BI[gn[ge.group]]] = ge.weight
ok = float((W.sum(1) > 1e-6).mean()); print("HEAT weighted %.3f" % ok)
bpy.data.objects.remove(wo, do_unlink=True)
dref = np.stack([seg_dist(U, Jf[PAR[j]], Jf[j])[0] for j in BJ], 1)
for i, n in enumerate(BN):                                    # a left bone never owns the far right side and vice versa
    if n[0] == "L" and n[1].isdigit(): dref[U[:, 0] < -0.15, i] = 1e9
    if n[0] == "R" and n[1].isdigit(): dref[U[:, 0] > 0.15, i] = 1e9
if ok < 0.99:
    # the fur makes the welded mesh non-manifold: bone heat on a voxel-remesh proxy (outer shell), transferred by nearest surface point
    print("HEAT failed on the real mesh -> voxel proxy")
    import bmesh
    from mathutils.bvhtree import BVHTree
    pm_ = bpy.data.meshes.new("proxy"); pm_.from_pydata([tuple(p) for p in U], [], [tuple(t) for t in tw if len(set(t)) == 3]); pm_.update()
    px = bpy.data.objects.new("proxy", pm_); bpy.context.scene.collection.objects.link(px)
    bpy.ops.object.select_all(action="DESELECT"); px.select_set(True); bpy.context.view_layer.objects.active = px
    rm = px.modifiers.new("rm", "REMESH"); rm.mode = "VOXEL"; rm.voxel_size = float(ENV("VOXEL", "0.03")); rm.adaptivity = 0.0
    bpy.ops.object.modifier_apply(modifier="rm")
    bm = bmesh.new(); bm.from_mesh(px.data); seen = set(); isl = []
    for v in bm.verts:
        if v.index in seen: continue
        st = [v]; seen.add(v.index); cur = [v]
        while st:
            a_ = st.pop()
            for e_ in a_.link_edges:
                b_ = e_.other_vert(a_)
                if b_.index not in seen: seen.add(b_.index); st.append(b_); cur.append(b_)
        isl.append(cur)
    isl.sort(key=len, reverse=True)
    for cur in isl[1:]:
        for v in cur: bm.verts.remove(v)
    bm.to_mesh(px.data); bm.free(); px.data.update()
    bpy.ops.object.select_all(action="DESELECT"); px.select_set(True); fit.select_set(True); bpy.context.view_layer.objects.active = fit
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    gnp = [vg.name for vg in px.vertex_groups]; Wp = np.zeros((len(px.data.vertices), NB))
    for v in px.data.vertices:
        for ge in v.groups:
            if gnp[ge.group] in BI: Wp[v.index, BI[gnp[ge.group]]] = ge.weight
    okp = float((Wp.sum(1) > 1e-6).mean()); print("HEAT proxy verts %d islands %d weighted %.3f" % (len(px.data.vertices), len(isl), okp))
    if okp > 0.9:
        PV = np.array([tuple(v.co) for v in px.data.vertices]); PP = [tuple(p.vertices) for p in px.data.polygons]
        bvh = BVHTree.FromPolygons([tuple(p) for p in PV], PP); W = np.zeros((nu, NB))
        for i in range(nu):
            loc, nrm, idx, dist = bvh.find_nearest(Vector(U[i]))
            vs = list(PP[idx]); d_ = np.linalg.norm(PV[vs] - np.array(loc), axis=1); w_ = 1.0 / (d_ + 1e-4); W[i] = (w_[:, None] * Wp[vs]).sum(0) / w_.sum()
        ok = 1.0
    else:
        print("HEAT failed on the proxy too -> nearest bone + smoothing")
    bpy.data.objects.remove(px, do_unlink=True)
zero = W.sum(1) < 1e-6
if zero.any():
    W[zero] = 0; W[np.nonzero(zero)[0], dref[zero].argmin(1)] = 1
W = np.maximum(W, 0)
if (~main).any():                                             # loose bits take the weights of the nearest main-shell vertex
    kd = kdtree.KDTree(int(main.sum())); mi = np.nonzero(main)[0]
    for k, i in enumerate(mi): kd.insert(Vector(U[i]), k)
    kd.balance()
    for i in np.nonzero(~main)[0]: W[i] = W[mi[kd.find(Vector(U[i]))[1]]]
if ENV("OWN_LEGS", "0") == "1":
    # a leg part belongs to ONE leg: connected components of the leg zone (the mesh without the body; legs were separated by
    # blender_unfuse_legs.py) take the leg of the nearest landmark chain; the bones of the other legs get no weight there
    BRs, ARs = float(ENV("BODY_R", "0.40")) * sg, float(ENV("ABD_R", "0.66")) * sg
    lz = (seg_dist(U, Jf[1], Jf[2])[0] > BRs) & (seg_dist(U, Jf[2] + (Jf[3] - Jf[2]) * 0.25, Jf[3])[0] > ARs)
    ez = ed[lz[ed[:, 0]] & lz[ed[:, 1]]]; cz = np.arange(nu)
    for _ in range(10000):
        m = np.minimum(cz[ez[:, 0]], cz[ez[:, 1]]); c2 = cz.copy(); np.minimum.at(c2, ez[:, 0], m); np.minimum.at(c2, ez[:, 1], m)
        if (c2 == cz).all(): break
        cz = c2
    legof = np.array([next((li for li, l in enumerate(LEGS) if j in l), -1) for j in BJ])          # bone -> leg index (-1 = not a leg bone)
    dleg = np.stack([dref[:, legof == li].min(1) for li in range(len(LEGS))], 1); near = dleg.argmin(1)
    nfix = 0; own_comp = {}
    for c_ in np.unique(cz[lz]):
        m = (cz == c_) & lz
        if m.sum() < 8: continue
        li = int(np.bincount(near[m], minlength=len(LEGS)).argmax()); own_comp[li] = own_comp.get(li, 0) + int(m.sum())
        other = (legof >= 0) & (legof != li); nfix += int((W[np.ix_(m, other)].sum(1) > 0.01).sum())
        Wm = W[m]; Wm[:, other] = 0; z_ = Wm.sum(1) < 1e-6
        if z_.any():
            cols = np.nonzero(legof == li)[0]; Wm[np.nonzero(z_)[0], cols[dref[m][z_][:, cols].argmin(1)]] = 1
        W[m] = Wm
    print("OWN_LEGS vertices per leg", {NAMES[LEGS[k][0]][:2]: v for k, v in sorted(own_comp.items())}, "vertices that had weights of another leg:", nfix)
W /= W.sum(1, keepdims=True); W = lap_smooth(W, 2)
# deform groups: root (rigid cephalothorax) + abdomen_end + palps + 3 segments per leg
RIGID = ["head", "abdomen"] + [NAMES[l[0]] for l in LEGS]
GN = ["root"] + [n for n in BN if n not in RIGID]; GI = {n: i for i, n in enumerate(GN)}; NG = len(GN)
G = np.zeros((nu, NG))
for n in BN: G[:, GI["root" if n in RIGID else n]] += W[:, BI[n]]
W = G
ref = np.array([GI["root" if n in RIGID else n] for n in BN])[dref.argmin(1)]
# regions for the stretch metric: 0 body/saddle, 1 abdomen, 2 leg roots (hip-knee segment), 3 legs distal, 4 palps
def reg_of(n):
    if n == "root": return 0
    if n == "abdomen_end": return 1
    if n.startswith("palp"): return 4
    return 2 if n.endswith("knee") else 3
region = np.array([reg_of(GN[k]) for k in ref])
RL = lm.get("rules", {})
PRM = lambda k, d: float(ENV(k, str(RL.get(k, d))))
# locks: rigid gear (saddle, plates, band)
lam = np.zeros(nu); Tg = np.zeros_like(W)
for Lk in RL.get("locks", []):
    bx = np.array([(-99 if i % 2 == 0 else 99) if v is None else v for i, v in enumerate(Lk["box"])], float); fe = Lk.get("feather", 0.05)
    bx = (bx.reshape(3, 2) - np.array(lm["joints"][0])[:, None]) * sg + Js[0][:, None]; bx = bx.ravel()     # boxes are given in the landmark frame (before the global scale)
    d = np.maximum(np.maximum(bx[0::2] - U, U - bx[1::2]).max(1), 0); l = 1 - sstep(d, 0, fe)
    t = np.zeros(NG)
    for n, w in Lk["bones"].items(): t[GI[n]] = w
    t /= t.sum(); up = l > lam; Tg[up] = t; lam[up] = l[up]
    if Lk.get("region") is not None: region[l > 0.5] = int(Lk["region"])
    print("RULES lock %-12s %5d vertices" % (Lk.get("name", "?"), int((l > 0.5).sum())))
W = (1 - lam[:, None]) * W + lam[:, None] * Tg
# smooth zone: body + abdomen + leg roots (distance from the nearest hip joint), not the distal legs
dh = np.min(np.stack([np.linalg.norm(U - Jf[l[0]], axis=1) for l in LEGS], 1), 1)
legw = sum(W[:, GI[NAMES[l[k]]]] for l in LEGS for k in (1, 2, 3))
zone = np.maximum(1 - sstep(dh, PRM("ROOT_R0", 0.22), PRM("ROOT_R1", 0.5)), 1 - sstep(legw, 0.3, 0.7))
NP = int(PRM("BODY_PASSES", 150))
W = lap_smooth(W, NP, 0.5, zone * (1 - lam))
print("RULES smooth zone: %d passes, %.0f%% of the vertices" % (NP, 100 * (zone > 0.5).mean()))
# rigid leg segments: sharpen the weights outside the zone (short blend bands at knee / ankle)
k = PRM("LEG_POW", 3.0)
if k > 1:
    Wk = np.maximum(W, 0) ** k; Wk /= np.maximum(Wk.sum(1, keepdims=True), 1e-12)
    b = ((1 - zone) * (1 - lam))[:, None]; W = (1 - b) * W + b * Wk
SOFT = float(ENV("SOFT", "0.001")); W = np.maximum(np.nan_to_num(W) - SOFT, 0)
bad = W.sum(1) < 1e-6
if bad.any(): W[bad] = 0; W[np.nonzero(bad)[0], ref[bad]] = 1
W /= W.sum(1, keepdims=True)
print("INFLUENCES mean %.2f, >4: %.1f%%, >8: %.1f%% of vertices (weights > 0.01)" % ((W > 0.01).sum(1).mean(), 100 * ((W > 0.01).sum(1) > 4).mean(), 100 * ((W > 0.01).sum(1) > 8).mean()))
# bake: landmark pose -> SOMA rest pose
def swing(a, b):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b); v = np.cross(a, b); cc = float(a @ b)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]); return np.eye(3) + K + K @ K / (1 + cc)
# the rigid cephalothorax: one similarity-free rigid transform (Kabsch on body, head, junction, hips)
ids = [0, 1, 2] + [l[0] for l in LEGS]
A = Jf[ids] - Jf[ids].mean(0); B = Js[ids] - Js[ids].mean(0)
Uu, Dd, Vt = np.linalg.svd(B.T @ A); dd = np.sign(np.linalg.det(Uu @ Vt)); R0 = Uu @ np.diag([1, 1, dd]) @ Vt
t0 = Js[ids].mean(0) - R0 @ Jf[ids].mean(0)
print("BAKE cephalothorax rigid fit error (m): max %.3f" % np.linalg.norm(Jf[ids] @ R0.T + t0 - Js[ids], axis=1).max())
Rj = {0: R0}; M = {"root": (R0, t0)}
for j in BJ:
    p = PAR[j]; n = NAMES[j]
    if n in RIGID: Rj[j] = R0; continue
    df = Jf[j] - Jf[p]; ds = Js[j] - Js[p]; lf = np.linalg.norm(df); df /= lf
    R = swing(Rj[p] @ df, ds) @ Rj[p]; Rj[j] = R
    Sx = np.eye(3) + (np.linalg.norm(ds) / lf - 1) * np.outer(df, df)
    M[n] = (R @ Sx, Js[p] - R @ Sx @ Jf[p])
Ur = np.zeros_like(U)
for i, n in enumerate(GN):
    Am, tm = M[n]; Ur += W[:, i:i + 1] * (U @ Am.T + tm)
rr = np.linalg.norm(Ur[ed[:, 0]] - Ur[ed[:, 1]], axis=1) / np.maximum(np.linalg.norm(U[ed[:, 0]] - U[ed[:, 1]], axis=1), 1e-9)
print("BAKE edge ratio (SOMA rest / generated mesh): p1 %.3f p5 %.3f p50 %.3f p95 %.3f p99 %.3f max %.3f" % tuple(np.percentile(rr, [1, 5, 50, 95, 99, 100])))
Vr = Ur[inv]; me.vertices.foreach_set("co", Vr.ravel()); me.update()
bpy.data.objects.remove(fit, do_unlink=True)
g.vertex_groups.clear(); Wv = W[inv]
for i, n in enumerate(GN):
    vg = g.vertex_groups.new(name=n); colw = Wv[:, i]
    for w in np.unique(np.round(colw[colw > 1e-4], 4)):
        vg.add([int(k_) for k_ in np.nonzero(np.round(colw, 4) == w)[0]], float(w), "REPLACE")
bpy.ops.object.select_all(action="DESELECT"); g.select_set(True); bpy.context.view_layer.objects.active = g
bpy.ops.object.vertex_group_normalize_all(group_select_mode="ALL", lock_active=False)
at = me.attributes.new("region", "INT", "POINT"); at.data.foreach_set("value", region[inv].astype(np.int32))
am = g.modifiers.new("arm", "ARMATURE"); am.object = arm; am.use_deform_preserve_volume = ENV("DQ", "1") == "1"
g.parent = arm; g.matrix_parent_inverse = arm.matrix_world.inverted(); g.name = "SpiderGen_mesh"
zmin = float(Vr[:, 2].min())
try: bpy.ops.file.pack_all()
except Exception as e: print("pack_all:", e)
bpy.ops.wm.save_as_mainfile(filepath=OUT)
spec["rest_mesh_min_z"] = zmin; spec["mesh"] = GEN; spec["groups"] = GN
spec["fit"] = dict(method="landmarks + bone heat + rules + bake (blender_fit_spider.py)", landmarks=LMF, body_passes=NP, leg_pow=k)
json.dump(spec, open(OUT.replace(".blend", "_spec.json"), "w"), indent=1)
print("FIT_SPIDER_DONE", OUT, "min z", round(zmin, 4))
