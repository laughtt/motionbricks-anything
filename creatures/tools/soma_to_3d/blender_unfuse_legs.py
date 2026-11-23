"""Separate fused legs of an image-to-3D creature mesh (skin sheets between neighbour legs).
blender -b --factory-startup --python blender_unfuse_legs.py -- <gen.glb> <landmarks.json> <soma_spec.json> <out.glb> <out_mesh.npz> [report.json]
Method (landmark rules): every vertex in the leg zone (outside the body capsules) belongs to the nearest leg chain (hip-knee-ankle-tip of the
landmarks). A face whose vertices belong to different legs BRIDGES two legs: it is given to the leg nearest to its centre and its foreign
vertices are replaced by copies (a rip along the border between the two legs: no edge joins two legs any more). The open cut of each leg is
closed with a cap (the cut surface). Output: the same mesh frame as the input (the landmark transform still applies) + npz for spider_landmarks.py.
Env: BODY_R (0.40), ABD_R (0.66) = leg zone as in spider_landmarks.py."""
import bpy, bmesh, sys, os, json
import numpy as np
from mathutils import Vector
argv = sys.argv[sys.argv.index("--") + 1:]
GEN, LMF, SPECF, OUTG, OUTN = argv[:5]; REP = argv[5] if len(argv) > 5 else OUTG.replace(".glb", "_unfuse.json")
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=GEN)
gm = [o for o in bpy.data.objects if o.type == "MESH"]
bpy.ops.object.select_all(action="DESELECT")
for o in gm: o.select_set(True)
bpy.context.view_layer.objects.active = gm[0]
if len(gm) > 1: bpy.ops.object.join()
g = bpy.context.view_layer.objects.active
for o in list(bpy.data.objects):
    if o is not g: bpy.data.objects.remove(o, do_unlink=True)
mw = g.matrix_world.copy(); g.parent = None; g.matrix_world = mw
bpy.ops.object.select_all(action="DESELECT"); g.select_set(True); bpy.context.view_layer.objects.active = g
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
lm = json.load(open(LMF)); spec = json.load(open(SPECF)); LEGS = spec["legs"]; N = spec["joints"]
ROT = np.array(lm["rot"], float); SC = lm["scale"]; OFF = np.array(lm["offset"]); Jf = np.array(lm["joints"], float)
for n, p in lm.get("hand", {}).items(): Jf[N.index(n)] = p
me = g.data; bm = bmesh.new(); bm.from_mesh(me)
nv0, nf0 = len(bm.verts), len(bm.faces)
bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5 / SC)
bm.verts.ensure_lookup_table(); bm.faces.ensure_lookup_table(); bm.verts.index_update(); bm.edges.index_update()
uvl = bm.loops.layers.uv.active


def seg_dist(P, a, b):
    ab = b - a; t = np.clip(((P - a) @ ab) / max(ab @ ab, 1e-12), 0, 1); return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def chain_dist(P):
    return np.stack([np.min([seg_dist(P, Jf[l[k]], Jf[l[k + 1]]) for k in range(3)], 0) for l in LEGS], 1)


X = np.array([tuple(v.co) for v in bm.verts]); C = X @ ROT.T * SC + OFF
BR = float(os.environ.get("BODY_R", "0.40")); AR = float(os.environ.get("ABD_R", "0.66"))
leg = (seg_dist(C, Jf[1], Jf[2]) > BR) & (seg_dist(C, Jf[2] + (Jf[3] - Jf[2]) * 0.25, Jf[3]) > AR)
D = chain_dist(C); lab = D.argmin(1)
old_b = {frozenset((e.verts[0].index, e.verts[1].index)) for e in bm.edges if e.is_boundary}
# (only the leg-zone vertices of a face count: an edge between two legs at the border of the zone is cut too)
bridging = [f for f in bm.faces if len({int(lab[v.index]) for v in f.verts if leg[v.index]}) > 1]
pairs = {}
owner = {v: int(lab[v.index]) for v in bm.verts if leg[v.index]}; orig = {v: v.index for v in bm.verts}
dup = {}; rebuilt = []
for f in bridging:
    ls = sorted({int(lab[v.index]) for v in f.verts if leg[v.index]}); k = "-".join(N[LEGS[i][0]][:2] for i in ls); pairs[k] = pairs.get(k, 0) + 1
    cen = np.mean([C[v.index] for v in f.verts], 0); fl = ls[int(np.argmin(chain_dist(cen[None])[0][ls]))]
    cen_raw = Vector(np.mean([X[v.index] for v in f.verts], 0)); vs = []
    for l in f.loops:
        v = l.vert
        if leg[v.index] and int(lab[v.index]) != fl:
            key = (v.index, fl)
            if key not in dup:
                d = cen_raw - v.co; nv = bm.verts.new(v.co + d.normalized() * min(0.0005 / SC, d.length * 0.2))
                dup[key] = nv; owner[nv] = fl; orig[nv] = v.index
            vs.append(dup[key])
        else: vs.append(v)
    rebuilt.append((vs, [l[uvl].uv.copy() for l in f.loops] if uvl else None, f.material_index, f.smooth))
bmesh.ops.delete(bm, geom=bridging, context="FACES_ONLY")
made = 0
for vs, uvs, mi, sm in rebuilt:
    try: nf = bm.faces.new(vs)
    except ValueError: continue
    nf.material_index = mi; nf.smooth = sm; made += 1
    if uvs:
        for l, u in zip(nf.loops, uvs): l[uvl].uv = u
# loose edges left by the removed faces
bmesh.ops.delete(bm, geom=[e for e in bm.edges if not e.link_faces], context="EDGES")
bm.verts.index_update(); bm.edges.ensure_lookup_table()
# caps: the new boundary of each leg (the cut) -> one polygon per chain of boundary edges
newb = [e for e in bm.edges if e.is_boundary and e.verts[0] in owner and e.verts[1] in owner and owner[e.verts[0]] == owner[e.verts[1]]
        and frozenset((orig.get(e.verts[0], -1), orig.get(e.verts[1], -2))) not in old_b]
caps, skipped, cap_faces = 0, 0, 0
if os.environ.get("CAPS", "1") == "1":
    adj = {}
    for e in newb:
        a, b = e.verts; adj.setdefault(a, []).append(b); adj.setdefault(b, []).append(a)
    seen = set()
    for v0 in list(adj):
        if v0 in seen: continue
        comp = []; st = [v0]; seen.add(v0)
        while st:
            a = st.pop(); comp.append(a)
            for b in adj[a]:
                if b not in seen: seen.add(b); st.append(b)
        if len(comp) < 5: continue
        if max(len(adj[v]) for v in comp) > 2: skipped += 1; continue
        ends = [v for v in comp if len(adj[v]) == 1]; cur = ends[0] if ends else comp[0]; order = [cur]; prev = None
        while True:
            nx = [b for b in adj[cur] if b is not prev and b not in order[:1] or (b is order[0] and len(order) > 2 and False)]
            nx = [b for b in adj[cur] if b is not prev and b not in order]
            if not nx: break
            prev, cur = cur, nx[0]; order.append(cur)
        if len(order) < 5: continue
        try: nf = bm.faces.new(order)
        except ValueError: skipped += 1; continue
        A = owner[order[0]]; cc = np.mean([np.array(v.co) for v in order], 0) @ ROT.T * SC + OFF
        l_ = LEGS[A]; dd = [seg_dist(cc[None], Jf[l_[k]], Jf[l_[k + 1]])[0] for k in range(3)]; k_ = int(np.argmin(dd))
        a_, b_ = Jf[l_[k_]], Jf[l_[k_ + 1]]; t_ = np.clip(((cc - a_) @ (b_ - a_)) / ((b_ - a_) @ (b_ - a_)), 0, 1); out_dir = (cc - (a_ + t_ * (b_ - a_))) @ ROT       # canonical -> raw direction
        nf.normal_update()
        if np.array(nf.normal) @ out_dir < 0: nf.normal_flip()
        if uvl:
            for l in nf.loops:
                oth = [ll for ll in l.vert.link_loops if ll.face is not nf]
                if oth: l[uvl].uv = oth[0][uvl].uv
        r = bmesh.ops.triangulate(bm, faces=[nf]); caps += 1; cap_faces += len(r["faces"])
bm.normal_update(); bm.to_mesh(me); me.update()
V = np.array([tuple(v.co) for v in me.vertices]); tri = []
for p in me.polygons:
    vs = list(p.vertices)
    for i in range(1, len(vs) - 1): tri.append((vs[0], vs[i], vs[i + 1]))
np.savez(OUTN, V=V, tri=np.array(tri), col=np.full((len(V), 3), 0.6))
bpy.ops.object.select_all(action="DESELECT"); g.select_set(True)
bpy.ops.export_scene.gltf(filepath=OUTG, export_format="GLB", use_selection=True)
rep = dict(src=GEN, out=OUTG, verts_in=nv0, faces_in=nf0, verts_welded=len(X), leg_zone_verts=int(leg.sum()), bridging_faces=len(bridging), bridging_by_legs=pairs,
           copied_vertices=len(dup), faces_rebuilt=made, cut_boundary_edges=len(newb), caps=caps, cap_faces=cap_faces, cap_chains_skipped=skipped, verts_out=len(V), faces_out=len(tri))
json.dump(rep, open(REP, "w"), indent=1); print("UNFUSE_DONE", json.dumps(rep))
