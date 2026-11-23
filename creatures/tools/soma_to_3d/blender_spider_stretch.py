"""Stretch metric for a skinned spider rig: edge length ratio vs the rest mesh while a motion plays.
blender -b --factory-startup --python blender_spider_stretch.py -- <motion.npz> <rig.blend> <out_prefix> [step=2]
Regions (mesh attribute 'region' from blender_fit_spider.py; else from the max-weight group): body/saddle, abdomen, leg roots, legs distal, palps.
Reports p1/p5/p95/p99/min/max and % outside [0.7, 1.4]; saves <out>.json and heat maps of the worst frame."""
import bpy, sys, os, json, math
import numpy as np
from mathutils import Vector
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spider_drive import drive
argv = sys.argv[sys.argv.index("--") + 1:]
MOT, RIG, OUT = argv[:3]; STEP = int(argv[3]) if len(argv) > 3 else 2
LO, HI = 0.7, 1.4; REG = ["body/saddle", "abdomen", "leg roots", "legs distal", "palps"]
arm, meshes, P, spec = drive(MOT, RIG); ob = max(meshes, key=lambda o: len(o.data.vertices)); sc = bpy.context.scene
me = ob.data; nv = len(me.vertices)
rest = np.zeros(nv * 3); me.vertices.foreach_get("co", rest); rest = rest.reshape(-1, 3)
key = np.round(rest / 1e-5).astype(np.int64)
_, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True); inv = inv.ravel()
tri = []
for p in me.polygons:
    vs = list(p.vertices)
    for i in range(1, len(vs) - 1): tri.append((vs[0], vs[i], vs[i + 1]))
tri = np.array(tri); ed = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
edw = np.sort(inv[ed], 1); edw = edw[edw[:, 0] != edw[:, 1]]; _, ue = np.unique(edw, axis=0, return_index=True)
E = np.stack([first[edw[ue, 0]], first[edw[ue, 1]]], 1)
gname = [g.name for g in ob.vertex_groups]
def reg_of(n):
    if n in ("root", "head", "abdomen") or n.endswith("hip"): return 0
    if n == "abdomen_end": return 1
    if n.startswith("palp"): return 4
    return 2 if n.endswith("knee") else 3
reg = np.zeros(nv, int)
if "region" in me.attributes:
    r_ = np.zeros(nv, np.int32); me.attributes["region"].data.foreach_get("value", r_); reg = r_.astype(int)
else:
    for v in me.vertices:
        if v.groups: reg[v.index] = reg_of(gname[max(v.groups, key=lambda x: x.weight).group])
def elen(X): return np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1)
L0 = elen(rest); okE = L0 > 1e-5
ereg = [((reg[E[:, 0]] == r) & (reg[E[:, 1]] == r)) & okE for r in range(5)]
frames = list(range(1, sc.frame_end + 1, STEP)); RE = []; worst = (-1, None, None); co = np.zeros(nv * 3)
for f in frames:
    sc.frame_set(f); dg = bpy.context.evaluated_depsgraph_get(); em = ob.evaluated_get(dg).data
    em.vertices.foreach_get("co", co); X = co.reshape(-1, 3)
    r = elen(X) / np.maximum(L0, 1e-12); RE.append(r.astype(np.float32))
    b = float(np.mean((r[okE] < LO) | (r[okE] > HI)) * 100)
    if b > worst[0]: worst = (b, f, r.copy())
RE = np.array(RE)
def stats(x):
    if x.size == 0: return None
    q = np.percentile(x, [1, 5, 50, 95, 99])
    return dict(p1=round(float(q[0]), 3), p5=round(float(q[1]), 3), p50=round(float(q[2]), 3), p95=round(float(q[3]), 3), p99=round(float(q[4]), 3),
                min=round(float(x.min()), 3), max=round(float(x.max()), 3), pct_out=round(float(np.mean((x < LO) | (x > HI)) * 100), 3), edges=int(x.shape[1]))
res = dict(rig=RIG, motion=MOT, frames=len(frames), step=STEP, edges=int(okE.sum()), band=[LO, HI], worst_frame=worst[1], worst_frame_pct_out=round(worst[0], 3),
           edge={"all": stats(RE[:, okE])})
for r, nm in enumerate(REG): res["edge"][nm] = stats(RE[:, ereg[r]])
if os.environ.get("FUSED_R"):
    # "fused-leg region": vertices of a leg (max-weight group) that have a vertex of ANOTHER leg nearer than FUSED_R metres in the rest mesh
    from mathutils import kdtree
    FR_ = float(os.environ["FUSED_R"]); legid = np.full(nv, -1)
    for v in me.vertices:
        if v.groups:
            n_ = gname[max(v.groups, key=lambda x: x.weight).group]
            if n_[:1] in "LR" and n_[1:2].isdigit(): legid[v.index] = (0 if n_[0] == "L" else 4) + int(n_[1]) - 1
    inreg = np.zeros(nv, bool)
    for li in range(8):
        oth = np.nonzero((legid >= 0) & (legid != li))[0]; mine = np.nonzero(legid == li)[0]
        if not len(oth) or not len(mine): continue
        kd = kdtree.KDTree(len(oth))
        for k_, i_ in enumerate(oth): kd.insert(Vector(rest[i_]), k_)
        kd.balance()
        for i_ in mine:
            if kd.find(Vector(rest[i_]))[2] < FR_: inreg[i_] = True
    em_ = inreg[E[:, 0]] & inreg[E[:, 1]] & okE
    res["edge"]["fused-leg region"] = stats(RE[:, em_]); res["fused_region"] = dict(radius=FR_, vertices=int(inreg.sum()), edges=int(em_.sum()))
    s = res["edge"]["fused-leg region"]
    if s: print("%-14s %6.3f %6.3f %6.3f %6.3f %6.3f %7.3f %7.3f %7d" % ("fused region", s["p1"], s["p5"], s["p95"], s["p99"], s["min"], s["max"], s["pct_out"], s["edges"]))
if os.environ.get("DUMP_WORST"):
    mxr = RE.max(0); mnr = RE.min(0); bad = np.nonzero(((mxr > 2.0) | (mnr < 0.4)) & okE)[0]
    mid = (rest[E[bad, 0]] + rest[E[bad, 1]]) / 2
    gsets = [sorted(gname[g_.group] for g_ in me.vertices[int(E[b, k])].groups if g_.weight > 0.2) for b in bad[:400] for k in (0, 1)]
    from collections import Counter
    res["worst_edges"] = dict(n=int(len(bad)), bbox=[mid.min(0).round(3).tolist(), mid.max(0).round(3).tolist()] if len(bad) else None,
                              by_groups=Counter(" + ".join(sorted(set(gsets[2 * i]) | set(gsets[2 * i + 1]))) for i in range(len(gsets) // 2)).most_common(12),
                              by_region=Counter(REG[reg[E[b, 0]]] for b in bad).most_common())
    print("WORST", json.dumps(res["worst_edges"]))
json.dump(res, open(OUT + ".json", "w"), indent=1)
print("STRETCH %s frames %d worst frame %d (%.2f%% out)" % (os.path.basename(RIG), len(frames), worst[1], worst[0]))
print("%-14s %6s %6s %6s %6s %6s %7s %7s %7s" % ("region", "p1", "p5", "p95", "p99", "min", "max", "%out", "edges"))
for nm in ["all"] + REG:
    s = res["edge"][nm]
    if s: print("%-14s %6.3f %6.3f %6.3f %6.3f %6.3f %7.3f %7.3f %7d" % (nm, s["p1"], s["p5"], s["p95"], s["p99"], s["min"], s["max"], s["pct_out"], s["edges"]))
if os.environ.get("NO_HEAT") != "1":
    r = worst[2]; lr = np.log(np.maximum(r, 1e-6)); lr[~okE] = 0; lv = np.zeros(nv)
    for k in (0, 1):
        idx = E[:, k]; o = np.argsort(np.abs(lr)); lv[idx[o]] = lr[o]
    lv = lv[first[inv]]; t = np.clip(lv / math.log(1.6), -1, 1)
    col = np.ones((nv, 4)); gy = 0.72
    col[:, 0] = np.where(t > 0, gy + (1 - gy) * t, gy * (1 + t)); col[:, 1] = gy * (1 - np.abs(t)); col[:, 2] = np.where(t < 0, gy + (1 - gy) * -t, gy * (1 - t))
    at = me.color_attributes.new("stretch", "FLOAT_COLOR", "POINT"); at.data.foreach_set("color", col.ravel()); me.color_attributes.active_color = at
    sc.frame_set(worst[1])
    sc.render.engine = "BLENDER_WORKBENCH"; sc.display.shading.light = "FLAT"; sc.display.shading.color_type = "VERTEX"
    sc.view_settings.view_transform = "Standard"; sc.world = bpy.data.worlds.new("w"); sc.world.color = (0.2, 0.2, 0.22)
    sc.render.resolution_x, sc.render.resolution_y = 1100, 800; sc.render.image_settings.file_format = "PNG"
    cam = bpy.data.objects.new("hcam", bpy.data.cameras.new("hcam")); sc.collection.objects.link(cam); sc.camera = cam
    cam.data.type = "ORTHO"; cam.data.ortho_scale = 5.0
    c = Vector(P[worst[1] - 1, 0]); c.z = 0.7
    fwd = P[worst[1] - 1, 1, :2] - P[worst[1] - 1, 0, :2]; ang = math.atan2(fwd[1], fwd[0])
    for nm, da, dz in (("top34", -math.pi / 3, 0.9), ("under", -math.pi / 4, -0.8)):
        d = Vector((math.cos(ang + da), math.sin(ang + da), dz)).normalized()
        cam.location = c + d * 20; cam.rotation_euler = (c - cam.location).to_track_quat("-Z", "Y").to_euler()
        sc.render.filepath = OUT + "_heat_" + nm + ".png"; bpy.ops.render.render(write_still=True)
print("STRETCH_DONE", OUT + ".json")
