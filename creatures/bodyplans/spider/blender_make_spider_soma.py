"""Spider SOMA: neutral procedural body from the canonical spider skeleton (Skin modifier + subdivision), rigged with automatic weights.
blender -b --factory-startup --python blender_make_spider_soma.py -- <spider_spec.json> <out_dir>
Out: spider_soma.blend / .glb / spider_soma_spec.json (rest joints, rest_mesh_min_z). Blender axes: Z up, forward -Y, left +X, 1 unit = 1 m."""
import bpy, sys, json, math
import numpy as np
from mathutils import Vector
argv = sys.argv[sys.argv.index("--") + 1:]
spec = json.load(open(argv[0])); OUTD = argv[1]
NAMES, PAR = spec["joints"], spec["parents"]; J = len(NAMES)
rest = np.array(spec["rest_metres"])
T = np.stack([rest[:, 0], -rest[:, 2], rest[:, 1]], -1)
bpy.ops.wm.read_factory_settings(use_empty=True)
verts, edges, rad = [], [], []
def add(p, r):
    verts.append(tuple(p)); rad.append(r); return len(verts) - 1
def radius(n):
    if n == "body": return (0.34, 0.24)
    if n == "head": return (0.22, 0.17)
    if n == "abdomen": return (0.2, 0.17)
    if n == "abdomen_end": return (0.2, 0.2)
    if n.startswith("palp"): return (0.035, 0.035)
    k = n.split("_")[1]
    return {"hip": (0.085, 0.085), "knee": (0.07, 0.07), "ankle": (0.05, 0.05), "tip": (0.022, 0.022)}[k]
idx = {j: add(T[j], radius(NAMES[j])) for j in range(J)}
ia, ie = NAMES.index("abdomen"), NAMES.index("abdomen_end")
for j in range(1, J):
    if j == ie: continue
    edges.append((idx[PAR[j]], idx[j]))
# abdomen: a big egg along junction -> end
prev = idx[ia]
for t, r in ((0.3, 0.42), (0.6, 0.52), (0.85, 0.4)):
    v = add(T[ia] + (T[ie] - T[ia]) * t, (r, r * 0.9)); edges.append((prev, v)); prev = v
edges.append((prev, idx[ie]))
# palps: a thicker base
for n in ("palp_L", "palp_R"):
    j = NAMES.index(n); verts_mid = add(T[PAR[j]] + (T[j] - T[PAR[j]]) * 0.45, (0.06, 0.06))
    edges.remove((idx[PAR[j]], idx[j])); edges.append((idx[PAR[j]], verts_mid)); edges.append((verts_mid, idx[j]))
me = bpy.data.meshes.new("spider_skel"); me.from_pydata(verts, edges, []); me.update()
ob = bpy.data.objects.new("SpiderSOMA_mesh", me); bpy.context.scene.collection.objects.link(ob)
bpy.context.view_layer.objects.active = ob; ob.select_set(True)
sk = ob.modifiers.new("skin", "SKIN"); sk.use_smooth_shade = True
for i, r in enumerate(rad):
    me.skin_vertices[0].data[i].radius = r
me.skin_vertices[0].data[idx[0]].use_root = True
sub = ob.modifiers.new("sub", "SUBSURF"); sub.levels = 2
bpy.ops.object.modifier_apply(modifier="skin"); bpy.ops.object.modifier_apply(modifier="sub")
mesh_min_z = min((ob.matrix_world @ v.co).z for v in ob.data.vertices)
mat = bpy.data.materials.new("chitin"); mat.diffuse_color = (0.45, 0.4, 0.36, 1); ob.data.materials.append(mat)
ad = bpy.data.armatures.new("SpiderSOMA"); arm = bpy.data.objects.new("SpiderSOMA", ad); bpy.context.scene.collection.objects.link(arm)
bpy.ops.object.select_all(action="DESELECT"); bpy.context.view_layer.objects.active = arm; bpy.ops.object.mode_set(mode="EDIT")
root = ad.edit_bones.new("root"); root.head = Vector(T[0]); root.tail = Vector(T[0]) + Vector((0, 0, 0.3))
for j in range(1, J):                              # one bone per joint, named by its END joint
    b = ad.edit_bones.new(NAMES[j]); b.head = Vector(T[PAR[j]]); b.tail = Vector(T[j])
    b.parent = ad.edit_bones["root" if PAR[j] == 0 else NAMES[PAR[j]]]; b.use_connect = False
bpy.ops.object.mode_set(mode="OBJECT")
ob.select_set(True); arm.select_set(True); bpy.context.view_layer.objects.active = arm
bpy.ops.object.parent_set(type="ARMATURE_AUTO")
# the cephalothorax is rigid: the root bone gets the weights of the short body->hip / body->abdomen-junction bones
bpy.ops.wm.save_as_mainfile(filepath=OUTD + "/spider_soma.blend")
bpy.ops.object.select_all(action="SELECT")
bpy.ops.export_scene.gltf(filepath=OUTD + "/spider_soma.glb", export_format="GLB", export_skins=True, export_animations=False)
out = dict(spec); out.update(name="SpiderSOMA", rest_blender=T.round(5).tolist(), glb_scale=1.0, rest_mesh_min_z=mesh_min_z,
           rigid=[0, 1, 2] + [l[0] for l in spec["legs"]], bones="one bone per joint, named by its END joint; root bone at 'body'",
           axes_blender="Z up, forward -Y, left +X (glTF export converts to Y up)", verts=len(ob.data.vertices),
           groups=[g.name for g in ob.vertex_groups])
json.dump(out, open(OUTD + "/spider_soma_spec.json", "w"), indent=1)
print("SPIDER_SOMA_DONE verts", len(ob.data.vertices), "min_z", mesh_min_z, "groups", len(ob.vertex_groups))
