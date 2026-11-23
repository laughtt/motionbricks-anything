"""Extract world joint positions (canonical spider joints, 30 fps) for every action of a source rig.
blender -b --factory-startup --python blender_extract.py -- <glb> <mapping.json> <out_dir>
Out: <out_dir>/<rig>__<action>.npz  pos [F,J,3] (source units, canonical axes: Y up, forward +Z, left +X), plus rest [J,3].
Leaf-bone tips come from the skinned mesh (the glTF importer guesses leaf bone tails)."""
import bpy, sys, json, os
import numpy as np
from mathutils import Vector
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spider_joints import NAMES
argv = sys.argv[sys.argv.index("--") + 1:]
GLB, MAPF, OUTD = argv[:3]
M = json.load(open(MAPF)); os.makedirs(OUTD, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=GLB)
sc = bpy.context.scene
arm = [o for o in bpy.data.objects if o.type == "ARMATURE"][0]
mesh = [o for o in bpy.data.objects if o.type == "MESH" and o.find_armature() == arm][0]
SRC_FPS = sc.render.fps
# mesh-derived tips (bone-local rest coordinates)
tips = {}
def tip_local(bname):
    if bname in tips: return tips[bname]
    b = arm.data.bones[bname]; gi = mesh.vertex_groups[bname].index
    pts = []
    for v in mesh.data.vertices:
        w = {g.group: g.weight for g in v.groups}
        if w.get(gi, 0) > 0.5 and w.get(gi, 0) >= max(w.values()) - 1e-6:
            pts.append(arm.matrix_world.inverted() @ (mesh.matrix_world @ v.co))
    Minv = b.matrix_local.inverted()
    if not pts:
        tips[bname] = Minv @ b.tail_local; return tips[bname]
    L = [Minv @ p for p in pts]
    c = sum(L, Vector()) / len(L); d = c.normalized()
    tips[bname] = d * max(p.dot(d) for p in L); return tips[bname]
def point(spec, rest=False):
    if isinstance(spec, list):
        ps = [point(s, rest) for s in spec[1:]]; return sum(ps, Vector()) / len(ps)
    if "@" in spec:                      # rigid point given in rest world coordinates: "bone@x,y,z"
        bn, xyz = spec.split("@"); b = arm.data.bones[bn]
        loc = b.matrix_local.inverted() @ (arm.matrix_world.inverted() @ Vector([float(x) for x in xyz.split(",")]))
        return arm.matrix_world @ ((b.matrix_local if rest else arm.pose.bones[bn].matrix) @ loc)
    bn, where = spec.rsplit(":", 1)
    Mb = arm.data.bones[bn].matrix_local if rest else arm.pose.bones[bn].matrix
    loc = Vector((0, 0, 0)) if where == "head" else tip_local(bn)
    return arm.matrix_world @ (Mb @ loc)
def canon(v):
    return (-v.x, v.z, v.y) if M["axes"] == "A" else (v.x, v.z, -v.y)
names = NAMES + sorted(k for k in M["joints"] if k not in NAMES)
rest = np.array([canon(point(M["joints"][n], True)) for n in names])
ad = arm.animation_data
for t in ad.nla_tracks: t.mute = True
rep = {}
for act in bpy.data.actions:
    ad.action = act
    try:
        if getattr(ad, "action_slot", None) is None and act.slots: ad.action_slot = act.slots[0]
    except Exception as e: print("slot", e)
    f0, f1 = act.frame_range; dur = (f1 - f0) / SRC_FPS
    n = int(round(dur * 30)) + 1; out = []
    for k in range(n):
        fr = f0 + min(k / 30.0, dur) * SRC_FPS
        sc.frame_set(int(np.floor(fr)), subframe=float(fr - np.floor(fr)))
        bpy.context.view_layer.update()
        out.append([canon(point(M["joints"][nm])) for nm in names])
    out = np.array(out)
    np.savez(f"{OUTD}/{M['name']}__{act.name}.npz", pos=out, rest=rest, names=np.array(names), fps=30, src_fps=SRC_FPS)
    rep[act.name] = dict(frames=n, seconds=round(dur, 3), root_travel=float(np.linalg.norm(out[-1, 0] - out[0, 0])))
json.dump(dict(rig=M["name"], clips=rep, rest=rest.tolist(), names=names), open(f"{OUTD}/{M['name']}__report.json", "w"), indent=1)
print("EXTRACT_DONE", json.dumps(rep))
