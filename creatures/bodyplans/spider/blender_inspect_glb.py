"""Inspect a glb: armatures, bones, actions. blender -b --factory-startup --python blender_inspect_glb.py -- <glb> <out.json>"""
import bpy, sys, json
argv = sys.argv[sys.argv.index("--") + 1:]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=argv[0])
out = {"armatures": [], "actions": [], "meshes": []}
for o in bpy.data.objects:
    if o.type == "ARMATURE":
        bones = [{"name": b.name, "parent": b.parent.name if b.parent else None,
                  "head": [round(x, 4) for x in (o.matrix_world @ b.head_local)], "tail": [round(x, 4) for x in (o.matrix_world @ b.tail_local)]} for b in o.data.bones]
        out["armatures"].append({"name": o.name, "scale": list(o.scale), "loc": list(o.location), "nbones": len(bones), "bones": bones,
                                 "action": o.animation_data.action.name if o.animation_data and o.animation_data.action else None,
                                 "nla": [[s.action.name for s in t.strips if s.action] for t in o.animation_data.nla_tracks] if o.animation_data else []})
    if o.type == "MESH":
        arm = o.find_armature()
        import mathutils
        bb = [o.matrix_world @ mathutils.Vector(c) for c in o.bound_box]
        out["meshes"].append({"name": o.name, "verts": len(o.data.vertices), "arm": arm.name if arm else None, "parent": o.parent.name if o.parent else None,
                              "min": [round(min(v[i] for v in bb), 3) for i in range(3)], "max": [round(max(v[i] for v in bb), 3) for i in range(3)]})
for a in bpy.data.actions:
    out["actions"].append({"name": a.name, "range": list(a.frame_range), "users": a.users})
out["fps"] = bpy.context.scene.render.fps
json.dump(out, open(argv[1], "w"), indent=1)
print("INSPECT_DONE")
