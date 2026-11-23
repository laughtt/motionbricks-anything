"""Export a dog SOMA rig (.blend) to a glb for Godot with at most 8 bone influences per vertex (Godot limit), JPEG textures.
blender -b --factory-startup --python blender_export_godot.py -- <rig.blend> <out.glb> [w8_rig.blend]
The 8-influence rig (LBS, no Preserve Volume = what Godot does) is also saved as [w8_rig.blend] (+ _spec.json)."""
import bpy, sys, shutil, os
import numpy as np
argv = sys.argv[sys.argv.index("--") + 1:]
RIG, OUT = argv[:2]; W8 = argv[2] if len(argv) > 2 else None
LIMIT = int(os.environ.get("LIMIT", "8"))
bpy.ops.wm.open_mainfile(filepath=RIG)
for ob in [o for o in bpy.data.objects if o.type == "MESH"]:
    me = ob.data; before = []; after = []
    for v in me.vertices:
        gs = sorted(v.groups, key=lambda g: -g.weight); before.append(len(gs))
        keep = gs[:LIMIT]
        # soft limit (default): subtract the weight of the first dropped influence from the kept ones, then renormalize. The weights
        # stay continuous over the surface when the 8th and 9th influence swap between neighbour vertices (a hard cut makes a step = a tear).
        MODE = os.environ.get("MODE", "pow2")          # hard | soft (w - w9) | powK (w * (1 - (w9 / w)^K): changes the big weights less)
        cut = gs[LIMIT].weight if (len(gs) > LIMIT and MODE != "hard") else 0.0
        if MODE.startswith("pow") and cut > 0:
            K = float(MODE[3:]); ws = [g.weight * (1.0 - (cut / g.weight) ** K) if g.weight > cut else 0.0 for g in keep]
        else:
            ws = [max(g.weight - cut, 0.0) for g in keep]
        tot = sum(ws) or 1.0
        new = [(g.group, w / tot) for g, w in zip(keep, ws) if w > 0]
        drop = [g.group for g in gs[LIMIT:]] + [g.group for g, w in zip(keep, ws) if w <= 0]
        for gi, w in new:
            ob.vertex_groups[gi].add([v.index], w, "REPLACE")
        for gi in drop:
            ob.vertex_groups[gi].remove([v.index])
        after.append(len(new))
    print("LIMIT", ob.name, "influences before mean %.2f max %d -> after mean %.2f max %d" % (np.mean(before), max(before), np.mean(after), max(after)))
    for m in ob.modifiers:
        if m.type == "ARMATURE":
            m.use_deform_preserve_volume = False
if W8:
    bpy.ops.wm.save_as_mainfile(filepath=W8)
    shutil.copy(RIG.replace(".blend", "_spec.json"), W8.replace(".blend", "_spec.json"))
kw = dict(filepath=OUT, export_format="GLB", export_animations=False, export_skins=True, export_all_influences=True,
          export_image_format="JPEG", export_yup=True, export_apply=False)
try:
    bpy.ops.export_scene.gltf(**kw, export_jpeg_quality=92)
except TypeError:
    bpy.ops.export_scene.gltf(**kw)
print("EXPORT_DONE", OUT, os.path.getsize(OUT))
