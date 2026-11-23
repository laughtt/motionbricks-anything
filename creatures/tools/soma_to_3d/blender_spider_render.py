"""Render a motion on a spider rig directly (same skeleton: no retarget).
blender -b --factory-startup --python blender_spider_render.py -- <motion.npz> <rig.blend> <out.mp4>
env: EEVEE=1, CAM_DIST (9), CAM_H (0.3), CAM_ANG (-0.45), CAM_FOLLOW (0.12), RES (1280x720), EVERY (1), TURNTABLE=1 (rest pose turntable, no motion)."""
import bpy, sys, math, subprocess, json, os
import numpy as np
from mathutils import Vector
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spider_drive import drive
argv = sys.argv[sys.argv.index("--") + 1:]
MOT, RIG, OUTV = argv[:3]
E = os.environ.get
TT = E("TURNTABLE") == "1"
if TT:
    bpy.ops.wm.open_mainfile(filepath=RIG); F = int(E("TT_FRAMES", "120")); P = None
    sc = bpy.context.scene; sc.frame_start, sc.frame_end = 1, F
else:
    arm, meshes, P, spec = drive(MOT, RIG); F = len(P); sc = bpy.context.scene
for o in bpy.data.objects:
    if o.type == "ARMATURE": o.hide_render = True
ms = [o for o in bpy.data.objects if o.type == "MESH"]
TEXTURED = any(m and m.use_nodes and any(n.type == "TEX_IMAGE" for n in m.node_tree.nodes) for o in ms for m in o.data.materials)
bpy.ops.mesh.primitive_plane_add(size=400, location=(0, 0, 0)); g = bpy.context.active_object
gm = bpy.data.materials.new("ground"); gm.diffuse_color = (0.33, 0.36, 0.31, 1); g.data.materials.append(gm)
lm = bpy.data.materials.new("line"); lm.diffuse_color = (0.23, 0.25, 0.22, 1)
for i in range(-80, 81, 2):
    for axis in (0, 1):
        bpy.ops.mesh.primitive_cube_add(size=1, location=(i if axis == 0 else 0, 0 if axis == 0 else i, 0.004))
        o = bpy.context.active_object; o.scale = (0.03, 320, 0.004) if axis == 0 else (320, 0.03, 0.004); o.data.materials.append(lm)
for m_ in (gm, lm):                        # Eevee reads the node colour, Workbench the viewport colour
    m_.use_nodes = True
    b_ = m_.node_tree.nodes.get("Principled BSDF")
    if b_: b_.inputs["Base Color"].default_value = m_.diffuse_color; b_.inputs["Roughness"].default_value = 0.9
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam); sc.camera = cam; cam.data.lens = 35
CD = float(E("CAM_DIST", "9")); kf = float(E("CAM_FOLLOW", "0.12"))
if TT:
    bb = [o.matrix_world @ Vector(c) for o in ms for c in o.bound_box]
    ctr = Vector([(min(v[i] for v in bb) + max(v[i] for v in bb)) / 2 for i in range(3)])
    for f in range(F):
        a = 2 * math.pi * f / F
        cam.location = ctr + Vector((CD * math.cos(a), CD * math.sin(a), CD * 0.35))
        cam.rotation_euler = (ctr - cam.location).to_track_quat("-Z", "Y").to_euler()
        cam.keyframe_insert("location", frame=f + 1); cam.keyframe_insert("rotation_euler", frame=f + 1)
else:
    fwd = P[:, 1, :2] - P[:, 0, :2]; ang = np.unwrap(np.arctan2(fwd[:, 1], fwd[:, 0]))
    k = 25; angS = np.convolve(np.pad(ang, (k, k), mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), mode="valid")
    sx, sy = P[0, 0, 0], P[0, 0, 1]
    for f in range(F):
        sx += (P[f, 0, 0] - sx) * kf; sy += (P[f, 0, 1] - sy) * kf
        a = angS[f] - math.pi / 2 + float(E("CAM_ANG", "-0.45"))      # -0.45 = from behind-left, +0.6 = from the front-left
        cam.location = Vector((sx + CD * math.cos(a), sy + CD * math.sin(a), 0.8 + CD * float(E("CAM_H", "0.3"))))      # CAM_H 1.4 + CAM_DIST 2.6 = close view on the legs from above
        cam.rotation_euler = (Vector((sx, sy, 0.7)) - cam.location).to_track_quat("-Z", "Y").to_euler()
        cam.keyframe_insert("location", frame=f + 1); cam.keyframe_insert("rotation_euler", frame=f + 1)
sc.render.fps = 30
rx, ry = [int(x) for x in E("RES", "1280x720").split("x")]; sc.render.resolution_x, sc.render.resolution_y = rx, ry
sc.frame_step = int(E("EVERY", "1"))
sc.render.engine = "BLENDER_WORKBENCH"; sc.display.shading.light = "STUDIO"; sc.display.shading.color_type = "TEXTURE" if TEXTURED else "MATERIAL"; sc.display.shading.show_shadows = True
if E("EEVEE") == "1":
    for eng in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
        try:
            sc.render.engine = eng; break
        except Exception: pass
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN")); sc.collection.objects.link(sun)
    sun.data.energy = 4.0; sun.rotation_euler = (math.radians(40), math.radians(10), math.radians(-30))
    if sc.world is None: sc.world = bpy.data.worlds.new("w")
    sc.world.use_nodes = True
    bg = sc.world.node_tree.nodes.get("Background")
    if bg: bg.inputs[0].default_value = (0.6, 0.68, 0.78, 1); bg.inputs[1].default_value = 1.2
    try: sc.eevee.taa_render_samples = 16
    except Exception: pass
    sc.view_settings.view_transform = "AgX" if "AgX" in [i.identifier for i in sc.view_settings.bl_rna.properties["view_transform"].enum_items] else "Filmic"
else:
    sc.view_settings.view_transform = "Standard"; sc.world = bpy.data.worlds.new("w"); sc.world.color = (0.55, 0.62, 0.7)
sc.render.image_settings.file_format = "PNG"; FR = OUTV.replace(".mp4", "_frames/f_"); sc.render.filepath = FR
bpy.ops.render.render(animation=True)
fps = 30 / sc.frame_step
import glob
fl = sorted(glob.glob(FR + "*.png"))
lst = OUTV.replace(".mp4", "_frames/list.txt"); open(lst, "w").write("".join(f"file '{p.replace(chr(92), '/')}'\nduration {1 / fps}\n" for p in fl))
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-vf", f"fps={fps}", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", OUTV])
print("RENDER_DONE", OUTV, F)
