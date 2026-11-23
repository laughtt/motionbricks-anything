"""Spider SOMA template for the image-edit step: grey mannequin, white background, 1536x1536, ortho three-quarter view from above-front
so that all 8 legs show. blender -b --factory-startup --python render_spider_template.py -- <rig.blend> <out.png> [yaw=35] [elev=32]"""
import bpy, sys, math
from mathutils import Vector
argv = sys.argv[sys.argv.index("--") + 1:]
bpy.ops.wm.open_mainfile(filepath=argv[0]); OUT = argv[1]
YAW = float(argv[2]) if len(argv) > 2 else 35.0; EL = float(argv[3]) if len(argv) > 3 else 32.0
sc = bpy.context.scene
ms = [o for o in bpy.data.objects if o.type == "MESH"]
for o in bpy.data.objects:
    if o.type == "ARMATURE": o.hide_render = True
bb = [o.matrix_world @ Vector(c) for o in ms for c in o.bound_box]
mn = Vector([min(v[i] for v in bb) for i in range(3)]); mx = Vector([max(v[i] for v in bb) for i in range(3)]); ctr = (mn + mx) / 2
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam); sc.camera = cam
cam.data.type = "ORTHO"; cam.data.ortho_scale = max(mx.x - mn.x, mx.y - mn.y) * 1.12
a, e = math.radians(YAW), math.radians(EL)          # spider faces -Y; camera on its left (+X), turned toward the front, raised
d = Vector((math.cos(a) * math.cos(e), -math.sin(a) * math.cos(e), math.sin(e)))
cam.location = ctr + d * 30
cam.rotation_euler = (ctr - cam.location).to_track_quat("-Z", "Y").to_euler()
if len(argv) <= 4:
    m = bpy.data.materials.new("grey"); m.diffuse_color = (0.62, 0.62, 0.62, 1)
    for o in ms: o.data.materials.clear(); o.data.materials.append(m)
sc.render.engine = "BLENDER_WORKBENCH"; sc.display.shading.light = "STUDIO"; sc.display.shading.color_type = "MATERIAL" if len(argv) <= 4 else "TEXTURE"
sc.display.shading.show_cavity = True; sc.display.shading.show_shadows = False
sc.view_settings.view_transform = "Standard"; sc.world = bpy.data.worlds.new("w"); sc.world.color = (1, 1, 1)
sc.render.resolution_x = sc.render.resolution_y = 1536; sc.render.film_transparent = False
sc.render.filepath = OUT; bpy.ops.render.render(write_still=True)
print("TEMPLATE_DONE", OUT)
