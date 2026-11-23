"""Drive a spider SOMA rig (.blend + <rig>_spec.json) with a motion npz (world [F,J,3], Y up, metres): empties + FK with the rig bone
lengths + COPY_LOCATION / DAMPED_TRACK chain + root bone turned with the cephalothorax (Kabsch on the rigid joints). Spec-driven."""
import bpy, json, os
import numpy as np
from mathutils import Vector, Matrix


def motion_points(mot, spec):
    z = np.load(mot); W = z["world"].astype(np.float64)
    P = np.stack([W[..., 0], -W[..., 2], W[..., 1]], -1)
    Tr = np.array(spec["rest_blender"]); PAR = spec["parents"]
    Q = np.zeros_like(P); Q[:, 0] = P[:, 0]
    for j in range(1, len(PAR)):
        p = PAR[j]; Lb = np.linalg.norm(Tr[j] - Tr[p]); d = P[:, j] - P[:, p]
        Q[:, j] = Q[:, p] + d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9) * Lb
    feet = spec["feet"]
    Q[..., 2] -= spec.get("rest_mesh_min_z", 0.0) - Tr[feet, 2].min()      # a foot joint on the floor (model y = 0) puts the mesh bottom on z = 0
    return Q


def drive(mot, rig_blend, open_file=True):
    spec = json.load(open(rig_blend.replace(".blend", "_spec.json")))
    P = motion_points(mot, spec); NAMES, PAR = spec["joints"], spec["parents"]
    if open_file: bpy.ops.wm.open_mainfile(filepath=rig_blend)
    sc = bpy.context.scene
    arm = [o for o in bpy.data.objects if o.type == "ARMATURE"][0]
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    F = len(P); E = []
    for j in range(len(NAMES)):
        e = bpy.data.objects.new(f"J{j}", None); sc.collection.objects.link(e); E.append(e)
        for f in range(F):
            e.location = Vector(P[f, j]); e.keyframe_insert("location", frame=f + 1)
    pb = arm.pose.bones
    c = pb["root"].constraints.new("COPY_LOCATION"); c.target = E[0]
    for j in range(1, len(NAMES)):
        if PAR[j] == 0:
            cl = pb[NAMES[j]].constraints.new("COPY_LOCATION"); cl.target = E[0]
        d = pb[NAMES[j]].constraints.new("DAMPED_TRACK"); d.target = E[j]; d.track_axis = "TRACK_Y"
    Tr = np.array(spec["rest_blender"]); ids = spec["rigid"]; A = Tr[ids] - Tr[0]
    rb = pb["root"]; rb.rotation_mode = "QUATERNION"; M0 = np.array(arm.data.bones["root"].matrix_local.to_3x3()); prev = None
    for f in range(F):
        B = P[f, ids] - P[f, 0]
        U, D, Vt = np.linalg.svd(B.T @ A); dd = np.sign(np.linalg.det(U @ Vt)); R = U @ np.diag([1, 1, dd]) @ Vt
        q = Matrix((M0.T @ R @ M0).tolist()).to_quaternion()
        if prev is not None and q.dot(prev) < 0: q.negate()
        rb.rotation_quaternion = q; rb.keyframe_insert("rotation_quaternion", frame=f + 1); prev = q.copy()
    sc.frame_start, sc.frame_end = 1, F
    return arm, meshes, P, spec
