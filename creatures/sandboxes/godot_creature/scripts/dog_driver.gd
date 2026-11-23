class_name DogDriver
extends Node3D
## Owns the DogBrain (C++ GDExtension: ONNX model + planner; "DogBrain" is the historic class name, it runs any creature model). Each physics frame: command in, pose out.
## Applies the pose to the Skeleton3D of the selected body and moves this node with the root.
## This node = the character: origin on the floor under the hips, +Z = heading.

signal body_changed(body_name: String)

@export var model_dir := "../../models/spider_v3"   ## res://, user://, absolute, or relative to the project folder
@export var input_path: NodePath
@export var body_index := 0
@export var threads := 4

var brain: DogBrain
var body_name := ""
var command := Vector3.ZERO
var measured_speed := 0.0            ## m/s, from the motion of this node (smoothed)

var _input: InputController
var _bodies: Array = []
var _holder: Node3D
var _skeleton: Skeleton3D
var _scale := 4.0
var _joint_names: PackedStringArray
var _parents: PackedInt32Array
var _rest: PackedVector3Array        # SOMA rest joints, metres
var _bone_joint: PackedInt32Array    # per skeleton bone: joint whose rotation it takes (-1 = keep rest)
var _bone_anchor: PackedInt32Array   # per skeleton bone: joint at the bone head
var _rest_global: Array[Transform3D] = []
var _last_pos := Vector3.ZERO
var _loaded_model := ""
var _brain_cfg: Variant = null
var _base_controls: Dictionary = {}
var _base_camera: Dictionary = {}


## Loads a model folder (dog_brain.json + ONNX files). A body may name its own "model_dir". --model_dir / CREATURE_MODEL_DIR override every body.
func _load_brain(dir: String) -> bool:
	dir = SandboxConfig.model_dir_for(dir)
	if dir == _loaded_model and brain != null and brain.is_loaded():
		return true
	var b := DogBrain.new()
	if not b.load(SandboxConfig.resolve_path(dir), threads):
		push_error("DogBrain: " + b.get_error() + " (" + dir + ")")
		return false
	brain = b
	_loaded_model = dir
	if _brain_cfg is Dictionary:
		brain.configure(_brain_cfg)
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--set="):
			var kv := a.trim_prefix("--set=").split("=")
			brain.configure({kv[0]: str_to_var(kv[1])})
	_joint_names = brain.get_joint_names()
	_parents = brain.get_joint_parents()
	_rest = brain.get_rest_positions()
	_scale = brain.get_rig_scale()
	_last_pos = Vector3.ZERO
	measured_speed = 0.0
	return true


func get_model_dir() -> String:
	return _loaded_model


func _ready() -> void:
	var cfg := SandboxConfig.data()
	model_dir = str(cfg.get("model_dir", model_dir))
	threads = int(cfg.get("threads", threads))
	for a in OS.get_cmdline_user_args():      # overrides: --threads=N, --body=N, --set=key=value (DogBrain.configure), --model_dir=<path>
		if a.begins_with("--threads="):
			threads = int(a.trim_prefix("--threads="))
		elif a.begins_with("--body="):
			body_index = int(a.trim_prefix("--body="))
	_bodies = cfg.get("bodies", [])
	_input = get_node_or_null(input_path) as InputController
	if _input:
		_input.body_switch_requested.connect(next_body)
		_input.reset_requested.connect(reset)
	_brain_cfg = cfg.get("brain", null)
	var cc: Variant = cfg.get("controls", null)
	if cc is Dictionary:
		_base_controls = cc
	brain = DogBrain.new()
	if not _load_brain(model_dir):
		set_physics_process(false)
		return
	set_body(body_index)


func reset() -> void:
	brain.reset()
	_last_pos = Vector3.ZERO
	measured_speed = 0.0


func next_body() -> void:
	if _bodies.size() > 0:
		set_body((body_index + 1) % _bodies.size())


func get_body_count() -> int:
	return _bodies.size()


func get_skeleton() -> Skeleton3D:
	return _skeleton


func set_body(index: int) -> void:
	if _bodies.is_empty():
		push_error("no bodies in config/sandbox.json")
		return
	body_index = clampi(index, 0, _bodies.size() - 1)
	var info: Dictionary = _bodies[body_index]
	if not _load_brain(str(info.get("model_dir", model_dir))):      # per-body model (falls back to the default model)
		return
	if info.get("brain", null) is Dictionary:
		brain.configure(info["brain"])
	if _input:                                                      # per-body speeds (a spider walks faster than a dog)
		for key: String in _base_controls:
			if key in _input:
				_input.set(key, _base_controls[key])
		if info.get("controls", null) is Dictionary:
			for key: String in info["controls"]:
				if key in _input:
					_input.set(key, info["controls"][key])
	var cam := get_node_or_null("../Camera")
	if cam:
		if _base_camera.is_empty():
			_base_camera = {"distance": cam.get("distance"), "focus_height": cam.get("focus_height")}
		var cs: Dictionary = _base_camera.duplicate()
		if info.get("camera", null) is Dictionary:
			cs.merge(info["camera"], true)
		for key: String in cs:
			if key in cam:
				cam.set(key, cs[key])
	if _holder:
		remove_child(_holder)
		_holder.queue_free()
		_holder = null
		_skeleton = null
	var inst := SandboxConfig.load_scene(str(info["scene"]))
	if inst == null:
		push_error("cannot load body " + str(info["scene"]))
		return
	_holder = Node3D.new()
	_holder.name = "Body"
	_holder.scale = Vector3.ONE / _scale          # SOMA rigs are in rig units (rig scale x metres)
	add_child(_holder)
	_holder.add_child(inst)
	_skeleton = _find_skeleton(inst)
	if _skeleton == null:
		push_error("body has no Skeleton3D: " + str(info["scene"]))
		return
	if info.has("color"):
		var c: Array = info["color"]
		var mat := StandardMaterial3D.new()
		mat.albedo_color = Color(c[0], c[1], c[2])
		mat.roughness = 0.85
		for mi: MeshInstance3D in inst.find_children("*", "MeshInstance3D", true, false):
			mi.material_override = mat
	_map_bones()
	# lowest paw JOINT height when the paw mesh touches the floor
	brain.configure({"ground_clearance": brain.get_paw_rest_height() - float(info.get("rest_mesh_min_y", 0.0))})
	body_name = info.get("name", "body")
	body_changed.emit(body_name)


func _find_skeleton(n: Node) -> Skeleton3D:
	if n is Skeleton3D:
		return n
	for c in n.get_children():
		var s := _find_skeleton(c)
		if s:
			return s
	return null


## Bones are named by their END joint ("Head" = neck). "root" takes the pelvis rotation (joint 0).
func _map_bones() -> void:
	var n := _skeleton.get_bone_count()
	_bone_joint.resize(n)
	_bone_anchor.resize(n)
	_rest_global.resize(n)
	var found := 0
	for b in n:
		var bname := _skeleton.get_bone_name(b)
		var j := 0 if bname == "root" else _joint_names.find(bname)
		_bone_joint[b] = j
		_bone_anchor[b] = 0 if j <= 0 else _parents[j]
		_rest_global[b] = _skeleton.get_bone_global_rest(b)
		if j >= 0:
			found += 1
	if found < _joint_names.size() - 2:
		push_warning("only %d of %d SOMA bones found in %s" % [found, _joint_names.size() - 1, _skeleton.name])


func _physics_process(delta: float) -> void:
	if _input:
		command = _input.get_command()
	brain.set_command(command.x, command.y, command.z)
	brain.step(delta)
	position = brain.get_root_position()
	rotation.y = brain.get_root_yaw()
	if delta > 0.0:
		measured_speed = lerpf(measured_speed, position.distance_to(_last_pos) / delta, 0.15)
	_last_pos = position
	if _skeleton:
		_apply_pose()


## Bone global pose (skeleton space, rig units) = joint rotation * rest. Poses are relative to each bone's REST in Godot,
## so the bone axes the glTF importer chose do not matter.
func _apply_pose() -> void:
	var pos := brain.get_joint_positions()
	var rot := brain.get_joint_rotations()
	var n := _skeleton.get_bone_count()
	var globals: Array[Transform3D] = []
	globals.resize(n)
	for b in n:
		var parent := _skeleton.get_bone_parent(b)
		var j := _bone_joint[b]
		var g: Transform3D
		if j < 0:
			g = (globals[parent] * _skeleton.get_bone_rest(b)) if parent >= 0 else _rest_global[b]
		else:
			var a := _bone_anchor[b]
			var r := Basis(rot[j] as Quaternion)
			g = Transform3D(r * _rest_global[b].basis, pos[a] * _scale + r * (_rest_global[b].origin - _rest[a] * _scale))
		globals[b] = g
		var local := g if parent < 0 else globals[parent].affine_inverse() * g
		_skeleton.set_bone_pose_position(b, local.origin)
		_skeleton.set_bone_pose_rotation(b, local.basis.get_rotation_quaternion())
