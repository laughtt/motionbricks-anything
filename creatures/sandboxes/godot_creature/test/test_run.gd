extends Node
## Scripted run (no keys): stand 1 s, walk 3 s, turn 3 s, fast 3 s, stop 2 s. Checks the motion and writes test_out/test_result.json.
## Headless:  Godot --headless --path . res://test/test_run.tscn
## Movie:     Godot --path . res://test/test_run.tscn --write-movie out.avi --fixed-fps 30 -- --shots
## User args (after --): --shots = save screenshots, --body=N = body index, --out=<name> = result file name.

var SEQUENCE := [   # seconds, forward m/s, yaw rate rad/s, name (speeds are replaced by the body's "test": {"walk", "fast", "turn"})
	[1.0, 0.0, 0.0, "stand"], [3.0, 1.0, 0.0, "walk"], [3.0, 1.0, 0.9, "turn"], [3.0, 2.2, 0.0, "trot"], [2.0, 0.0, 0.0, "stop"],
]
var SPEED_TOLERANCE := 0.15   # m/s, measured root speed vs command at the end of each segment (body "test": {"speed_tolerance"})
var YAW_TOLERANCE := 0.2
var _feet: PackedInt32Array = PackedInt32Array([10, 15, 19, 23])
var _tested_models := {}
const FK_TOLERANCE := 0.002     # m, skeleton bone positions vs model joints

var _driver: DogDriver
var _input: InputController
var _t := 0.0
var _seg := -1
var _seg_end := 0.0
var _result := {"segments": [], "ok": true, "errors": []}
var _nan := 0
var _fk_max := 0.0
var _paw_min := 1e9
var _paw_max_low := -1e9
var _paw_low_sum := 0.0
var _frames := 0
var _speed_win: Array[float] = []
var _yaw_win: Array[float] = []
var _last_pos := Vector3.ZERO
var _last_yaw := 0.0
var _shots := false
var _shot_times: Array[float] = []
var _out_name := "test_result.json"
var _body_arg := 0
var _render_frames := 0
var _wall_start := 0
var _process_ms: Array[float] = []
var _slide_sum := 0.0
var _slide_n := 0
var _seg_slide := 0.0
var _seg_slide_n := 0
var _last_paws: PackedVector3Array


func _ready() -> void:
	_driver = $Main/Dog
	_input = $Main/Input
	for a in OS.get_cmdline_user_args():
		if a == "--shots":
			_shots = true
		elif a.begins_with("--body="):
			_body_arg = int(a.trim_prefix("--body="))
		elif a.begins_with("--out="):
			_out_name = a.trim_prefix("--out=")
	if _shots:
		_shot_times = [0.5, 2.5, 3.6, 5.5, 6.5, 8.6, 9.3, 10.5, 11.8]
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path("res://test_out"))
	_input.scripted_command = Vector3.ZERO
	if not _driver.brain.is_loaded():
		_fail("brain not loaded: " + _driver.brain.get_error())
		_finish()
		return
	_result["bodies"] = []
	_result["self_tests"] = {}
	for i in _driver.get_body_count():
		_driver.set_body(i)
		var md := _driver.get_model_dir()
		if not _tested_models.has(md):                       # every model: the C++ plan must equal its PyTorch test vector
			_tested_models[md] = true
			var st := _driver.brain.self_test()
			_result["self_tests"][md] = st
			if i == 0:
				_result["self_test"] = st
			if st["max_abs_diff"] < 0.0 or st["max_abs_diff"] > 1e-3 or st["token_mismatches"] != 0:
				_fail("C++ plan differs from the PyTorch test vector: " + md)
		var need := _driver.brain.get_joint_count() - 1
		var have := 0
		if _driver.get_skeleton():
			for jn in _driver.brain.get_joint_names():
				if _driver.get_skeleton().find_bone(jn) >= 0:
					have += 1
		if _driver.get_skeleton() == null or have < need - 1:
			_fail("body %d (%s) did not load with the %d SOMA bones of its model (found %d)" % [i, _driver.body_name, need, have])
			continue
		var mi: MeshInstance3D = _driver.find_children("*", "MeshInstance3D", true, false)[0]
		var fmt: int = mi.mesh.surface_get_format(0)
		_result["bodies"].append({"name": _driver.body_name, "bones": _driver.get_skeleton().get_bone_count(), "model": md, "joints": need + 1,
			"weights_8": (fmt & Mesh.ARRAY_FLAG_USE_8_BONE_WEIGHTS) != 0, "vertices": mi.mesh.surface_get_array_len(0)})
	_driver.set_body(_body_arg)
	if _driver.brain.has_method("get_feet"):
		_feet = _driver.brain.get_feet()
	var bodies_cfg: Array = SandboxConfig.data().get("bodies", [])
	if _body_arg < bodies_cfg.size() and bodies_cfg[_body_arg].get("test", null) is Dictionary:
		var tc: Dictionary = bodies_cfg[_body_arg]["test"]
		var w: float = tc.get("walk", 1.0)
		var fa: float = tc.get("fast", 2.2)
		var tu: float = tc.get("turn", 0.9)
		SEQUENCE = [[1.0, 0.0, 0.0, "stand"], [3.0, w, 0.0, "walk"], [3.0, w, tu, "turn"], [3.0, fa, 0.0, "fast"], [2.0, 0.0, 0.0, "stop"]]
		SPEED_TOLERANCE = tc.get("speed_tolerance", SPEED_TOLERANCE)
		YAW_TOLERANCE = tc.get("yaw_tolerance", YAW_TOLERANCE)
	if not _result["bodies"][0]["weights_8"]:
		_fail("the body mesh was imported without 8 bone weights")
	_next_segment()


func _fail(msg: String) -> void:
	_result["ok"] = false
	_result["errors"].append(msg)
	push_error(msg)


func _next_segment() -> void:
	_seg += 1
	if _seg >= SEQUENCE.size():
		_finish()
		return
	var s: Array = SEQUENCE[_seg]
	_seg_end = _t + s[0]
	_input.scripted_command = Vector3(s[1], 0.0, s[2])
	_speed_win.clear()
	_yaw_win.clear()
	_seg_slide = 0.0
	_seg_slide_n = 0


func _physics_process(delta: float) -> void:
	if _seg < 0 or _seg >= SEQUENCE.size():
		return
	# this node is before Main in the tree order of _physics_process? No: parents run first, so the values are from the last driver step.
	_t += delta
	_frames += 1
	var brain := _driver.brain
	var pos := brain.get_joint_positions()
	for p in pos:
		if not (is_finite(p.x) and is_finite(p.y) and is_finite(p.z)):
			_nan += 1
	for q: Quaternion in brain.get_joint_rotations():
		if not (is_finite(q.x) and is_finite(q.w)) or absf(q.length() - 1.0) > 1e-3:
			_nan += 1
	# skeleton vs model joints (end-to-end check of the bone rotation math and the import axes)
	var skel := _driver.get_skeleton()
	var names := brain.get_joint_names()
	var parents := brain.get_joint_parents()
	var rest := brain.get_rest_positions()
	var scale := brain.get_rig_scale()
	for j in range(1, names.size()):
		var b := skel.find_bone(names[j])
		if b < 0:
			continue
		var g := skel.get_bone_global_pose(b)
		var head_err := (g.origin / scale).distance_to(pos[parents[j]])
		var tail := g * (skel.get_bone_global_rest(b).affine_inverse() * (rest[j] * scale))
		_fk_max = maxf(_fk_max, maxf(head_err, (tail / scale).distance_to(pos[j])))
	var low := 1e9
	var paws := PackedVector3Array()
	var world := brain.get_world_joint_positions()
	for f: int in _feet:
		low = minf(low, pos[f].y)
		paws.append(world[f])
	_paw_min = minf(_paw_min, low)
	_paw_max_low = maxf(_paw_max_low, low)
	_paw_low_sum += low
	var contacts := brain.get_contacts()
	if _last_paws.size() == _feet.size():
		for i in _feet.size():
			if contacts[i] > 0.5:
				_slide_sum += Vector2(paws[i].x - _last_paws[i].x, paws[i].z - _last_paws[i].z).length() / delta
				_slide_n += 1
				_seg_slide += Vector2(paws[i].x - _last_paws[i].x, paws[i].z - _last_paws[i].z).length() / delta
				_seg_slide_n += 1
	_last_paws = paws
	var p := _driver.position
	_speed_win.append(p.distance_to(_last_pos) / delta)
	_yaw_win.append((_driver.rotation.y - _last_yaw) / delta if absf(_driver.rotation.y - _last_yaw) < 1.0 else 0.0)
	_last_pos = p
	_last_yaw = _driver.rotation.y
	_process_ms.append(Performance.get_monitor(Performance.TIME_PHYSICS_PROCESS) * 1000.0)
	if _t >= _seg_end:
		var s: Array = SEQUENCE[_seg]
		var n := mini(_speed_win.size(), 30)   # last 0.5 s
		var sp := 0.0
		var yw := 0.0
		for i in n:
			sp += _speed_win[_speed_win.size() - 1 - i] / n
			yw += _yaw_win[_yaw_win.size() - 1 - i] / n
		var ok: bool = absf(sp - absf(s[1])) <= SPEED_TOLERANCE and absf(yw - s[2]) <= YAW_TOLERANCE
		_result["segments"].append({"name": s[3], "seconds": s[0], "cmd_forward": s[1], "cmd_yaw": s[2], "speed_end": snappedf(sp, 0.001),
			"yaw_rate_end": snappedf(yw, 0.001), "ok": ok, "contact_slide_mps": snappedf(_seg_slide / maxi(_seg_slide_n, 1), 0.001), "root": [snappedf(p.x, 0.01), snappedf(p.z, 0.01)], "hips_y": snappedf(pos[0].y, 0.001)})
		if not ok:
			_fail("segment %s: speed %.2f (command %.2f), yaw rate %.2f (command %.2f)" % [s[3], sp, s[1], yw, s[2]])
		_next_segment()


func _process(_delta: float) -> void:
	if _render_frames == 0:
		_wall_start = Time.get_ticks_usec()
	_render_frames += 1
	if _shots and not _shot_times.is_empty() and _t >= _shot_times[0]:
		var t: float = _shot_times.pop_front()
		get_viewport().get_texture().get_image().save_png(ProjectSettings.globalize_path("res://test_out/shot_%s_%04.1fs.png" % [str(_driver.body_index), t]))


func _finish() -> void:
	var brain := _driver.brain
	if brain.is_loaded():
		var st := brain.get_stats()
		_result["stats"] = st
		_result["config"] = brain.get_config()
		if st["nan_events"] > 0:
			_fail("the model produced non-finite values")
	_result["nan_values"] = _nan
	_result["fk_max_error_m"] = snappedf(_fk_max, 0.00001)
	_result["lowest_paw_joint_m"] = {"min": snappedf(_paw_min, 0.0001), "max": snappedf(_paw_max_low, 0.0001), "mean": snappedf(_paw_low_sum / maxi(_frames, 1), 0.0001)}
	_result["contact_paw_slide_mps_mean"] = snappedf(_slide_sum / maxi(_slide_n, 1), 0.001)
	_result["physics_frames"] = _frames
	_result["body"] = _driver.body_name
	_result["headless"] = DisplayServer.get_name() == "headless"
	_result["render_fps_avg"] = snappedf(_render_frames / maxf((Time.get_ticks_usec() - _wall_start) / 1e6, 1e-6), 0.1)
	_result["wall_seconds"] = snappedf((Time.get_ticks_usec() - _wall_start) / 1e6, 0.01)
	if not _process_ms.is_empty():
		_process_ms.sort()
		_result["physics_process_ms"] = {"median": snappedf(_process_ms[_process_ms.size() / 2], 0.001), "p99": snappedf(_process_ms[int(_process_ms.size() * 0.99)], 0.001),
			"max": snappedf(_process_ms[-1], 0.001)}
	if _nan > 0:
		_fail("%d non-finite or non-unit values in the pose" % _nan)
	if _fk_max > FK_TOLERANCE:
		_fail("skeleton does not match the model joints: %.4f m" % _fk_max)
	var path := ProjectSettings.globalize_path("res://test_out/" + _out_name)
	var f := FileAccess.open(path, FileAccess.WRITE)
	f.store_string(JSON.stringify(_result, " "))
	f.close()
	print("DOG_TEST ", "PASS" if _result["ok"] else "FAIL", " ", path)
	get_tree().quit(0 if _result["ok"] else 1)
