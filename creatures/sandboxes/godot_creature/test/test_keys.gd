extends Node
## Keyboard path test: sends real key events (W, Shift, A, Up, Space, Tab) and checks the command and the motion.
## Godot --headless --path . res://test/test_keys.tscn   -> test_out/test_keys.json, exit code 0 = pass

var _driver: DogDriver
var _input: InputController
var _checks: Array = []
var _ok := true


func _ready() -> void:
	_driver = $Main/Dog
	_input = $Main/Input
	_run.call_deferred()


func _key(code: Key, pressed: bool) -> void:
	var ev := InputEventKey.new()
	ev.physical_keycode = code
	ev.keycode = code
	ev.pressed = pressed
	Input.parse_input_event(ev)


func _wait(seconds: float) -> void:
	await get_tree().create_timer(seconds, false, true).timeout


## Yaw change expected after 1.5 s of A / Right (the turn rate ramps up, so about 30 % of the full turn).
func _min_turn() -> float:
	return 0.3 * _input.turn_rate_stand * 1.5


func _check(name: String, pass_: bool, detail: String) -> void:
	_checks.append({"name": name, "ok": pass_, "detail": detail})
	_ok = _ok and pass_


func _run() -> void:
	await _wait(0.5)
	_key(KEY_W, true)
	await _wait(2.5)
	var z1 := _driver.position.z
	_check("W = walk forward", is_equal_approx(_driver.command.x, _input.walk_speed) and z1 > 1.0 and absf(_driver.measured_speed - _input.walk_speed) < 0.15,
		"command %.2f, speed %.2f, z %.2f" % [_driver.command.x, _driver.measured_speed, z1])
	_key(KEY_SHIFT, true)
	await _wait(2.0)
	_check("Shift + W = trot", is_equal_approx(_driver.command.x, _input.trot_speed) and absf(_driver.measured_speed - _input.trot_speed) < 0.2,
		"command %.2f, speed %.2f" % [_driver.command.x, _driver.measured_speed])
	_key(KEY_SHIFT, false)
	_key(KEY_CTRL, true)
	await _wait(2.0)
	_check("Ctrl + W = slow walk", is_equal_approx(_driver.command.x, _input.slow_speed) and absf(_driver.measured_speed - _input.slow_speed) < 0.15,
		"command %.2f, speed %.2f" % [_driver.command.x, _driver.measured_speed])
	_key(KEY_CTRL, false)
	var yaw0 := _driver.rotation.y
	_key(KEY_A, true)
	await _wait(1.5)
	_check("A = turn left (yaw goes up)", _driver.command.z > 0.0 and _driver.rotation.y - yaw0 > _min_turn(), "yaw rate command %.2f, yaw change %.2f" % [_driver.command.z, _driver.rotation.y - yaw0])
	_key(KEY_A, false)
	yaw0 = _driver.rotation.y
	_key(KEY_RIGHT, true)
	await _wait(1.5)
	_check("Right arrow = turn right", _driver.command.z < 0.0 and _driver.rotation.y - yaw0 < -_min_turn(), "yaw rate command %.2f, yaw change %.2f" % [_driver.command.z, _driver.rotation.y - yaw0])
	_key(KEY_RIGHT, false)
	_key(KEY_SPACE, true)
	await _wait(1.5)
	_check("Space = stop (W still held)", _driver.command.x == 0.0 and _driver.measured_speed < 0.05, "command %.2f, speed %.2f" % [_driver.command.x, _driver.measured_speed])
	_key(KEY_SPACE, false)
	_key(KEY_W, false)
	await _wait(0.3)
	_key(KEY_UP, true)
	await _wait(1.5)
	_check("Up arrow = forward", is_equal_approx(_driver.command.x, _input.walk_speed), "command %.2f" % _driver.command.x)
	_key(KEY_UP, false)
	_key(KEY_S, true)
	await _wait(1.5)
	_check("S = back", is_equal_approx(_driver.command.x, -_input.back_speed), "command %.2f" % _driver.command.x)
	_key(KEY_S, false)
	var b0 := _driver.body_index
	_key(KEY_TAB, true)
	_key(KEY_TAB, false)
	await _wait(0.3)
	var tab_ok := _driver.get_skeleton() != null and (_driver.body_index != b0 or _driver.get_body_count() < 2)   # one body in the config: Tab keeps it
	_check("Tab = next body", tab_ok, "body %d -> %d (%s), %d bodies" % [b0, _driver.body_index, _driver.body_name, _driver.get_body_count()])
	await _wait(0.5)
	_check("no command = stand", _driver.command == Vector3.ZERO, str(_driver.command))
	_check("model healthy", _driver.brain.is_loaded() and _driver.brain.get_stats()["nan_events"] == 0, str(_driver.brain.get_stats()["plans"]) + " plans")
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path("res://test_out"))
	var f := FileAccess.open(ProjectSettings.globalize_path("res://test_out/test_keys.json"), FileAccess.WRITE)
	f.store_string(JSON.stringify({"ok": _ok, "checks": _checks}, " "))
	f.close()
	for c: Dictionary in _checks:
		print("  ", "ok  " if c["ok"] else "FAIL", " ", c["name"], "  (", c["detail"], ")")
	print("DOG_KEYS ", "PASS" if _ok else "FAIL")
	get_tree().quit(0 if _ok else 1)
