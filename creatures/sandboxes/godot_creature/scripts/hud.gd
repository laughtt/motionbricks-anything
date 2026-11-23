class_name SandboxHud
extends CanvasLayer
## Text overlay: speed (commanded / actual), gait estimate, model timing, body, key help. Reads state only.

@export var driver_path: NodePath
@export var input_path: NodePath
@export var stand_below := 0.12   ## m/s: gait name thresholds (estimate from the speed)
@export var walk_below := 1.5
@export var trot_below := 2.9

var _driver: DogDriver
var _input: InputController
var _label: Label
var _help: Label
var _show_help := true

const HELP := "W/S or Up/Down: forward / back    A/D or Left/Right: turn    Shift: trot    G: gallop (if the body sets gallop_speed)    Ctrl: slow walk    Space: stop\nC: cruise    Tab: next body    R: reset    Right mouse drag: orbit    Wheel: zoom    F1: hide help"


func _ready() -> void:
	_driver = get_node_or_null(driver_path) as DogDriver
	_input = get_node_or_null(input_path) as InputController
	_label = _make_label(Vector2(12, 8))
	_help = _make_label(Vector2(12, 0))
	_help.text = HELP
	_help.set_anchors_and_offsets_preset(Control.PRESET_BOTTOM_LEFT, Control.PRESET_MODE_MINSIZE, 12)
	if _input:
		_input.help_toggled.connect(func() -> void:
			_show_help = not _show_help
			_help.visible = _show_help)


func _make_label(at: Vector2) -> Label:
	var l := Label.new()
	l.position = at
	l.add_theme_font_size_override("font_size", 15)
	l.add_theme_color_override("font_color", Color(1, 1, 1))
	l.add_theme_color_override("font_outline_color", Color(0, 0, 0))
	l.add_theme_constant_override("outline_size", 5)
	add_child(l)
	return l


func gait_of(speed: float) -> String:
	if speed < stand_below:
		return "stand"
	if speed < walk_below:
		return "walk"
	if speed < trot_below:
		return "trot"
	return "gallop"


func _process(_delta: float) -> void:
	if _driver == null or _driver.brain == null or not _driver.brain.is_loaded():
		_label.text = "DogBrain is not loaded: " + (_driver.brain.get_error() if _driver and _driver.brain else "no driver")
		return
	var s := _driver.brain.get_stats()
	var c := _driver.command
	var lines := PackedStringArray()
	lines.append("Speed  command %.2f m/s   model %.2f m/s   actual %.2f m/s" % [c.x, s["speed_forward"], _driver.measured_speed])
	lines.append("Turn   command %.2f rad/s   model %.2f rad/s" % [c.z, s["yaw_rate"]])
	lines.append("Gait   %s (estimate from speed)%s" % [gait_of(absf(s["speed_forward"])), "   [cruise]" if _input and _input.cruise else ""])
	lines.append("Model  %.2f ms per plan (avg %.2f, max %.2f)   %d frames per plan   %d plans" % [s["plan_ms_last"], s["plan_ms_avg"], s["plan_ms_max"], s["frames_last_plan"], s["plans"]])
	lines.append("Body   %s   (Tab: next of %d)        %d fps" % [_driver.body_name, _driver.get_body_count(), Engine.get_frames_per_second()])
	_label.text = "\n".join(lines)
