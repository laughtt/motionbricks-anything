class_name CameraFollow
extends Camera3D
## Third-person follow camera. Right mouse drag = orbit, wheel = zoom. Smoothed.

@export var target_path: NodePath
@export var distance := 1.9
@export var min_distance := 0.8
@export var max_distance := 12.0
@export var focus_height := 0.3
@export var orbit_yaw_deg := 55.0     ## 0 = straight behind the dog
@export var orbit_pitch_deg := 14.0
@export var follow_heading := true    ## the camera turns with the dog
@export var position_smooth := 8.0    ## 1/s
@export var heading_smooth := 2.5     ## 1/s
@export var mouse_sensitivity := 0.3  ## deg per pixel

var _target: Node3D
var _focus := Vector3.ZERO
var _heading := 0.0
var _dragging := false


func _ready() -> void:
	_target = get_node_or_null(target_path) as Node3D
	if _target:
		_focus = _target.global_position + Vector3.UP * focus_height
		_heading = _target.global_rotation.y
	_place()


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventMouseButton:
		if event.button_index == MOUSE_BUTTON_RIGHT:
			_dragging = event.pressed
		elif event.pressed and event.button_index == MOUSE_BUTTON_WHEEL_UP:
			distance = maxf(min_distance, distance * 0.9)
		elif event.pressed and event.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			distance = minf(max_distance, distance * 1.1)
	elif event is InputEventMouseMotion and _dragging:
		orbit_yaw_deg -= event.relative.x * mouse_sensitivity
		orbit_pitch_deg = clampf(orbit_pitch_deg + event.relative.y * mouse_sensitivity, -5.0, 85.0)


func _process(delta: float) -> void:
	if _target == null:
		return
	var want := _target.global_position + Vector3.UP * focus_height
	_focus = _focus.lerp(want, 1.0 - exp(-position_smooth * delta))
	if follow_heading:
		_heading = lerp_angle(_heading, _target.global_rotation.y, 1.0 - exp(-heading_smooth * delta))
	_place()


func _place() -> void:
	var yaw := _heading + PI + deg_to_rad(orbit_yaw_deg)
	var pitch := deg_to_rad(orbit_pitch_deg)
	var dir := Vector3(sin(yaw) * cos(pitch), sin(pitch), cos(yaw) * cos(pitch))
	global_position = _focus + dir * distance
	look_at(_focus, Vector3.UP)
