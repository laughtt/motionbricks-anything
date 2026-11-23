class_name InputController
extends Node
## Keyboard -> command (forward m/s, side m/s, yaw rate rad/s). No model code here.
## W/S or Up/Down = forward / back, A/D or Left/Right = turn, Shift = trot, G = gallop (bodies with gallop_speed > 0), Ctrl = slow walk, Space = stop.
## C = cruise (keep walking without holding W), Tab = next body, R = reset, F1 = help.

signal body_switch_requested
signal reset_requested
signal help_toggled

@export var walk_speed := 1.0        ## m/s, W
@export var trot_speed := 2.2        ## m/s, Shift + W
@export var slow_speed := 0.5        ## m/s, Ctrl + W
@export var back_speed := 0.3        ## m/s, S (the data has little backward motion)
@export var turn_rate := 1.2         ## rad/s at walk
@export var turn_rate_trot := 0.9    ## rad/s at trot
@export var turn_rate_stand := 1.0   ## rad/s when A/D only (turn on the spot)
@export var gallop_speed := 0.0      ## m/s, G + W; 0 = no gallop for this body
@export var turn_rate_gallop := 0.45 ## rad/s at gallop

## A test sets this to drive the dog without keys: Vector3(forward, side, yaw rate). null = keyboard.
var scripted_command: Variant = null
var cruise := false
var gait_name := "walk"

const ACTIONS := {
	"dog_forward": [KEY_W, KEY_UP], "dog_back": [KEY_S, KEY_DOWN], "dog_left": [KEY_A, KEY_LEFT], "dog_right": [KEY_D, KEY_RIGHT],
	"dog_trot": [KEY_SHIFT], "dog_gallop": [KEY_G], "dog_slow": [KEY_CTRL], "dog_stop": [KEY_SPACE], "dog_cruise": [KEY_C], "dog_body": [KEY_TAB],
	"dog_reset": [KEY_R], "dog_help": [KEY_F1],
}


func _ready() -> void:
	SandboxConfig.apply_to(self, "controls")
	for action: String in ACTIONS:
		if not InputMap.has_action(action):
			InputMap.add_action(action)
			for key: int in ACTIONS[action]:
				var ev := InputEventKey.new()
				ev.physical_keycode = key
				InputMap.action_add_event(action, ev)


func _unhandled_input(event: InputEvent) -> void:
	if event.is_action_pressed("dog_body"):
		body_switch_requested.emit()
	elif event.is_action_pressed("dog_reset"):
		cruise = false
		reset_requested.emit()
	elif event.is_action_pressed("dog_help"):
		help_toggled.emit()
	elif event.is_action_pressed("dog_cruise"):
		cruise = not cruise
	elif event.is_action_pressed("dog_stop"):
		cruise = false


## Vector3(forward speed, side speed, yaw rate). Left turn = positive yaw rate.
func get_command() -> Vector3:
	if scripted_command != null:
		return scripted_command
	var gallop := gallop_speed > 0.0 and Input.is_action_pressed("dog_gallop")
	var trot := not gallop and Input.is_action_pressed("dog_trot")
	var slow := Input.is_action_pressed("dog_slow")
	var speed := gallop_speed if gallop else (trot_speed if trot else (slow_speed if slow else walk_speed))
	gait_name = "gallop" if gallop else ("trot" if trot else ("slow walk" if slow else "walk"))
	var forward := 0.0
	if Input.is_action_pressed("dog_stop"):
		forward = 0.0
	elif Input.is_action_pressed("dog_back"):
		forward = -back_speed
		cruise = false
	elif Input.is_action_pressed("dog_forward") or cruise:
		forward = speed
	var turn := Input.get_axis("dog_right", "dog_left")
	var rate := turn_rate_stand
	if forward > 0.0:
		rate = turn_rate_gallop if gallop else (turn_rate_trot if trot else turn_rate)
	return Vector3(forward, 0.0, turn * rate)
