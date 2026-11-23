# Godot creature sandbox

A small Godot 4.7 project that drives a creature model with the keyboard. The body is the generated spider SOMA mesh
(`assets/spider_soma.glb`) and the model is `../../models/spider_v3`. The model runs inside the C++ GDExtension `DogBrain`
(a historic class name; see [`../../runtime/README.md`](../../runtime/README.md)). GDScript only reads keys, moves the character and sets bone poses.

Tested on Windows x86_64 with Godot 4.7.2. Linux and macOS: not tested.

## Run it

1. Build the extension (in `creatures/runtime`, with `CREATURE_BUILD_GODOT=ON`). The copy step puts `dogbrain.dll` and `onnxruntime.dll` in `addons/dogbrain/bin/`.
2. Set the environment variable `GODOT` to your Godot 4.7 executable. (If you do not, the scripts call `godot` from PATH.)
3. Run `run_sandbox.bat` (Windows) or `./run_sandbox.sh`. The first start imports the glb (one minute at most).
   Arguments after the script name go to the project, for example `run_sandbox.bat --model_dir=../../models/spider_v3`.

## Controls

| Key | Action |
|---|---|
| W / S, Up / Down | forward / back |
| A / D, Left / Right | turn left / right |
| Shift | fast walk (trot speed) |
| Ctrl | slow walk |
| G | gallop (only for a body or model with `gallop_speed` above 0; the spider has none) |
| Space | stop |
| C | cruise (keep walking without holding W) |
| Tab | next body (there is only one body now) |
| R | reset |
| Right mouse drag, wheel | orbit, zoom |
| F1 | hide / show the help |

## Config (`config/sandbox.json`)

| Key | Meaning |
|---|---|
| `model_dir` | Default model folder. |
| `threads` | CPU threads for inference. |
| `controls` | Values of the exported properties of `input_controller.gd` (speeds in m/s, turn rates in rad/s). |
| `brain` | Values for `DogBrain.configure()` (replan and blend settings; the names are the fields of `dog::Config` in the runtime). |
| `bodies` | List of bodies (Tab switches). Per body: `name`, `scene`, `rest_mesh_min_y` (lowest mesh point in the rest pose, metres), optional `color`, `model_dir`, `controls`, `brain`, `camera` (`distance`, `focus_height`), `test` (`walk`, `fast`, `turn`, `speed_tolerance`, `yaw_tolerance` for the scripted test). |

Paths (`model_dir`, `scene`) can be `res://...`, `user://...`, absolute, or relative to the project folder (for example `../../models/spider_v3`).
A `res://` scene is an imported resource. Any other scene path is read as a glTF file at run time (no import needed).

To use another model folder, add `-- --model_dir=<path>` to the command line, or set the environment variable `CREATURE_MODEL_DIR`.
It replaces the model of every body. A relative path starts at the project folder. Other command-line options: `--threads=N`, `--body=N`, `--set=<key>=<value>` (a `DogBrain.configure` value).

## Add a body or a model

- **Body:** the mesh must use the SOMA skeleton of the model (bone names are the END joint names listed in `dog_brain.json`, plus `root`) with at most 8 bone influences per vertex.
  Put the glb in `assets/`, add an entry to `bodies`, and run `run_test.bat` once (it imports the file). The scale comes from the model (`rig_scale`).
- **Model:** put a folder with `dog_brain.json`, `pose_step.onnx`, `tok_dec.onnx` and `root_net.onnx` anywhere, and set `model_dir`
  (see [`../../models/spider_v3/MODEL_CONTRACT.md`](../../models/spider_v3/MODEL_CONTRACT.md)). A body can name its own `model_dir`. The number of joints must match the body.

## Tests

`run_test.bat` / `./run_test.sh` import the project and run two headless tests. Both write JSON to `test_out/` and return exit code 0 on pass.

- `res://test/test_run.tscn`: a scripted command sequence (stand, walk, turn, fast, stop). It checks that the C++ plan equals the PyTorch test vector of the model; speed and yaw rate at the end of each segment;
  skeleton bones against the model joints (FK error); no non-finite values. It also reports contact foot sliding. Result: `test_out/test_result.json`.
- `res://test/test_keys.tscn`: sends real key events and checks the commands and the motion. Result: `test_out/test_keys.json`.

Reference run (spider v3 + spider SOMA, headless, Windows): root speed 0.01 / 1.54 / 1.46 / 3.51 / 0.06 m/s at the commands 0 / 1.5 / 1.5 with 0.6 rad/s turn / 3.5 / 0 m/s, skeleton error 0.00 mm,
about 5 ms per plan inside Godot (2 threads), keys test 11 of 11 checks.

## Good to know

- Feet slide a little (contact foot speed about 0.2 to 0.6 m/s while the contact value is on). There is no foot locking or leg IK.
- Backward motion is small. There are no side-step keys.
- The gait name on the HUD is a rough guess from the speed.
- Inference runs on the main thread (a few ms every 12 frames). CPU only.
- Only one body ships (the spider SOMA). Other bodies are not included.
