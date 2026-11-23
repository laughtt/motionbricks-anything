# runtime/: the C++ runtime

This is the C++ runtime for the creature models. It uses ONNX Runtime on the CPU. It loads a model folder, takes a command (forward speed, side speed, yaw rate) and gives you the joint positions of the creature at any frame rate.

| Folder | What is inside |
|---|---|
| `src/` | `dog_brain_core.cpp/.hpp`: the core, with no engine types (class `dog::Brain`). `creature_runtime.h/.cpp`: a plain C API on top of the core. |
| `godot/` | `dog_brain_gd.cpp`: the Godot 4.7 GDExtension, class `DogBrain`. |
| `test/` | `dog_brain_test.cpp`: console test of the C++ core. `creature_c_api_test.cpp`: console test through the C API. |
| `tools/` | `fetch_deps.py`: downloads the official ONNX Runtime CPU package (from nuget.org by default; use `--ort-source github` for the GitHub release archive) and fetches godot-cpp at a pinned commit (`2a4c066e1e`). Git ignores the folder `creatures/deps/`. |

**Names:** the class `DogBrain`, the namespace `dog` and the spec file `dog_brain.json` are historic (the first model was a dog). They stay the same in v1.
The core works for any creature: the joint count, the feet and the body-frame joints come from `dog_brain.json` (the spider has 38 joints and 8 feet).
The model contract is in [`../models/spider_v3/MODEL_CONTRACT.md`](../models/spider_v3/MODEL_CONTRACT.md).

Axes: Y up, forward +Z, left +X, metres. A left turn is a positive yaw rate.

## Build on Windows (MSVC, tested)

You need Visual Studio 2022 (or Build Tools) with the C++ workload, and CMake 3.20 or newer.

1. Get the dependencies (or use your own copies, see "Settings" below):

       python tools\fetch_deps.py ..\deps
       set ONNXRUNTIME_DIR=<the path that fetch_deps.py prints>

2. Configure and build in an x64 developer prompt (the build folder can be anywhere):

       cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
       cmake --build build
       ctest --test-dir build --output-on-failure

   You get `build/bin/creature_runtime.dll`, `dog_brain_test.exe`, `creature_c_api_test.exe`, and `onnxruntime.dll` copied next to them.
   With the Visual Studio generator, use `-A x64` and `--config Release`.

3. Godot extension (optional): see "Godot extension" below.

## Build on Linux (NOT tested)

The CMake file has Linux branches (RPATH `$ORIGIN`, `libonnxruntime.so*` copy), but nobody has built it on Linux yet. The steps should be:

    python3 tools/fetch_deps.py ../deps
    export ONNXRUNTIME_DIR="$(cd ../deps/onnxruntime && pwd)"
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j
    ctest --test-dir build --output-on-failure

## Settings (environment variables and CMake options)

| Option / variable | Meaning |
|---|---|
| `ONNXRUNTIME_DIR` (-D or env) | Required. An unpacked official ONNX Runtime release (`include/`, `lib/`), or the NuGet layout (`build/native/include`, `runtimes/win-x64/native`). CPU only. DirectML is not used. |
| `NLOHMANN_JSON_DIR` (-D or env) | Optional. A folder that contains `nlohmann/json.hpp`. If you do not set it, CMake downloads nlohmann json 3.12.0 (FetchContent). |
| `CREATURE_BUILD_GODOT` | `ON` builds the GDExtension. Default `OFF`. |
| `GODOT_CPP_DIR` (-D or env) | Needed with `CREATURE_BUILD_GODOT=ON`: a godot-cpp checkout. By default CMake builds it with `add_subdirectory`. |
| `GODOT_CPP_LIB` (-D or env) | Optional: a prebuilt godot-cpp static library (Windows: Release, /MD). Then `GODOT_CPP_GEN_DIR` (default `<GODOT_CPP_DIR>/gen`) must hold the generated headers in `include/`. |
| `CREATURE_ADDON_BIN` | Where the copy step puts the GDExtension library and the ONNX Runtime library. Default: `../sandboxes/godot_creature/addons/dogbrain/bin`. |
| `CREATURE_BUILD_TESTS` | Build the console tests. Default `ON`. |
| `CREATURE_TEST_MODEL_DIR` | Model folder for `ctest`. Default `../models/spider_v3`. |

About the godot-cpp version: Godot 4.7 has no godot-cpp release branch yet (the branches stop at 4.5), so `tools/fetch_deps.py` clones `master` (`--godot-cpp-ref` picks another branch or tag). The `.gdextension` file asks for Godot 4.7 or newer.

## Godot extension

    set GODOT_CPP_DIR=<the path that fetch_deps.py prints>
    cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCREATURE_BUILD_GODOT=ON
    cmake --build build

The build copies `dogbrain.dll` and `onnxruntime.dll` into `CREATURE_ADDON_BIN` (the sandbox addon folder by default).
Then run the sandbox (see [`../sandboxes/godot_creature/README.md`](../sandboxes/godot_creature/README.md)).

## C API (`src/creature_runtime.h`)

| Function | What it does |
|---|---|
| `creature_create()` / `creature_destroy(rt)` | Make / free a handle. |
| `creature_load(rt, model_dir, threads)` | Load a model folder. Returns 1 on success, 0 on failure. |
| `creature_reset(rt)` | Go back to the start pose, with the root at the origin. |
| `creature_set_command(rt, forward, side, yaw_rate)` | Set the command in m/s and rad/s. |
| `creature_step(rt, dt)` | Advance by dt seconds. Returns the number of 30 fps model frames advanced, or -1 on error. |
| `creature_joint_count(rt)`, `creature_contact_count(rt)` | Sizes. |
| `creature_get_world_joints(rt, out_xyz, max_joints)` | Joint world positions (x y z per joint) into your buffer. |
| `creature_get_root(rt, out_xyz, out_yaw)` | Root position (y = 0) and heading yaw in radians. |
| `creature_get_contacts(rt, out, max)` | Foot contact values, about 0..1, one per foot. |
| `creature_nan_events(rt)` | Count of non-finite model outputs (0 = healthy). |
| `creature_last_error(rt)` | Error text of the last failure. |

Use one handle from one thread at a time. The core needs only the model folder: `dog_brain.json`, `pose_step.onnx`, `tok_dec.onnx`, `root_net.onnx`.
The C API gives joint positions only. The Godot class `DogBrain` also gives bone rotations, settings (`configure`) and statistics.

## Tests

Both tests take the model folder as the first argument (a relative path is fine). They need no other data.
The model folder holds a PyTorch test vector inside `dog_brain.json` (block `test`; the root net and the seam contract have their own test blocks).

    build/bin/dog_brain_test ../models/spider_v3 2 walk 1.5 fast 3.5
    build/bin/creature_c_api_test ../models/spider_v3 2 1.5

- `dog_brain_test <model_dir> [threads] [walk <m/s>] [fast <m/s>] [replan <n>] [crossfade <n>] [idle <n>] [rootnet 0|1] [seam 0|1]`:
  runs the C++ plan on the stored test input and compares it with the PyTorch output (the max abs difference must be below 1e-3 and the token mismatches must be 0).
  It does the same for the root net and the seam contract when the model has them. Then it drives a scripted command sequence and checks for non-finite values.
  Exit code 0 means pass.
- `creature_c_api_test <model_dir> [threads] [walk <m/s>]`: drives the model through the C API for 9 seconds (stand, walk, turn, stop). It checks for non-finite values,
  that the root moved, and that reset works. It prints `C_API_TEST PASS` and exits with 0.

Reference results (spider v3, Windows, CPU, 2 threads): test vector difference 1.4e-6, 0 token mismatches, one plan about 3 ms.

## Good to know

- CPU only (ONNX Runtime CPU provider).
- Tested on: Windows x86_64, MSVC 2022, Godot 4.7.2, ONNX Runtime 1.24.4 (nuget.org package). Linux: not tested.
- We built the Godot extension with a prebuilt godot-cpp library (`GODOT_CPP_LIB`). Building godot-cpp from source (`add_subdirectory`) is not tested.
- The runtime has no collision, terrain or IK. Foot contact values come from the model.
