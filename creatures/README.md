# MotionBricks Anything: a motion model for any animal

<p align="center">
  <a href="https://ko-fi.com/josemauriciocarpioherrera"><img src="https://ko-fi.com/img/githubbutton_sm.svg" alt="Support me on Ko-fi" height="54"></a>
</p>
<p align="center">
  <b>This work is free. If it helps you, please support it on Ko-fi.</b><br>
  <a href="https://x.com/carpio_her8717">X</a> ·
  <a href="https://www.linkedin.com/in/mauriciocarpioherrera/">LinkedIn</a>
</p>

<table>
  <tr><th>Spider (shipped model)</th><th>Husky (dog model, train it yourself)</th></tr>
  <tr>
    <td><img src="docs/media/sandbox_spider.gif" alt="Spider driven by the shipped model in the Godot sandbox" width="100%"></td>
    <td><img src="docs/media/sandbox_husky.gif" alt="Husky driven by the dog model in the Godot sandbox" width="100%"></td>
  </tr>
</table>

The colored ghosts are the frames the model plans next (sandbox plan-preview overlay; the overlay itself ships in the next release).

> **Status (2026-11-23):** version 1. The dog and spider weights, the code and the Godot sandbox are in this folder.

## What this is

**Unofficial extension of NVIDIA MotionBricks. Not affiliated with NVIDIA.** NVIDIA's MotionBricks makes motion for a human character. We changed it so you can train the same kind of model for **any animal**, then play it live in the Godot game engine.

Each animal gets its own small model. It trains in minutes on one GPU. It runs on the CPU inside Godot, and you drive it with the keyboard. There is no Python in the game.

## Results

### Dog (husky) model

- **Data:** one dog, 41 minutes of motion capture. Weights: [`models/dog_v2/`](models/dog_v2/MODEL_CARD.md) (non-commercial). You can also train your own in about 10 minutes on one GPU.
- **Skeleton:** 27 joints and 4 paw contacts.
- **Size and speed:** 24 MB of ONNX files. One plan takes 2.4 ms on the CPU.
- **Quality** (60-second keyboard scripts): paws planted 54.5 % of the time at the walk and 19.1 % at the trot. A planted paw slides 5.0 cm/s at the walk and 15.3 cm/s at the trot. The error on unseen motion is 2.80 cm per plan.
- **One model, many meshes:** any mesh rigged to the same 27-joint skeleton runs on the same model. The husky and wolf meshes in the GIFs are not in this repo.

<table>
  <tr><th>Husky</th><th>Skeleton wolf mount</th></tr>
  <tr>
    <td><img src="docs/media/sandbox_husky.gif" alt="Husky driven by the dog model in Godot" width="100%"></td>
    <td><img src="docs/media/sandbox_skeleton_wolf.gif" alt="Skeleton wolf mount driven by the dog model in Godot" width="100%"></td>
  </tr>
</table>

<p align="center"><img src="docs/media/dog_soma_v3.gif" alt="Dog SOMA body on real dog motion capture (reference)" width="60%"><br><i>Dog SOMA body on real dog motion capture (reference)</i></p>

### Spider model

- **Data:** only 23 seconds of real clips (two Fab packs by Bestgameassets and Mixall, CC BY per the Fab listing). We added 26 minutes of generated walking built from the real walk. [Attribution](bodyplans/spider/ATTRIBUTION.md).
- **Skeleton:** 38 joints and 8 legs.
- **Size and speed:** 24 MB of ONNX files. One plan takes 2.4 ms on the CPU.
- **Quality** (60-second keyboard scripts): 81.7 % of stance feet are planted. They slide 0.21 m/s (the legs are 3.5 m wide). The error on unseen motion is 2.75 cm.
- **Weights:** they are in [`models/spider_v3`](models/spider_v3/MODEL_CARD.md) (Git LFS). Give the attribution when you use them.
- **Train it again:** get the two packs from Fab and follow [`bodyplans/spider/README.md`](bodyplans/spider/README.md). The war-spider mesh in the GIF is not in this repo. The grey spider body (the SOMA) is.

### Mounts

<p align="center"><img src="docs/media/sandbox_spider_mount.gif" alt="War-spider mount driven by the spider model in Godot" width="70%"></p>

The husky, the skeleton wolf and the war spider are generated meshes. Each one is rigged to the skeleton of its model (or its output is retargeted), so the model drives it.

## How it works

1. **Skeleton spec.** A small JSON file lists the joints, their parents, the feet and the front joint. This is the only thing that is different for each animal.
2. **A SOMA body.** The SOMA is a plain grey body with the proportions of your training data. Every new character is built on it.
3. **Data.** Motion capture or generated clips become one file, `dataset.npz`, with joint positions at 30 frames per second.
4. **Three small networks.**
   - The *tokenizer* turns a short piece of pose (4 frames) into a few codes (5 codes, 64 choices each).
   - The *pose net* is a small transformer. It picks the codes for the next 32 frames.
   - The *root net* is a small MLP. It turns your command (forward, side, turn) into the motion of the root: speed, turn and height.
5. **Export to ONNX.** Three ONNX files plus one JSON file with sizes and test values. The JSON is called `dog_brain.json` for historic reasons. It is used for every animal.
6. **C++ runtime.** It runs the ONNX files on the CPU with ONNX Runtime. It makes a new plan every 12 frames and blends the old plan into the new one.
7. **Godot.** A GDExtension wraps the runtime. A short script reads the keys, moves the character and sets the bones.

## What we learned

- **A root net stops foot sliding.** Without it, the step rhythm is not tied to the speed and the paws slide. The root net learns how the real animal speeds up, turns and slows down, so the feet stay put.
- **Add a foot-skate loss.** The loss punishes the decoded paws for moving while they are planted.
- **Many legs need collision-free data.** A body with many legs gets crossing legs unless the training data is clean. We gave each leg its own lane in the generator and ran a capsule repair on every clip. Legs now overlap on 0.10 % of frames.
- **A small model beats adapting the big one.** With 10 minutes or more of data, a small model trained from scratch (4 layers, 256 wide) is as good, and much smaller. Only with less than 5 minutes of data is it worth freezing the big MotionBricks transformer and adding LoRA (about 4-5 % better there).
- **Same skeleton, new mesh needs no training.** Rig the new mesh to the same skeleton.

## Try it

### Prerequisites

Windows 10/11 x64 is the only tested system. Linux and macOS are not tested.

- [Godot](https://godotengine.org/) 4.7.2
- Visual Studio 2022 Build Tools (C++ workload)
- CMake 3.20 or newer
- Ninja
- Python 3.11 or 3.12 (also with `pip install -r creatures/requirements.txt` for the smoke test and for training)
- [Git LFS](https://git-lfs.com/)

**Prebuilt binaries: coming soon (GitHub Release).** Until then you build the Godot extension yourself, as below.

### Run the spider (no training)

Get the files. A normal clone fetches only `creatures/**` from Git LFS (see [`.lfsconfig`](../.lfsconfig)), about 70 MB. To also get the NVIDIA human models, run `git lfs pull --include="*"` (about 2.4 GB).

**Windows** (x64 developer prompt, from the repository root):

```bat
pip install -r creatures\requirements.txt
python creatures\tools\smoke_test.py
python creatures\runtime\tools\fetch_deps.py creatures\deps
rem fetch_deps.py prints two lines: set ONNXRUNTIME_DIR=... and set GODOT_CPP_DIR=...  Run both.
cmake -S creatures\runtime -B creatures\runtime\build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCREATURE_BUILD_GODOT=ON
cmake --build creatures\runtime\build
ctest --test-dir creatures\runtime\build --output-on-failure
set GODOT=<path to your Godot 4.7.2 executable>
creatures\sandboxes\godot_creature\run_test.bat
creatures\sandboxes\godot_creature\run_sandbox.bat
```

The last command opens the spider. Keys: W A S D, Shift, Space. `smoke_test.py` is optional (about 1 minute). `ctest` compares the C++ plan with the PyTorch test vector. `run_test.bat` is the headless Godot test.

**Linux (not tested):** the same steps with the `.sh` scripts: `creatures/runtime/tools/fetch_deps.py`, `cmake` without `-G Ninja` is fine, `creatures/sandboxes/godot_creature/run_test.sh` and `run_sandbox.sh` (set `GODOT` to your Godot 4.7 executable).

`fetch_deps.py` pins godot-cpp to the commit `2a4c066e1e8c121046e9d35fe91cc8e726861c48` (the one used for the local build, a 4.8-beta3 sync of the master branch; there is no 4.7 release branch yet) and ONNX Runtime 1.24.4.
More details: [`runtime/README.md`](runtime/README.md) and [`sandboxes/godot_creature/README.md`](sandboxes/godot_creature/README.md).

### Train the dog

```sh
python -I creatures/bodyplans/quadruped/download_mann.py --out creatures/work/dog/mann --accept-license
python -I creatures/bodyplans/quadruped/build_dataset.py --bvh_dir creatures/work/dog/mann --out creatures/work/dog/data/dataset.npz
CFG=creatures/bodyplans/quadruped/creature.json     # then run the commands in "Train your own animal"
```

### Train your own animal

```sh
CFG=creatures/bodyplans/<animal>/creature.json        # skeleton spec + dataset path + output folder (default: creatures/work/<animal>/)
python -I creatures/creature/train_tok.py  --cfg $CFG --tag tok --steps 6000
python -I creatures/creature/train_pose.py --cfg $CFG --tok tok --tag pose --steps 6000 --rollout 0 --aux 1 --aux_mode st
python -I creatures/creature/train_pose.py --cfg $CFG --tok tok --tag pose_base --steps 4000 --rollout 0 --aux 0 --balance 0
python -I creatures/creature/train_root.py --cfg $CFG --tok tok --pose pose_base --tag root --steps 4000
python -I creatures/creature/eval_rollout.py --cfg $CFG --model m=tok,pose,root --out creatures/work/<animal>/eval.json
python -I creatures/creature/export_onnx.py --cfg $CFG --tok tok --pose pose --root root --out creatures/work/<animal>/onnx
```

Install the Python packages with `pip install -r creatures/requirements.txt`. To see your model in Godot, run `run_sandbox.sh --model_dir=<folder>`. The body in the sandbox must have the same skeleton as the model.

## From a SOMA to 3D assets

Each motion model has a **SOMA**: a plain body with the proportions of its training data. You build new characters on top of it, so the mesh fits the skeleton and needs no retargeting.

1. **SOMA.** Build it from the skeleton spec and render it (grey, white background, three-quarter view).
2. **Image.** Run image-to-image on that render. Keep the pose, the proportions and the view. Keep gear tight, with nothing hanging.
3. **3D.** Cut out the background, then run image-to-3D (about 40k triangles).

   Steps 2 and 3 work with hosted tools (**Nano Banana + Tripo**) or with local models (**MiniMax H3 in ComfyUI + Pixal3D**, 1-2 minutes each on an RTX 4090).
4. **Rig.** Place landmarks inside the mesh, bake to the SOMA rest pose, add bone-heat weights and smooth them, and keep gear and leg segments rigid.

<img src="docs/media/soma_to_3d/4_spider_template_and_image.png" width="640">
<img src="docs/media/soma_to_3d/5_spider_3d_views.png" width="640">

> Check the output terms of the image and 3D tools before you publish generated assets.

The optional Blender helpers for these steps are in [`tools/soma_to_3d/`](tools/README.md).

## The 100 MB MotionBricks model

NVIDIA's SOMA model has 189 M parameters and ships as a 758 MB ONNX file. We made it **99.9 MB** and 24 % faster, in four steps:

1. **Drop layers.** Keep 10 of the 16 transformer layers.
2. **Distill.** Train the smaller model to copy the original (the teacher). All the data comes from the teacher itself.
3. **Quantize.** Store the weights in 4-bit and 8-bit numbers, and the decoder in 16-bit.
4. **Check.** Foot skate and jitter must stay within 5 % of the original. We got +4.8 %. With 8 layers it failed (+7.8 %).

Result: 11.8 ms for a batch of 8 on DirectML. 78.8 % of tokens match the original. The mean joint difference is 4.10 cm.

The scripts are in [`distill/`](distill/README.md). The 99.9 MB weights are in [`models/motionbricks_100mb/`](models/motionbricks_100mb/MODEL_CARD.md).

**Honesty note:** these numbers come from our own run. That run also used extra request data that is not public. The public scripts make their data from the teacher only, so your numbers can differ.

## Licences

- **Code:** Apache-2.0. See [`NOTICE`](NOTICE). Third-party data, libraries and tools are listed in [`THIRD_PARTY.md`](THIRD_PARTY.md).
- **Dog weights:** in this repo, non-commercial use only. See the [model card](models/dog_v2/MODEL_CARD.md).
- **Spider weights:** in this repo. They come from data that is CC BY per the Fab listing. Commercial use is allowed with the attribution in the [model card](models/spider_v3/MODEL_CARD.md).
- **100 MB model:** in this repo. A modified NVIDIA model, under the NVIDIA Open Model Agreement. See the [model card](models/motionbricks_100mb/MODEL_CARD.md).
- **Generated meshes** (husky, skeleton wolf, war spider): not in this repo. Check the image model's output terms before you publish such meshes.

## Folder layout

```
creatures/
├─ creature/            shared Python library: data features, training, evaluation, ONNX export
├─ bodyplans/quadruped/ dog config, data download, dataset builder
├─ bodyplans/spider/    spider skeleton, clip extraction, dataset builder, gait generator, collision spec
├─ distill/             the 100 MB MotionBricks model: layer pick, distillation, quantization, check
├─ runtime/             C++ runtime, C API, Godot extension, tests
├─ sandboxes/godot_creature/   Godot 4.7 project and the grey spider SOMA body
├─ models/              dog, spider and 100 MB MotionBricks weights, model cards (Git LFS)
├─ tools/               smoke test, release audit, optional Blender helpers
├─ docs/media/          GIFs and images
└─ work/, deps/         your data, checkpoints and dependencies (ignored by git)
```

## Credits

Dog data: MANN dog motion capture (Zhang, Starke, Komura, Saito, SIGGRAPH 2018), CC BY-NC 4.0. Spider data: Fab packs by Bestgameassets and Mixall, CC BY (confirmed on the Fab listing). Based on NVIDIA MotionBricks.

## Support

<p align="center">
  <a href="https://ko-fi.com/josemauriciocarpioherrera"><img src="https://ko-fi.com/img/githubbutton_sm.svg" alt="Support me on Ko-fi" height="54"></a>
</p>
<p align="center">
  <b>This work is free. If it helps you, please support it on Ko-fi.</b><br>
  <a href="https://x.com/carpio_her8717">X</a> ·
  <a href="https://www.linkedin.com/in/mauriciocarpioherrera/">LinkedIn</a>
</p>
