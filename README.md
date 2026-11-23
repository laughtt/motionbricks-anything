# MotionBricks Anything

**Unofficial extension of NVIDIA MotionBricks. Not affiliated with NVIDIA.**

<p align="center">
  <a href="https://ko-fi.com/josemauriciocarpioherrera"><img src="https://ko-fi.com/img/githubbutton_sm.svg" alt="Support me on Ko-fi" height="54"></a>
</p>
<p align="center">
  <b>This work is free. If it helps you, please support it on Ko-fi.</b><br>
  <a href="https://x.com/carpio_her8717">X</a> ·
  <a href="https://www.linkedin.com/in/mauriciocarpioherrera/">LinkedIn</a>
</p>

**I extended NVIDIA MotionBricks to animate animals, and compressed it from 758 MB to 100 MB.** Train a small motion model for any animal, then play it live in Godot with the keyboard.

<table>
  <tr><th>Spider (shipped model)</th><th>Husky (dog model)</th></tr>
  <tr>
    <td><img src="creatures/docs/media/sandbox_spider.gif" alt="Spider driven by the shipped model in the Godot sandbox" width="100%"></td>
    <td><img src="creatures/docs/media/sandbox_husky.gif" alt="Husky driven by the dog model in the Godot sandbox" width="100%"></td>
  </tr>
</table>

The colored ghosts are the frames the model plans next (sandbox plan-preview overlay; the overlay itself ships in the next release).

> **Status (2026-11-23):** version 1. Dog and spider weights, the 100 MB MotionBricks model, training code, the C++ runtime and the Godot sandbox are all in this repo.

## Contents

- [What this is](#what-this-is)
- [Numbers at a glance](#numbers-at-a-glance)
- [How it works](#how-it-works)
- [Results](#results)
- [What was hard, and what fixed it](#what-was-hard-and-what-fixed-it)
- [Try it](#try-it)
- [Train your own animal](#train-your-own-animal)
- [From a SOMA to 3D assets](#from-a-soma-to-3d-assets)
- [The 100 MB MotionBricks model](#the-100-mb-motionbricks-model)
- [Limits](#limits)
- [FAQ](#faq)
- [Licences](#licences)
- [Folder layout](#folder-layout)

## What this is

NVIDIA's MotionBricks makes real-time motion for a human character. You tell it where to go and how fast, and it makes the animation frame by frame. It does not play back stored clips.

I wanted the same thing for the animals in my game, so I extended it:

- You train **one small motion model per body type**: one for four-legged animals and one for spiders.
- In the game you steer with the keyboard. The model makes walk, trot, run and turn motion live. There are **no animation clips** at runtime.
- **One skeleton can drive many characters.** The same dog model moves a husky, an orc war-wolf and a skeleton wolf.
- It runs **on the CPU** inside Godot 4 through a C++ runtime. There is **no Python** in the game.

## Numbers at a glance

| | Dog (husky) | Spider |
|---|---|---|
| Skeleton | 27 joints, 4 paws | 38 joints, 8 legs |
| Training data | 41 min motion capture | 23 s real clips + 26 min generated |
| Model size | about 3 M parameters, 24 MB ONNX | about 3 M parameters, 24 MB ONNX |
| Training time | about 10 min on one GPU | minutes on one GPU |
| Speed | 2.4 ms per plan on the CPU | 2.4 ms per plan on the CPU |
| Error on unseen motion | 2.80 cm | 2.75 cm |
| Weights | [`creatures/models/dog_v2`](creatures/models/dog_v2/MODEL_CARD.md) (non-commercial) | [`creatures/models/spider_v3`](creatures/models/spider_v3/MODEL_CARD.md) (CC BY) |

**Bonus:** NVIDIA's MotionBricks (SOMA) model compressed from **758 MB to 99.9 MB**, 24 % faster. Weights in [`creatures/models/motionbricks_100mb`](creatures/models/motionbricks_100mb/MODEL_CARD.md).

## How it works

1. **Skeleton spec.** A small JSON file lists the joints, their parents, the feet and the front joint. This is the only thing that is different for each animal.
2. **A SOMA body.** The SOMA is a plain grey body with the proportions of your training data. Every new character is built on it.
3. **Data.** Motion capture or generated clips become one file, `dataset.npz`, with joint positions at 30 frames per second.
4. **Three small networks.**
   - The *tokenizer* turns a short piece of pose (4 frames) into a few codes (5 codes, 64 choices each). Think of each code as a word: the motion becomes a sentence.
   - The *pose net* is a small transformer (4 layers, 256 wide). It picks the codes for the next 32 frames from your input (direction, speed) and the last poses.
   - The *root net* is a small MLP. It turns your command (forward, side, turn) into the motion of the root: speed, turn and height. This keeps the feet planted.
5. **Export to ONNX.** Three ONNX files plus one JSON file with sizes and test values. The JSON is called `dog_brain.json` for historic reasons. It is used for every animal.
6. **C++ runtime.** It runs the ONNX files on the CPU with ONNX Runtime. It makes a new plan every 12 frames and blends the old plan into the new one, so there is no pop.
7. **Godot.** A GDExtension wraps the runtime. A short script reads the keys, moves the character and sets the bones. The C API can be used from any other engine too.

## Results

### Dog (husky)

- **Data:** one dog, 41 minutes of motion capture (MANN).
- **Quality** (60-second keyboard scripts): paws planted 54.5 % of the time at the walk (the real dog data: 49 %) and 19.1 % at the trot. A planted paw slides 5.0 cm/s at the walk and 15.3 cm/s at the trot.
- **One model, many meshes:** any mesh rigged to the same 27-joint skeleton runs on the same model, with no new training. The husky and wolf meshes in the GIFs are not in this repo.

<table>
  <tr><th>Husky</th><th>Skeleton wolf mount</th></tr>
  <tr>
    <td><img src="creatures/docs/media/sandbox_husky.gif" alt="Husky driven by the dog model in Godot" width="100%"></td>
    <td><img src="creatures/docs/media/sandbox_skeleton_wolf.gif" alt="Skeleton wolf mount driven by the dog model in Godot" width="100%"></td>
  </tr>
</table>

<p align="center"><img src="creatures/docs/media/dog_soma_v3.gif" alt="Dog SOMA body on real dog motion capture (reference)" width="60%"><br><i>Dog SOMA body on real dog motion capture (reference)</i></p>

### Spider

- **Data:** only 23 seconds of real clips (two Fab packs by Bestgameassets and Mixall, CC BY per the Fab listing). A gait generator built 26 minutes of extra walking from the real walk. [Attribution](creatures/bodyplans/spider/ATTRIBUTION.md).
- **Quality** (60-second keyboard scripts): 81.7 % of stance feet are planted. They slide 0.21 m/s (the legs are 3.5 m wide). Legs overlap on only 0.10 % of frames.
- **Train it again:** get the two packs from Fab and follow [`creatures/bodyplans/spider/README.md`](creatures/bodyplans/spider/README.md). The war-spider mesh in the GIF is not in this repo. The grey spider body (the SOMA) is.

### Mounts

<p align="center"><img src="creatures/docs/media/sandbox_spider_mount.gif" alt="War-spider mount driven by the spider model in Godot" width="70%"></p>

The husky, the skeleton wolf and the war spider are generated meshes. Each one is rigged to the skeleton of its model, so the model drives it.

## What was hard, and what fixed it

- **Sliding feet.** The first dog looked like it was ice skating: the step rhythm was not tied to the speed. A separate **root net** fixed most of it. It learns how the real animal speeds up, turns and slows down. Planted paws at the walk went from **18 % to 54.5 %** (the real dog: 49 %).
- **A foot-skate loss helps too.** The tokenizer loss punishes the decoded paws for moving while they are planted.
- **Balance the speeds.** Most motion capture is slow walking. Sampling the training clips by speed keeps the trot and the run good too.
- **Clean seams between plans.** Each new plan keeps 2 tokens of the old plan plus 2 tokens of context, so the switch from plan to plan does not jump.
- **Spider legs going through each other.** A "do not overlap" loss did not help much. **Clean training data did:** each leg got its own lane in the gait generator, and a capsule repair fixed every clip. Leg overlap went from **88.8 % to 0.10 %** of frames.
- **Big model vs small model.** Adapting the big human MotionBricks model (frozen transformer + LoRA) to a dog only helped with **less than 5 minutes** of data (about 4-5 % better). With 10 minutes or more, a small model trained from scratch was just as good and much smaller.
- **Rigging generated meshes.** Image-to-3D meshes stretch badly with default weights. What worked: landmarks inside the mesh, bone-heat weights on a welded mesh, many smoothing passes, rigid gear and leg segments, and Preserve Volume.

## Try it

### Prerequisites

Windows 10/11 x64 is the only tested system. Linux and macOS are not tested.

- [Godot](https://godotengine.org/) 4.7.2
- Visual Studio 2022 Build Tools (C++ workload)
- CMake 3.20 or newer
- Ninja
- Python 3.11 or 3.12
- [Git LFS](https://git-lfs.com/)

### Get the files (Git LFS)

Install Git LFS first. A normal `git clone` then downloads only the creature models and media (`creatures/**`, set in [`.lfsconfig`](.lfsconfig)), about 175 MB.
To also get the large NVIDIA human models (about 2.4 GB in total), run:

```sh
git lfs pull --include="*"
```

**Prebuilt binaries: coming soon (GitHub Release).** Until then, build the Godot extension yourself, as below.

### Run the spider (no training)

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

The last command opens the spider. **Keys:** W A S D to move, Shift to go faster (trot), Space to stop. `smoke_test.py` is optional (about 1 minute). `ctest` compares the C++ plan with the PyTorch test vector. `run_test.bat` is the headless Godot test.

**Linux (not tested):** the same steps with the `.sh` scripts (`run_test.sh`, `run_sandbox.sh`). Set `GODOT` to your Godot 4.7 executable.

More details: [`creatures/runtime/README.md`](creatures/runtime/README.md) and [`creatures/sandboxes/godot_creature/README.md`](creatures/sandboxes/godot_creature/README.md).

## Train your own animal

### Train the dog

```sh
python -I creatures/bodyplans/quadruped/download_mann.py --out creatures/work/dog/mann --accept-license
python -I creatures/bodyplans/quadruped/build_dataset.py --bvh_dir creatures/work/dog/mann --out creatures/work/dog/data/dataset.npz
CFG=creatures/bodyplans/quadruped/creature.json     # then run the commands below
```

### Any animal

```sh
CFG=creatures/bodyplans/<animal>/creature.json        # skeleton spec + dataset path + output folder (default: creatures/work/<animal>/)
python -I creatures/creature/train_tok.py  --cfg $CFG --tag tok --steps 6000
python -I creatures/creature/train_pose.py --cfg $CFG --tok tok --tag pose --steps 6000 --rollout 0 --aux 1 --aux_mode st
python -I creatures/creature/train_pose.py --cfg $CFG --tok tok --tag pose_base --steps 4000 --rollout 0 --aux 0 --balance 0
python -I creatures/creature/train_root.py --cfg $CFG --tok tok --pose pose_base --tag root --steps 4000
python -I creatures/creature/eval_rollout.py --cfg $CFG --model m=tok,pose,root --out creatures/work/<animal>/eval.json
python -I creatures/creature/export_onnx.py --cfg $CFG --tok tok --pose pose --root root --out creatures/work/<animal>/onnx
```

To see your model in Godot, run `run_sandbox.sh --model_dir=<folder>`. The body in the sandbox must have the same skeleton as the model.

**Tips for a new animal:**
- Ten minutes of clean motion or more is enough for a small model. With less than 5 minutes, think about adapting the big model with LoRA.
- For bodies with many legs, clean the data first (no crossing legs). Check it with `creatures/creature/collision.py`.
- Keep the same skeleton for all meshes of one animal. Then a new mesh needs no new training.

## From a SOMA to 3D assets

Each motion model has a **SOMA**: a plain body with the proportions of its training data. You build new characters on top of it, so the mesh fits the skeleton and needs no retargeting.

1. **SOMA.** Build it from the skeleton spec and render it (grey, white background, three-quarter view).
2. **Image.** Run image-to-image on that render. Keep the pose, the proportions and the view. Keep gear tight, with nothing hanging.
3. **3D.** Cut out the background, then run image-to-3D (about 40k triangles).

   Steps 2 and 3 work with hosted tools (**Nano Banana + Tripo**) or with local models (**MiniMax H3 in ComfyUI + Pixal3D**, 1-2 minutes each on an RTX 4090).
4. **Rig.** Place landmarks inside the mesh, bake to the SOMA rest pose, add bone-heat weights and smooth them, and keep gear and leg segments rigid.

<img src="creatures/docs/media/soma_to_3d/4_spider_template_and_image.png" width="640">
<img src="creatures/docs/media/soma_to_3d/5_spider_3d_views.png" width="640">

> Check the output terms of the image and 3D tools before you publish generated assets.

The optional Blender helpers for these steps are in [`creatures/tools/soma_to_3d/`](creatures/tools/README.md).

## The 100 MB MotionBricks model

NVIDIA's SOMA model has 189 M parameters and ships as a 758 MB ONNX file. We made it **99.9 MB** and 24 % faster, in four steps:

1. **Drop layers.** Keep 10 of the 16 transformer layers.
2. **Distill.** Train the smaller model to copy the original (the teacher). All the data comes from the teacher itself.
3. **Quantize.** Store the weights in 4-bit and 8-bit numbers, and the decoder in 16-bit.
4. **Check.** Foot skate and jitter must stay within 5 % of the original. We got +4.8 %. With 8 layers it failed (+7.8 %).

Result: 11.8 ms for a batch of 8 on DirectML. 78.8 % of tokens match the original. The mean joint difference is 4.10 cm.

The scripts are in [`creatures/distill/`](creatures/distill/README.md). The weights are in [`creatures/models/motionbricks_100mb/`](creatures/models/motionbricks_100mb/MODEL_CARD.md).

**Honesty note:** these numbers come from our own run. That run also used extra request data that is not public. The public scripts make their data from the teacher only, so your numbers can differ.

## Limits

- **Dog weights are non-commercial only**, because the dog motion data (MANN) is CC BY-NC 4.0. Train on your own data for commercial use.
- **Two body types for now:** four legs and spiders. A new body type needs its own motion data and a short training run.
- **Rigging generated meshes still takes manual work** (landmarks, weight checks).
- **Only Windows is tested.** Linux scripts exist but are not tested. macOS is not tested.
- **No prebuilt binaries yet.** You build the Godot extension yourself.
- **The plan-preview ghosts** in the GIFs are not in the shipped runtime yet.

## FAQ

**Is this just blending animation clips?**
No. There are no clips at runtime. The model makes each new pose from your input and the last poses.

**Can I use it for a horse, a cat or a dragon?**
A four-legged animal can use the dog body plan if you map its skeleton. A new shape (wings, six legs, a long tail that moves a lot) needs its own data and a short training run.

**Does it need a GPU in the game?**
No. It runs on the CPU at about 2.4 ms per plan.

**Does it work outside Godot?**
The core is C++ with a plain C API, and the models are ONNX. Godot is the shipped integration; other engines can call the same C API.

**What are the ghosts in the GIFs?**
The poses the model plans for the next frames.

**Can I use it commercially?**
The code is Apache-2.0 and the spider weights are CC BY (give the attribution). The dog weights are non-commercial. The 100 MB model follows the NVIDIA Open Model Agreement.

## Licences

- **Code:** Apache-2.0. See [`creatures/NOTICE`](creatures/NOTICE). Third-party data, libraries and tools are listed in [`creatures/THIRD_PARTY.md`](creatures/THIRD_PARTY.md).
- **Dog weights:** non-commercial use only. See the [model card](creatures/models/dog_v2/MODEL_CARD.md).
- **Spider weights:** from data that is CC BY per the Fab listing. Commercial use is allowed with the attribution in the [model card](creatures/models/spider_v3/MODEL_CARD.md).
- **100 MB model:** a modified NVIDIA model, under the NVIDIA Open Model Agreement. See the [model card](creatures/models/motionbricks_100mb/MODEL_CARD.md).
- **Generated meshes** (husky, skeleton wolf, war spider): not in this repo.

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

The rest of the repository is the original NVIDIA MotionBricks code.

## NVIDIA MotionBricks (upstream)

This repository started from [NVIDIA MotionBricks](https://github.com/NVlabs/motionbricks) (Apache-2.0). The original NVIDIA README is unchanged in [README_UPSTREAM.md](README_UPSTREAM.md).
The upstream code, models and licences (`LICENSE`, `licenses/`) are kept as they are. The additions are in [`creatures/`](creatures/README.md); see [`creatures/NOTICE`](creatures/NOTICE).

## Credits

Dog data: MANN dog motion capture (Zhang, Starke, Komura, Saito, SIGGRAPH 2018), CC BY-NC 4.0. Spider data: Fab packs by Bestgameassets and Mixall, CC BY. Based on NVIDIA MotionBricks.

## Support

<p align="center">
  <a href="https://ko-fi.com/josemauriciocarpioherrera"><img src="https://ko-fi.com/img/githubbutton_sm.svg" alt="Support me on Ko-fi" height="54"></a>
</p>
<p align="center">
  <b>This work is free. If it helps you, please support it on Ko-fi.</b><br>
  <a href="https://x.com/carpio_her8717">X</a> ·
  <a href="https://www.linkedin.com/in/mauriciocarpioherrera/">LinkedIn</a>
</p>
