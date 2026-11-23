# Third-party data, libraries and tools

Nothing in this list is in the repository, except where the column "In the repo" says so.

## Data

| Item | Licence | In the repo | Notes |
|---|---|---|---|
| MANN dog motion capture (Zhang, Starke, Komura, Saito, SIGGRAPH 2018), https://github.com/sebastianstarke/AI4Animation | CC BY-NC 4.0 | no (link only) | `bodyplans/quadruped/download_mann.py` downloads it after you accept the licence. The dog weights in `models/dog_v2/` are trained on it: non-commercial use only |
| "Spider animated character" by Bestgameassets, https://www.fab.com/listings/62456bda-b5f6-4372-8673-6471359c5e50 | CC BY (confirmed on the Fab listing) | no | Attribution: `bodyplans/spider/ATTRIBUTION.md`. Derived: spider spec, spider weights |
| "Spiders - characters with animations" by Mixall, https://www.fab.com/listings/41eeeaca-af9e-43b2-8a71-0874356cebb2 | CC BY (confirmed on the Fab listing) | no | Attribution: `bodyplans/spider/ATTRIBUTION.md`. Derived: spider weights |

## Libraries

| Item | Licence | In the repo | Notes |
|---|---|---|---|
| NVIDIA MotionBricks, https://github.com/NVlabs/MotionBricks | Apache-2.0 (code); NVIDIA Open Model Agreement (weights) | code: yes (upstream, unchanged) | See `NOTICE`. `distill/` needs the NVIDIA checkpoint, which you download from NVIDIA |
| ONNX Runtime, https://github.com/microsoft/onnxruntime | MIT | no | `runtime/` links it. `runtime/tools/fetch_deps.py` downloads the official release |
| godot-cpp, https://github.com/godotengine/godot-cpp | MIT | no | Only for the Godot extension |
| nlohmann/json, https://github.com/nlohmann/json | MIT | no | The runtime reads the JSON spec with it |
| Godot Engine, https://godotengine.org | MIT | no | The sandbox is a Godot 4 project |
| PyTorch, NumPy, ONNX | BSD-3-Clause / BSD-3-Clause / Apache-2.0 | no | Training and export |
| Blender, https://www.blender.org | GPL (the tool; our scripts call it from the command line) | no | Optional: clip extraction, SOMA generator, rig tools |

## Optional external tools (the section "From a SOMA to 3D assets" in the README)

You do not need these tools to train or run a model. Check the output terms of each tool before you publish a generated asset.

| Tool | Use | Notes |
|---|---|---|
| Nano Banana (Google image model) | image-to-image on the SOMA render | hosted; check its output terms |
| Tripo | image-to-3D | hosted; check its output terms |
| MiniMax H3 (image model, in ComfyUI) | image-to-image, local | check the model licence and its output terms |
| Pixal3D | image-to-3D, local | check the model licence and its output terms |
| SuperAnimal-Quadruped (DeepLabCut model zoo) | 2D keypoints from video (no shipped model uses it) | non-commercial weights; link only: https://github.com/DeepLabCut/DeepLabCut |
