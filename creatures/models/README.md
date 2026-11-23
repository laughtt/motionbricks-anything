# models/: released weights

This folder holds the weights we are allowed to release. Version 1 has a spider model, a dog model and the 100 MB MotionBricks human model.

| Folder | What is inside | Licence |
|---|---|---|
| `spider_v3/` | `pose_step.onnx`, `tok_dec.onnx`, `root_net.onnx`, `dog_brain.json`, `MODEL_CARD.md`, `MODEL_CONTRACT.md` | CC BY data (confirmed on the Fab listing): use requires the attribution in the model card |
| `dog_v2/` | the same files for the dog | non-commercial only: see the model card |
| `motionbricks_100mb/` | `motionbricks_soma_100mb.onnx` (99.9 MB), `MODEL_CARD.md` | NVIDIA Open Model Agreement (modified NVIDIA model) |

## How to use it

1. Run `git lfs pull`. The ONNX files are in Git LFS.
2. Read the model card: [`spider_v3`](spider_v3/MODEL_CARD.md) or [`dog_v2`](dog_v2/MODEL_CARD.md).
3. Run the model with `creatures/runtime`, or read [`spider_v3/MODEL_CONTRACT.md`](spider_v3/MODEL_CONTRACT.md) and write your own runtime.

## Good to know

- Put your own models in `creatures/work/`. Git ignores every other folder in `models/`.
