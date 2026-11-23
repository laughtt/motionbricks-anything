# creature/: the shared Python library

This folder trains, tests and exports a motion model for one skeleton. The code never names a joint. Everything comes from a config file.

## What you give it

- A creature config JSON file. Examples: `../bodyplans/quadruped/creature.json` and `../bodyplans/spider/creature.json`.
- A dataset file (`dataset.npz`). The body plan folder builds it.

| Config key | Meaning |
|---|---|
| `name` | name of the creature |
| `data`, `out` | dataset file and output folder. A relative path starts at the folder of the config file. The env vars `CREATURE_DATA` and `CREATURE_OUT` replace them |
| `skeleton` | a JSON object, or the path of a skeleton spec: `joints`, `parents`, `contacts` (or `feet`), `root`, `front`, optional `mirror`, optional `orient` |
| `fps`, `heads`, `levels` | 30, 5, [8, 8] (5 heads x 64 codes) |
| `val_split` | name of the held-out split (default `val`) |
| `eval` | command speeds for the eval scripts: `walk`, `trot`, `slow`, `back`, `turn`, ... |

The `orient` block names four joints that are rigid with the body: `left`, `right`, `front`, `back`. The C++ runtime needs it for every skeleton that is not the 27-joint dog.
The env var `CREATURE_DEV` sets the device (`cpu` or `cuda`).

## What you get

- `<out>/<tag>.pt` and `<out>/<tag>.json` from each train script.
- An eval report (JSON).
- `pose_step.onnx`, `tok_dec.onnx`, `root_net.onnx` and `dog_brain.json` from `export_onnx.py`. The name `dog_brain.json` is historic. The runtime uses it for every creature.

## Files

| File | What it does |
|---|---|
| `core.py` | config, data, FSQ tokenizer, pose net, root net, MaskGIT decode |
| `features.py` | turns world joint positions into features (6 x joints + contacts) and root values |
| `losses.py` | foot-skate losses |
| `collision.py` | capsule collision metric, penalty and repair |
| `rollout.py` | the runtime written in Python (the reference for the model contract) and the rollout metrics |
| `train_tok.py`, `train_pose.py`, `train_root.py` | the three train scripts |
| `eval_tok.py`, `eval_rollout.py` | tokenizer round trip; 60-second keyboard scripts |
| `export_onnx.py` | ONNX export, check against PyTorch, JSON spec with test vectors |

## How to run it

1. Set `CFG` to the path of the creature config.
2. `python -I creatures/creature/train_tok.py --cfg $CFG --tag tok --steps 6000`
3. `python -I creatures/creature/train_pose.py --cfg $CFG --tok tok --tag pose --steps 6000 --rollout 0 --aux 1 --aux_mode st`
4. `python -I creatures/creature/train_pose.py --cfg $CFG --tok tok --tag pose_base --steps 4000 --rollout 0 --aux 0 --balance 0`
5. `python -I creatures/creature/train_root.py --cfg $CFG --tok tok --pose pose_base --tag root --steps 4000`
6. `python -I creatures/creature/eval_rollout.py --cfg $CFG --model m=tok,pose,root --out eval.json`
7. `python -I creatures/creature/export_onnx.py --cfg $CFG --tok tok --pose pose --root root --out <model folder>`

Every script shows its arguments with `--help`. To test all the steps fast on random data, run `python -I creatures/tools/smoke_test.py`.

## Good to know

- You need `torch`, `numpy`, `onnx` and `onnxruntime`.
- A dataset needs a `train` split and a held-out split with windows of 128 frames or more.
- The released spider weights come from an earlier copy of these scripts made for the spider. This library has the same model parts and the same file format.
