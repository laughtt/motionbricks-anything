# distill/: the 100 MB MotionBricks model

These scripts shrink the NVIDIA MotionBricks SOMA model (189 M parameters, pose transformer 136.5 M, 758 MB FP32 ONNX) to 99.9 MB.
Every script shows its options with `python -I creatures/distill/<script>.py --help`.
The NVIDIA model stays under the NVIDIA Open Model Agreement. The code here is Apache-2.0.

**What is not here:** any training data, and the exact request sets behind the published numbers (see "What you must make yourself"). This folder alone does not reproduce the published numbers. The trained weights are in [`models/motionbricks_100mb/`](../models/motionbricks_100mb/MODEL_CARD.md).

## The four steps

| Step | What | Code |
|---|---|---|
| 1 | Pick which layers to keep (by angular distance between hidden states) | `layer_pick.py` + `distill.py select` |
| 2 | Distill: train the small model to copy the big one (KL + hidden L2, MaskGIT masking, data labelled by the teacher) | `distill.py` (gen, genkp, label, train, eval, export) |
| 3 | Quantize: GPTQ INT4, INT8, FP16 decoder | `gptq.py`, `export_compact.py`, `quantize.py` |
| 4 | Gate: skate and jitter must stay within +5 % | `gate.py` |

## What you need

- The upstream package in this repo and NVIDIA's exporter (`tools/onnx`). Download the NVIDIA SOMA checkpoint (see `tools/onnx/README.md`) and set `MB_CKPT` (or pass `--ckpt`) to its folder, for example `checkpoints/MotionBricks/soma`.
- The FP32 teacher ONNX `mmm_v6_soma_fp32.onnx` (default `motionbricks/assets/onnx/soma/`, or `MB_FP32_ONNX` / `--fp32-onnx`).
- The sample BVH clips (default `assets/motions/soma`, or `MB_MOTIONS` / `--motions`).
- Python 3.12 with torch, numpy, scipy, onnx and onnxruntime (1.20 or newer, for INT4). Step 2 and the gate on PyTorch models need a CUDA GPU. Our numbers come from one RTX 4090 (24 GB).
- A work folder: `--work DIR` or `DISTILL_WORK` (default `./distill_work`). All data, runs and results go there. Use a big disk (tens of GB).

Run everything with `python -I` from the repo root. `--help` works without a GPU, a checkpoint or data.

## Run it

```sh
W=distill_work
# 0. golden set: 244 requests built from the sample BVH clips + the FP32 teacher outputs (CPU, about 3 min)
python -I creatures/distill/build_golden.py --work $W

# 1+2. data labelled by the teacher (GPU)
python -I creatures/distill/distill.py gen   --work $W --rounds 10 --per-round 10000 --seed 21     # teacher self-rollouts
python -I creatures/distill/distill.py genkp --work $W --rounds 4 --per-round 20000 --seed 31       # key-pose targets
# optional: python -I creatures/distill/distill.py label --work $W --requests my_requests/ --out $W/runs/mine.npz

# 1. layer pick: writes $W/runs/angular.json
python -I creatures/distill/distill.py select --work $W --pool $W/runs/teacher_pool_s21.npz
python -I creatures/distill/layer_pick.py --table $W/runs/angular.json --keep 10     # prints the list of layers to keep

# 2. distill (the rounds on the student's own rollouts happen inside train)
python -I creatures/distill/distill.py train --work $W --name S10 --keep auto10 --steps 12000 \
    --pool $W/runs/teacher_pool_s21.npz,$W/runs/keypose_pool_s31.npz
python -I creatures/distill/distill.py eval   --work $W --name base,S10
python -I creatures/distill/distill.py export --work $W --name S10        # FP32 ONNX via NVIDIA's exporter

# 3. quantize: GPTQ INT4 (transformer layers), INT8 (input/output + conv head), FP16 decoder
python -I creatures/distill/quantize.py --work $W --name S10

# 4. gate: 7 scenarios x 8 variants, PASS if skate and jitter are within +5 % of the teacher
python -I creatures/distill/gate.py --work $W --n 8 --ref base --tol 0.05 --out gate.json base $W/runs/S10/S10_compact_fp16dec.onnx
```

## Files

| File | What it does |
|---|---|
| `layer_pick.py` | angular distance maths and greedy layer pick (numpy / torch, CPU) |
| `distill.py` | data generation, layer table, training, eval, FP32 export |
| `dataset.py`, `features.py` | request sampling and the teacher feature path (library) |
| `mbtorch.py` | loads the NVIDIA PyTorch model (uses `motionbricks_onnx`, installed or from `tools/onnx`) |
| `gptq.py` | Hessian calibration and GPTQ / clip / RTN INT4 levels |
| `export_compact.py` | writes INT4 / INT8 weights + DequantizeLinear, FP16 decoder, opset 21 |
| `quantize.py` | runs calibration, quantization, export and a score on the compact file |
| `gate.py`, `rollout.py`, `quality.py`, `footlock.py` | the gate, the 7 rollout scenarios, scoring, foot lock |
| `build_golden.py`, `mb_common.py`, `onnx_utils.py` | golden set, numpy port of the engine inputs and outputs, ONNX graph helpers |
| `smoke_test.py` | CPU test of layer pick, GPTQ, INT4 packing and foot lock |

## What you must make yourself

- Teacher-labelled data: `gen` and `genkp` make it from the teacher. You need no outside data.
  `genkp` uses windows of the sample BVH clips as key poses, or your own key poses (`--keypose-npz`, arrays `<name>__pos` and `<name>__rot`).
- Your own captured requests are optional. Put them in npz files with the 7 request arrays
  (`global_root_values`, `has_global_root_values`, `local_root_values`, `has_local_root_values`, `local_poses`, `has_local_poses`, `allowed_pred_num_tokens`),
  then run `distill.py label --requests FOLDER --out FILE.npz` and pass the file to `--pool`.
- The first pool file also gives the held-out set (its last `--holdout` requests, default 1500).

## Expected numbers (our own run, not re-measured here)

99.9 MB file, 11.8 ms for batch 8 on DirectML, 78.8 % of tokens equal to FP32, 4.10 cm mean joint difference.
The gate passes at +4.8 % (8 layers failed at +7.8 %). Training took 12k steps (28 min) plus 6k fine-tune steps (11 min) on one RTX 4090.
The 6k fine-tune is `train --init S10 --steps 6000 --pool ...` with the key-pose pool.

## Good to know

- Only the pose transformer is distilled. The root model, the tokenizer and the decoder stay as NVIDIA made them.
- The INT4 file needs ONNX Runtime with opset 21 and graph optimisation BASIC.
- The 7 gate scenarios use sample clips named in `rollout.py`. They must be in your `--motions` folder.
