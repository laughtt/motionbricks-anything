# Model card: MotionBricks 100 MB (SOMA human)

A smaller copy of NVIDIA's MotionBricks SOMA model. It is a drop-in replacement for `mmm_v6_soma_fp32.onnx`: same 7 inputs, same outputs.

| Item | Value |
|---|---|
| File | `motionbricks_soma_100mb.onnx`, 99.9 MB (original: 758 MB) |
| What changed | 10 of the 16 transformer layers kept, distilled from the original, weights in 4-bit / 8-bit, decoder in 16-bit |
| Inputs | `global_root_values`, `has_global_root_values`, `local_root_values`, `has_local_root_values`, `local_poses`, `has_local_poses`, `allowed_pred_num_tokens` (as upstream) |
| Outputs | `pred_global_motions`, `pred_num_tokens` (+ token outputs) |
| Format | ONNX opset 21 |
| Speed | 11.8 ms for a batch of 8 on DirectML (24 % faster than the original) |

## Quality (against the original)

| Metric | Value |
|---|---|
| Foot skate and jitter | +4.8 % (limit +5 %) |
| Tokens equal to the original | 78.8 % |
| Mean joint difference | 4.10 cm |

## How to use it

1. Run `git lfs pull --include="creatures/models/motionbricks_100mb/*"`.
2. Use it wherever the upstream code loads `mmm_v6_soma_fp32.onnx`.

## How it was made

With the scripts in [`creatures/distill/`](../../distill/README.md). Our run also used extra request data that is not public, so a run with the public scripts only can give slightly different numbers.

## Licence

Licensed by NVIDIA Corporation under the NVIDIA Open Model Agreement. This is a modified version of NVIDIA's MotionBricks SOMA model. Commercial use is allowed under that agreement, with this attribution.
