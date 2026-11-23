# Model card: spider v3 (MotionBricks Anything)

| Item | Value |
|---|---|
| Task | keyboard command (forward speed, side speed, yaw rate) in, 38-joint spider motion out, 30 frames per second |
| Files | `pose_step.onnx` 12.2 MB, `tok_dec.onnx` 8.6 MB, `root_net.onnx` 2.8 MB, `dog_brain.json` (spec + test vectors) |
| Parts | FSQ tokenizer (5 heads x 64 codes, 4 frames per token), pose net (transformer 4 layers x 256, 3.02 M parameters), root net (MLP, 0.71 M parameters) |
| Format | ONNX opset 17, FP32, CPU. The runtime contract is [`MODEL_CONTRACT.md`](MODEL_CONTRACT.md) |
| Speed | about 2.4 ms per 32-frame plan on a desktop CPU (C++ ONNX Runtime). One plan every 12 frames |

## Training data

- Real clips: 23 s from two packs on Fab (CC BY per the Fab listing) (see the attribution below). The clips are idle, walk, attack and damage.
- Generated clips: 26 min from a gait generator. It uses the gait of the real walk (period 0.40 s, duty 0.58, 8 leg phases).
- Each leg has its own lane in the generator. A capsule repair step moves the legs apart in every clip.
- The real clips are retargeted to one spider skeleton (`creatures/bodyplans/spider/spider_spec.json`).
- No pack file is in this repository. The model has only the mean pose, the rest pose and the weights.

## Results (60 s keyboard scripts)

| Metric | Value |
|---|---|
| Stance feet planted | 81.7 % |
| Foot slide while planted | 0.21 m/s (leg span 3.5 m) |
| Frames with leg overlap (capsule metric) | 0.10 % |
| Error on unseen motion (tokenizer / pose net) | 1.40 cm / 2.75 cm |

## How it was trained

The weights come from the spider prototype of the training scripts (tags `tok_v3a`, `pose_v3a`, `rootc_v3`, seed 0).
The shared library `creatures/creature/` has the same model parts and the same data format. To train the spider again, follow `creatures/bodyplans/spider/README.md`.

## Limits

- The spider is large (body height 0.5 m, leg span 3.5 m). Scale the output if you need a small spider.
- There are only 23 s of real motion. The attack and damage clips are in the data, but no command starts them.
- Backward and side motion are weak. The feet slide a little. The model does not lock the feet.

## Licence and attribution

The weights come from data that is CC BY per the Fab listing. You can use them, also in a commercial product, if you give the attribution below.

- "Spider animated character" by Bestgameassets, CC BY (per the Fab listing), https://www.fab.com/listings/62456bda-b5f6-4372-8673-6471359c5e50
- "Spiders - characters with animations" by Mixall, CC BY (per the Fab listing), https://www.fab.com/listings/41eeeaca-af9e-43b2-8a71-0874356cebb2

The model design comes from NVIDIA MotionBricks (code: Apache-2.0). No NVIDIA weights are in this model.
