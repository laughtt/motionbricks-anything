# Model card: dog v2 (MotionBricks Anything)

| Item | Value |
|---|---|
| Task | keyboard command (forward speed, side speed, yaw rate) in, 27-joint dog motion out, 30 frames per second |
| Files | `pose_step.onnx` 12.1 MB, `tok_dec.onnx` 8.4 MB, `root_net.onnx` 3.7 MB, `dog_brain.json` (spec + test vectors) |
| Parts | FSQ tokenizer (5 heads x 64 codes, 4 frames per token), pose net (transformer 4 layers x 256), root net |
| Format | ONNX, FP32, CPU. The runtime contract is [`MODEL_CONTRACT.md`](MODEL_CONTRACT.md) |
| Speed | about 3.2 ms per 32-frame plan on a desktop CPU (C++ ONNX Runtime). One plan every 12 frames |

## Results (60 s keyboard scripts)

| Metric | Value |
|---|---|
| Paws planted, walk / trot | 54.5 % / 19.1 % |
| Paw slide while planted, walk / trot | 5.0 / 15.3 cm/s |
| Error on unseen motion, one plan / 10 chained plans | 2.80 / 4.05 cm |

## Limits

- One dog only: other breeds and sizes need retargeting or more data.
- The trot data is thin, so the trot still slides a little. The model does not lock the feet.
- No jumps, attacks or hit reactions on command.

## Licence

Non-commercial use only (CC BY-NC 4.0, the same as the training data). Credit: trained on the MANN dog motion capture, Zhang, Starke, Komura, Saito, SIGGRAPH 2018.
