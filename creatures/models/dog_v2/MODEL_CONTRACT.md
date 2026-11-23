# Dog MotionBricks model v2: runtime contract

This file tells an engine (Godot, Unreal, Unity, Python, ...) how to run the model. It needs only an ONNX runtime and the JSON spec.
Research only: the model is trained on the MANN dog mocap (CC BY-NC 4.0).

Reference implementations (same behaviour): Python `creatures/creature/rollout.py` (class `Runtime`),
C++ `creatures/runtime/src/`.

## 1. Files
| File | Input | Output |
|---|---|---|
| `dog_brain.json` | - | all sizes, normalization, skeleton, start state, test vectors |
| `root_net.onnx` | `cmd` float [1,3], `root_hist` float [1,4,4], `ctx` float [1,4,166] | `root` float [1,28,4] |
| `pose_step.onnx` | `tokens` int64 [1,8,5], `ctx` float [1,4,166], `root` float [1,32,4] | `logits` float [1,8,5,64] |
| `tok_dec.onnx` | `tokens` int64 [1,N,5] (N is free; 8 or 10 are used) | `pose` float [1,4N,166] |

All float tensors are NORMALIZED. Read every size from the JSON (`tokens`, `frames`, `context_frames`, `heads`, `codes`, `pose_dim`, `joints_n`); the numbers here are the dog values.

## 2. Conventions
- Axes: Y up, forward +Z, left +X. Units: metres, seconds, radians. Model rate: 30 frames per second (`fps`).
- Heading frame: origin on the floor below the root joint (Hips), rotated about Y by the character yaw.
- Pose feature of one frame (166 values, `pose_dim`):
  - [0, 81): 27 joint positions (x, y, z) in the heading frame. Joint order = `joints`; tree = `parents`.
  - [81, 162): 27 joint velocities: WORLD velocity, written in the heading frame (m/s). The runtime does not need them (they are context for the nets).
  - [162, 166): contact head, 4 values for the paws `contact.joints` (LeftHand_end, RightHand_end, LeftFoot_end, RightFoot_end). After denormalization: > 0.5 = the paw is planted.
- Root values of one frame (4 values): [vx = side speed (+X, left), vz = forward speed, yaw rate (rad/s, positive turns +Z toward +X), root height (m)].
- Normalization: `normalized = (raw - mean) / std`. Pose: `pose_mean`, `pose_std` (166 values). Root: `root_mean`, `root_std` (4 values).
- Command from the game: [forward speed m/s, side speed m/s, yaw rate rad/s]. Normalize it as root values: `cmd = [(side - root_mean[0]) / root_std[0], (forward - root_mean[1]) / root_std[1], (yaw - root_mean[2]) / root_std[2]]`.
- Tokens: a plan has 8 tokens (`tokens`); one token = 4 frames (`frames_per_token`); each token has 5 heads (`heads`); each head has a code 0..63 (`codes` = 64). The value 64 (`mask_id`) means "not known yet".

## 3. State that the runtime keeps
| Name | Size | Content |
|---|---|---|
| `hist` | 4 x 166, normalized | the last 4 frames that were played (after the crossfade) |
| `hist_root` | 4 x 4, raw | the root values that were applied on those 4 frames |
| `plan` | 32 x 166, normalized | the current plan; frames 0..3 are the context again, frames 4..31 are new |
| `plan_root` | 32 x 4, raw | root values of the plan |
| `plan_tokens` | 8 x 5 | tokens of the current plan |
| `cursor` | int | next plan frame to play (starts at 4) |
| `x, z, yaw` | world | character position and heading |

Start / reset: `hist = ctx_start` (4 x 166), `hist_root = root_net.root_start` (4 x 4), `x = z = yaw = 0`, no plan.

## 4. Make a plan (call order)
1. **Root path.** Run `root_net.onnx` with `cmd` (section 2), `root_hist = (hist_root - root_mean) / root_std`, `ctx = hist`.
   The output is the normalized root path of the plan frames 4..31. Denormalize it. `plan_root[0..3] = hist_root`, `plan_root[4..31] = output`.
   The root net gives the speed, the turn and the root height the way the real dog moves (acceleration, step rhythm). Do not smooth it and do not clamp it to the command.
2. **Pose tokens** (MaskGIT loop, `decode_steps` = 6 calls of `pose_step.onnx`). Inputs of each call: the current `tokens`, `ctx = hist`, `root = (plan_root - root_mean) / root_std` (all 32 frames).
   - Start: all 40 token slots = 64 (`mask_id`), none known.
   - In step s (0..5): for each slot take the softmax of its 64 logits; `pred` = the best code, `conf` = its probability; a known slot has conf = 2.
     Let k = `keep_schedule[s]` and thr = the k-th largest conf. Each slot that is not known and has conf >= thr becomes known with its `pred`.
   - After step 5 all slots are known (`keep_schedule[5]` = 40).
3. **Decode.** Run `tok_dec.onnx` on the 8 tokens. The output is the plan (32 normalized frames). Set `cursor = 4`.

Seam contract (block `seam` in the JSON; it makes the new plan continue the old plan). Use it on a regular replan only: the cursor c of the old plan is a multiple of 4,
the command did not change (section 6), `c/4 - 1 >= seam.left_ctx` and `c/4 + seam.carry_tokens <= 8`:
- In step 2 the first `1 + seam.carry_tokens` tokens are GIVEN and known from the start: new token i = old plan token `c/4 - 1 + i`. (Token 0 covers the context frames; token 1 covers the first 4 new frames, which the old plan had planned already.) Use `seam.keep_schedule` in place of `keep_schedule`.
- In step 3 put the `seam.left_ctx` old plan tokens `c/4 - 1 - left_ctx .. c/4 - 2` in front of the 8 new tokens (N = 10), run `tok_dec.onnx`, and drop the first `4 * left_ctx` output frames.
- In all other cases (first plan, command change, cursor not on a token border) use the plain steps 2 and 3.

Old root path (model v1, only if `root_net` is absent or disabled): move the last root values toward the command with acceleration limits (2.5 m/s^2 forward, 1.5 side, 5 rad/s^2 yaw); root height from the table `height_speed` / `height_value`. This path makes the paws slide (the step rhythm is not tied to the speed).

## 5. Play a frame (30 times per second)
1. `f = plan[cursor]`. For the first `crossfade_frames` (4) frames after a replan, blend with the old plan: `f = old[old_cursor + i] + (f - old[old_cursor + i]) * (i + 1) / (crossfade_frames + 1)`, i = 0..3.
2. `r = plan_root[cursor]`. Push f to `hist` and r to `hist_root` (drop the oldest).
3. Denormalize f. Joint j in the heading frame = values [3j, 3j+3). Contacts = values [162, 166).
4. World position of a joint p: `world = ( cos(yaw)*p.x + sin(yaw)*p.z + x,  p.y,  -sin(yaw)*p.x + cos(yaw)*p.z + z )`.
5. Then move the character with r: `x += (cos(yaw)*r.vx + sin(yaw)*r.vz) / 30`, `z += (-sin(yaw)*r.vx + cos(yaw)*r.vz) / 30`, `yaw += r.yaw_rate / 30`. `cursor += 1`.
6. Engine side (not part of the model): interpolate between two model frames for the render frame; set bone lengths to the rig lengths (walk the tree, keep directions);
   keep the lowest paw on the floor; make bone rotations from the joint positions.

## 6. When to make a new plan
- No plan yet, or `cursor` = 32.
- Every 12 played frames (`replan_frames`; the models are trained for 12).
- The command changed since the last plan by more than 0.25 m/s or 0.35 rad/s, and at least 4 frames were played from the current plan.
- Standing (command = 0 and |vx| + |vz| + |yaw rate| of the last applied root < `root_net.idle_eps` = 0.08): every 28 frames is enough.

## 7. Contact head
Values [162, 166) of each denormalized frame; > `contact.threshold` (0.5) = planted. On held-out mocap the decoder gives the right contact state on 99.2 % of paw-frames.
Use it for foot locking / leg IK, footstep sounds and dust. The model does not lock the paws by itself.

## 8. Checks for a new port
The JSON holds PyTorch results. A port is correct when it reproduces them:
| Block | Inputs | Expected |
|---|---|---|
| `test` | `ctx = ctx_start`, `root = test.root` (normalized 32 x 4), plain steps 2 and 3 | `test.tokens` equal; `test.pose` max abs diff < 1e-3 |
| `root_net.test` | `cmd`, `root_hist`, `ctx` | `root` max abs diff < 1e-3 |
| `seam.test` | `ctx_start`, `test.root`, given tokens `init`, decoder left context `left` | `tokens` equal; `pose` max abs diff < 1e-3 |
`verify` in the JSON has the ONNX Runtime vs PyTorch differences measured at export.

## 9. Limits
- Commands inside the data: forward 0..2.5 m/s (`data_stats`), backward to about -0.3 m/s, yaw rate to about 1.2 rad/s. The mocap has only about 90 s above 1.2 m/s: the trot is weaker than the walk.
- The root net follows the command like the real dog: a start, a stop or a turn reversal takes 0.5-1 s.
- One plan costs 6 calls of `pose_step` + 1 `tok_dec` + 1 `root_net` (CPU: about 3 ms in C++ with 2 threads).
