"""Shared creature MotionBricks code (skeleton-agnostic): tokenizer, pose net, root net, foot-skate losses, runtime simulator, eval, ONNX export.

Every script takes a creature config JSON (see bodyplans/quadruped/creature.json and bodyplans/spider/creature.json):
  {"name": "dog", "data": "<dataset.npz>", "out": "<train dir>", "fps": 30, "heads": 5, "levels": [8, 8],
   "val_split": "val" (optional: the name of the held-out split),
   "skeleton": {"joints": [...], "parents": [...], "contacts": [joint ids of the feet], "root": 0, "front": 3} or "<skeleton spec>.json"}
Relative paths are relative to the folder of the config file. Env CREATURE_DATA / CREATURE_OUT replace "data" / "out"; CREATURE_DEV sets the device.
Dataset npz (made by bodyplans/*/build_dataset.py with creature/features.py):
  pose [F,D] (J*3 local positions in the heading frame + J*3 WORLD velocities in the heading frame + C contacts),
  root [F,4] = [vx side, vz forward, yaw rate, root height], local [F,J,3], starts, lens, names, split ("train"/"val"), pose_mean/std, root_mean/std.
Order: train_tok.py -> train_pose.py -> train_root.py -> eval_rollout.py -> export_onnx.py.
"""
