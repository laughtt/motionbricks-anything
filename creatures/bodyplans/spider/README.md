# bodyplans/spider/: the spider

This folder builds the spider dataset. It mixes real clips from two Fab packs (CC BY per the Fab listing) with a generated walk. The generator gives each leg its own lane, so legs do not cross.

**Warning:** do not commit the pack files or the extracted clips. Give the attribution in [`ATTRIBUTION.md`](ATTRIBUTION.md) when you show a result.

## What you need

- Your own glb copies of the two packs in `ATTRIBUTION.md` (free on Fab).
- Blender 4 or newer for `extract_clips.py` (tested with Blender 5.2). Set the env var `BLENDER` or pass `--blender`.

## What you get

- A clips folder with `<rig>__<action>.npz` files (world joint positions, 30 frames per second).
- `dataset.npz`, `data_report.json` (minutes of each part, gait values, leg overlap before and after the repair) and `spider_spec.json`.

## Files

| File | What it does |
|---|---|
| `spider_joints.py`, `spider_spec.json` | the skeleton: 38 joints, 8 feet, rest pose, mirror map, `orient` joints |
| `map_bga_spider.json`, `map_mixall_argiope.json` | rig maps: source bone to our joint |
| `extract_clips.py`, `blender_extract.py`, `blender_inspect_glb.py` | clip extraction with Blender |
| `spider_lib.py`, `spider_data.py` | retarget, leg IK, the gait of the real walk, the gait generator with leg lanes |
| `spider_lanes.json` | lane values for the generator |
| `spider_collision.json`, `make_collision_spec.py` | capsule spec for the leg overlap metric and repair |
| `build_dataset.py` | real clips + mirror + time warp + generated walk + capsule repair, saved as `dataset.npz` |
| `tune_lanes.py`, `eval_collision.py` | lane search; leg overlap report |
| `blender_make_spider_soma.py` | makes the grey spider SOMA body from `spider_spec.json` |
| `creature.json` | spider config for `creature/` (`creatures/work/spider/`, held-out split `valsyn`) |

## How to run it

1. Get the two packs from Fab as glb files.
2. `python -I creatures/bodyplans/spider/extract_clips.py --bga <pack 1 glb> --mixall <pack 2 glb> --out creatures/work/spider/raw`
3. `python -I creatures/bodyplans/spider/build_dataset.py --raw creatures/work/spider/raw --out creatures/work/spider/data`
4. Open the `collision` block in `data_report.json`. The generated clips must have almost no leg overlap.
5. Set `CFG=creatures/bodyplans/spider/creature.json`.
6. Train with the steps in [`../../creature/README.md`](../../creature/README.md).
7. Optional: `python -I creatures/bodyplans/spider/eval_collision.py --out col.json --data creatures/work/spider/data/dataset.npz --roll <rollout npz>`

To make the SOMA body yourself (optional; the sandbox already has a copy):

```sh
blender -b --factory-startup --python creatures/bodyplans/spider/blender_make_spider_soma.py -- creatures/bodyplans/spider/spider_spec.json <existing output folder>
```

## Good to know

- There are only 23 seconds of real clips. The walk in the dataset is generated from the gait of one real walk.
- `build_dataset.py` uses the GPU for the capsule repair if there is one (about 1 minute). On the CPU it is slow.
- The released weights (`creatures/models/spider_v3`) come from an earlier copy of the train scripts. A new train run gives a similar model, not the same weights.
