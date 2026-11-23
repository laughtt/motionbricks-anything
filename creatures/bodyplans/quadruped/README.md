# bodyplans/quadruped/: the dog

This folder builds the dog dataset from the MANN motion capture files.

**Warning:** the MANN data is CC BY-NC 4.0 (non-commercial). Do not commit the data. Weights you train from it are non-commercial only.

## What goes in and what comes out

- In: the MANN BVH files (Zhang, Starke, Komura, Saito, SIGGRAPH 2018). `download_mann.py` gets them from the official AI4Animation link (33 MB).
- Out: `dataset.npz`. It has 41 minutes at 30 frames per second, 27 joints and 4 paw contacts. Every eighth file is held out for testing.

## Files

| File | What it does |
|---|---|
| `creature.json` | dog config: 27 joints, parents, contacts, eval speeds, data and output paths (`creatures/work/dog/`) |
| `download_mann.py` | downloads and unpacks the BVH files. It stops if you do not pass `--accept-license` |
| `bvh.py` | BVH reader (joint positions from the hierarchy and the motion) |
| `build_dataset.py` | turns the BVH files into `dataset.npz` |

## How to run it

1. Read the licence of the data.
2. `python -I creatures/bodyplans/quadruped/download_mann.py --out creatures/work/dog/mann --accept-license`
3. `python -I creatures/bodyplans/quadruped/build_dataset.py --bvh_dir creatures/work/dog/mann --out creatures/work/dog/data/dataset.npz`
4. Set `CFG=creatures/bodyplans/quadruped/creature.json`.
5. Train with the steps in [`../../creature/README.md`](../../creature/README.md).

## Good to know

- The data has one dog. It has almost no backward motion and only 90 seconds above 1.2 m/s.
- No dog mesh is in this repository. The runtime tests and the sandbox use the spider.
