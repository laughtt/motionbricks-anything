# tools/

Two test scripts and some optional Blender scripts.

| File | What it does | Needs |
|---|---|---|
| `smoke_test.py` | Trains a very small model on random data, exports it to ONNX and compares ONNX with PyTorch. CPU, about 1 minute | torch, onnx, onnxruntime |
| `release_audit.py` | Looks for absolute paths, secrets, private names, large files and file types that must not be in a commit | git |
| `soma_to_3d/` | Optional scripts for "From a SOMA to 3D assets" in the main README (SOMA, image, 3D, rig) | Blender; `spider_landmarks.py` also needs scipy and matplotlib |

## How to run it

1. Run `python -I creatures/tools/smoke_test.py`. The last line must be `SMOKE_PASS`.
2. Run `python -I creatures/tools/release_audit.py` before each commit. The last line must be `AUDIT_CLEAN`. If it is not, fix every item it lists.
3. To add your own private words to the audit, put them in the untracked file `creatures/tools/.audit_private_words.txt` (one per line; `regex:` and `allow:` prefixes work), or set the env var `AUDIT_PRIVATE_WORDS` (comma separated), or pass `--words <file>`. No private word is stored in the script. The audit scans itself too.

## soma_to_3d/ (optional, not tested in this release)

These scripts come from our research work. They take all paths from the command line. The first lines of each file show its command.

| File | What it does |
|---|---|
| `render_spider_template.py` | Grey render of a SOMA rig for the image-to-image step |
| `spider_landmarks.py` | Joint landmarks inside a generated spider mesh |
| `blender_unfuse_legs.py` | Cuts the skin between legs that are joined in a generated mesh |
| `blender_fit_spider.py` | Fits and skins a generated spider mesh on the spider SOMA |
| `spider_drive.py`, `blender_spider_render.py`, `blender_spider_stretch.py` | Play a motion on a rig; render; stretch metric |
| `blender_export_godot.py` | glb export for Godot with at most 8 bone weights per vertex |

The image and 3D tools (Nano Banana, Tripo, MiniMax H3, Pixal3D) are external. Check their output terms before you publish a generated asset.
