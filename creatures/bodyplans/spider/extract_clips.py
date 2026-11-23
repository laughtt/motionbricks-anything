"""Extract the spider clips from YOUR OWN copies of the two CC-BY Fab packs (no pack file is in this repo; see ATTRIBUTION.md).
python -I extract_clips.py --bga <spider_animated_character.glb> --mixall <spiders_characters_with_animations.glb> --out <raw clips folder> [--blender <blender exe>]
Needs Blender 4.x (env BLENDER or --blender; default "blender" on PATH). It runs blender_extract.py once per pack with the rig maps
map_bga_spider.json and map_mixall_argiope.json. Out: <out>/<rig>__<action>.npz (world joint positions, 30 fps) + <rig>__report.json.
Get the packs from Fab (free, CC-BY) and export / download them as glb:
  "Spider animated character" by Bestgameassets, "Spiders - characters with animations" by Mixall (the Argiope spider is used)."""
import argparse, os, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bga", required=True, help="your glb of 'Spider animated character' (Bestgameassets)")
    ap.add_argument("--mixall", required=True, help="your glb of 'Spiders - characters with animations' (Mixall)")
    ap.add_argument("--out", required=True, help="output folder for the extracted clips")
    ap.add_argument("--blender", default=os.environ.get("BLENDER", "blender"), help="Blender executable (env BLENDER)")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for glb, mp in ((a.bga, "map_bga_spider.json"), (a.mixall, "map_mixall_argiope.json")):
        if not Path(glb).is_file():
            sys.exit(f"file not found: {glb}")
        cmd = [a.blender, "-b", "--factory-startup", "--python", str(HERE / "blender_extract.py"), "--", str(glb), str(HERE / mp), str(a.out)]
        print("RUN", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    print("EXTRACT_ALL_DONE", a.out)


if __name__ == "__main__":
    main()
