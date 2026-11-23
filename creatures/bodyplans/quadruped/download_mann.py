"""Download the MANN dog motion capture (Zhang, Starke, Komura, Saito: "Mode-Adaptive Neural Networks for Quadruped Motion Control", SIGGRAPH 2018)
from the official AI4Animation link and unpack the BVH files. No MANN data is in this repo.
python -I download_mann.py --out <folder> --accept-license
LICENCE: the data is CC BY-NC 4.0 (non-commercial). A model that you train on it is for research only. Do not commit the data or the weights.
Source page: https://github.com/sebastianstarke/AI4Animation (SIGGRAPH 2018, "Mocap Data")."""
import argparse, sys, urllib.request, zipfile
from pathlib import Path

URL = "https://starke-consult.de/AI4Animation/SIGGRAPH_2018/MotionCapture.zip"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="folder for the zip and the BVH files (keep it out of git, e.g. creatures/work/dog/mann)")
    ap.add_argument("--accept-license", action="store_true", help="you accept the CC BY-NC 4.0 licence of the data (non-commercial use only)")
    ap.add_argument("--url", default=URL)
    a = ap.parse_args()
    if not a.accept_license:
        sys.exit("The MANN dog data is CC BY-NC 4.0 (non-commercial). Read the licence, then run again with --accept-license.")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); zp = out / "MotionCapture.zip"
    if not zp.is_file():
        print("download", a.url, flush=True)
        req = urllib.request.Request(a.url, headers={"User-Agent": "python-urllib"})
        with urllib.request.urlopen(req) as r, open(str(zp) + ".part", "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        Path(str(zp) + ".part").rename(zp)
    n = 0
    with zipfile.ZipFile(zp) as z:
        for m in z.infolist():
            if m.filename.lower().endswith(".bvh"):
                (out / Path(m.filename).name).write_bytes(z.read(m)); n += 1
    print("MANN_DONE", n, "bvh files in", out)


if __name__ == "__main__":
    main()
