"""CPU smoke test of the creature library on RANDOM synthetic data (no dataset needed). It takes about 1 minute.
python -I creatures/tools/smoke_test.py [--work <temp dir>] [--steps 30]
Steps: make a fake 6-joint creature and a random-walk dataset -> train_tok -> train_pose -> train_root -> eval_tok -> eval_rollout -> export_onnx
(the exporter checks ONNX Runtime against PyTorch) -> load the ONNX files again and compare with the test vector in the JSON spec."""
import argparse, importlib.util, json, os, subprocess, sys, tempfile
from pathlib import Path
_missing = [m for m in ("numpy", "torch", "onnx", "onnxruntime") if importlib.util.find_spec(m) is None]
if _missing:
    sys.exit("smoke_test: missing Python packages: " + ", ".join(_missing) + "\nInstall them with:  pip install -r creatures/requirements.txt")
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def fake_dataset(path, seed=0):
    from creature.features import features
    rng = np.random.default_rng(seed)
    joints = ["body", "head", "L_foot", "R_foot", "L_hand", "R_hand"]; parents = [-1, 0, 0, 0, 1, 1]
    spec = dict(root=0, front=1, feet=[2, 3, 4, 5])
    rest = np.array([[0, .5, 0], [0, .55, .4], [.2, 0, -.2], [-.2, 0, -.2], [.2, 0, .3], [-.2, 0, .3]])
    out = dict(pose=[], root=[], local=[], starts=[], lens=[], names=[], split=[]); n = 0
    for c in range(10):
        F = 300; t = np.arange(F) / 30.0
        v = np.clip(np.cumsum(rng.normal(0, .05, F)) + 1.0, 0, 2.5); w = np.cumsum(rng.normal(0, .03, F)).clip(-1, 1)
        yaw = np.cumsum(w) / 30; x = np.cumsum(np.sin(yaw) * v) / 30; z = np.cumsum(np.cos(yaw) * v) / 30
        P = np.zeros((F, 6, 3))
        for j in range(6):
            r = rest[j].copy()
            sw = np.sin(t * 8 + j * 1.6) * (0.1 if j >= 2 else 0.0)
            lx = r[0]; lz = r[2] + sw; ly = r[1] + (np.maximum(np.cos(t * 8 + j * 1.6), 0) * 0.08 if j >= 2 else 0.0)
            P[:, j, 0] = x + np.cos(yaw) * lx + np.sin(yaw) * lz; P[:, j, 2] = z - np.sin(yaw) * lx + np.cos(yaw) * lz; P[:, j, 1] = ly
        f = features(P, spec, floor=0.0); L = len(f["pose"])
        out["pose"].append(f["pose"]); out["root"].append(f["root"]); out["local"].append(f["local"])
        out["starts"].append(n); out["lens"].append(L); out["names"].append(f"clip_{c}"); out["split"].append("val" if c >= 8 else "train"); n += L
    pose = np.concatenate(out["pose"]); root = np.concatenate(out["root"])
    pm, ps = pose.mean(0), pose.std(0) + 1e-3; pm[-4:] = 0; ps[-4:] = 1
    np.savez(path, pose=pose, root=root, local=np.concatenate(out["local"]), starts=np.array(out["starts"]), lens=np.array(out["lens"]), names=np.array(out["names"]),
             split=np.array(out["split"]), pose_mean=pm, pose_std=ps, root_mean=root.mean(0), root_std=root.std(0) + 1e-3)
    return dict(joints=joints, parents=parents, contacts=[2, 3, 4, 5], root=0, front=1, mirror=[0, 1, 3, 2, 5, 4],
                orient=dict(left="L_foot", right="R_foot", front="head", back="body"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default=""); ap.add_argument("--steps", type=int, default=30)
    a = ap.parse_args()
    work = Path(a.work or tempfile.mkdtemp(prefix="creature_smoke_")); work.mkdir(parents=True, exist_ok=True)
    sk = fake_dataset(work / "dataset.npz")
    cfg = work / "creature.json"
    json.dump(dict(name="fake", data="dataset.npz", out="train", fps=30, heads=5, levels=[8, 8], skeleton=sk,
                   eval=dict(walk=1.0, trot=2.0, back=-0.3, slow=0.5, turn=0.6, turn_trot=0.5, turn_stand=0.6)), open(cfg, "w"))
    env = dict(os.environ, CREATURE_DEV="cpu"); env.pop("CREATURE_DATA", None); env.pop("CREATURE_OUT", None)
    py = [sys.executable, "-I"]; s = str(a.steps); C = str(ROOT / "creature")

    def run(*args):
        print("RUN", " ".join(args), flush=True)
        subprocess.run(py + list(args), check=True, env=env)

    run(C + "/train_tok.py", "--cfg", str(cfg), "--tag", "tok", "--steps", s)
    run(C + "/train_pose.py", "--cfg", str(cfg), "--tok", "tok", "--tag", "pose", "--steps", s, "--rollout", "0", "--aux", "1", "--aux_mode", "st")
    run(C + "/train_root.py", "--cfg", str(cfg), "--tok", "tok", "--pose", "pose", "--tag", "root", "--steps", s)
    run(C + "/eval_tok.py", "--cfg", str(cfg), "--tok", "tok", "--out", str(work / "eval_tok.json"))
    run(C + "/eval_rollout.py", "--cfg", str(cfg), "--model", "m=tok,pose,root", "--out", str(work / "eval.json"), "--no_heldout")
    run(C + "/export_onnx.py", "--cfg", str(cfg), "--tok", "tok", "--pose", "pose", "--root", "root", "--out", str(work / "onnx"))
    # independent check: ONNX Runtime on the exported files against the PyTorch test vector in the spec
    import onnxruntime as ort
    spec = json.load(open(work / "onnx" / "dog_brain.json")); L, H, NC, D, T = spec["tokens"], spec["heads"], spec["codes"], spec["pose_dim"], spec["frames"]
    s1 = ort.InferenceSession(str(work / "onnx" / "pose_step.onnx"), providers=["CPUExecutionProvider"])
    s2 = ort.InferenceSession(str(work / "onnx" / "tok_dec.onnx"), providers=["CPUExecutionProvider"])
    ctx = np.array(spec["ctx_start"], np.float32).reshape(1, 4, D); root = np.array(spec["test"]["root"], np.float32).reshape(1, T, 4)
    tok = np.full((1, L, H), NC, np.int64); known = np.zeros(L * H, bool)
    for k in spec["keep_schedule"]:
        lg = s1.run(None, {"tokens": tok, "ctx": ctx, "root": root})[0].reshape(L * H, NC).astype(np.float64)
        e = np.exp(lg - lg.max(-1, keepdims=True)); pr = e / e.sum(-1, keepdims=True); conf = pr.max(-1); pred = pr.argmax(-1); conf[known] = 2.0
        new = (conf >= np.sort(conf)[::-1][k - 1]) & ~known; tok.reshape(-1)[new] = pred[new]; known |= new
    pose = s2.run(None, {"tokens": tok})[0]
    same = tok.ravel().tolist() == spec["test"]["tokens"]; d = float(np.abs(pose.ravel() - np.array(spec["test"]["pose"])).max()) if same else float("nan")
    print("SMOKE onnx tokens equal:", same, "| pose max abs diff %.2e" % d, "| verify", json.dumps(spec["verify"]))
    ok = same and d < 1e-3 and "orient" in spec and (work / "onnx" / "root_net.onnx").is_file()
    print("SMOKE_PASS" if ok else "SMOKE_FAIL", work)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
