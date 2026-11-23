"""Release audit: scan every new or changed file that git would add for things that must not be published.
python -I creatures/tools/release_audit.py [--all] [--words <file>]
Checks: absolute paths, secret patterns, private names, e-mail addresses, the local user name and machine name, files over 25 MB that are not
in Git LFS, file types that are not allowed (checkpoints, data, binaries, backups). Binary files are scanned as bytes too.
Default file set: untracked (not ignored) files + files changed against HEAD. --all: every tracked file under creatures/ as well.
Exit code 0 = clean. This script also scans itself.
Private words are NOT stored in this file. They come from (1) the untracked file creatures/tools/.audit_private_words.txt (git-ignored),
(2) the env var AUDIT_PRIVATE_WORDS (comma separated), (3) --words <file>. File format: one word per line; "regex: <pattern>" for a pattern;
"allow: <text>" for a text that may appear (for example your own support links); "#" starts a comment."""
import argparse, getpass, os, platform, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
MAX_BYTES = 25 * 1024 * 1024
BAD_EXT = {".pt", ".pth", ".ckpt", ".npz", ".npy", ".blend", ".blend1", ".bvh", ".fbx", ".dll", ".exe", ".lib", ".so", ".dylib", ".pdb", ".zip", ".7z",
           ".pyc", ".uid", ".import", ".safetensors", ".bin", ".obj", ".uasset", ".umap", ".mp4", ".jpg", ".jpeg"}
BINARY_OK = {".onnx", ".png", ".gif", ".glb"}                     # allowed binary types (Git LFS)
PRIVATE_FILE = ROOT / "creatures" / "tools" / ".audit_private_words.txt"   # untracked (see .gitignore)
PATTERNS = [
    ("absolute path (drive)", rb"(?<![A-Za-z])[A-Za-z]:[\\/](?![\\/])[\w.\- ]"),
    ("absolute path (msys / wsl)", rb"(?<![\w.:/<])/(?:[a-z]|mnt/[a-z])/(?:Users|[\w.\-]+_[\w.\-]+|[\w.\-]+/[\w.\-]+/)"),
    ("home path", rb"(?:/home/|/Users/|\\Users\\)[\w.\-]+"),
    ("secret: GitHub token", rb"gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}"),
    ("secret: API key", rb"sk-[A-Za-z0-9_\-]{24,}|hf_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9\-]{10,}|AIza[0-9A-Za-z_\-]{30,}"),
    ("secret: private key", rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    ("secret: password", rb"(?i)(?:password|passwd|secret|api_key|apikey|token)\s*[=:]\s*['\"][^'\"\s]{8,}['\"]"),
    ("e-mail address", rb"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+\.[A-Za-z]{2,}"),
    ("MANN data file name", rb"D1_\d{3}_KAN\d+"),
]
EMAIL_OK = (b"git@github.com", b"noreply@")


def git(*a):
    return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True, check=True).stdout


def files(all_tracked):
    out = set(git("ls-files", "--others", "--exclude-standard", "-z").split("\0")) | set(git("diff", "--name-only", "-z", "HEAD").split("\0"))
    if all_tracked:
        out |= set(git("ls-files", "-z", "creatures").split("\0"))
    return sorted(f for f in out if f and (ROOT / f).is_file())


def lfs_tracked(paths):
    res = {}
    for i in range(0, len(paths), 200):
        o = git("check-attr", "filter", "--", *paths[i:i + 200])
        for line in o.splitlines():
            p, _, v = line.rsplit(": ", 2); res[p] = v.strip() == "lfs"
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true"); ap.add_argument("--words", default="")
    a = ap.parse_args()
    words, regexes, allowed = [], [], []

    def load(text):
        for w in text.splitlines():
            w = w.strip()
            if not w or w.startswith("#"):
                continue
            if w.lower().startswith("regex:"):
                regexes.append(w[6:].strip())
            elif w.lower().startswith("allow:"):
                allowed.append(w[6:].strip())
            else:
                words.append(w)
    for env in ("AUDIT_PRIVATE_WORDS", "RELEASE_AUDIT_WORDS"):
        words.extend(w.strip() for w in os.environ.get(env, "").split(",") if w.strip())
    if PRIVATE_FILE.is_file():
        load(PRIVATE_FILE.read_text(encoding="utf-8"))
    if a.words:
        load(open(a.words, encoding="utf-8").read())
    generic = {"user", "admin", "administrator", "root", "runner", "desktop", "localhost", "pc", "home", "owner"}
    for w in (getpass.getuser(), platform.node()):                # the local user name and machine name (not stored in this file)
        if w and w.lower() not in generic and len(w) >= 4:
            words.append(w)
    pats = [(n, re.compile(p)) for n, p in PATTERNS] + [(f"private word #{i + 1}", re.compile(re.escape(w.encode()), re.I)) for i, w in enumerate(words)] \
        + [(f"private pattern #{i + 1}", re.compile(r.encode(), re.I)) for i, r in enumerate(regexes)]
    fs = files(a.all); lfs = lfs_tracked(fs); bad = []; total = 0
    for f in fs:
        p = ROOT / f; size = p.stat().st_size; total += size; ext = p.suffix.lower()
        if size > MAX_BYTES and not lfs.get(f):
            bad.append((f, 0, "file over 25 MB outside Git LFS", f"{size / 1e6:.1f} MB"))
        if ext in BAD_EXT or ".bak" in p.name or p.name.endswith("~") or "__pycache__" in f:
            bad.append((f, 0, "file type not allowed", p.name))
        data = p.read_bytes(); binary = b"\0" in data[:8192]
        if binary and ext not in BINARY_OK:
            bad.append((f, 0, "binary file type not allowed", ext or p.name))
        if ext in BINARY_OK and not lfs.get(f):
            bad.append((f, 0, "binary file is not tracked by Git LFS", ext))
        if ext in (".png", ".gif"):
            continue                                              # compressed pixels: random bytes match short patterns
        for ln, line in enumerate(data.split(b"\n"), 1):
            if binary and len(line) > 20000:
                chunks = re.findall(rb"[\x20-\x7e]{6,}", line)    # printable runs of a binary blob
            else:
                chunks = [line]
            for c in chunks:
                for ok in allowed:
                    c = c.replace(ok.encode(), b"")
                for name, rx in pats:
                    m = rx.search(c)
                    if not m:
                        continue
                    if name == "e-mail address" and any(m.group(0).startswith(e) or e in m.group(0) for e in EMAIL_OK):
                        continue
                    if binary and name.startswith(("private", "absolute path", "e-mail", "home", "secret")) and not re.fullmatch(rb"[\x20-\x7e]*", c):
                        continue
                    bad.append((f, ln, name, c[max(0, m.start() - 30):m.end() + 30].decode("utf-8", "replace").strip()))
    print(f"release audit: {len(fs)} files, {total / 1e6:.1f} MB, {len(words) + len(regexes)} private words/patterns loaded")
    for f, ln, name, ctx in bad:
        print(f"  FAIL {f}:{ln}: {name}: {ctx}")
    print("AUDIT_CLEAN" if not bad else f"AUDIT_FAILED ({len(bad)} findings)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
