#!/usr/bin/env python3
"""Fetch the build dependencies of the creature runtime into a folder.

  python fetch_deps.py <deps_dir> [--ort-version 1.24.4] [--ort-source nuget|github] [--godot-cpp-ref <commit, tag or branch>] [--no-godot-cpp] [--dry-run]

Downloads the official ONNX Runtime CPU package and unpacks it to <deps_dir>/onnxruntime.
Source "nuget" (default, tested on Windows): the Microsoft.ML.OnnxRuntime package from nuget.org (all platforms in one file, about 125 MB).
Source "github": the release archive of github.com/microsoft/onnxruntime for this platform (smaller; not tested here). Clones godot-cpp (github.com/godotengine/godot-cpp) to <deps_dir>/godot-cpp.
Then it prints the environment variables for the CMake build. Only the Python standard library and git are needed.

godot-cpp ref: pinned to the commit GODOT_CPP_COMMIT below (the one used for the tested build; a master-branch sync of 2026-11-12).
Godot 4.7 has no godot-cpp release branch yet. When a branch or tag for your Godot version exists, pass it with --godot-cpp-ref.
nlohmann json is not fetched here: CMake downloads it itself (FetchContent) when NLOHMANN_JSON_DIR is not set.
"""
import argparse
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile

ORT_URL = "https://github.com/microsoft/onnxruntime/releases/download/v{v}/onnxruntime-{plat}-{v}.{ext}"
ORT_NUGET_URL = "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/{v}/microsoft.ml.onnxruntime.{v}.nupkg"
GODOT_CPP_URL = "https://github.com/godotengine/godot-cpp.git"
GODOT_CPP_COMMIT = "2a4c066e1e8c121046e9d35fe91cc8e726861c48"


def ort_platform():
    machine = platform.machine().lower()
    arch = "aarch64" if machine in ("arm64", "aarch64") else "x64"
    if sys.platform.startswith("win"):
        return ("win-arm64" if arch == "aarch64" else "win-x64"), "zip"
    if sys.platform == "darwin":
        return ("osx-arm64" if arch == "aarch64" else "osx-x86_64"), "tgz"
    return ("linux-aarch64" if arch == "aarch64" else "linux-x64"), "tgz"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("deps_dir", help="folder that receives onnxruntime/ and godot-cpp/ (created when missing)")
    ap.add_argument("--ort-version", default="1.24.4", help="ONNX Runtime release (default: %(default)s)")
    ap.add_argument("--ort-source", default="nuget", choices=["nuget", "github"], help="where the ONNX Runtime package comes from (default: %(default)s)")
    ap.add_argument("--ort-url", default="", help="full archive URL (overrides the version pattern)")
    ap.add_argument("--godot-cpp-ref", default=GODOT_CPP_COMMIT, help="godot-cpp commit, tag or branch (default: %(default)s)")
    ap.add_argument("--no-godot-cpp", action="store_true", help="skip godot-cpp (console tools and C API only)")
    ap.add_argument("--dry-run", action="store_true", help="print the URLs and the environment variables, download nothing")
    a = ap.parse_args()

    deps = os.path.abspath(a.deps_dir)
    plat, ext = ort_platform()
    url = a.ort_url or (ORT_NUGET_URL.format(v=a.ort_version) if a.ort_source == "nuget" else ORT_URL.format(v=a.ort_version, plat=plat, ext=ext))
    ort_dir = os.path.join(deps, "onnxruntime")
    gcpp_dir = os.path.join(deps, "godot-cpp")
    print("ONNX Runtime archive: " + url)
    print("godot-cpp: %s (ref %s)" % (GODOT_CPP_URL, a.godot_cpp_ref))

    if not a.dry_run:
        os.makedirs(deps, exist_ok=True)
        if any(os.path.isfile(os.path.join(ort_dir, p, "onnxruntime_cxx_api.h")) for p in ("include", os.path.join("build", "native", "include"))):
            print("ONNX Runtime already in " + ort_dir)
        else:
            archive = os.path.join(deps, os.path.basename(url))
            print("downloading ...")
            req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
            with urllib.request.urlopen(req) as r, open(archive, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            tmp = os.path.join(deps, "_ort_unpack")
            shutil.rmtree(tmp, ignore_errors=True)
            os.makedirs(tmp)
            if archive.endswith((".zip", ".nupkg")):
                with zipfile.ZipFile(archive) as z:
                    z.extractall(tmp)
            else:
                with tarfile.open(archive) as t:
                    t.extractall(tmp)
            inner = [os.path.join(tmp, n) for n in os.listdir(tmp)]
            src = inner[0] if len(inner) == 1 and os.path.isdir(inner[0]) else tmp
            shutil.rmtree(ort_dir, ignore_errors=True)
            shutil.move(src, ort_dir)
            shutil.rmtree(tmp, ignore_errors=True)
            os.remove(archive)
            print("ONNX Runtime unpacked to " + ort_dir)
        if not a.no_godot_cpp:
            if os.path.isdir(os.path.join(gcpp_dir, "include")):
                print("godot-cpp already in " + gcpp_dir)
            else:
                if len(a.godot_cpp_ref) == 40 and all(ch in "0123456789abcdef" for ch in a.godot_cpp_ref):     # a commit id: fetch exactly this commit
                    os.makedirs(gcpp_dir, exist_ok=True)
                    subprocess.check_call(["git", "init", "-q", gcpp_dir])
                    subprocess.check_call(["git", "-C", gcpp_dir, "fetch", "--depth", "1", GODOT_CPP_URL, a.godot_cpp_ref])
                    subprocess.check_call(["git", "-C", gcpp_dir, "checkout", "-q", "FETCH_HEAD"])
                else:
                    subprocess.check_call(["git", "clone", "--depth", "1", "--branch", a.godot_cpp_ref, GODOT_CPP_URL, gcpp_dir])

    print("")
    print("Environment variables for the build:")
    if sys.platform.startswith("win"):
        print("  set ONNXRUNTIME_DIR=" + ort_dir)
        if not a.no_godot_cpp:
            print("  set GODOT_CPP_DIR=" + gcpp_dir)
    else:
        print("  export ONNXRUNTIME_DIR=" + ort_dir)
        if not a.no_godot_cpp:
            print("  export GODOT_CPP_DIR=" + gcpp_dir)
    print("Then: cmake -S <creatures>/runtime -B <build_dir> [-DCREATURE_BUILD_GODOT=ON]")


if __name__ == "__main__":
    main()
