#!/bin/sh
# Opens the creature sandbox. Set GODOT to the Godot 4.7 executable (default: "godot" on PATH).
# Extra arguments go to the project, for example:  ./run_sandbox.sh --model_dir=../../models/spider_v3
here=$(cd "$(dirname "$0")" && pwd)
godot_bin=${GODOT:-godot}
if [ ! -d "$here/.godot/imported" ]; then "$godot_bin" --headless --path "$here" --import; fi
exec "$godot_bin" --path "$here" -- "$@"
