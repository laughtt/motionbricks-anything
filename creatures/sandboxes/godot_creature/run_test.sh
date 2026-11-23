#!/bin/sh
# Headless tests: the scripted run and the key test. Results: test_out/test_result.json, test_out/test_keys.json. Exit code 0 = pass.
# Set GODOT to the Godot 4.7 executable (default: "godot" on PATH).
here=$(cd "$(dirname "$0")" && pwd)
godot_bin=${GODOT:-godot}
"$godot_bin" --headless --path "$here" --import
"$godot_bin" --headless --path "$here" res://test/test_run.tscn -- "$@" || exit 1
"$godot_bin" --headless --path "$here" res://test/test_keys.tscn -- "$@"
