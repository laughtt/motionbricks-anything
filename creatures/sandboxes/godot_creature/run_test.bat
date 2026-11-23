@echo off
rem Headless tests: the scripted run and the key test. Results: test_out\test_result.json, test_out\test_keys.json. Exit code 0 = pass.
rem Set GODOT to the Godot 4.7 executable (default: "godot" on PATH). Extra arguments go to the scripted run, for example --body=0
setlocal
if not defined GODOT set GODOT=godot
rem always import: a new glb in assets\ has no .import file yet (fast when nothing changed)
"%GODOT%" --headless --path "%~dp0." --import
"%GODOT%" --headless --path "%~dp0." res://test/test_run.tscn -- %* || exit /b 1
"%GODOT%" --headless --path "%~dp0." res://test/test_keys.tscn -- %*
