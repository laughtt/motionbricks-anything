@echo off
rem Opens the creature sandbox. Set GODOT to the Godot 4.7 executable (default: "godot" on PATH).
rem Extra arguments go to the project, for example:  run_sandbox.bat --model_dir=../../models/spider_v3
setlocal
if not defined GODOT set GODOT=godot
rem import when the project is new or a glb in assets\ has no .import file yet
set NEEDIMPORT=0
if not exist "%~dp0.godot\imported" set NEEDIMPORT=1
for %%F in ("%~dp0assets\*.glb") do if not exist "%%F.import" set NEEDIMPORT=1
if %NEEDIMPORT%==1 "%GODOT%" --headless --path "%~dp0." --import
"%GODOT%" --path "%~dp0." -- %*
