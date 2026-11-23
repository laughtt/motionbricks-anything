class_name SandboxConfig
extends RefCounted
## Loads config/sandbox.json once. Every module reads its own section; missing keys keep the exported defaults.
## Also resolves paths: a path may be res://, user://, absolute, or relative to the project folder (for example "../../models/spider_v3").

const PATH := "res://config/sandbox.json"
static var _data: Dictionary = {}
static var _loaded := false


static func data() -> Dictionary:
	if not _loaded:
		_loaded = true
		if FileAccess.file_exists(PATH):
			var parsed: Variant = JSON.parse_string(FileAccess.get_file_as_string(PATH))
			if parsed is Dictionary:
				_data = parsed
			else:
				push_warning("sandbox.json is not valid JSON; defaults are used")
	return _data


static func section(name: String) -> Variant:
	return data().get(name, null)


## Copies the values of a config section to the exported properties of a node (same names).
static func apply_to(node: Object, name: String) -> void:
	var s: Variant = section(name)
	if s is Dictionary:
		for key: String in s:
			if key in node:
				node.set(key, s[key])


## File system path of a config path. res:// and user:// are converted; a relative path starts at the project folder.
static func resolve_path(path: String) -> String:
	if path.begins_with("res://") or path.begins_with("user://"):
		return ProjectSettings.globalize_path(path)
	if path.is_absolute_path():
		return path
	return ProjectSettings.globalize_path("res://").path_join(path).simplify_path()


## The model folder to use: "--model_dir=<path>" after "--" on the command line, else env CREATURE_MODEL_DIR, else the configured one.
static func model_dir_for(configured: String) -> String:
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--model_dir="):
			return a.trim_prefix("--model_dir=")
	var env := OS.get_environment("CREATURE_MODEL_DIR")
	return env if env != "" else configured


## Instances a body scene. res:// scenes are imported resources (load); any other path is read as a glTF file at run time.
static func load_scene(path: String) -> Node:
	if path.begins_with("res://"):
		var packed := load(path) as PackedScene
		return packed.instantiate() if packed else null
	var doc := GLTFDocument.new()
	var state := GLTFState.new()
	if doc.append_from_file(resolve_path(path), state) != OK:
		return null
	return doc.generate_scene(state)
