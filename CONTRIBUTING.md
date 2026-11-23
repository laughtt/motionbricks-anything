# Contributing

Issues and pull requests are welcome.

Before you open a pull request:

1. Run the smoke test: `python creatures/tools/smoke_test.py` (install `creatures/requirements.txt` first). The last line must be `SMOKE_PASS`.
2. Run the release audit: `python creatures/tools/release_audit.py`. The last line must be `AUDIT_CLEAN`.
3. If you changed the C++ runtime, run `ctest` (see `creatures/runtime/README.md`).

Keep changes small and say what you tested and on which OS. Windows is the only tested OS today.
