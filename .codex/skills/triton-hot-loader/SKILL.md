---
name: "triton-hot-loader"
description: "Use when working in this repository on Triton hot-loading workflows, especially to run or document `cli.py`, explain or debug `apply/status/list/unload/reload`, update the FastAPI/UI docs, or reason about runtime state under `runtime/`."
---

# Triton Hot Loader

Project skill for this repository.

## When to use

- The user asks how this repo's CLI works
- The user wants command examples, docs, or README updates
- The user wants to debug hot-loading, unload, reload, or state-file behavior
- The user asks about the Web UI or HTTP API backed by the same runtime

## Source of truth

Read code before trusting prose:

- `cli.py`: command surface and argparse flags
- `hot_loader.py`: real behavior for apply/unload/reload/state
- `server.py`: HTTP API routes and header overrides
- `docs/cli/README.md`: user-facing CLI handbook
- `tests/test_hot_loader.py` and `tests/test_server.py`: edge cases that the docs should not contradict

## Repo facts

- CLI entry is `python3 cli.py`
- Subcommands are `serve`, `apply`, `status`, `list`, `unload`, `reload`
- All commands share the runtime args defined in `add_common_runtime_args(...)`
- Default runtime paths live under `runtime/`
- `apply` accepts `--config-file` or `--json`
- JSON keys are placeholders; the managed identity is the image ref in the value
- `mlman_config` and `mlmanconfig` entries are skipped during config parsing
- `reload` triggers Triton `load` only; it does not do an explicit `unload`
- `unload --versions model@123` removes version directories and reloads the model when versions remain

## Triton operating rules

Before claiming that load or unload should work, verify that Triton is running with:

- `--model-control-mode=EXPLICIT`
- `--repository-poll-secs=0`

If the user sees `explicit model load / unload is not allowed if polling is enabled`, route the diagnosis to those two flags first.

## Workflow

1. Inspect `cli.py` or run `python3 cli.py --help` before documenting flags.
2. If behavior is ambiguous, confirm in `hot_loader.py` rather than inferring from README text.
3. Prefer relative command examples like `python3 cli.py apply --config-file sample_config.json`.
4. When editing docs, keep command examples aligned with the current repo path `triton_hot_loader`, not the older `hot_triton` path.
5. If code changes affect behavior, update docs and tests together.

## Validation

- `python3 cli.py --help`
- `python3 cli.py <subcommand> --help`
- `python3 -m unittest tests.test_hot_loader tests.test_server` when behavior or API contracts changed
