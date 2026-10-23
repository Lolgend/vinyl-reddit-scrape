## Environment & Tooling

### Python & Runner

Package manager & runner is `uv`.
- Run commands: `uv run <cmd>` (e.g. `uv run python script.py`)
- Manage packages: `uv add <pkg>` or `uv pip list`
- Windows UTF-8: Set `PYTHONUTF8=1` in environment or commands to prevent `cp1252` encoding errors.

### Asynchronous Tasks

Long-running commands (>5-10s) transition to background tasks automatically. Do not poll `manage_task(Action="status")` in a loop; the system provides reactive wakeup upon task completion.

## Agent skills

### Issue tracker

GitHub Issues via `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Canonical triage roles (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context (`GLOSSARY.md` and `docs/adr/` at repo root). See `docs/agents/domain.md`.
