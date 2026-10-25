## Environment & Tooling

### Python & Runner

Package manager & runner is `uv`.
- Run commands: `uv run <cmd>` (e.g. `uv run python script.py`)
- Manage packages: `uv add <pkg>` or `uv pip list`

### Toolchain & Commands
- Run tests: `uv run pytest` (Always use `BypassSandbox: true`)
- Type check: `uv run mypy src tests` (Always use `BypassSandbox: true`)
- Lint / format: `uv run ruff check .` (Always use `BypassSandbox: true`)
- GitHub CLI: `gh` (Always use `BypassSandbox: true`)
- Git commands (`git status`, `git diff`, `git log`, `git branch`, `git worktree`, and any `git` operation inside`.worktrees/`): Always use `BypassSandbox: true`
- PowerShell: Chain commands with `;`, not `&&`

### Asynchronous Tasks

End your turn immediately** (call no more tools) after calling `invoke_subagent` or when `run_command` transitions to a background task. The runtime wakes you automatically when the subagent replies or the task finishes. Never call `manage_subagents(Action="list")` or `manage_task(Action="status")` to wait for completion.

## Agent skills

### Issue tracker

GitHub Issues via `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Canonical triage roles (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context (`GLOSSARY.md` and `docs/adr/` at repo root). See `docs/agents/domain.md`.

### Pull Request (PR)

Use `/pr` skill to form PR body before marking it ready for review.
