# Coding Standards

Mechanical formatting, import sorting, and type safety are enforced by `uv run ruff check .` and `uv run mypy src tests`. This document covers design and review judgement calls not enforced by linters.

## Domain & Architecture Documentation Sync

- When a change modifies domain terms or invariants in `GLOSSARY.md`, or alters schemas/decisions recorded in `docs/adr/` or `docs/spec/`, update those documents in the same branch. Updating `GLOSSARY.md`, `docs/adr/`, or `docs/spec/` to match a ticket's schema/behavior change is required maintenance, **not** scope creep.

## Internal Type Contracts

- Trust internal dataclass and SQLAlchemy model annotations (`RawSubmission`, `Submission`, `Comment`, `RawComment`). Access declared attributes directly (`sub.updated_at`) rather than using defensive `hasattr()` or `getattr()` checks on internal types.

## Test Fixtures & Factories

- Prefer shared builder/factory helpers (in `tests/conftest.py`) with sensible defaults for constructing domain dataclasses (`RawSubmission`, `Submission`, `RawComment`, `Comment`) so adding a field to a model does not require shotgun edits across unrelated test files.