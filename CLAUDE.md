# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`eflips-eval` turns eflips-model databases (SQLAlchemy, PostgreSQL/PostGIS) into DataFrames and plots. `eflips` is a
namespace package. `eflips.model` and `eflips.depot` come from other distributions, so don't add `eflips/__init__.py`.

## Commands (always via poetry)
- Format: `poetry run black .`
- Types: `poetry run mypy eflips --explicit-package-bases --strict`
- Docs: `sphinx-build doc/ doc/_build -W`
- Tests: see `tests/CLAUDE.md` (they need a disposable PostgreSQL database)

Hooks in `.claude/settings.json` run black after edits and black + mypy before finishing.

## Pull requests
Every PR must bump the version in `pyproject.toml` **and** `release` in `doc/conf.py`, and the two must match.
CI rejects a version that is not higher than the one on PyPI. Merging to `main` publishes to PyPI.
- patch: no API change
- minor: backwards-compatible API change (e.g. a new prepare/visualize pair)
- major: breaking change

## Dependencies
Add packages only with `poetry add` (`--group dev` for tooling), never by editing `poetry.lock`. Propose any new
runtime dependency before adding it.
