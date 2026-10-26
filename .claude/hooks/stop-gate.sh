#!/bin/bash
# Stop: block finishing until black, mypy and the version check pass (mirrors CI).
input=$(cat)
[ "$(jq -r '.stop_hook_active' <<<"$input")" = "true" ] && exit 0
cd "$CLAUDE_PROJECT_DIR" || exit 0

# Skip if nothing Python-related changed in the working tree.
git status --porcelain | grep -qE '\.py$|pyproject\.toml$|doc/conf\.py$' || exit 0

errors=""
pkg=$(grep -m1 -oE '^version = "[^"]+"' pyproject.toml | cut -d'"' -f2)
doc=$(grep -m1 -oE '^release = "[^"]+"' doc/conf.py | cut -d'"' -f2)
[ "$pkg" != "$doc" ] && errors+="Version mismatch: pyproject.toml=$pkg, doc/conf.py=$doc"$'\n'

if command -v poetry >/dev/null; then
  out=$(poetry run black --check . 2>&1) || errors+="black --check failed:"$'\n'"$out"$'\n'
  out=$(poetry run mypy eflips --explicit-package-bases --strict 2>&1) || errors+="mypy failed:"$'\n'"$(tail -40 <<<"$out")"$'\n'
else
  echo "poetry not on PATH; skipped black/mypy" >&2
fi

[ -n "$errors" ] && jq -n --arg r "$errors" '{decision: "block", reason: $r}'
exit 0
