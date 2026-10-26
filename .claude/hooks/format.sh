#!/bin/bash
# PostToolUse (Edit|Write): run black on the edited Python file.
f=$(jq -r '.tool_input.file_path // empty')
case "$f" in
  *.py) command -v poetry >/dev/null && cd "$CLAUDE_PROJECT_DIR" && poetry run black -q "$f" ;;
esac
exit 0
