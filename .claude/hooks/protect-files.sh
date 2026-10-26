#!/bin/bash
# PreToolUse (Edit|Write): block hand edits to the lockfile.
f=$(jq -r '.tool_input.file_path // empty')
case "$f" in
  */poetry.lock)
    echo "Don't edit poetry.lock by hand. Use poetry add/remove/lock." >&2
    exit 2 ;;
esac
exit 0
