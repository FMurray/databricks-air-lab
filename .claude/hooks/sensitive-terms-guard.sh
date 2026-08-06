#!/usr/bin/env bash
# PreToolUse guard: block customer identifiers / workspace hosts from entering committed files.
#
# Two nets (wired in .claude/settings.json):
#   1. Write|Edit|NotebookEdit — scans the content being written.
#   2. Bash `git commit` — scans ADDED lines of the staged diff (catches sed -i / script
#      writes that never went through Write/Edit — how a real leak happened, 2026-08-06).
#
# Patterns live in docs/private/sensitive-terms.txt (gitignored ON PURPOSE — the list IS the
# identifiers; never commit it). Two sections:
#   [everywhere]  — forbidden in any repo file outside docs/private/
#   [code-only]   — forbidden only in code files (workspace hosts are allowed in docs by
#                   convention, but must not be hardcoded in code — keep per-target values
#                   blank, cf. uat_config.py TARGET)
# Lines are extended regexes (grep -E), case-sensitive; '#' comments allowed.
# No terms file => allow everything (colleagues without docs/private are unaffected).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TERMS="$REPO_ROOT/docs/private/sensitive-terms.txt"
[ -f "$TERMS" ] || exit 0

CODE_EXT_RE='\.(py|ipynb|sh|bash|js|jsx|ts|tsx|scala|java|sql|go|rs)$'

INPUT="$(cat)"
TOOL="$(printf '%s' "$INPUT" | jq -r '.tool_name // empty')"

section() {  # section <name> -> patterns of that section, comments/blanks stripped
  awk -v want="[$1]" '
    /^\[/ { cur=$0; next }
    cur==want && !/^[[:space:]]*(#|$)/ { print }
  ' "$TERMS"
}

deny() {
  printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"Sensitive-term guard: %s. Customer identifiers/workspace hosts must not enter committed files (list: docs/private/sensitive-terms.txt; exempt: docs/private/). Rephrase per customer-refs conventions."}}\n' "$1"
  exit 0
}

scan() {  # scan <text> <section> <label> ; denies on first match
  local text="$1" sec="$2" label="$3" pat
  [ -n "$text" ] || return 0
  while IFS= read -r pat; do
    [ -n "$pat" ] || continue
    if printf '%s' "$text" | grep -qE "$pat" 2>/dev/null; then
      deny "$label matched pattern #$(printf '%s' "$pat" | cksum | cut -d' ' -f1)"
    fi
  done < <(section "$sec")
}

case "$TOOL" in
  Write|Edit|NotebookEdit|MultiEdit)
    FILE="$(printf '%s' "$INPUT" | jq -r '.tool_input.file_path // .tool_input.notebook_path // empty')"
    case "$FILE" in
      "$REPO_ROOT"/docs/private/*|*/scratchpad/*|*/memory/*|*/tool-results/*) exit 0 ;;
      "$REPO_ROOT"/*) ;;   # in-repo: scan
      *) exit 0 ;;         # outside the repo: not our problem
    esac
    TEXT="$(printf '%s' "$INPUT" | jq -r '[.tool_input.content // empty, .tool_input.new_string // empty, .tool_input.new_source // empty, (.tool_input.edits // [] | map(.new_string // empty) | join("\n"))] | join("\n")')"
    scan "$TEXT" everywhere "write to ${FILE#$REPO_ROOT/}"
    if printf '%s' "$FILE" | grep -qE "$CODE_EXT_RE"; then
      scan "$TEXT" code-only "code write to ${FILE#$REPO_ROOT/}"
    fi
    ;;
  Bash)
    CMD="$(printf '%s' "$INPUT" | jq -r '.tool_input.command // empty')"
    printf '%s' "$CMD" | grep -qE '(^|[;&|[:space:]])git([[:space:]]+-C[[:space:]]+[^[:space:]]+)?[[:space:]]+commit' || exit 0
    # commit message text itself
    scan "$CMD" everywhere "git commit command text"
    STAGED_ADDED="$(git -C "$REPO_ROOT" diff --cached --diff-filter=ACM -- . ':!docs/private' 2>/dev/null | grep '^+' || true)"
    scan "$STAGED_ADDED" everywhere "staged diff"
    STAGED_CODE="$(git -C "$REPO_ROOT" diff --cached --diff-filter=ACM -- '*.py' '*.ipynb' '*.sh' '*.js' '*.ts' '*.tsx' '*.scala' '*.java' '*.sql' '*.go' '*.rs' 2>/dev/null | grep '^+' || true)"
    scan "$STAGED_CODE" code-only "staged code diff"
    ;;
esac
exit 0
