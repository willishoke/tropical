#!/usr/bin/env bash
# tropical demo — one engine, two TUIs side by side (requires tmux + bun).
#
# Starts a single `frontend --serve` engine, launches the scrub (which loads and
# drives its patch) and the scope (which attaches, discovers the live patch's
# nodes, and taps them) against the SAME socket. The scrub owns the patch and
# drives params; the scope is a pure observer — wire-to-`scope`, read-only.
#
#   tui/launch-demo.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/lean/.lake/build/bin/frontend"
SOCK="${TMPDIR:-/tmp}/tropical-demo-$$.sock"

[ -x "$BIN" ] || { echo "engine not built: $BIN (run 'make lean')" >&2; exit 1; }
command -v tmux >/dev/null || { echo "tmux required (brew install tmux)" >&2; exit 1; }

"$BIN" --serve "$SOCK" &
ENGINE=$!
trap 'kill "$ENGINE" 2>/dev/null || true; rm -f "$SOCK"' EXIT

# Wait for the socket to come up.
for _ in $(seq 1 50); do [ -S "$SOCK" ] && break; sleep 0.1; done
[ -S "$SOCK" ] || { echo "engine socket never appeared: $SOCK" >&2; exit 1; }

# Scrub owns the patch (starts first); scope attaches and taps it.
tmux new-session  -d -s tropical -c "$ROOT" "TROPICAL_SOCK=$SOCK bun tui/scrub/app.tsx"
tmux split-window -h -t tropical -c "$ROOT" "TROPICAL_SOCK=$SOCK bun tui/scope/scope.tsx"
tmux set -t tropical mouse on
tmux attach -t tropical
