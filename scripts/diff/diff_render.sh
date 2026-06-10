#!/bin/bash
# diff_render.sh — render gate: the Lean FFI path (diffcli render-bytes)
# must be byte-for-byte the TS koffi path (render_bytes.ts) on the same
# compiled plan, for every corpus patch. 16×256 samples each, compared
# by sha256 of the raw float64 stream.
#
# Usage: scripts/diff/diff_render.sh   (from the repo root, after
#        `make build lean`)
set -u
fail=0
tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

for p in patches/*.json; do
  n=$(basename "$p" .json)
  plan="$tmpdir/$n.plan.json"
  if ! bun run scripts/diff/compile_patch.ts "$p" 2>/dev/null > "$plan"; then
    echo "  ERROR  $n  (compile failed)"
    fail=1
    continue
  fi
  ts=$(bun run scripts/diff/render_bytes.ts "$plan" 2>/dev/null | shasum -a 256 | cut -d' ' -f1)
  ln=$(./lean/.lake/build/bin/diffcli render-bytes "$plan" | shasum -a 256 | cut -d' ' -f1)
  if [ "$ts" = "$ln" ]; then
    echo "  PASS   $n  ${ts:0:16}"
  else
    echo "  FAIL   $n  ts=${ts:0:16} lean=${ln:0:16}"
    fail=1
  fi
done

if [ "$fail" = 0 ]; then
  echo "all renders agree byte-for-byte (koffi vs Lean FFI)"
else
  echo "RENDER DIVERGENCE"
fi
exit $fail
