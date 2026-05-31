#!/usr/bin/env bash
#
# Play a quick tone through the Lean front-door to confirm the whole stack —
# Turnstile (Lean) → relay → ir_service (TS) → C++ engine — actually makes sound.
#
# Builds an oscillator → DAC graph, starts audio for a few seconds, stops.
#
# Usage:  lean/play_demo.sh [seconds] [program] [output_port] [freq_hz] [gain]
#   lean/play_demo.sh                 # 8s of BlepSaw @ 220Hz
#   lean/play_demo.sh 4 Clock out 0   # 4s of Clock's `out`
#
# Note: tropical oscillators are quiet by design (BlepSaw.saw ≈ -26 dBFS), so
# the signal is scaled up by GAIN before the DAC. Start with your system volume
# low and raise it to taste — it's a tone, not a full-scale blast.

set -euo pipefail

# Always run from the repo root, so `mcp/ir_service.ts` resolves.
cd "$(dirname "$0")/.."

SECS="${1:-8}"
PROGRAM="${2:-BlepSaw}"
OUT_PORT="${3:-saw}"
FREQ="${4:-220}"
GAIN="${5:-6}"
BIN="lean/.lake/build/bin/frontend"

command -v bun >/dev/null || { echo "error: 'bun' not on PATH" >&2; exit 1; }

if [[ ! -x "$BIN" ]]; then
  echo "front-door not built — running 'make lean'…" >&2
  PATH="$HOME/.elan/bin:$PATH" make lean
fi

echo "▶  $PROGRAM.$OUT_PORT @ ${FREQ}Hz → VCA(×${GAIN}) → dac for ${SECS}s (Ctrl-C to stop early)" >&2

# The DAC boundary only accepts a ref-shaped expr, so the gain can't be inlined
# at the dac wire — it goes through a VCA (out = audio * cv).
FREQ_SET=""
[ "${FREQ}" != "0" ] && FREQ_SET="{\"instance\":\"osc1\",\"input\":\"freq\",\"expr\":${FREQ}},"

{
  printf '%s\n' \
    '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}' \
    "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/call\",\"params\":{\"name\":\"add_instance\",\"arguments\":{\"program\":\"${PROGRAM}\",\"instance_name\":\"osc1\"}}}" \
    '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"add_instance","arguments":{"program":"VCA","instance_name":"amp"}}}' \
    "{\"jsonrpc\":\"2.0\",\"id\":4,\"method\":\"tools/call\",\"params\":{\"name\":\"wire\",\"arguments\":{\"set\":[${FREQ_SET}{\"instance\":\"amp\",\"input\":\"audio\",\"expr\":{\"op\":\"ref\",\"instance\":\"osc1\",\"output\":\"${OUT_PORT}\"}},{\"instance\":\"amp\",\"input\":\"cv\",\"expr\":${GAIN}},{\"instance\":\"dac\",\"input\":\"out\",\"expr\":{\"op\":\"ref\",\"instance\":\"amp\",\"output\":\"out\"}}]}}}" \
    '{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"start_audio","arguments":{}}}'
  sleep "$SECS"
  printf '%s\n' '{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"stop_audio","arguments":{}}}'
} | "$BIN"

echo "■  stopped." >&2
