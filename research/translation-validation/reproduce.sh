#!/bin/sh
# EXPERIMENTAL. Reproduce the translation-validation spike (SinOscQ15, 1 frame, 48 kHz, thumbv6m).
# Needs: python3 venv with z3-solver (requirements.txt), llvm@22 (llvm-objdump), Lean via elan.
#   PY=path/to/venv/bin/python DIFFCLI=path/to/diffcli ./reproduce.sh
set -e
cd "$(dirname "$0")"
PY=${PY:-./venv/bin/python}
DIFFCLI=${DIFFCLI:-../../lean/.lake/build/bin/diffcli}
export TROPICAL_KERNEL_CACHE_ROOT=${TROPICAL_KERNEL_CACHE_ROOT:-$PWD/.cache}
# 1. the artifacts: object + the exact LLVM text handed to the object compiler
$DIFFCLI compile-object lfo_knob2.json --out lfo_knob2-m0.o --frames 1 --rate 48000 \
  --triple thumbv6m-none-eabi --cpu cortex-m0plus --emit-ir
# 2. z3: spec == LLVM text == thumbv6m object (monolithic, ~2.5 min)
$PY tv.py 1
# 3. z3: the same with cut points (< 1 s)
$PY cuts.py
$PY cuts2.py
# 4. z3: spec-level properties from the Numerics.lean doc comments
$PY props.py
$PY inc_lia.py
# 5. Lean bv_decide on the lifted object (generated defs) vs a hand-written Lean spec
$PY tolean.py && CUTS=1 $PY tolean.py
lean/run_lean.sh Check2.lean   # cut-point proof, composed in Lean (cuts time out at 600 s)
lean/run_lean.sh Check.lean    # monolithic attempt (did not finish)
