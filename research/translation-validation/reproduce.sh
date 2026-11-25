#!/bin/bash
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

# ---- phase 2: the spec GENERATED from the Plan reference semantics ----
mkdir -p gen
# (diffcli emit-spec = Semantics.Plan over Semantics.Symbolic.algebra)
$DIFFCLI emit-spec lfo_knob2.json --rate 48000 > gen_lfo_knob2.spec.smt2
diff <(grep -v '^;' gen_lfo_knob2.spec.smt2) <(grep -v '^;' lfo_knob2.spec.smt2) && echo "spec matches the committed one"
# generated spec vs the thumbv6m object, automatic merges, and vs the hand spec
$PY tv_gen.py lfo_knob2.json --diffcli $DIFFCLI --cuts --compare-handspec
# a second kernel with no kernel-specific code
$PY tv_gen.py ../../tests/fixtures/embedded_lfo.json --diffcli $DIFFCLI --cuts
# the frame loop: a 128-frame object, every frame
$DIFFCLI compile-object lfo_knob2.json --out gen/lfo_knob2-f128.o --frames 128 --rate 48000 \
  --triple thumbv6m-none-eabi --cpu cortex-m0plus
$PY tv_frames.py --spec lfo_knob2.spec.smt2 --obj gen/lfo_knob2-f128.o --cuts
# differential check of the symbolic carrier against the concrete reference (bounded)
#   lake env lean --run tools/RefOfPlanJson.lean <plan.json> <vectors> > ref_out.txt
#   $PY diff_ref.py lfo_knob2.spec.smt2 <vectors> ref_out.txt param:freq param:phase
# kernel-checked version: z3 only picks the merges; Lean (bv_decide + rewriting) checks
$PY tolean2.py lfo_knob2.spec.smt2 gen/lfo_knob2.o gen/TVlfo.lean
(cd ../../lean && lake env lean --tstack=200000 ../research/translation-validation/gen/TVlfo.lean)
