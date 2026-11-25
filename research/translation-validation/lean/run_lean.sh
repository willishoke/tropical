#!/bin/sh
# usage: run_lean.sh Check.lean  -> concatenates Spec + Lifted + the check file, runs lean
cd "$(dirname "$0")"
LEAN=$HOME/.elan/toolchains/leanprover--lean4---v4.29.1/bin/lean
{ echo "import Std.Tactic.BVDecide"; grep -v '^import' Spec.lean; grep -v "^import" Lifted.lean; [ "$1" = Check2.lean ] && grep -v "^--" LiftedCuts.lean; grep -v '^import' "$1"; } > _all_$1
/usr/bin/time -l $LEAN _all_$1 2>&1 | grep -E -v '^\s+[0-9]+\s+(page|block|volun|invol|messages|signals|instructions|cycles|swaps|average)'
