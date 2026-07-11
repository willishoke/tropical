/-!
# The banks-realization flag (`TROPICAL_BANKS_UNROLL`)

Banks-as-data is the default: uniform (deg-0) modal banks and summing folds
lower through an indexed reduction (a `bankSum` region) instead of unrolling.
`TROPICAL_BANKS_UNROLL` is the escape hatch back to the unrolled form — the
bisection ladder, where the naive realization stays reachable. Read once here,
at load, so the two consumers that branch on it — the strata array-lowering
pass (`Ir.Strata.ArrayLower`) and the arrow modal builder
(`EmitArrow.Modal`) — share ONE environment read rather than each doing its own.
-/

namespace Tropical.Ir

/-- `true` when banking is on (the default; `TROPICAL_BANKS_UNROLL` unset). The
    single source of truth for the banks-vs-unroll realization choice. -/
initialize banksEnabled : Bool ← do
  return (← IO.getEnv "TROPICAL_BANKS_UNROLL").isNone

end Tropical.Ir
