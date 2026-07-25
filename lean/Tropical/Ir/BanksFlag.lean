/-!
# The banks-realization flag (`TROPICAL_BANKS_UNROLL`)

Banks-as-data is the default: uniform (deg-0) modal banks lower through an
indexed reduction (a `bankSum` region) instead of unrolling.
`TROPICAL_BANKS_UNROLL` is the escape hatch back to the unrolled form — the
bisection ladder, where the naive realization stays reachable. Read once here,
at load; the consumer that branches on it is the arrow modal builder
(`EmitArrow.Modal`), which chooses the banked or unrolled lowering at
authoring time. (Hand-authored `Sig.bankSum` always stays a region — the
flag governs the builder's choice, not the emit.)
-/

namespace Tropical.Ir

/-- `true` when banking is on (the default; `TROPICAL_BANKS_UNROLL` unset). The
    single source of truth for the banks-vs-unroll realization choice. -/
initialize banksEnabled : Bool ← do
  return (← IO.getEnv "TROPICAL_BANKS_UNROLL").isNone

end Tropical.Ir
