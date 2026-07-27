/-!
# The banks-realization flag (`TROPICAL_BANKS_UNROLL`)

Banks-as-data is the default: uniform (deg-0) modal banks lower through an
indexed reduction (a `bankSum` region) instead of unrolling.
`TROPICAL_BANKS_UNROLL` is a DEBUGGING convenience — a bisection flag that
keeps the naive realization reachable — not a correctness ladder: since
slice 3c, banked ≡ unrolled is a theorem
(`EmitArrow/BankOrder.lean` + `Ir/EmitBankLaws.lean`, trusted base = the one
named assumption `REDUCE_REGION_EXECUTES_IN_ARRAY_ORDER`), so flipping the
flag can only ever isolate an emitter/runtime bug, never rescue the
semantics. Read once here, at load; the consumer that branches on it is the
arrow modal builder (`EmitArrow.Modal`), which chooses the banked or
unrolled lowering at authoring time. (Hand-authored `Sig.bankSum` always
stays a region — the flag governs the builder's choice, not the emit.)
-/

namespace Tropical.Ir

/-- `true` when banking is on (the default; `TROPICAL_BANKS_UNROLL` unset). The
    single source of truth for the banks-vs-unroll realization choice. -/
initialize banksEnabled : Bool ← do
  return (← IO.getEnv "TROPICAL_BANKS_UNROLL").isNone

end Tropical.Ir
