Status: Current

# Stale-token audit

This is the checked-in review record for vocabulary that commonly survives an
architecture retirement. Run from the repository root:

```bash
git grep -n -E 'ParsedProgram|Ir/Elaborator|CoreArena|ArrayLower|SumLower|Specialize'
git grep -n -E -i 'two backends|sub-millisecond|state transfer'
git grep -n -E 'reg_decl|delay_decl|next_update|state_init|register_targets'
git grep -n -E -i 'compiler service|TS session|inputExprNodes|compileSessionSlotted|runtime\.loadPlan|by-name transfer'
git grep -n -E 'paramExpr|triggerParamExpr|arrayLiteral|array_set|sampleClock|sample_clock|sessionSlot|sessionArraySlot|audio_outputs|breaks_cycles'
```

The commands are intentionally broader than “must return zero.” A zero-result
rule would erase useful history and compatibility tests. Every remaining hit
must fit one of these reviewed buckets.

## Allowed current hits

- A negated statement that explicitly says a former type/pass/path is gone.
- A rejection boundary or test naming a retired schema or field.
- A refusal test whose purpose is to ensure a deleted source construct cannot
  cross the boundary.
- A decoder or normalization rejection message naming a retired Program-2
  alias or root carrier.
- A safety/correctness counterexample saying that agreement between two
  backends is insufficient; this does not claim that Tropical has only two
  execution targets.
- A retained diagnostic string whose historical function prefix does not
  describe a reachable compiler stage.
- This audit, the mismatch ledger, or the compatibility matrix.

## Historical hits

The following documents are retained as design/migration records and carry an
at-point-of-use historical or superseded status:

- [`archive/TESTING.md`](archive/TESTING.md)
- [`bugs/lean_port_findings.md`](bugs/lean_port_findings.md)
- [`emitarrow-cutover-handoff.md`](emitarrow-cutover-handoff.md)
- [`lean_port.md`](lean_port.md)
- [`native-dag-plan.md`](native-dag-plan.md)
- [`voice-admission-handoff.md`](voice-admission-handoff.md)
- [`voice-ports-handoff.md`](voice-ports-handoff.md)

The benchmark spike
[`active_set_spike/findings.md`](../benchmarks/llvm/active_set_spike/findings.md)
is also explicitly historical; it measures a rejected/retired scheduler design.
The current [`playground/README.md`](../playground/README.md) labels its older
compile-scaling subsection as a historical measurement record.

## Retired-boundary hits

- [`NumericProgramParser.hpp`](../engine/runtime/NumericProgramParser.hpp)
  names retired fields only to reject them.
- [`FlatRuntime.cpp`](../engine/runtime/FlatRuntime.cpp) explicitly states that
  publication performs no by-name state transfer.
- [`engine/tests`](../engine/tests/) names `tropical_plan_4` only in a negative
  boundary test.
- [`PlanDecode.lean`](../lean/Tropical/PlanDecode.lean) names retired fields
  only to reject them.
- [`WireExpr.lean`](../lean/Tropical/WireExpr.lean),
  [`Raise.lean`](../lean/Tropical/Parse/Raise.lean), and
  [`mcp/errors.test.ts`](../mcp/errors.test.ts) name retired Program-2
  spellings only at rejection boundaries and their tests.

## Audit outcome

The authoritative path contains no positive reference to a deleted elaborator,
rich-to-core pass sequence, separate core arena, production state transfer, or
two-target architecture. Quantitative latency claims are delegated to the
dated performance report. Historical records remain searchable and useful
because they are labeled rather than mechanically rewritten.
