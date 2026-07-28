Status: Current

# Stale-token audit

This is the checked-in review record for vocabulary that commonly survives an
architecture retirement. Run from the repository root:

```bash
git grep -n -E 'ParsedProgram|Ir/Elaborator|CoreArena|ArrayLower|SumLower|Specialize'
git grep -n -E -i 'two backends|sub-millisecond|state transfer'
git grep -n -E 'reg_decl|delay_decl|next_update|state_init|register_targets'
```

The commands are intentionally broader than “must return zero.” A zero-result
rule would erase useful history and compatibility tests. Every remaining hit
must fit one of these reviewed buckets.

## Allowed current hits

- A negated statement that explicitly says a former type/pass/path is gone.
- A compatibility comment naming a field that is ignored or accepted only by
  the plan-4 branch.
- A refusal test whose purpose is to ensure a deleted source construct cannot
  cross the boundary.
- A safety/correctness counterexample saying that agreement between two
  backends is insufficient; this does not claim that Tropical has only two
  execution targets.
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

## Compatibility-only hits

- [`NumericProgramParser.hpp`](../engine/runtime/NumericProgramParser.hpp)
  names ignored state keys and the bounded plan-4 lift.
- [`FlatRuntime.cpp`](../engine/runtime/FlatRuntime.cpp) explicitly states that
  publication performs no by-name state transfer.
- [`engine/tests`](../engine/tests/) names `legacy_plan4` only in the dedicated
  compatibility test/CTest.
- [`web/runtime/manifest.ts`](../web/runtime/manifest.ts) retains the empty
  `stateInit`/`registerTypes` carrier classified in the compatibility matrix.
- The migration fixture
  [`stdlib_sin.json`](../tests/fixtures/flat_plan/stdlib_sin.json) retains a
  dead `expected_plan` snapshot that current golden code does not consume.

## Audit outcome

The authoritative path contains no positive reference to a deleted elaborator,
rich-to-core pass sequence, separate core arena, production state transfer, or
two-target architecture. Quantitative latency claims are delegated to the
dated performance report. Historical records remain searchable and useful
because they are labeled rather than mechanically rewritten.
