# tropical

Tropical is a substrate for closed-form realtime computation. Every oscillator,
filter, envelope, and wire in a patch compiles to one function
`f(τ, params)`. The production language has no per-sample register, delay, or
update state.

That contract provides:

- random access: evaluate any coordinate without warming a recurrence;
- reversible clocks: change the coordinate map instead of unwinding state;
- fixed-topology fusion: one optimized kernel covers a whole patch;
- clickless publication: a replacement kernel resumes at the current
  coordinate rather than copying hidden DSP state;
- multiple realizations of one typed plan: native LLVM JIT, LLVM-to-wasm32,
  and MSL/Metal.

Audio is the first product instance. The graph, slot, source, sink, and kernel
concepts are not intrinsically audio-specific.

## The current path

There are three supported construction paths:

```text
Lean arrow builders → assemble → per-program direct lowering → registered types
                                                                  │
MCP mutations / program_2 JSON → typed SessionSt → synthetic resolved root
                                                                  │
                                                                  ▼
                                                   partition + stage-0 split
                                                                  │
                                                        tropical_plan_5
                                                   ┌──────────────┼───────────┐
                                                   ▼              ▼           ▼
                                              LLVM → JIT    LLVM → wasm32  MSL → Metal
```

The standard library is authored as fourteen-constructor
[`Tropical.EmitArrow.Sig`](lean/Tropical/EmitArrow/Sig.lean) trees and
assembled directly into the resolved `ENode`/`ExprArena` vocabulary. There is
no surface parser, name-resolution elaborator, generic specialization pass,
sum-lowering pass, or separate core-expression arena.

MCP sessions and `tropical_program_2` files are patch bays over registered
types. Their wiring decodes into the closed
[`Tropical.WireExpr`](lean/Tropical/WireExpr.lean) grammar. A patch cannot
define a program body, and no front door can spell a state operation or
feedback edge that reaches a backend.

Per-program direct lowering checks acyclicity, optionally inlines nested
instances, eliminates identities, and copies the evaluator-reachable graph
into the same vocabulary. Session compilation links those already-resolved
types into a synthetic root, checks the session graph, then partitions
instance functions, allocates typed slots/sources/sinks, and performs a typed
stage-0 split for parameter-derived coefficient work.

See [`design/architecture.md`](design/architecture.md) for the complete
source-to-sound walkthrough and invariant index.

## Live edits

Parameter and structural edits are different operations:

- A parameter edit applies its plan-carried host discipline and writes slots.
  It does not relower the graph. If stage-0 coefficient work exists, that
  kernel reruns after the write.
- A structural selector change or topology edit rebuilds, lowers, emits, and
  publishes a fresh kernel.
- Native publication carries the sample coordinate only. Current parameter
  values are supplied by the session/control layer; registers, arrays, and
  slots are not migrated as arbitrary kernel state.

There is no universal compile-latency promise. Dated measurements, cache
conditions, machines, and percentiles live in the
[`current_baseline` report](benchmarks/current_baseline/findings.md).

## Execution targets

| Target | Numeric regime | Role |
|---|---|---|
| LLVM → ORC JIT | f64 values plus i64 rails | native reference, random-access scopes, portable native audio |
| LLVM → wasm32 | shared LLVM f64/i64 semantics | precompiled browser player |
| MSL → Metal | f32 values plus exact i64 clock rail | heavy live modal audio on supported Apple builds |

Metal-enabled sessions dual-load the JIT artifact: a dedicated worker renders
exact-epoch Metal tiles while the audio callback consumes prepared slices;
scopes and the CPU reference keep using JIT. See the
[`metal_live` findings](benchmarks/metal_live/findings.md) for current
qualification evidence.

## Why you can trust the output

Correctness is not defined as “backends agree.” Agreement can show a refinement
regression; both implementations can still share a mistake.

The checked surface includes:

- frozen audio goldens;
- stdlib plan/port goldens;
- native realization checks;
- wasm-vs-JIT sample agreement;
- Metal-vs-JIT tolerance/SNR and runtime tests;
- cycle, patch-bay, Plan-5 schema-rejection, and production non-emission gates.

The exact statement, evidence, limitation, and owner of each load-bearing
claim is recorded in [`design/trust-boundary.md`](design/trust-boundary.md).
Retired schemas and state-shaped carriers are removed; their rejection
boundary and residual historical references are recorded in
[`design/compatibility-matrix.md`](design/compatibility-matrix.md).

## Build and test

```bash
make build
make lean
make validate
```

See [`CLAUDE.md`](CLAUDE.md) for requirements, direct binary commands, and the
repository map. See [`INSTALL.md`](INSTALL.md) for installation prerequisites.

## Where to read next

- [`design/architecture.md`](design/architecture.md) — current architecture.
- [`design/trust-boundary.md`](design/trust-boundary.md) — theorem,
  implementation, and empirical obligations.
- [`design/compatibility-matrix.md`](design/compatibility-matrix.md) —
  current Plan-5 reachability and retired-surface rejection.
- [`mcp/CLAUDE.md`](mcp/CLAUDE.md) — MCP tools and session behavior.
- [`lean/Main.lean`](lean/Main.lean) — the Lean/Turnstile frontend.

## License

MIT.
