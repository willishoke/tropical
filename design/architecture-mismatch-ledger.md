Status: Current

# Architecture mismatch ledger

Audit base: `9c492dd` on 2026-07-27. “Resolved” means the checked-in description
now matches the reachable source; it does not mean a historical implementation
was deleted.

| # | Mismatch found by source audit | Classification | Resolution |
|---:|---|---|---|
| 1 | `design/architecture.md` routed patch JSON and sessions through `ParsedProgram` and an elaborator that no longer exist. | wrong | Resolved by the current source-to-sound rewrite in [`architecture.md`](architecture.md). |
| 2 | The architecture described specialization, sum lowering, array lowering, and a separate core arena as live stages. | wrong | Resolved: the authority now names direct lowering and the single `ENode`/`ExprArena` vocabulary. |
| 3 | Root docs said there were two backends even though MSL/Metal is a live execution target. | wrong | Resolved in `README.md`, `CLAUDE.md`, and the backend matrix. |
| 4 | `README.md` claimed live name resolution, monomorphization, sum lowering, and a universal sub-millisecond compile budget. | wrong / unsupported performance claim | Resolved: current vocabulary and a link to the dated performance report replace those claims. |
| 5 | `Engine/Compile.lean` and `Engine.lean` module comments said the session was elaborated/downcast. | wrong | Resolved in module comments; definitions were unchanged. |
| 6 | `engine/tests/CLAUDE.md` described ten plan-4/state tests, including named-state transfer, although `test_module_process.cpp` had only current textual-IR/ABI tests and no state transfer. | wrong | Resolved with the current four-case suite, including explicit Plan-5-only rejection coverage. |
| 7 | `playground/README.md` performance notes predated banks-as-data and named deleted lowering passes. | historical data presented as current | Resolved by historical status and a pointer to the current baseline. |
| 8 | `engine/CLAUDE.md` described `NumericProgramParser → FlatProgram → OrcJitEngine::compile_flat_program`, but current codegen is Lean `EmitLlvm` followed by C++ `compile_ir_text`; the instruction graph is metadata-only in C++. | wrong | Resolved in the engine guide. |
| 9 | `patches/CLAUDE.md` advertised `reg_decl`, `delay_decl`, `next_update`, and `program_decl` as accepted patch fields. Ingest/decoder refuse them. | wrong | Resolved: the patch schema is documented as instances, params, outputs, and typed wires only. |
| 10 | Root docs said params/control slots transfer between kernels by name. `FlatRuntime::publish_state` carries only `sample_index`; current values come from plan defaults/control-host writes. | wrong | Resolved in architecture and runtime docs. |
| 11 | `Diffcli.lean` still framed live verbs as a TypeScript differential/parsed-layer migration surface. | wrong module comment | Resolved in its module comment. |
| 12 | C++ headers retained comments saying arrays/slots/state transfer on hot-swap even though no such copy occurs. | wrong comment over current code | Resolved in `FlatRuntime.hpp` and `test_module_process.cpp`; behavior unchanged. |
| 13 | The Lane F handoff assumed current C++ tests and plan-4 parsing retained register-state execution. The parser ignored state keys, allocated no registers, and C++ tests no longer exercised state. | stale sprint premise | Audited in [`compatibility-matrix.md`](compatibility-matrix.md); S-09 then retired Plan-4 acceptance immediately. |
| 14 | State-shaped C++ types (`StateReg`, writebacks, register target/type vectors) remained compiled but had no parser, emitter, runtime, or test producer/consumer. | historical/dead | Removed under S-09. |
| 15 | Browser `KernelManifest.stateInit`/`registerTypes` and its initialization loop survived, while every production build supplied empty arrays. | historical/dead carrier | Removed from the browser manifest contract under S-09. |
| 16 | The lone checked-in plan-4 fixture was nested as `expected_plan` in a migration golden, but current golden code compiled only its `input` and never read that plan. | historical/dead fixture payload | Removed; the useful `tropical_program_2` input and migration golden remain. |
| 17 | Several completed design handoffs and bug reports used deleted vocabulary without an at-point-of-use status. | historical, not wrong in original context | Scoped status labels added only to documents returned by the stale-token audit. |
| 18 | `PlanDecode.FlatPlan.ofWire` was documented as Plan-5-only but did not itself validate the top-level schema tag. | wrong boundary behavior | Resolved: `FlatPlan.ofWire` accepts exact Plan 5/6, enforces the immutable-asset schema split, and rejects retired carriers; `plan6-asset-abi` gates it. |

## Outstanding assignments

- The [trusted-boundary ledger](trust-boundary.md) owns the typed Plan-5/6
  serialized-plan obligation and maintained gates.
- The [performance baseline](../benchmarks/current_baseline/findings.md) owns
  all quantitative compile/control/runtime claims.
- The [Metal qualification report](../benchmarks/metal_live/findings.md) owns
  supported hardware, latency, and soak claims.
- Runtime and web-runtime owners preserve the Plan-5-only boundary; the former
  2026-10-01 compatibility review was superseded by S-09.
