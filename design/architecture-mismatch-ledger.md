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
| 6 | `engine/tests/CLAUDE.md` described ten plan-4/state tests, including named-state transfer, although `test_module_process.cpp` now has three current textual-IR/ABI tests and no state transfer. | wrong | Resolved with current/compatibility sections and explicit CTest labels. |
| 7 | `playground/README.md` performance notes predated banks-as-data and named deleted lowering passes. | historical data presented as current | Resolved by historical status and a pointer to the current baseline. |
| 8 | `engine/CLAUDE.md` described `NumericProgramParser → FlatProgram → OrcJitEngine::compile_flat_program`, but current codegen is Lean `EmitLlvm` followed by C++ `compile_ir_text`; the instruction graph is metadata-only in C++. | wrong | Resolved in the engine guide. |
| 9 | `patches/CLAUDE.md` advertised `reg_decl`, `delay_decl`, `next_update`, and `program_decl` as accepted patch fields. Ingest/decoder refuse them. | wrong | Resolved: the patch schema is documented as instances, params, outputs, and typed wires only. |
| 10 | Root docs said params/control slots transfer between kernels by name. `FlatRuntime::publish_state` carries only `sample_index`; current values come from plan defaults/control-host writes. | wrong | Resolved in architecture and runtime docs. |
| 11 | `Diffcli.lean` still framed live verbs as a TypeScript differential/parsed-layer migration surface. | wrong module comment | Resolved in its module comment. |
| 12 | C++ headers retained comments saying arrays/slots/state transfer on hot-swap even though no such copy occurs. | wrong comment over current code | Resolved in `FlatRuntime.hpp` and `test_module_process.cpp`; behavior unchanged. |
| 13 | The Lane F handoff assumed current C++ tests and plan-4 parsing retained register-state execution. The parser ignores state keys, allocates no registers, and C++ tests no longer exercise it. | stale sprint premise | Classified row-by-row in [`compatibility-matrix.md`](compatibility-matrix.md). |
| 14 | State-shaped C++ types (`StateReg`, writebacks, register target/type vectors) remain compiled but have no parser, emitter, runtime, or test producer/consumer. | historical/dead | Kept behavior-neutral this sprint; assigned to the runtime owner for the 2026-10-01 review. |
| 15 | Browser `KernelManifest.stateInit`/`registerTypes` and its initialization loop survive, while every production build supplies empty arrays. | compatibility-supported / sister-runtime candidate | Explicitly classified; assigned to the web-runtime owner for the 2026-10-01 extraction/API review. |
| 16 | The lone checked-in plan-4 fixture is nested as `expected_plan` in a migration golden, but current golden code compiles only its `input` and never reads that plan. | historical/dead fixture payload | Classified as a migration record; no silent deletion in this sprint. |
| 17 | Several completed design handoffs and bug reports used deleted vocabulary without an at-point-of-use status. | historical, not wrong in original context | Scoped status labels added only to documents returned by the stale-token audit. |
| 18 | `PlanDecode.FlatPlan.ofWire` is documented as plan-5-only but does not itself validate the top-level schema tag. Production emitters always produce plan 5; `FlatRuntime` does validate before native load. | trust-boundary ambiguity, not legacy-state reachability | Assigned to the trust-boundary owner; no behavior change in the documentation lane. |

## Outstanding assignments

- The [trusted-boundary ledger](trust-boundary.md) owns row 18 and any claim
  about what schema validation proves.
- The [performance baseline](../benchmarks/current_baseline/findings.md) owns
  all quantitative compile/control/runtime claims.
- The [Metal qualification report](../benchmarks/metal_live/findings.md) owns
  supported hardware, latency, and soak claims.
- Runtime and web-runtime owners review the bounded compatibility surface on
  2026-10-01; no legacy behavior was deleted in this pass.
