# The Lean port — top-down migration of the compiler

Status: **COMPLETE — all eight phases landed. The production stack is
Lean end to end, and the TypeScript compiler and MCP engine are
deleted.** This document is the finished record of that port; it is no
longer a live roadmap. `design/architecture.md` stays the authority on
what the compiler *is*.

The Lean frontend binary (`lean/.lake/build/bin/frontend`) is now the
whole stack — compiler, MCP server, and session engine in one process.
It owns: raise (`Tropical/Parse/Raise.lean`), the surface parser
(`Tropical/Parse/Surface/*` + `Tropical/Parse/Nodes.lean`), the
elaborator (`Tropical/Ir/Elaborator.lean`), the codec
(`Tropical/Ir/Codec.lean`), all six strata passes
(`Tropical/Ir/Strata/*` + `Tropical/Ir/Strata.lean`), the Core
downcast (`Tropical/Ir/Core.lean`, carrying resolved port types), emit
(`Tropical/Ir/Emit.lean` — the structural-CSE emitter over Core with
total matches), the per-program boundary
(`Tropical/Ir/CompileResolved.lean`), the partitioner + session
compile (`Tropical/Compile.lean` — two-phase slot preallocation incl.
the array-input alias quotient, recursive partitionKernel, remap,
sinks, slot metadata), the typed plan layer + wire codec
(`Tropical/Plan.lean`), the native runtime FFI (`Tropical/Ffi.lean`
over `lean/ffi/shim.c` / `engine/c_api/tropical_c.h`), the v2 ingest
(load/merge walk the normalized JsonV node engine-side), save/export
(extraction + reconstruction / the full exportSessionAsProgram port),
catalog-entry rendering (`Tropical/Entries.lean`), and the MCP
resources/prompts (`Tropical/Resources.lean`). `make mcp-lean`
launches one binary; no bun subprocess exists on any path. The stdlib
boots from the pre-parsed bridge (`stdlib/parsed/`) in manifest order.

The port's correctness floor is the frozen audio **goldens** plus the
developer's ear, cross-checked by native mode-equivalence (fused vs
microkernel). The differential gates against the TS oracle
(`diff-plan`, `diff-audio`, `diff-engine`, `diff-emit`, `diff-strata`,
`diff-elab`, `diff-raise`, `diff-render`) did their job phase by phase
and were retired with the oracle in Phase 8 — a differential proves
*agreement*, not correctness, and the TS oracle was migration
scaffolding with no forward value (see the interpreter note below).
The production defects the gates surfaced are written up postmortem-
style in `design/bugs/lean_port_findings.md`.

Known non-gated divergences accepted at Phase 4 (documented, not
papered over): (1) on a nested define, typeDef registries register
nested-first rather than parent-first, so a parent/nested typeDef name
collision resolves to a different last-writer — unobservable through
any tool surface; (2) a generic template introduced via `load`/`merge`
(inline programDecl in a patch) exists only service-side (state-dump
entries ship `resolved: null` for generics), so a subsequent
`define_program` referencing it fails engine-side elaboration where TS
resolved it. Closing (2) means shipping raw templates in generic
entries; do it when a real session hits it or at Phase 5, whichever
comes first.

Production defects surfaced by the port's gates are written up
postmortem-style in `design/bugs/lean_port_findings.md` (append-only).

Golden hashes (`tests/golden/*.hash`) are now owned and rewritten
engine-side by `tropicaltest`'s write mode (`tropicaltest --write`),
which replaced the former `make validate-write` / `scripts/
validate_stdlib.ts` path. The goldens are the correctness floor: a
hash is intentional iff the developer's ear confirms it.

## Shape of the migration

The starting point: `lean/Main.lean` (Turnstile) validated 23 tools
against typed schemas and relayed them over newline JSON-RPC to a TS
relay (`mcp/ir_service.ts` → `mcp/engine.ts` → `compileSession` → plan
JSON → koffi → C++ engine). The port marched that relay boundary
**top-down** — engine/session first, elaborator next, surface parser
after that — collapsing it one layer per phase until no relay remained.

End state (realized): TypeScript is deleted. Lean owns the
engine/session and the whole compiler pipeline, and talks to the C++
engine through native `@[extern]` FFI (`Tropical/Ffi.lean` over
`engine/c_api/tropical_c.h`). The C++ engine (LLVM JIT, DAC) stays
C++. Browser TS survives as the web backend: `web/` host + worklet
plus the WASM emitter and memory layout, now under `web/wasm/`
(`emit_wasm.ts` + `wasm_memory_layout.ts`), consuming plan JSON
precompiled by the Lean engine (`diffcli compile`).

Design stance: **proof-ready, prove later.** Port unverified, but the
Lean IR is a thing proofs can later attach to: decl stores + typed-id
refs (the completion of the de Bruijn migration the TS elaborator
already made — refs are branded integer `idx` values into typed decl
tables), total functions, and `Except`-shaped errors mapped onto the
`mcp/ERRORS.md` envelope taxonomy. The Phase-2-planned plan-level
*reference interpreter* was **not** built — see the closing interpreter
note: a differential proves agreement, not correctness, so the oracle
needs no replacement; frozen audio goldens + the developer's ear are
the semantic anchor.

## Why this was tractable

- **ParsedProgram is a free seam.** The parsed program (now
  `Tropical/Parse/Nodes.lean`) is plain JSON-serializable data (NameRef
  placeholders, no object identity), and the session serializer routes
  sessions through the single `elaborate` front door.
- **No plan-JSON byte comparison anywhere.** Golden gates hash rendered
  audio, so float-formatting divergence between TS and Lean serializers
  could not break a gate; only double round-trip fidelity mattered.
- **Refs are already indices.** Post-elaboration identity is positional
  (decl tables), not pointer-based — a Resolved⇄JSON codec is
  mechanical.
- **The C FFI surface is tiny** (~27 opaque-pointer functions), and
  session-path params are slot-based (`param:name` module slots) — no
  native pointers inside plans.

## The harness (Phase 0 — built to drive the migration, retired in Phase 8)

The migration was driven by command-pair *differs* (originally under
`scripts/diff/`, all green TS-vs-TS from day one). Each compared two
*commands*; a Lean layer landed by becoming side B and keeping the diff
empty. The headline three, plus the per-layer differs added later
(`diff-emit`, `diff-strata`, `diff-elab`, `diff-raise`, `diff-render`):

| target | sides implement | compared |
|---|---|---|
| `diff-plan` | `<cmd> <patch> [--mode=<m>]` → plan JSON on stdout | structural: key-order-insensitive, numbers by IEEE-754 bits |
| `diff-audio` | same contract | byte equality of 16×256 rendered samples per side, fresh runtime each |
| `diff-engine` | engine protocol (newline JSON-RPC, method = tool) | normalized ToolResults per step of recorded scripts in `tests/fixtures/mcp_scripts/` |

The Lean half of the comparator was `Tropical.Json.diff`
(`lean/Tropical/Json.lean`). The whole differ harness — every `make
diff-*` target — was removed in Phase 8 along with the TS oracle it
compared against; it was migration scaffolding, not a forward gate.

## Phases (all landed)

Each phase had a hard exit criterion: the relayed/TS entrypoint it
replaced was **deleted**, and the relevant differs + existing suites
(`bun test`, `ctest`, stdlib golden hashes) were green.

1. **Engine + session layer in Lean** ✅ (landed). `mcp/engine.ts`'s 23
   handlers + the session ported to `Tropical/{Engine,Session,Errors,
   Expr,Wiring,Client,Rpc}`; the Lean session stores wires canonically
   pre-extraction; TS shrank to `mcp/compiler_service.ts` (snapshot
   `sync` in → compile + hot-swap; save/export/load/merge + audio/params
   relay until their phases). Gates passed: protocol suites unmodified
   against `frontend --rpc`, diff-engine 41/41, full bun suite.
   Bonus: the differential gate exposed and fixed a production bug —
   re-wiring an already-compiled input died with
   "duplicate reg/delay '__autodelay:…'" (stale delaySlotRegistry);
   `wire()` now canonicalizes + rebuilds slot state per compile.
2. **Native FFI + ownership** ✅ (landed; interpreter deferred).
   `Tropical/Ffi` + `lean/ffi/shim.c` over `tropical_c.h`; the DAC
   holds its Runtime alive as a Lean field; plans load over Lean FFI;
   the TS service is stateless pure compile. Two findings: set_param
   was audibly inert on the session path (the kernel reads `param:`
   slots, never Param handles) — both engines now drive the slot; and
   instance input wires are canonicalized to port-declaration order
   (JSON key order isn't preserved by clients/relays). Gate:
   `make diff-render` — Lean FFI rendering byte-for-byte the koffi
   path across the corpus.
   **Deferred from this phase**: `Tropical/Plan` (typed FlatPlan/NInstr
   + codec) and `Tropical/Plan/Interp` (the pure plan-level reference
   interpreter — the proof-ready semantic anchor and bun-independent
   oracle). It needs the engine's exact per-op semantics done
   carefully, and it gates Phase 6, not Phase 3 — it lands as its own
   focused change before then.
3. **Session lowering; seam lands on ParsedProgram** ✅ (landed).
   `Tropical/Lowering.lean` runs slot allocation (canonical order:
   params → outputs → delay slots), delay extraction, the Tarjan
   acyclicity tripwire, topo ordering, and session→ParsedProgram
   serialization per compile; the service's `compile` just elaborates
   + partitions the shipped ParsedProgram against the shipped slot
   bookkeeping. Wire lifting stays a service method (it needs strata),
   driven by the engine's pure `needsWireLift` detection and its
   `__wire` counter. Gates: diff-engine with `debug_render` audio
   probes (renders hex-compared after every mutation — this caught a
   real bug: `reconstructWireDelays` dropped delay ids, renaming
   registers per recompile, so hot-swap state transfer missed them and
   feedback graphs hit absorbing zero states; reconstruction is now
   id-preserving). Param slots also survive rebuilds now (they were
   silently dropped after any rewire, killing set_param's slot).
4. **Elaborator front + decl stores + Resolved codec** ✅ (landed, five
   gated stages). (1) TS `resolved_codec.ts` (`tropical_resolved_1`):
   three identity pools — programs, typeDefs, TypeParamDecls — where
   positional indices are provably insufficient (scope-chain lookups,
   post-inline lifted aliases, specialize keying substitution on decl
   identity); property-gated by round-trip + recompile-equality over
   stdlib post-elaborate AND post-strata, specialized generics, sums,
   nested programs, and session roots, *before* any Lean consumer.
   (2) Typed ParsedProgram AST + raise port (`Tropical/Parse/*`;
   order-preserving JSON layer because `Object.entries` order is
   load-bearing); stdlib bridge `stdlib/parsed/` committed; gate
   `diff-raise`. (3) The elaborator (`Tropical/Ir/*`): pool-shaped
   arena IR (pool-index sharing is the Lean image of TS pointer
   identity; decl tables are computed projections), codec with the TS
   encoder's canonical post-order renumbering, line-faithful
   `elaborate` (programDecls-first decl reordering, sequential-let
   scoping, exact binder-minting order, transitive registry merge in
   insertion order, `findInstanceCycles` + byte-exact CycleViolation
   incl. the unsorted-SCC suggested-fix target); gate `diff-elab`.
   (4a) Compile seam: `syncCompile` elaborates the root over the
   engine's arena store (resolver = `sessionTypeResolver` parity:
   per-instance `resolvedIdx` snapshots, first-instance-wins, keyed by
   stored `prog.name` — the BASE name for specialized generics, not
   the catalog display key) and ships `resolved_root`; failed compiles
   still mirror-sync the service session before surfacing the verbatim
   `internal_error` envelope. (4b) Registration: the engine raises +
   elaborates defines (nested programDecls post-order, one service
   call per item, adopting each post-strata response before
   elaborating the next); stdlib boots from the bridge with
   `loadStdlibFromResolved`'s relink discipline (concrete relinked to
   post-strata, generic templates never relinked); oracle-probed
   partial-registration semantics on mid-batch failure reproduced.
   Scope deviation from the original list: `recursion.ts` is
   strata-internal (Phase 5) and `port_type`/`branded_names` are
   session-layer — none are elaborator dependencies; ported instead:
   `decl_tables` semantics, `findInstanceCycles`,
   `elaboration_diagnostics`.
5. **Strata pipeline.** ✅ (landed: passes at stages 0–6a, seam
   cutover at 6b). Port of the six passes (`Tropical/Ir/Recursion.lean` +
   `Tropical/Ir/Strata/{Basic,Specialize,SumLower,InlineInstances,
   ArrayLower,IdentityElim}.lean`), `inlineNested` fractal path kept.
   **Gate methodology revision** (deliberate, replacing the per-pass
   differential): whole-strata comparison with a hybrid prefix —
   `make diff-strata` runs Lean passes `1..K` (`STRATA_K`, ratcheted
   per stage, = `Strata.portedPasses`), ships the prefix through the
   resolved codec, the TS suffix (`strata_cmd.ts strata-suffix`)
   completes `K+1..5`, and only final post-strata output is compared —
   divergence at stage K localizes to pass K with no per-intermediate
   diff semantics and no codec-fidelity obligation at intermediate
   strata. At K=5 the suffix is skipped: pure Lean-vs-TS. Corpus: 86
   outputs × both modes (stdlib manifest, generics re-run all-args=8,
   elaborable raise fixtures incl. the new `identity_wire.json` —
   added because identityElim was a no-op on the prior corpus —
   specialize error probes; byte-exact error parity). TS identity
   discipline mirrored where observable (typeArg subst keyed on
   typeParam pool idx; sumLower slotKey lookups name-keyed; arrayLower
   registry recursion per entry so aliased keys stay aliased only on
   the no-op fast path); TS WeakMap memos are sharing-only (invisible
   to the structural codec) and dropped. Stage 6a: `Tropical/Ir/
   Core.lean` — the post-strata sub-IR as a type + `check`, the
   executable downcast asserted in the K=5 inline gate; it is the spec
   for any later typed-boundary refactor and the domain Phase 6
   emit/partition consume with total matches.
   **Stage 6b (landed, three gated sub-stages):** the production
   strata call sites moved engine-side and left `compiler_service.ts`.
   (6b-1) `register_program`: `relinkProgramRegistry` ported to
   `Strata/Basic.lean`; `registerOne` relinks against the concrete
   `templateByName` mirror (load-bearing on the boot path, structural
   no-op on define), runs the full pipeline + the Core downcast, and
   ships post-strata; the service wraps via `makeCompiled`. Generics
   still ship raw. (6b-2) specialization: `compiler/specialize.ts`
   ported to `Tropical/TypeArgs.lean` (byte-exact `instance of 'X'`
   messages — all specialize-path failures map to `invalid_type_args`,
   matching the old failure relay); the engine caches
   `(ProgMeta, store idx)` per `Type<N=8>` key and a hit skips the
   service round trip; the compile payload ships per-instance resolved
   snapshots, so `handleCompile` instantiates verbatim
   (`resolveProgramType` left the service). (6b-3) wire lifting:
   `wire_program.ts` ported to `Tropical/Ir/WireProgram.lean`;
   `liftIfNeeded` builds + stratas + rewires each `__wire_N` lift;
   `lift_wires` became `register_lifted` (raw form into
   `session.programs` — what export's resolver elaborates against —
   post-strata Compiled into typeRegistry). Two gate fixtures added
   because the prior corpus was blind to both paths: `generics.json`
   (specialize miss/hit/second key, MCP-defined generic with required
   param, default fill, five error shapes) and `lifted_wires.json`
   (multi-ref lift, param-in-lift, bare-array wire with array reg-init
   broadcast, save/export after lift). Gates: diff-engine 7 scripts
   incl. `debug_render`, test-lean-engine, full bun suite,
   `make validate` (modulo the three documented stale goldens),
   diff-strata K=5.
   **Post-oracle restructure decision (recorded, NOT Phase 5 work):**
   inlining is a realization/cost knob, not a lowering — both backends
   consume nested plans; the depth-vs-flat benches show the ~25%
   flat-path runtime win comes from slot *removal* (IR-level), which
   LLVM legally cannot do (slots are observable hot-swap state), while
   microkernel-vs-fused shows compile-time crossover at ~8 voices.
   After the oracle retires, `inlineInstances` leaves the strata
   pipeline and becomes an optional post-strata `Core → Core`
   normalization (gated by `nested_vs_inlined` at 1e-12; audio goldens
   re-baseline expected). That also linearizes the stratum lattice,
   simplifying any later typed-boundary refactor of the passes —
   which is deliberately deferred until `check` can gate it
   Lean-internally.
6. **Emit + partition** ✅ (landed; six gated stages, one session).
   Port `slots`,
   `compile_resolved`, `emit_resolved`, `partition_recursive`,
   `compile_session_slotted*`. CSE memos key on arena `ExprId`s that
   replicate TS identity semantics — no aggressive hash-consing until
   the TS oracle retires. The TS compiler service is **deleted**.
   Gate: `diff-plan` + `diff-audio` across the corpus × all three
   compilation modes; goldens pass without re-baselining.
   **Staging (recorded at phase start):**
   - **6a** `Tropical/Plan.lean`: typed plan_5 (NOperand/DstSlot/NInstr/
     PerInstancePlan/InstanceFunction/Sink/Source/FlatPlan) + the
     `toWirePlan` JSON encoder with its key-omission rules. Core
     enrichment: port types (input/output decls, reg scalar) survive
     the downcast — emit consumes them.
   - **6b** Emit: `slots` + `emit_resolved` + `compile_resolved` over
     Core with total matches. CSE port note: TS interns structural key
     strings to dense ids — only the *partition* the keys induce must
     match, not the key text; numbers key by value-equivalence (TS keys
     on JS number toString of the parsed double, so distinct decimal
     texts of one double must collide in Lean too). New gate
     `make diff-emit`: TS `emit_cmd.ts` vs `diffcli emit-*`, the
     diff-strata corpus, inline mode, structural compare of
     PerInstancePlan wire JSON (fractal emit paths are exercised at
     6c/6d through partition).
   - **6c** Partition + session compile: `partition_recursive`,
     `compile_session_slotted*` (translateNode / remapInstancePlan /
     emitSinks / buildSlotMetadata), and the `session.ts` slot
     allocators incl. the array-input alias quotient. `syncCompile`
     builds and loads its own plan; the service `compile` method dies.
     Engine renders its own catalog entries. Gates: diff-engine (all
     scripts, debug_render probes), test-lean-engine, bun suite.
   - **6d** load/merge engine-side (v2 ingest → session
     materialization; raise is already Lean) + `diffcli compile`
     implementing the compile_patch contract. Makefile diff-plan /
     diff-audio gain the Lean side B × all three modes. The phase
     gate proper lands here.
   - **6e** save/export engine-side (`v2NodeToFile`, `prettyExpr`,
     `saveProgramFromSession`, `exportSessionAsProgram`). Gate:
     diff-engine program_io (extend the fixture if coverage is thin).
   - **6f** `mcp/compiler_service.ts` deleted; the frontend spawns no
     bun process. Full battery; goldens unchanged; docs.
   **Landed notes:** save is extraction + `reconstructWireDelays`, not
   a canonical-wire echo (extraction-minted delay ids are observable);
   the ingest walks the normalized JsonV node (key order preserved);
   entry rendering moved to `Tropical/Entries.lean` with the codec
   self-round-trip preserving the store-adoption discipline; the
   service `compile` became `sync` at 6c and the whole service died at
   6f along with `Tropical/Client.lean`. `compileMirrorPlan` (the
   diffcli harness rebuild) can collapse into `syncCompile` now that
   the sync payload is gone — deferred housekeeping.
   **Interpreter decision (final, reversed at Phase 8):** the
   Phase-2-deferred `Tropical/Plan/Interp` reference interpreter — long
   carried as "the post-oracle semantic anchor" that would land in
   Phase 8 when the TS oracle retired — was ultimately **not built**,
   and is not planned. The realization: a reference interpreter is a
   *differential* oracle, and a differential proves *agreement*, not
   correctness. The TS oracle was migration scaffolding — it gave the
   port a concrete moving target to converge on, which was its entire
   value, and that value expired the moment the port completed. A Lean
   interpreter built afterward would only re-establish agreement
   between the Lean compiler and a second Lean artifact authored from
   the same understanding — circular, with no independent purchase on
   correctness. The correctness floor that actually holds is the frozen
   audio **goldens** plus the developer's ear (re-baselined only when a
   change to the math is heard and confirmed), cross-checked by native
   mode-equivalence (fused vs microkernel, both shipping). So the
   oracle needs no replacement; `Tropical/Plan.lean` (6a) remains the
   typed substrate a *verified* semantics could one day attach to, but
   that is proof work, not an oracle.
7. **Surface parser + markdown + printer + stdlib.** ✅ (landed).
   `compiler/parse/*` ported to `Tropical/Parse/Surface/*` (lexer,
   cursor, expressions, statements, declarations, bounds, markdown) +
   `Tropical/Parse/Nodes.lean` + `Tropical/Parse/Raise.lean` (the JSON
   ingest adapter). Gated by dual-parsing every `stdlib/*.md` and
   comparing ParsedProgram JSON, plus parse∘print fixpoint tests in
   Lean. The committed `stdlib/parsed/` bridge is now regenerated
   straight from the Lean surface parser (`make parse-all` =
   `diffcli parse-all`), retiring `scripts/build_parsed_stdlib.ts`.
8. **TS deletion + CI consolidation.** ✅ (landed). `compiler/` was
   deleted entirely; the TS MCP engine (`mcp/engine.ts`,
   `ir_service.ts`, `resources.ts`, `envelope.ts`,
   `program_format_example*`, `test_patch.ts`, `remove_feedback.test.ts`)
   was deleted — `mcp/` now holds only the two behavioral suites
   (`errors.test.ts`, `wire_dac.test.ts`) plus `CLAUDE.md` / `ERRORS.md`.
   `koffi` and the bun compiler-service subprocess left the server path.
   The differ scaffolding went with the oracle: `scripts/diff/` and the
   remaining one-off scripts (`build_parsed_stdlib`, `validate_stdlib`,
   `eval_program`, `capture_old_pipeline_audio`) were removed —
   `scripts/` is now empty — and every `make diff-*` + `make
   validate-write` target was dropped from the Makefile.
   The WASM survivors (`emit_wasm.ts` + `wasm_memory_layout.ts`, plus
   the shared `flat_plan` / `plan_types` / `slot_indices`) moved under
   `web/wasm/`; the browser fetches plans precompiled by the Lean engine
   (`diffcli compile`) and runs only the WASM emitter + runtime.
   The golden/equiv runners became `lake exe tropicaltest` (the native
   `tests/bench/` and the native equiv suites — `microkernel_vs_fused`,
   `nested_vs_inlined`, `microkernel_deep`, `hotswap_root`,
   `migration_audio` — were absorbed or retired). The surviving bun
   suites are the WASM≡JIT equivalence (`tests/web/`) and the MCP
   behavioral tests, both run against the Lean engine via
   `TROPICAL_ENGINE_CMD`. CI consolidated onto `make validate`
   (`tropicaltest` goldens + native mode-equiv, web build, `bun test`,
   `ctest`). **No differential gates remain** — correctness is anchored
   by frozen audio goldens + the developer's ear, per the interpreter
   decision above. Schemas unchanged throughout: `tropical_program_2`,
   `tropical_plan_5`.

## Top risks (all retired)

These were the risks that shaped the migration; each was handled by the
strategy noted, and none drew blood post-port:

1. **Emit divergence** (CSE/order → different plans/audio) — arena-id
   memos mirroring TS identity semantics; deterministic traversal; the
   audio differ localized divergence to a fixture and sample index.
2. **Phase 1 behavioral fidelity** — `mcp/*.test.ts` as executable
   spec, run unmodified against the Lean engine (still the case:
   `errors.test.ts` + `wire_dac.test.ts` run against `frontend --rpc`).
3. **Resolved codec infidelity** — property-gated before any consumer.
4. **FFI lifetimes** (finalizers vs. DAC audio thread) — held
   back-references, explicit dispose, ASan smoke.
5. **Dual-maintenance drag** — feature work froze per layer once its
   port phase started; the replaced entrypoint was deleted at phase
   exit. The dual-maintenance window is now closed: there is one stack.
