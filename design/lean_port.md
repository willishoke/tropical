# The Lean port — top-down migration of the compiler

Status: **Phase 6 complete — the production compiler is Lean,
end to end, and the TS compiler service is deleted.** The Lean engine
is the whole stack: raise (`Tropical/Parse/Raise.lean`), the
elaborator (`Tropical/Ir/Elaborator.lean`), the codec, all six strata
passes (`Tropical/Ir/Strata/*`), the Core downcast
(`Tropical/Ir/Core.lean`, now carrying resolved port types), emit
(`Tropical/Ir/Emit.lean` — the structural-CSE emitter over Core with
total matches), the per-program boundary
(`Tropical/Ir/CompileResolved.lean`), the partitioner + session
compile (`Tropical/Compile.lean` — two-phase slot preallocation incl.
the array-input alias quotient, recursive partitionKernel, remap,
sinks, slot metadata), the typed plan layer + wire codec
(`Tropical/Plan.lean`), the v2 ingest (load/merge walk the normalized
JsonV node engine-side), save/export (extraction + reconstruction /
the full exportSessionAsProgram port), catalog-entry rendering
(`Tropical/Entries.lean`), and the MCP resources/prompts
(`Tropical/Resources.lean`). `make mcp-lean` launches one binary; no
bun subprocess exists on any production path. The stdlib boots from
the pre-parsed bridge in manifest order.

Gates at phase exit, all green: `diff-plan` + `diff-audio` — the
phase headline — 8/8 patches structurally equal AND byte-for-byte
through the JIT across all three compilation modes (Lean `diffcli
compile` vs the TS oracle); `diff-engine` 8 scripts / 146 steps incl.
hex-compared `debug_render` audio after every mutation (`ingest.json`
added at 6d/6e to discriminate the moved load/merge/save/export
paths); `diff-emit` 43 outputs (per-program emit, errors byte-exact);
`diff-strata` 86 at K=5; `test-lean-engine` 67; the full bun suite;
`make validate` modulo the three documented stale goldens on
origin/main. Two more production defects surfaced and fixed (findings
#7 and the export-sessionSlot appendix note in
`design/bugs/lean_port_findings.md`). The TS engine
(`mcp/engine.ts` + `ir_service.ts`) and the TS compiler under
`compiler/` remain ONLY as the differential oracle until Phase 8.
This document is the roadmap; `design/architecture.md` stays the
authority on what the compiler *is*.

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
postmortem-style in `design/bugs/lean_port_findings.md` (append-only;
six findings so far).

Known main-branch issue surfaced by the render gate: the three
committed golden hashes (`tests/golden/*.hash`) are stale on
origin/main itself — both engine implementations agree with each other
and disagree with the goldens identically. Regenerate with
`make validate-write` once confirmed intentional.

## Shape of the migration

The MCP surface is already Lean: `lean/Main.lean` (Turnstile) validates
23 tools against typed schemas and relays them over newline JSON-RPC to
`mcp/ir_service.ts` → `mcp/engine.ts` → `compileSession` → plan JSON →
koffi → C++ engine. The port marches that relay boundary **top-down**:
engine/session first, elaborator last, surface parser after that.

End state: TypeScript is deleted. Lean owns the engine/session, the
whole compiler pipeline, and talks to the C++ engine through native
`@[extern]` FFI (`engine/c_api/tropical_c.h`). The C++ engine (LLVM
JIT, DAC) stays C++. Browser TS (`web/` host + worklet, `emit_wasm`
consuming plan JSON) survives as the web backend.

Design stance: **proof-ready, prove later.** Port unverified, but make
the Lean IR a thing proofs can later attach to: decl stores + typed-id
refs (the completion of the de Bruijn migration the TS elaborator
already made — refs are branded integer `idx` values into typed decl
tables), total functions, `Except`-shaped errors mapped onto the
`mcp/ERRORS.md` envelope taxonomy, and a plan-level reference
interpreter as the semantic anchor.

## Why this is tractable

- **ParsedProgram is a free seam.** `compiler/parse/nodes.ts` is plain
  JSON-serializable data (NameRef placeholders, no object identity),
  and `session_to_parsed.ts` already routes sessions through the single
  `elaborate` front door.
- **No plan-JSON byte comparison anywhere.** Golden gates hash rendered
  audio (`scripts/validate_stdlib.ts`), so float-formatting divergence
  between TS and Lean serializers cannot break a gate; only double
  round-trip fidelity matters.
- **Refs are already indices.** Post-elaboration identity is positional
  (decl tables), not pointer-based — a Resolved⇄JSON codec is
  mechanical.
- **The C FFI surface is tiny** (~27 opaque-pointer functions), and
  session-path params are slot-based (`param:name` module slots) — no
  native pointers inside plans.

## The harness (Phase 0, landed)

Three differs under `scripts/diff/`, all green TS-vs-TS from day one.
Each compares two *commands*; a Lean layer lands by becoming side B and
keeping the diff empty.

| target | sides implement | compares |
|---|---|---|
| `make diff-plan` | `<cmd> <patch> [--mode=<m>]` → plan JSON on stdout | structural: key-order-insensitive, numbers by IEEE-754 bits |
| `make diff-audio` | same contract | byte equality of 16×256 rendered samples per side, fresh runtime each |
| `make diff-engine` | ir_service protocol (newline JSON-RPC, method = tool) | normalized ToolResults per step of recorded scripts in `tests/fixtures/mcp_scripts/` |

The Lean half of the comparator is `Tropical.Json.diff`
(`lean/Tropical/Json.lean`).

## Phases

Each phase has a hard exit criterion: the relayed/TS entrypoint it
replaces is **deleted**, and the relevant differs + existing suites
(`bun test`, `ctest`, stdlib golden hashes, `tests/equiv/*`) are green.

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
   **Interpreter decision (recorded):** the Phase-2-deferred
   `Tropical/Plan/Interp` reference interpreter is re-deferred past
   Phase 6. While the TS oracle lives, the differential gates
   (diff-plan structural, diff-audio byte-for-byte through the real
   JIT) are strictly stronger than interpreter agreement; the
   interpreter's value is as the post-oracle semantic anchor, so it
   lands with Phase 8 (when bun/koffi leave and the equiv runners move
   to `lake exe tropicaltest`). `Tropical/Plan.lean` (6a) is the typed
   substrate it will attach to.
7. **Surface parser + markdown + printer + stdlib.** Port
   `compiler/parse/*` → `Tropical/Parse/*`; dual-parse every
   `stdlib/*.md`, compare ParsedProgram JSON; parse∘print fixpoint
   tests in Lean.
8. **TS deletion + CI consolidation.** `mcp/` and `compiler/` die
   except web survivors (`emit_wasm` + memory layout move under
   `web/`); equiv/golden runners become `lake exe tropicaltest`;
   bun/koffi leave the server path.

## Top risks

1. **Emit divergence** (CSE/order → different plans/audio) — arena-id
   memos mirroring TS identity semantics; deterministic traversal; the
   audio differ localizes divergence to a fixture and sample index.
2. **Phase 1 behavioral fidelity** — `mcp/*.test.ts` as executable
   spec, run unmodified against the Lean engine.
3. **Resolved codec infidelity** — property-gate in TS before use.
4. **FFI lifetimes** (finalizers vs. DAC audio thread) — held
   back-references, explicit dispose, ASan smoke.
5. **Dual-maintenance drag** — freeze feature work per layer once its
   port phase starts; delete the replaced entrypoint at phase exit.
