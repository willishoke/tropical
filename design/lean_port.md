# The Lean port — top-down migration of the compiler

Status: **Phase 1 landed** — the engine (session state + 23 tool
handlers) is Lean (`lean/Tropical/Engine.lean`), gated by the protocol
test suites running unmodified against `frontend --rpc`
(`make test-lean-engine`) and by `make diff-engine` (TS oracle vs Lean,
41/41 recorded steps). The TS engine (`mcp/engine.ts` + `ir_service.ts`)
is off the production path but retained as the differential oracle;
its deletion is deferred to Phase 2 (it also backs two in-process test
suites). This document is the roadmap; `design/architecture.md` stays
the authority on what the compiler *is*.

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
2. **Native FFI + plan types + plan interpreter.** `Tropical/Ffi` +
   `lean/ffi/shim.c` over all of `tropical_c.h` (external classes with
   finalizers; the DAC object holds its Runtime alive; thread-local
   `tropical_last_error` captured in the shim). `Tropical/Plan`
   (FlatPlan/NInstr/SinkSpec/SourceSpec/RegTarget + wire codec, with
   round-trip-safe float emission). `Tropical/Plan/Interp` — pure
   plan-level reference interpreter (the proof-ready semantics; also a
   bun-independent oracle). Audio/param tools + `loadPlan` move to
   Lean; the TS service becomes stateless pure compile. Gate: stdlib
   golden hashes reproduced exactly through the Lean-owned runtime.
3. **Session lowering; seam lands on ParsedProgram.** Port
   `lift_wires`, `extract_session_delays`, `session_cycle_check`,
   `session_to_parsed`, slot pre-allocation. Service entrypoint:
   `compile_root {parsed_program} → plan`. Gate: Lean-vs-TS
   ParsedProgram structural compare per corpus entry.
4. **Elaborator front + decl stores + Resolved codec** (HIGH risk).
   Port `raise`, `schema`, `ir/nodes` (as decl stores), `elaborator`,
   diagnostics, `recursion`, `port_type`, `branded_names`. One
   Resolved⇄JSON codec (TS side keyed on decl-table positions),
   property-gated by round-trip + recompile-equality *before* any Lean
   consumer. Stdlib bridge: pre-parse `.md` → ParsedProgram JSON until
   Phase 7. Service shrinks to `strata` + `emit_root`.
5. **Strata pipeline.** Port the six passes (keep the `inlineNested`
   fractal path). Gate: per-pass differential over canonical resolved
   JSON; `nested_vs_inlined` / `microkernel_*` suites stay green.
6. **Emit + partition** (HIGH risk; strictest gates). Port `slots`,
   `compile_resolved`, `emit_resolved`, `partition_recursive`,
   `compile_session_slotted*`. CSE memos key on arena `ExprId`s that
   replicate TS identity semantics — no aggressive hash-consing until
   the TS oracle retires. The TS compiler service is **deleted**.
   Gate: `diff-plan` + `diff-audio` across the corpus × all three
   compilation modes; goldens pass without re-baselining.
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
