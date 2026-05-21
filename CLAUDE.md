# tropical

Realtime audio synthesis. The whole patch — every oscillator, filter,
envelope, and wire — compiles to a single per-sample kernel. There is no
runtime interpreter and no module boundary in the audio callback. Every
edit hot-swaps a fresh kernel; matching state transfers by name so
delays and oscillators don't click.

## Build

```bash
make build          # C++ core, outputs build/libtropical.dylib
make mcp-ts         # build + launch MCP server on stdio (requires Bun)
make validate       # build + bun test + ctest + stdlib validator
make clean          # remove build directories
```

**Requirements:** CMake 3.20+, C++20, LLVM ≥ 19 (Homebrew: `/opt/homebrew/opt/llvm`), Bun.

## Test

```bash
cmake --build build -j4 && ctest --test-dir build   # C++ tests (JIT + C API, no audio device)
bun test                                              # TS compiler tests
bun test --exclude compiler/apply_plan.test.ts        # pure-TS subset (no native FFI)
```

`apply_plan.test.ts` and the WASM-vs-JIT equivalence tests load
`build/libtropical.dylib` via koffi. Run `make build` first or use the
exclude form above.

## Ideological backbone

If you want a single sentence to hang the whole codebase off:

> tropical's IR is a DAG-shaped operad — programs are typed signal-flow
> graphs with cycles broken explicitly by a single state primitive
> (`RegDecl`), and the compiler is a functor between this operad and
> the slot-operational operad consumed by the runtime.

That's not load-bearing vocabulary you have to use day-to-day, but it
*is* the shape of the system: programs are graphs, parallel composition
is the cartesian product, sequential composition is graph wiring,
feedback is broken at the source-language layer (cycles in source code
must pass through an explicit user `reg` or `delay`; the elaborator
throws `CycleViolation` otherwise). The strata pipeline is what makes
this concrete: each pass takes a graph, retires some structure that's
already been consumed, and hands the next pass a smaller graph in the
same category. Backends interpret the final, fully-reduced graph into
different runtime targets.

In practical terms, every pass in `compiler/parse/`, `compiler/`, and
`compiler/ir/` is structure-preserving — it produces an IR that's
strictly poorer than its input, where the dropped structure is something
the next pass doesn't have to reason about. Reading the pipeline from
top to bottom:

```
.trop source / tropical_program_2 JSON / MCP mutations
  │
  │  parse  — drops layout, comments, sugar (`in [lo, hi]` → clamp/select)
  ▼
ParsedProgram (compiler/parse/nodes.ts)
  refs are NameRefNode placeholders; the parser does no scope analysis
  │
  │  raise  (legacy JSON ingest) — pass-through into the same parsed shape
  ▼
ParsedProgram
  │
  │  elaborate (compiler/ir/elaborator.ts) — drops names, enforces
  │              the acyclic-source invariant
  │              every NameRef is replaced by a direct decl-object pointer.
  │              inter-instance cycles in source code throw
  │              CycleViolation here (Tier-2 port-detailed error
  │              with a suggested explicit-delay fix).
  ▼
ResolvedProgram (compiler/ir/nodes.ts)
  DAG-shaped graph IR — cycles are not representable in a valid
  resolved program (they're rejected upstream by the elaborator
  or the session materializer). State lives in a single primitive
  `RegDecl { name, init, update? }`; `delay name = u init v` is
  surface sugar for `reg name { init: v, update: u }`.
  │
  │  strata pipeline (compiler/ir/strata.ts):
  │  ────────────────────────────────────────
  │   assertAcyclic    — confirms the caller honored the contract
  │   specialize       — drops type parameters
  │   sumLower         — drops sum types (variants → tag + scalar bundles)
  │   inlineInstances  — drops nesting (inner bodies lifted, _liftedFrom kept as provenance)
  │   arrayLower       — drops shapes and combinators (fold/generate/let/etc. unroll)
  │   identityElim     — categorical identity-law rewrite
  ▼
ResolvedProgram (post-strata)
  scalar-only · monomorphic · acyclic · non-nested · combinator-free.
  the smallest sub-IR sufficient for any per-sample evaluator.
```

Sessions (the MCP/runtime view of a graph in flight) plug into this
pipeline through one extra step that lifts a partially-typed session
graph into the same `ResolvedProgram` shape the per-program path
produces:

```
SessionState  (instances + wiring + dac.out + params)
  │
  │  materializeSession (compiler/ir/materialize_session.ts)
  │     lift a partially-typed session graph into a top-level
  │     ResolvedProgram. handles gateable wraps, paramDecl synthesis,
  │     session-level delay() extraction.
  ▼
ResolvedProgram (top-level synthetic)  →  strata pipeline  →  post-strata
```

**The IR is acyclic by construction.** Source-level cycles that
don't pass through an explicit user register are rejected at the
elaborator. Session-level cycles in MCP-built graphs are broken at
the wire layer: every wire stored via `setWireExpr` is wrapped in a
unit delay (`{op:'delay', args:[expr], init:0}`), and
`extractSessionDelays` (a compileSession pre-pass) hoists every
`delay()` op in any wire to a fresh session-level module slot
updated in the scheduler's `state_evolution` phase. Hand-written
JSON patches with cross-coupled instances must wrap their own
back-edges in `delay()` to break the session-level cycle —
`assertSessionAcyclic` (`compiler/ir/lowering/session_cycle_check.ts`)
runs as a defensive invariant at compileSession's entry. Every MCP
wire gains exactly one sample of latency (~21µs at 48kHz), matching
VCV Rack's per-wire-delay mental model.

## What sits below post-strata

Three *backends* consume the post-strata `ResolvedProgram`. They are
not further compiler stages — they are interpretations of the same
fully-reduced IR into different targets, and the equivalence test
suites assert they agree pointwise.

```
post-strata ResolvedProgram (per-program path)  /  SessionState (session path)
        │
        ├─→ compileSession (compiler/ir/compile_session.ts)
        │      liftWiresToInstances → extractSessionDelays →
        │      assertSessionAcyclic → compileSessionSlotted:
        │      per-instance compileResolved → tropical_plan_5 JSON
        │      (instance_functions[] + scheduler).
        │      ──── C API boundary (engine/c_api/tropical_c.h, koffi FFI) ────
        │      NumericProgramParser → FlatProgram (multi-function)
        │      OrcJitEngine → LLVM IR — one kernel function whose body is:
        │          for each sample:
        │            preamble (currently empty; reserved for future setup)
        │            for each instance: body + writebacks
        │            state_evolution (delay-slot WriteSlots)
        │            postamble (DAC stitch reads)
        │            output mix
        │      FlatRuntime → buffer loop, double-buffered hot-swap
        │      TropicalDAC (RtAudio) → audio output
        │
        ├─→ interpret_resolved (compiler/interpret_resolved.ts)
        │      pure-TS evaluator over a materialized ResolvedProgram.
        │      `materialize_session.ts` flattens the session for the
        │      oracle. Independent of the JIT structure.
        │
        └─→ emit_wasm (compiler/emit_wasm.ts + compiler/wasm_memory_layout.ts)
               tropical_plan_5 → WebAssembly bytes + linear-memory layout.
               Same per-sample sequencing as the C++ scheduler.
               compilePlan (web/host/compiler.ts)
               WasmRuntime (web/worklet/runtime.ts) — same hot-swap logic as FlatRuntime,
               state transfer by name, smoothstep fade
               AudioWorkletProcessor (web/worklet/processor.ts) → audio output
```

**Fixed-topology compilation.** Tropical compiles a session graph to
one monolithic kernel; the topology is fixed for the lifetime of the
kernel. Topology changes (adding/removing instances, rewiring) trigger
hot-swap to a freshly compiled kernel with state transferred by name.
There is no per-instance runtime gating — every instance runs every
sample, and the JIT fuses across instances aggressively. This is the
shape of synthesis the language is good at; dynamic-lifecycle
semantics belong in a different language with a different runtime.

Param handles are the only thing that differs between
backends. Wiring expressions reference parameters by name
(`{op:'param', name}` / `{op:'trigger', name}`); the materializer
resolves names to handles at compile time. For the JIT path the handle
is a native pointer (`tropical_param_t`); for the WASM path it's a
SAB slot index, stringified to keep the `tropical_plan_5` schema
backend-agnostic.

## Equivalence gates

The pipeline is correct only if every pass and every backend agrees
with the per-sample semantics on the input. Three test suites pin
that down by cross-checking outputs:

- `tests/equiv/jit_vs_interp_stdlib.test.ts` — JIT and
  `interpret_resolved` agree sample-for-sample across the stdlib corpus.
- `tests/equiv/wasm_vs_jit.test.ts` — WASM and JIT agree
  sample-for-sample.
- `tests/equiv/web_plans_vs_jit.test.ts` — every precompiled plan in
  `web/dist/patches/` matches the JIT output.

Any disagreement is a strata, materialize, or backend bug, and the
suite localises it.

## Schema versions

Two distinct JSON schemas; do not confuse them.

| Schema | Produced by | Purpose |
|--------|-------------|---------|
| `tropical_program_2` | `compiler/program.ts`, `compiler/parse/raise.ts` | The high-detail input shape: a program with typed ports, a body block of decls/assigns, optionally generic in `type_params`. Authored by humans (in `.trop`) or by agents (over MCP). |
| `tropical_plan_5`    | `compiler/ir/compile_session_slotted.ts` (`compiler/flat_plan.ts` schema) | The low-detail output: per-instance instruction streams plus a scheduler postamble (DAC stitch) and state-evolution phase (delay-slot updates). The C++ JIT and the WASM emitter both consume this shape. The engine still accepts the older `tropical_plan_4` (single-kernel form) for hand-crafted unit tests; it's lifted into a one-instance plan_5 at parse time. |

Going from the first to the second without losing meaning is exactly
what the strata pipeline does.

## Layout

```
compiler/             TS: parse → elaborate → strata → emit
  parse/              .trop surface syntax + JSON-ingest adapter (raise.ts)
  ir/                 strata pipeline + resolved-IR emit boundary
  runtime/            FFI bridge to C++ (koffi bindings, Runtime, DAC, Param)
engine/               C++: plan parsing, LLVM JIT, per-sample execution, audio output
  c_api/              Stable C API — the boundary between TS and C++
  jit/                LLVM ORC JIT engine
  runtime/            FlatRuntime (plan loading, kernel execution)
  dac/                Audio output (RtAudio)
mcp/                  MCP server — primary agent interface over stdio
web/                  WASM/browser backend — host (main thread), worklet (audio thread), build
patches/              Example patches (tropical_program_2 JSON)
stdlib/               32 .trop programs; see stdlib/README.md
tests/                Cross-cutting test surface (see tests/ for layout)
  equiv/              Cross-backend equivalence suites (the integration layer)
  bench/              Compile-time and runtime benchmarks
  fixtures/           Shared fixtures (flat_plan JSONs, equiv edge_cases)
  golden/             Golden hashes / migration goldens
design/               Architecture and design notes (architecture.md is authoritative)
```

Unit tests live next to the code they test (`compiler/**/*.test.ts`).
Tests that cross compilation/backend boundaries live under `tests/equiv/`.

## Conventions

- Commit messages: `type(scope): description` (e.g., `fix(jit):`, `feat(compiler):`, `refactor:`)
- Program types: PascalCase (`LadderFilter`, `OnePole`, `Clock`)
- Input/output names: lowercase (`freq`, `signal`, `out`, `saw`)
- C++ is header-heavy by design (templates, inlining for audio perf)
- JIT failures are fatal — no interpreter fallback on the audio path
  (`interpret_resolved` is an oracle for tests, not a runtime)
