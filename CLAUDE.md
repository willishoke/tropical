# tropical

Realtime audio synthesis. The whole patch — every oscillator, filter,
envelope, and wire — compiles to a single per-sample kernel. There is no
runtime interpreter and no module boundary in the audio callback. Every
edit hot-swaps a fresh kernel; matching state transfers by name so
delays and oscillators don't click.

## Build

```bash
make build          # C++ core, outputs build/libtropical.dylib
make mcp-lean       # build C++ + Lean front door, then launch the MCP server
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

**If equivalence tests fail unexpectedly, clear the JIT cache first.**
`rm -rf ~/.cache/tropical/kernels` — the LLVM-IR cache is keyed by an MD5
of the serialized plan plus the dylib's build-id, but a fix that changes
*runtime* behavior without changing the emitted plan (e.g., the engine
emits an existing instruction field that the parser previously ignored)
can leave stale kernels in place. Clearing the cache is cheap; chasing
a phantom regression is not.

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
literate .md source / tropical_program_2 JSON / MCP mutations
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

Sessions (the MCP/runtime view of a graph in flight) reuse the
per-program pipeline at the instance level — each instance type is
elaborated and strata-processed once at load (`resolveProgramType`) —
and, by default, serialize the whole session **back into a
`ParsedProgram` and run it through the SAME `elaborate` front door** the
surface path uses, producing one synthetic root `ResolvedProgram`
lowered through the same fractal path:

```
SessionState  (instances + wiring + dac.out + params)
  │
  │  compileSession (compiler/ir/compile_session.ts)
  │     each instance is already a post-strata ResolvedProgram;
  │     liftWiresToInstances + extractSessionDelays normalize the wiring;
  │     then compileSessionSlotted (default = root-program lowering):
  │       buildSessionRoot: sessionToParsedProgram → elaborate
  │         (instances → InstanceDecls; per-wire delays → `delay` decls
  │          the elaborator folds into root RegDecls; the instances'
  │          already-resolved types are supplied via the elaborator's
  │          ExternalProgramResolver hook — LINK, not re-elaboration)
  │       → partitionKernel → tropical_plan_5
  ▼
tropical_plan_5  (instance_functions[] + sinks[])
```

**The IR is acyclic by construction.** Source-level cycles that
don't pass through an explicit user register are rejected at the
elaborator. Session-level cycles in MCP-built graphs are broken at
the wire layer: every wire stored via `setWireExpr` is wrapped in a
unit delay (`{op:'delay', args:[expr], init:0}`), and
`extractSessionDelays` (a compileSession pre-pass) hoists every
`delay()` op in any wire to a fresh slot recorded in
`session.delaySlotRegistry` — which the root-program lowering
realizes as a root `RegDecl` (read-old/write-new register writeback)
inside the kernel. Hand-written
JSON patches with cross-coupled instances must wrap their own
back-edges in `delay()` to break the session-level cycle —
`assertSessionAcyclic` (`compiler/ir/lowering/session_cycle_check.ts`)
runs as a defensive invariant at compileSession's entry. Every MCP
wire gains exactly one sample of latency (~21µs at 48kHz), matching
VCV Rack's per-wire-delay mental model.

## What sits below post-strata

Two *backends* consume the post-strata IR (as `tropical_plan_5`). They
are not further compiler stages — they are interpretations of the same
fully-reduced plan into different targets, and the equivalence test
suites assert they agree pointwise.

```
post-strata ResolvedProgram (per-program path)  /  SessionState (session path)
        │
        ├─→ compileSession (compiler/ir/compile_session.ts)
        │      liftWiresToInstances → extractSessionDelays →
        │      assertSessionAcyclic → compileSessionSlotted:
        │      root-program (sessionToParsedProgram → elaborate → partitionKernel);
        │      instance_functions = [root] with the session instances as
        │      nested children, delays as root RegDecl writebacks.
        │      ──── C API boundary (engine/c_api/tropical_c.h, koffi FFI) ────
        │      NumericProgramParser → FlatProgram (multi-function)
        │      OrcJitEngine → LLVM IR — one kernel function whose body is:
        │          for each sample:
        │            for each instance_function (recursively: preamble,
        │              per-child {pre_input, child}, body, writebacks)
        │            for each sink: output[target] = gain · Σ slots[inputs]
        │      FlatRuntime → buffer loop, double-buffered hot-swap
        │      TropicalDAC (RtAudio) → audio output
        │
        └─→ emit_wasm (compiler/emit_wasm.ts + compiler/wasm_memory_layout.ts)
               tropical_plan_5 → WebAssembly bytes + linear-memory layout.
               Same per-sample sequencing as the C++ engine.
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

Params. Wiring expressions reference parameters by name
(`{op:'param', name}` / `{op:'trigger', name}`). The **session**
compiler resolves each to a `param:name` **module slot** read — the
control plane drives it via `setSlot`, and hot-swap transfers it by
name like any other slot (both session lowerings do this; the root
path threads it as `paramSlots` so the root kernel's `ParamRef` lowers
to the slot). The standalone **per-program** path instead binds params
to FFI handles — a native pointer (`tropical_param_t`) on the JIT, a
SAB slot index (stringified to keep `tropical_plan_5` backend-agnostic)
on the WASM path.

## Equivalence gates

The pipeline is correct only if every pass and every backend agrees
with the per-sample semantics on the input. The cross-checking suites:

- `tests/equiv/wasm_vs_jit.test.ts` — WASM and JIT agree
  sample-for-sample (the two backends, both off `tropical_plan_5`).
- `tests/equiv/microkernel_vs_fused.test.ts`,
  `tests/equiv/nested_vs_inlined.test.ts`,
  `tests/equiv/microkernel_deep.test.ts` — realization-variant
  differentials *within* the JIT: fused vs. per-instance microkernel,
  flat vs. nested. These exercise the kernel/slot layer that the
  JIT↔WASM pair shares.
- `tests/equiv/migration_audio.test.ts` — byte-for-byte audio goldens
  against frozen reference output.
- `tests/equiv/web_plans_vs_jit.test.ts` — every precompiled plan in
  `web/dist/patches/` matches the JIT output.

Any disagreement is a strata or backend bug, and the suite localises
it. (Property/invariant-based coverage to replace the former pure-TS
interpreter oracle is a planned follow-up; unit tests now render the
JIT directly via `renderFramesJit`.)

## Schema versions

Two distinct JSON schemas; do not confuse them.

| Schema | Produced by | Purpose |
|--------|-------------|---------|
| `tropical_program_2` | `compiler/program.ts`, `compiler/parse/raise.ts` | The high-detail input shape: a program with typed ports, a body block of decls/assigns, optionally generic in `type_params`. Authored by humans (in literate `.md`) or by agents (over MCP). |
| `tropical_plan_5`    | `compiler/ir/compile_session_slotted.ts` (`compiler/flat_plan.ts` schema) | The low-detail output: a root instruction stream (instances nested as `children`) plus `sinks[]` (device-bound outputs: sum input slots × gain → channel). The C++ JIT and the WASM emitter both consume this shape. The engine still accepts the older `tropical_plan_4` (single-kernel form, top-level `output_targets` temp-mix) for hand-crafted unit tests; it's lifted into a one-instance plan_5 at parse time. |

Going from the first to the second without losing meaning is exactly
what the strata pipeline does.

## Layout

```
compiler/             TS: parse → elaborate → strata → emit
  parse/              literate surface syntax + JSON-ingest adapter (raise.ts)
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
stdlib/               33 literate .md programs; see stdlib/README.md
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
  (cross-backend agreement is gated by `tests/equiv/wasm_vs_jit`)

## Don't bake in audio-specific assumptions where it's free not to

Tropical's v2 roadmap includes video synthesis, control-rate signals,
and multi-backend codegen at independent rates. Most of the
infrastructure to support that doesn't need to exist yet — but where
it costs nothing to be neutral, prefer neutral. The asymmetry: small
unforced assumptions today compound into many call-site refactors
when a second signal type or rate or backend arrives.

This is *hygiene*, not abstraction. Don't build generic frameworks
ahead of need; do avoid accidentally precluding generality.

Concrete forms:
- **Naming.** If a parameter is `rate` (just a rate), don't name it
  `sampleRate` — that bakes in "audio." If a slot's element type is
  configurable, don't name the field `audioValue`. Names that don't
  forbid generality cost nothing extra to write.
- **Element types.** Don't hardcode `double` as a slot/wire/array
  element type where the type is already parameterized or could
  trivially be. Audio-rate-double-precision is what v1 uses, but
  there's no reason for that to be load-bearing at every layer.
- **Hardware/timing assumptions.** Don't assume "44100" or "48000"
  or "block of 256 samples" in places where the actual sample rate
  / block size is already available as a parameter. The audio
  callback knows its own rate; pass it, don't redeclare it.
- **Cross-cutting concepts.** "Signal" and "wire" and "slot" are
  rate- and color-agnostic at the conceptual level. Keep them that
  way in code where it's free. The same goes for "kernel",
  "instance", "port" — these are not "audio kernels" / "audio
  ports" / "audio instances" by nature, they're nature-agnostic.

What this isn't a license to do: build abstract frameworks, define
type-level operad machinery, parameterize over backend types
speculatively. The cost-asymmetry is on **unforced** assumptions —
the things you'd be neutral about anyway if you weren't on autopilot.
Anything that requires a deliberate design choice (a new type
parameter, a new module, a new abstraction layer) should wait until
there's concrete evidence the abstraction pays off — typically: a
second concrete consumer.
