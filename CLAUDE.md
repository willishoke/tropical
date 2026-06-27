# tropical

Realtime audio synthesis. The whole patch — every oscillator, filter,
envelope, and wire — compiles to a single per-sample kernel. There is no
runtime interpreter and no module boundary in the audio callback. Every
edit hot-swaps a fresh kernel; matching state transfers by name so
delays and oscillators don't click.

## Build

```bash
make build          # C++ core, outputs build/libtropical.dylib
make lean           # Lean frontend + diffcli (the production compiler + MCP server)
make mcp-lean       # build C++ + Lean front door, then launch the MCP server
make validate       # build + lean + lake exe tropicaltest + web build + bun test + ctest
make parse-all      # regenerate stdlib/parsed/*.json from stdlib/*.md (Lean surface parser)
make clean          # remove build directories
```

**Requirements:** CMake 3.20+, C++20, LLVM 22 (Homebrew: `/opt/homebrew/opt/llvm`;
the JIT-only core builds on LLVM ≥ 19, but `make build`/CI and the in-process
wasm emitter target 22), Lean 4 (via elan), Bun (for the surviving behavioral
suites). The wasm emitter (`TROPICAL_WASM_EMIT`, on by default in `make build`)
also needs lld (`brew install lld`) and LLVM ≥ 21 for its Triple-based codegen API.

## Test

```bash
cmake --build build -j4 && ctest --test-dir build   # C++ tests (JIT + C API, no audio device)
./lean/.lake/build/bin/tropicaltest                 # audio goldens + native mode-equiv
TROPICAL_ENGINE_CMD="./lean/.lake/build/bin/frontend --rpc" bun test   # WASM≡JIT + MCP behavioral
```

The compiler is the Lean `frontend` binary — there is no TS compiler and no
koffi FFI subprocess. `tropicaltest` (the built binary at
`./lean/.lake/build/bin/tropicaltest`) drives the goldens and the native
realization-variant equivalence directly through the engine. The bun suites
are the surviving cross-backend gate (`tests/web`, wasm vs. JIT) and the MCP
protocol tests (`mcp/`), both run against the live Lean engine via
`TROPICAL_ENGINE_CMD`.

**Run these binaries directly — never via `lake exe`.** `libtropical`
hard-links Homebrew's `liblldWasm` (the wasm emitter), which in turn binds the
absolute `/opt/homebrew/opt/llvm/lib/libLLVM.dylib` (the one with the AMDGPU
target). `lake exe` forces the Lean toolchain's lib dir onto
`DYLD_LIBRARY_PATH` so the process can find `libleanshared`, and that shadows
`libLLVM.dylib` *by leaf name* with Lean's own bundled libLLVM (no AMDGPU
target) — so the binary dies at load with
`Symbol not found: _LLVMInitializeAMDGPUAsmParser`. The built binary instead
resolves `libleanshared` via rpath (which only applies to `@rpath/…` refs and
never shadows the absolute Homebrew libLLVM), so it runs clean. This is why
`make validate` invokes `./lean/.lake/build/bin/…` directly throughout and
never `lake exe`. (Not a version mismatch — pinning won't help; it's the
`lake exe` DYLD environment. Run the binary, not the wrapper.)

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

In practical terms, every pass under `lean/Tropical/Parse/` and
`lean/Tropical/Ir/` is structure-preserving — it produces an IR that's
strictly poorer than its input, where the dropped structure is something
the next pass doesn't have to reason about. Reading the pipeline from
top to bottom:

```
literate .md source / tropical_program_2 JSON / MCP mutations
  │
  │  parse  — drops layout, comments, sugar (`in [lo, hi]` → clamp/select)
  ▼
ParsedProgram (lean/Tropical/Parse/Nodes.lean)
  refs are NameRefNode placeholders; the parser does no scope analysis.
  surface syntax is a two-stage combinator-lexer + token-array parser
  under lean/Tropical/Parse/Surface/ (Lexer, Cursor, Expr, Statements,
  Declarations, Bounds, Markdown).
  │
  │  raise  (legacy JSON ingest, lean/Tropical/Parse/Raise.lean) —
  │         pass-through into the same parsed shape
  ▼
ParsedProgram
  │
  │  elaborate (lean/Tropical/Ir/Elaborator.lean) — drops names, enforces
  │              the acyclic-source invariant
  │              every NameRef is replaced by a direct decl-object pointer.
  │              inter-instance cycles in source code throw
  │              CycleViolation here (Tier-2 port-detailed error
  │              with a suggested explicit-delay fix).
  ▼
ResolvedProgram (lean/Tropical/Ir/Nodes.lean)
  DAG-shaped graph IR — cycles are not representable in a valid
  resolved program (they're rejected upstream by the elaborator
  or the session materializer). State lives in a single primitive
  `RegDecl { name, init, update? }`; `delay name = u init v` is
  surface sugar for `reg name { init: v, update: u }`.
  │
  │  strata pipeline (lean/Tropical/Ir/Strata.lean,
  │                   passes under lean/Tropical/Ir/Strata/):
  │  ────────────────────────────────────────
  │   assertAcyclic    — confirms the caller honored the contract
  │   specialize       — drops type parameters       (Specialize.lean)
  │   sumLower         — drops sum types (variants → tag + scalar bundles)  (SumLower.lean)
  │   inlineInstances  — drops nesting (inner bodies lifted, _liftedFrom kept as provenance)  (InlineInstances.lean)
  │   arrayLower       — drops shapes and combinators (fold/generate/let/etc. unroll)  (ArrayLower.lean)
  │   identityElim     — categorical identity-law rewrite  (IdentityElim.lean)
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
  │  compileSession (engine-side: lean/Tropical/{Engine,Compile,Lowering,Wiring}.lean)
  │     each instance is already a post-strata ResolvedProgram;
  │     liftWiresToInstances + extractSessionDelays normalize the wiring;
  │     then the slotted root-program lowering:
  │       buildSessionRoot: sessionToParsedProgram → elaborate
  │         (instances → InstanceDecls; per-wire delays → `delay` decls
  │          the elaborator folds into root RegDecls; the instances'
  │          already-resolved types are supplied via the elaborator's
  │          external-program-resolver hook — LINK, not re-elaboration)
  │       → partitionKernel → tropical_plan_5
  ▼
tropical_plan_5  (instance_functions[] + sinks[] + sources[])
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
the session-acyclicity check (`lean/Tropical/{Engine,Lowering}.lean`)
runs as a defensive invariant at compileSession's entry. Every MCP
wire gains exactly one sample of latency (~21µs at 48kHz), matching
VCV Rack's per-wire-delay mental model.

## What sits below post-strata

Two *backends* consume the post-strata IR (as `tropical_plan_5`). They
are not further compiler stages — they are interpretations of the same
fully-reduced plan into different targets, and the surviving equivalence
suite (wasm vs. JIT) plus the frozen audio goldens assert they
agree pointwise.

```
post-strata ResolvedProgram (per-program path)  /  SessionState (session path)
        │
        ├─→ compileSession (engine-side: lean/Tropical/{Engine,Compile,Lowering,Wiring}.lean)
        │      liftWiresToInstances → extractSessionDelays →
        │      session-acyclicity check → slotted root-program lowering:
        │      (sessionToParsedProgram → elaborate → partitionKernel);
        │      instance_functions = [root] with the session instances as
        │      nested children, delays as root RegDecl writebacks.
        │      ──── C API boundary (engine/c_api/tropical_c.h, lean/Tropical/Ffi.lean) ────
        │      NumericProgramParser → FlatProgram (multi-function)
        │      OrcJitEngine → LLVM IR — one kernel function whose body is:
        │          for each sample:
        │            for each instance_function (recursively: preamble,
        │              per-child {pre_input, child}, body, writebacks)
        │            for each sink: output[target] = gain · Σ slots[inputs]
        │      FlatRuntime → buffer loop, double-buffered hot-swap
        │      TropicalDAC (RtAudio) → audio output
        │
        └─→ compile-wasm (engine LLVM + lld, in-process: the *same* IR the
               JIT runs, lowered to wasm32 — there is no second emitter).
               Per patch the build ships <slug>.wasm + a trimmed
               <slug>.manifest.json (web/patches/*.json + `diffcli
               compile-wasm` → web/dist/patches/). The browser is a
               precompiled-patch player: it fetches .wasm + manifest and
               instantiates via the runtime package (web/runtime/) — no
               recompile, no hot-swap, no SharedArrayBuffer. Smoothstep
               fade, AudioWorkletProcessor → audio output.
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

With the TS implementation gone there is no second implementation to
diff against — a differential proves *agreement*, not correctness.
Correctness is anchored by **frozen audio goldens** (and the
developer's ear); the surviving cross-checks pin the two `tropical_plan_5`
backends and the JIT's own realization variants to those goldens:

- `tests/web/wasm_vs_jit.test.ts` — WASM and JIT agree sample-for-sample
  (the two backends, both off `tropical_plan_5`), run against the live
  Lean engine via `TROPICAL_ENGINE_CMD`.
- `tests/web/web_plans_vs_jit.test.ts` — every precompiled plan in
  `web/dist/patches/` matches the JIT output.
- `lake exe tropicaltest` (`lean/Tropicaltest.lean`) — byte-for-byte
  audio goldens (`tests/golden/`) plus the native realization-variant
  equivalence *within* the JIT (fused vs. per-instance microkernel,
  flat vs. nested), driven directly through the engine.

Any disagreement is a strata or backend bug. The former
`make diff-*` differential harness (Lean-vs-TS) is gone along with the
TS implementation; there are no longer differential gates.

## Schema versions

Two distinct JSON schemas; do not confuse them.

| Schema | Produced by | Purpose |
|--------|-------------|---------|
| `tropical_program_2` | `lean/Tropical/Parse/Raise.lean` (JSON ingest) and the surface parser under `lean/Tropical/Parse/Surface/` | The high-detail input shape: a program with typed ports, a body block of decls/assigns, optionally generic in `type_params`. Authored by humans (in literate `.md`) or by agents (over MCP). |
| `tropical_plan_5`    | `lean/Tropical/Compile.lean` (`lean/Tropical/Plan.lean` schema) | The low-detail output: a root instruction stream (instances nested as `children`) plus `sinks[]` (device-bound outputs: sum input slots × gain → channel) and `sources[]` (runtime-bound inputs: canonical `[tick, rate]`; the dual of sinks). The engine consumes it as the codegen manifest; the web build derives a `.wasm` + a trimmed `KernelManifest` from it. The engine still accepts the older `tropical_plan_4` (single-kernel form, top-level `output_targets` temp-mix) for hand-crafted unit tests; it's lifted into a one-instance plan_5 with the canonical sources at parse time. |

Going from the first to the second without losing meaning is exactly
what the strata pipeline does.

## Layout

The Lean `frontend` binary (`lean/.lake/build/bin/frontend`) is the
whole stack — compiler + session + runtime FFI + MCP server, one binary.

```
lean/                 Lean 4: the production compiler + MCP server (one binary)
  Main.lean           the MCP front door → the `frontend` binary
  Diffcli.lean        the `diffcli` CLI (compile / parse-all)
  Tropicaltest.lean   golden + native realization-variant equivalence runner
  ffi/                C shim to libtropical (shim.c, built by `make lean`)
  Tropical/
    Parse/            parse: Nodes, Raise (JSON ingest), OrderedJson
      Surface/        two-stage combinator-lexer + token-array surface parser
                        (Lexer, Cursor, Expr, Statements, Declarations, Bounds, Markdown)
    Ir/               elaborate → strata → emit
      Elaborator.lean   names → decl pointers; CycleViolation on cyclic source
      Strata.lean + Strata/{Specialize,SumLower,InlineInstances,ArrayLower,IdentityElim}
      Core, Nodes, Emit, CompileResolved, Codec, WireProgram, Recursion
    Engine, Session, Compile, Lowering, Wiring   engine-side session compile
    Plan.lean         tropical_plan_5 schema
    Ffi.lean          FFI bridge to libtropical (Runtime, DAC, Param)
    Tools, Rpc, Relay, Resources, Frontend, …    MCP tool surface + RPC
engine/               C++: plan parsing, LLVM JIT, per-sample execution, audio output
  c_api/              Stable C API — the boundary between Lean and C++
  jit/                LLVM ORC JIT engine
  runtime/            FlatRuntime (plan loading, kernel execution)
  dac/                Audio output (RtAudio)
mcp/                  MCP behavioral tests (errors.test.ts, wire_dac.test.ts) + docs (CLAUDE.md, ERRORS.md)
web/                  WASM/browser backend (precompiled-patch player)
  runtime/            extractable runtime package: KernelManifest + layout + WasmKernel
  patches/            curated source patches; build_patches.ts → .wasm + manifest via `diffcli compile-wasm`
  dist/patches/       precompiled <slug>.wasm + <slug>.manifest.json
patches/              Example patches (tropical_program_2 JSON)
stdlib/               literate .md programs (see stdlib/README.md); stdlib/parsed/ is the committed parse bridge
tests/                Cross-cutting test surface
  web/                WASM≡JIT + precompiled-plan equivalence (run vs. the Lean engine)
  fixtures/           Shared fixtures (flat_plan JSONs, surface/elab/raise/mcp cases)
  golden/             Audio golden hashes / migration goldens
design/               Architecture and design notes (architecture.md is authoritative)
```

Unit tests for the compiler live inside the Lean tree (`lake exe tropicaltest`).
The behavioral bun suites (`tests/web`, `mcp/`) run against the live Lean
engine via `TROPICAL_ENGINE_CMD`.

## Conventions

- Commit messages: `type(scope): description` (e.g., `fix(jit):`, `feat(lean):`, `refactor:`)
- Program types: PascalCase (`LadderFilter`, `OnePole`, `Clock`)
- Input/output names: lowercase (`freq`, `signal`, `out`, `saw`)
- C++ is header-heavy by design (templates, inlining for audio perf)
- JIT failures are fatal — no interpreter fallback on the audio path
  (cross-backend agreement is gated by `tests/web/wasm_vs_jit`)

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
