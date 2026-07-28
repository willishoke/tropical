# tropical

Realtime audio synthesis. The whole patch — every oscillator, filter,
envelope, and wire — compiles to a single per-sample kernel. There is no
runtime interpreter and no module boundary in the audio callback. Every
kernel is closed-form: a pure function `f(τ, params)` of a time
coordinate, with no per-sample state. Parameter edits write live slots;
structural and topology edits hot-swap a fresh kernel at the current
coordinate. Nothing clicks because there is no hidden DSP state to carry —
oscillator phase is computed from the time coordinate, not latched.

## Build

```bash
make build          # C++ core, outputs build/libtropical.dylib
make lean           # Lean frontend + diffcli (the production compiler + MCP server)
make mcp-lean       # build C++ + Lean front door, then launch the MCP server
make validate       # build + lean + tropicaltest + web build + bun test + ctest
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
> graphs with cycles rejected outright (there is no state primitive to
> break them through), and the compiler is a functor between this operad
> and the slot-operational operad consumed by the runtime.

That's not load-bearing vocabulary you have to use day-to-day, but it
*is* the shape of the system: programs are graphs, parallel composition
is the cartesian product, sequential composition is graph wiring,
feedback is forbidden at the source-language layer (any cycle is a
compile error at the boundary that constructs the graph — the session
compile, export's direct construction (`Ir/Cycles.lean`) — there is no
`reg`/`delay` escape hatch; recursive filtering of live/external input
is the ceded island, deferred to a future stateful sister runtime). The lowering is
what makes this concrete — and it is DIRECT, not a pipeline: the
authoring surface (`Sig`, fourteen constructors) is already the trunk
IR, so there is nothing left to progressively retire. The historical
five-pass strata drop sequence (specialize → sumLower →
inlineInstances → arrayLower → identityElim) was retired 2026-07-25:
four of the five passes had no live producer for the structure they
existed to erase — the literate surface parser and generics that
produced it are gone, and `Sig` cannot spell it (a type-level fact).
Backends interpret the lowered graph into different runtime targets.

Reading the path from top to bottom:

```
arrow-combinator builders (Tropical.Stdlib / EmitArrow)  ·  MCP patch graphs  ·  tropical_program_2 JSON (load / export)
  │
  │  There is no surface language: the literate .md parser was retired,
  │  and the ELABORATOR WITH IT (2026-07-26) — there is no ParsedProgram
  │  and no name-resolution pass. The stdlib and new instruments are
  │  authored as arrow builders that `assemble` DIRECTLY into the
  │  resolved-IR DAG (EmitArrow.Sig → Nodes). The JSON front door is a
  │  PATCH BAY: `normalizeProgramFile` (lean/Tropical/Parse/Raise.lean,
  │  schema check + node/metadata split) → the session ingest
  │  (Engine/ProgramIO.lean) walks the node directly — instances +
  │  wiring + params of REGISTERED types. Ingest is the refusal site:
  │  a programDecl (a program body over the wire) dies with the
  │  retirement message; wire expressions are a TYPED inductive
  │  (Tropical.WireExpr — the decoder is the refusal site; no
  │  combinator/binder/state-op spellings exist) —
  │  the grammar you can spell is the language that compiles.
  ▼
ResolvedProgram (lean/Tropical/Ir/Nodes.lean)
  DAG-shaped graph IR — cycles are not representable in a valid
  resolved program (they're rejected at every constructing boundary
  via Ir/Cycles.lean). There is no state primitive: kernels
  are closed-form `f(τ, params)`. (`reg`/`next`/`delay` are gone —
  there is no IR node for them and no way to author one; recursive
  feedback is the ceded island, deferred to a future stateful sister runtime.)
  │
  │  the direct lowering (lean/Tropical/Ir/Strata.lean — two named
  │  rewrites and a type boundary, not a pass pipeline):
  │  ────────────────────────────────────────
  │   assertAcyclic    — confirms the caller honored the contract
  │   inlineInstances  — OPTIONAL (opts.inlineNested): inner bodies lifted
  │                      in place; the fractal session path skips it and
  │                      keeps instances as kernel boundaries  (InlineInstances.lean)
  │   identityElim     — categorical identity-law peephole  (IdentityElim.lean)
  │   toResolved       — the reachability GC (Strata/EArena.lean): copy the
  │                      evaluator-reachable graph into a fresh arena of the
  │                      SAME vocabulary (there is one expression type,
  │                      `ENode`; the former CNode/CoreArena twin dissolved
  │                      into it). No refusal remains here — retired
  │                      constructors are refused at raise and are
  │                      unspellable in the IR. Combinator programs are
  │                      authored in Lean — the host language is the
  │                      meta-level (Array.map builds coefficient tables at
  │                      assemble time); only structure a backend wants
  │                      AS DATA earns an IR node, the way `bankSum` did.
  ▼
ResolvedProgram (lowered)
  scalar-only · monomorphic · acyclic · non-nested (inline path) ·
  DECISION-FREE: no combinator exists except `bankSum`, the bounded
  reduction that is itself the normal form for uniform indexed families
  (modal banks, reverbs, partial banks). Whether a bank is REALIZED as a
  `bankSum` region or unrolled is chosen at AUTHORING time, by the arrow
  modal builder reading the one shared flag (`Ir/BanksFlag.lean`,
  `TROPICAL_BANKS_UNROLL` — a debugging bisection, banked is the
  default); backends realize whatever arrives. Order preservation makes
  every realization bit-identical, floats included (no associativity
  precondition) — a theorem since slice 3c (`EmitArrow/BankOrder.lean` +
  `Ir/EmitBankLaws.lean`; trusted base = one named assumption,
  `REDUCE_REGION_EXECUTES_IN_ARRAY_ORDER`). The waist of the
  hourglass: the smallest sub-IR sufficient for any per-sample
  evaluator — and, because `Sig` is this same constructor set, the
  authoring layer and the trunk are ONE vocabulary with the `assemble`
  seam between them.
```

Sessions (the MCP/runtime view of a graph in flight) reuse the
per-program pipeline at the instance level — each instance type is
lowered once at load — and build the whole session
**directly into one synthetic root `ResolvedProgram`**
(`sessionToResolvedRoot`, no ParsedProgram round-trip), lowered
through the same fractal path:

```
SessionState  (instances + wiring + dac.out + params)
  │
  │  compileSession (engine-side: lean/Tropical/{Engine,Compile,Lowering,Wiring}.lean)
  │     each instance is already a lowered ResolvedProgram;
  │     liftWiresToInstances normalizes the wiring;
  │     then the slotted root-program lowering:
  │       sessionToResolvedRoot: the session graph is already
  │         post-elaborate-shaped, so the resolved root `Program` is
  │         built DIRECTLY — instances → InstanceDecls linking each
  │         instance's already-resolved type snapshot, wires → Ir.Expr
  │       → partitionKernel → tropical_plan_5
  ▼
tropical_plan_5  (instance_functions[] + sinks[] + sources[])
```

**The IR is acyclic by construction.** There is no register to break a
cycle through, so every boundary that constructs a graph rejects them
(`Ir/Cycles.lean`): the session-acyclicity check
(`lean/Tropical/{Engine,Lowering}.lean`) is a plain "no cycles at all"
rule, run as an invariant at compileSession's entry, and
`export_program`'s direct construction enforces the same contract. Nothing breaks cycles for you — there are no
per-wire delays, so wires add no latency, and a back-edge (in MCP
mutations or hand-written JSON) is a compile error, not a one-sample
feedback path. Recursive feedback on live/external input is the ceded
island, deferred to a future stateful sister runtime.

## What sits below the waist

Three execution targets consume the same typed `FlatPlan`; they are target
interpretations, not more source-language stages.

```
tropical_plan_5 / FlatPlan
  ├─ EmitLlvm → textual LLVM → OrcJitEngine::compile_ir_text → FlatRuntime
  ├─ EmitLlvm → the same LLVM → wasm32 TargetMachine + lld → browser player
  └─ EmitMsl  → MSL → MetalKernel → supported Apple live audio
```

The JIT uses f64 values plus i64 rails and is the native reference and scope
path. WebAssembly shares the LLVM f64/i64 semantics. Metal uses an f32 value
path plus the exact i64 clock rail; on Metal sessions the JIT remains
dual-loaded for `render_window` and reference comparisons.

`NumericProgramParser` is now a manifest reader. Lean owns code generation:
the C++ runtime receives textual LLVM plus plan metadata, compiles the text, and
publishes the resulting kernel. The parser's instruction graph does not
generate native code.

**Fixed-topology compilation.** Topology and structural selector changes
rebuild the synthetic root, lower, emit, and hot-swap. Ordinary parameter
changes write `param:<name>` slots without relowering. `FlatRuntime` carries
only `sample_index` across publication; it does not transfer registers, arrays,
or slots by name. Current parameter values come from the session/control layer
and the fresh plan defaults.

There is no per-instance runtime gating: every included instance runs every
sample. Current timing data, split by parameter writes versus structural
recompiles, lives in `benchmarks/current_baseline/findings.md`.

## Equivalence gates

With the TS implementation gone there is no second implementation to
diff against — a differential proves *agreement*, not correctness.
Correctness is anchored by **frozen audio goldens**; the surviving
cross-checks pin target refinements and the JIT's realization variants:

- `tests/web/wasm_vs_jit.test.ts` — WASM and JIT agree sample-for-sample,
  run against the live Lean engine via `TROPICAL_ENGINE_CMD`.
- `tests/web/web_plans_vs_jit.test.ts` — every precompiled plan in
  `web/dist/patches/` matches the JIT output.
- `tests/web/metal_vs_jit.test.ts` — Metal meets the documented f32
  tolerance/SNR boundary against the f64 JIT.
- `tropicaltest` (`lean/Tropicaltest.lean`, run as the built binary) — byte-for-byte
  audio goldens (`tests/golden/`) plus the native realization-variant
  equivalence *within* the JIT (fused vs. per-instance microkernel,
  flat vs. nested), driven directly through the engine.

Any disagreement is a lowering or backend bug, but agreement alone does not
prove source semantics; see `design/trust-boundary.md`. The former
`make diff-*` differential harness (Lean-vs-TS) is gone along with the
TS implementation; there are no longer differential gates.

## Schema versions

Two distinct JSON schemas; do not confuse them.

| Schema | Produced by | Purpose |
|--------|-------------|---------|
| `tropical_program_2` | `lean/Tropical/Parse/Raise.lean` (JSON ingest) | The PATCH-BAY shape: instances of registered types + wiring + params, a body block of instanceDecls/paramDecls/outputAssigns. The JSON front door for `load`/`merge` (and `save`/`export_program`'s output serialization). Program DEFINITIONS over the wire are retired — a programDecl is refused at ingest with the retirement message; wire expressions decode into the typed session grammar (`Tropical.WireExpr`), which cannot spell combinators/binders/state ops. (Programs are authored as `Tropical.Stdlib`/EmitArrow arrow builders, not this schema.) |
| `tropical_plan_5`    | `lean/Tropical/Compile.lean` (`lean/Tropical/Plan.lean` schema) | The low-detail output: nested instance blocks plus `sinks[]`, `sources[]`, typed slots, stage metadata, and optional bank regions. Lean emits LLVM/MSL from the typed plan; native and web hosts consume trimmed metadata manifests. The native C load APIs retain a bounded `tropical_plan_4` metadata lift for direct callers, but current Tropical never emits it and legacy state keys are ignored. See `design/compatibility-matrix.md`. |

Going from the first to the second without losing meaning is exactly
what the session compile does (ingest → sessionToResolvedRoot →
inline → identityElim → toResolved → partitionKernel). A
`tropical_program_2` file that carries a program body (`programDecl`)
or spells a retired construct in a wire (`fold`/`scan`/…, generics,
state ops) does not load: ingest refuses it with the retirement
message, and the session wire grammar has no spelling for the rest.
Nothing past the codec can carry it.

## Layout

The Lean `frontend` binary (`lean/.lake/build/bin/frontend`) is the
whole stack — compiler + session + runtime FFI + MCP server, one binary.

```
lean/                 Lean 4: the production compiler + MCP server (one binary)
  Main.lean           the MCP front door → the `frontend` binary
  Diffcli.lean        the `diffcli` CLI (compile / compile-wasm / render / emit-ir / emit-msl)
  Tropicaltest.lean   golden + native realization-variant equivalence runner
  ffi/                C shim to libtropical (shim.c, built by `make lean`)
  Tropical/
    Parse/            JSON ingest: Raise (normalizeProgramFile — the patch-bay
                        front door), Nodes (ScalarKind), OrderedJson (JsonV)
                        (the surface parser AND the elaborator are retired)
    Stdlib.lean       the stdlib as 15 arrow-combinator builders — the boot chain
    EmitArrow/        the arrow authoring substrate (Sig, Term, Numerics, Patch, Modal, Gong)
    Ir/               lower → emit (authoring assembles directly; no elaborate)
      Strata.lean (the direct lowering) + Strata/{Basic,EArena,InlineInstances,IdentityElim}
      Core, Nodes, Cycles, Emit, CompileResolved, Codec, WireProgram
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
tests/                Cross-cutting test surface
  web/                WASM≡JIT + precompiled-plan equivalence (run vs. the Lean engine)
  fixtures/           Shared fixtures (flat_plan JSONs)
  golden/             Audio golden hashes + the per-program stdlib wire+port goldens (golden/stdlib/)
design/               Architecture and design notes (architecture.md is authoritative)
```

Unit tests for the compiler live inside the Lean tree (the `tropicaltest` binary).
The behavioral bun suites (`tests/web`, `mcp/`) run against the live Lean
engine via `TROPICAL_ENGINE_CMD`.

## Conventions

- Commit messages: `type(scope): description` (e.g., `fix(jit):`, `feat(lean):`, `refactor:`)
- Program types: PascalCase (`FixedSinOsc`, `SoftClip`, `ModalVoice`)
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
