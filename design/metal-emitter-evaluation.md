# Metal emitter — scoping evaluation

What it would take to add a Metal (MSL) backend alongside the LLVM emitter, so heavy
closed-form patches can run realtime on the GPU (the win the `benchmarks/gpu_time_partition`
study measured). Grounded in a read of the current emitter, plan, runtime, and the wasm
precedent.

## Verdict up front

The plumbing is a bounded, well-shaped sibling of the existing emitter — smaller than it
looks, because the op surface is tiny and the transcendentals are pre-inlined. The **real
decision is not the plumbing: it's f64.** Metal is f32-native; tropical's entire numeric
model and *both* equivalence gates are f64/i64 bit-exact. A Metal backend cannot be
bit-exact, so it can't join the existing correctness harness — it needs its own
tolerance-based correctness story. That is the load-bearing cost, and it should be decided
deliberately, not discovered mid-port.

> **Status update (scope A, `fixed-carrier` branch, 2026-07): the f64 obstacle has been
> deliberately shrunk.** The sample datapath is now integer where it matters: the clock
> and every phase are i64/ℤ-2³² (they always were on the stdlib path; the modal island
> joined them), the sine is a Q2.30 integer polynomial (`stdlib/FixedSin.md` /
> `fixedSinCycSig`), modal envelope×weight products land once per mode in Q4.28, and
> mode/tap sums are modular i64 adds. Every multiply in that datapath is
> 32×32→64-decomposable (MSL `mulhi`/`mullo`), so **byte-identical CPU/GPU is achievable
> for the datapath**, and the bit-exact golden regime survives the port. The residual
> float surface — the only part needing a tolerance gate — is enumerated and small:
> the envelope `exp` (bounded argument by construction), the per-param landings
> (`toInt(x·2³²)` / `toInt(x·2²⁸)`), and the one DAC scale (`toFloat/2³⁰`). The one
> genuine-i64 (non-32×32) op is the sum accumulator: Apple-silicon MSL has 64-bit
> integer add; a 32-bit pair-with-carry is the fallback. See `design/fixed-carrier.md`
> for the Q-format ledger and gates. The section below is kept as written for the
> pre-scope-A analysis it records.

## The current shape (what a Metal backend mirrors)

- **One codegen, in Lean.** `lean/Tropical/Ir/EmitLlvm.lean` — `emitKernel : FlatPlan →
  Except String String` — emits LLVM IR *text*. The C++ engine is a pure consumer
  (`OrcJitEngine::compile_ir_text` parses+JITs; `compile_ir_to_wasm` parses+retargets).
- **The JIT and wasm share the IR text.** wasm reuses it *because wasm is an upstream LLVM
  target* — same `parseIR`, different triple + lld. This reuse **does not transfer to
  Metal**: Metal is not an LLVM target, so none of the `TargetMachine`/`addPassesToEmitFile`
  path applies. The shared surface for a Metal backend moves up one level, to the
  `FlatPlan` itself.
- **Kernel shape.** `@tropical_kernel` is one flat function with an outer per-sample loop;
  inside, a fractal instance walk (preamble → children → body) emits instruction-by-
  instruction; sinks compute `output[s] = gain · Σ slots[inputs]`. Time is a `source`:
  `tick → start_sample_index + s`, `rate → sampleRate`.
- **Host boundary.** The DAC (`TropicalDACImpl<AudioSource>`) is a duck-typed template —
  a new runtime exposing `process/outputBuffer/getBufferLength/fade` drops in free. But
  runtime↔kernel is hardwired: `FlatRuntime` is concrete, stores a `NumericKernelFn`
  fn-ptr, no `IRuntime` interface. The convention (proven by the TS `WasmKernel`) is that
  a new backend gets its own **sibling host** re-implementing the ABI, not a shared base.

## What's genuinely easy

1. **The op set is tiny.** ~37 string tags (arithmetic, comparisons, logical/bitwise,
   unary, casts, `Ldexp`, `Clamp`, `Select`, array `Pack`/`SetElement`/`Index`/elementwise,
   `WriteSlot`). All mechanical to map to MSL.
2. **Only 5 hardware math intrinsics** — `sqrt`, `floor`, `ceil`, `round`, `fabs` → all in
   `metal_stdlib`. **Transcendentals are already gone:** `sin`/`cos`/`exp`/`pow` are
   literate `.md` programs inlined to pure arithmetic by the `inlineInstances` strata pass
   *before* emission. A Metal emitter inherits that for free — no accuracy fight over
   `metal::sin`.
3. **The per-sample→per-thread transform is already validated.** The MSL kernel is the loop
   *body* keyed on `thread_position_in_grid` = sample index — exactly the dispatch the
   `gpu_time_partition` benchmark proved realtime-feasible on UMA. This is the one real
   semantic change, and it's known to work.
4. **MSL emission is arguably simpler than LLVM emission** — structured C-like text, no SSA
   / basic-block bookkeeping, thread-private locals instead of the i64-backed temp pool.
5. **The DAC seam is free**, and the slot/hot-swap/fade logic (~50 lines) is backend-
   independent copy.
6. **The wasm precedent gives the scaffolding shape** (not the codegen reuse): one CLI verb,
   one FFI extern, one C-API load path, a trimmed manifest, a runtime package — additive,
   no refactor of existing code.

## The load-bearing obstacle: f64 vs f32

- Everything is **f64 (`double`)**; temps are **i64**-backed (bitcast); slots are native
  `double[]`. Metal has **no `double`** — Apple GPUs are f32-native (f64 absent/emulated,
  i64 limited).
- Consequences, in order of severity:
  - **Bit-exactness breaks.** An f32 Metal kernel will not agree sample-for-sample with the
    JIT/wasm. It cannot join `tests/web/wasm_vs_jit` (sample equality) or the frozen f64
    audio goldens (hash equality). The *entire* existing correctness floor assumes one
    shared bit-exact numeric semantics.
  - **A new correctness regime is required.** Metal needs a tolerance gate — ULP or dB-error
    bounds against the f64 reference — not hash equality. That gate does not exist today; it
    is genuinely new work, and it is a philosophical shift for a project whose floor is
    bit-exact goldens.
  - **i64 bit-tricks need rework.** `Ldexp`/`FloatExponent` and the i64-backed temp model
    assume f64 bit layout + i64 ops. Re-express for f32 (simpler: drop the i64 pool, use
    thread-private f32 locals; extract f32 exponent bits via `as_type<uint>`).
  - **Software f64 is not the answer.** Double-float (pair-of-f32) is ~10–20× slower and
    kills the performance that motivated the port.
- **Net:** adding Metal means accepting a *second numeric regime*. Not a blocker — but the
  actual decision, and everything else is downstream of it.

## Phased plan

- **Phase 0 — decide the numeric contract (do first).** Commit to f32 + a tolerance gate
  (ULP/dB vs the f64 reference), accepting non-bit-exactness as the price of GPU. Build a
  small comparison harness. *This gates whether the rest is worth starting.*
- **Phase 1 — `EmitMsl.lean`.** Sibling of `EmitLlvm` over the same `FlatPlan`: the same
  three parts (`resolveOperand`, `emitOp`, fractal walk + sinks), but a per-thread kernel
  body instead of a per-sample loop. ~37 ops, 5 intrinsics, f32 locals. Bounded (EmitLlvm
  is 562 lines; the MSL sibling is comparable or smaller). Gate with a golden-MSL text test.
- **Phase 2 — host `MetalRuntime`.** Sibling of `FlatRuntime` satisfying the DAC concept;
  `process()` is the pipelined dispatch from the benchmark, productionized (ring +
  async-completion). `slots`→`MTLBuffer` (shared), `output`→`MTLBuffer`, `temps`/`registers`
  thread-private. Hot-swap via `newLibraryWithSource`. New C entry point
  (`tropical_runtime_load_metal`) or introduce the missing `IRuntime` seam.
- **Phase 3 — correctness + integration.** The tolerance gate vs f64 goldens; `diffcli
  compile-metal` verb + a `TROPICAL_METAL_EMIT` CMake gate (mirror `TROPICAL_WASM_EMIT`);
  the reserved-GPU/contention robustness test flagged by the benchmark.

**Size:** the emitter is the biggest chunk but bounded by the small op set; the host runtime
is small (DAC seam clean, dispatch already prototyped); the tolerance harness is the new
work. It is a *medium-additive sibling plus one new correctness regime* — not a rewrite. The
wasm backend was "small additive"; Metal is that plus the emitter-from-scratch and the f32
gate.

## Recommended de-risking step before committing

Don't build `EmitMsl` to find out if it's worth it. First, hand-translate **one genuine
heavy patch** (a fat modal sum / deep arrow graph — above the benchmark's `K·B ≈ 10⁵`
crossover) to MSL by hand, run it on the GPU, and compare against the f64 reference for both
(a) speedup and (b) error. That single experiment answers the two open questions — *is the
heavy-patch win real on a real graph?* and *how bad is f32 error in practice, perceptually?*
— for a fraction of the cost of the full backend. If the speedup is real and the f32 error
is inaudible, build `EmitMsl`. If not, the CPU/NEON path remains the right home and this
stays a documented dead-end.
