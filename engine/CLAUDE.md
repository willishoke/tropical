# engine/

C++20 execution core. Lean owns plan construction and kernel emission; this
directory loads emitted artifacts, hosts them, and connects them to native
audio and control APIs.

## Layout

```text
c_api/     tropical_c.h/.cpp      stable opaque-handle API
           tropical_socket.*      native control/data socket
runtime/   FlatRuntime.*          metadata load, staged init, atomic publish
           NumericProgramParser.hpp
jit/       OrcJitEngine.*         textual LLVM → ORC; LLVM → wasm32 support
metal/     MetalKernel.*          MSL compilation and block dispatch
dac/       TropicalDAC.hpp        RtAudio device boundary
tests/                              current ABI and schema-boundary CTests
```

## Kernel load path

The production load is:

```text
Lean FlatPlan
  ├─ EmitLlvm → audio LLVM IR
  ├─ Stage0    → optional coefficient LLVM IR
  ├─ EmitMsl   → optional MSL
  └─ toWire    → tropical_plan_6 metadata
        │
        ▼
tropical_runtime_load_ir_staged
  → NumericProgramParser (metadata only)
  → OrcJitEngine::compile_ir_text
  → optional MetalKernel
  → run coefficient kernel once
  → FlatRuntime::publish_state
```

The C++ plan instruction structs and parser are not a code generator.
`OrcJitEngine` compiles the LLVM text emitted by Lean. The manifest sizes
scratch/array/slot regions, supplies defaults and parameter disciplines, and
names stage-0 coefficient columns.

Three C entry points share the same tail:

- `tropical_runtime_load_ir`;
- `tropical_runtime_load_ir_msl`;
- `tropical_runtime_load_ir_staged`.

All three require a `tropical_plan_6` manifest. Older or unknown schemas and
retired state/output carriers fail at the load boundary; see
[`design/compatibility-matrix.md`](../design/compatibility-matrix.md).

## State and publication contract

Production Tropical kernels are closed-form `f(τ, params)`. In the native
runtime:

- `register_count` sizes the SSA temp pool; it is not persistent state;
- the `%registers` kernel-ABI argument is retained but its backing buffer is
  empty;
- retired state keys such as `state_init`, `register_names`,
  `register_types`, and `register_targets` are rejected;
- a fresh load initializes scratch, arrays, and slots from its own manifest;
- `publish_state` carries only `sample_index`;
- there is no by-name register, array, or slot migration.

The session/control layer keeps current parameter values and supplies them
through fresh plan defaults and live slot writes. That is not kernel state
transfer.

Stage-0 coefficient storage is also not per-sample state. The coefficient
kernel is a closed-form control-time computation. Scalar results go to slots;
bank coefficient columns use a three-generation publication protocol so an
audio block sees one complete generation.

## FlatRuntime

`KernelState` owns:

- the JIT kernel and optional coefficient kernel;
- optional `MetalKernel`;
- SSA temp scratch;
- array storage and coefficient generations;
- module slots/defaults/names;
- parameter write-discipline metadata;
- sample rate and sample coordinate.

`process()` captures the active state, calls the selected audio kernel for one
buffer, advances the coordinate, and applies the smoothstep fade. The active
state is published with a release store after the inactive state is fully
built.

On the JIT callback path, `process()`:

1. applies any stable even-sequence clock request;
2. CAS-owns, captures, and revalidates the active state and one coherent
   slot/coefficient generation;
3. copies published slots into preallocated audio scratch and calls the kernel;
4. advances the runtime-global audio clock and applies the smoothstep fade;
5. publishes the completed sample boundary and releases ownership.

Control writes serialize under `build_mutex_` and run the coefficient stage in
control-only scratch. JIT release-publishes the complete slot/coefficient set
as one generation; Metal forms an immutable exact-epoch worker request. State
and generation ownership atomics live outside movable kernel state, closing an
ABA window during JIT hot-swap. Neither audio path locks or allocates; a
bounded JIT ownership failure or Metal handoff fault emits silence and
telemetry.

Configure with `-DTROPICAL_TSAN=ON`, then build `check_runtime_tsan`, to run
the barrier-driven clock/state/generation tests plus epoch-queue and
render-worker mailbox publication stress under ThreadSanitizer when the
compiler provides it.

`render_window` evaluates the fused JIT kernel at arbitrary coordinates for
scope/slave consumers. Metal sessions remain dual-loaded so this path stays on
the f64 JIT reference.

## Execution targets

### ORC JIT

`OrcJitEngine::compile_ir_text` parses a one-function LLVM module, verifies it,
renames the function to a content-addressed symbol, compiles it through ORC,
and caches the result. The stage-0 coefficient module can request the
unoptimized JIT because it runs at control time and compile latency dominates
its execution cost.

Kernel ABI:

```text
(inputs, registers, arrays, array_sizes, temps,
 rate, start_sample_index, param_ptrs,
 output_buffer, buffer_length, slots) → void
```

Lean emission defines the semantics of those arguments. Do not resurrect C++
instruction codegen.

Benchmark/qualification runs may set `TROPICAL_KERNEL_CACHE_ROOT` to move the
build-id subtree under a harness-owned directory, or
`TROPICAL_KERNEL_CACHE_DISABLE=1` to disable disk cache reads/writes. Defaults
are unchanged; harnesses must never clear the user's ordinary cache.

### JIT compile controls

Two independent dials, and they are often confused:

- `TROPICAL_JIT_OPT_LEVEL` (`O0`..`O3`/`Os`/`Oz`, default `O2`) selects the
  **IR optimization pipeline** — vectorize, SLP, CSE, instcombine. This is
  where the kernel's runtime performance comes from.
- `TROPICAL_JIT_CODEGEN_OPT` (`none`/`less`/`default`/`aggressive`, **default
  `none`**) selects `CodeGenOptLevel` — instruction selection, pre-RA
  scheduling, register allocation. This is where the compile wall goes.

The codegen dial defaults OFF because on tropical's kernels it buys no
measurable runtime while costing 6x–232x of compile. A closed-form kernel is
long straight-line f64 arithmetic (or one `bankSum` reduction) that is
dependency- and memory-bound, so an aggressive scheduler and a graph-colouring
allocator have little to exploit. Measured across oscillator sweeps, gong,
resonator→reverb, and a 256-partial banked (loop) kernel, the runtime delta is
within ±1%; audio goldens are byte-identical. Set
`TROPICAL_JIT_CODEGEN_OPT=default` to restore LLVM's choice — no rebuild
needed. Re-measure before assuming this holds for a kernel shape with tight
loops, branchy control flow, or high register pressure.

`TROPICAL_JIT_PRERA_SCHED` (`fast`, `list-ilp`, `source`, `linearize`, …)
overrides just the pre-RA scheduler, a middle point on the same curve. It is
mostly a diagnostic now that codegen defaults to `none`; the default `source`
scheduler selects by linear scan over the ready list and goes superlinear on
one very large straight-line block.

Diagnostics, both default-off and free when unset:
`TROPICAL_JIT_TRACE=1` reports per compile the tier, IR byte count, instruction
counts either side of the IR pipeline, each phase's wall, and a call counter;
`TROPICAL_JIT_TRACE_DUMP=<dir>` writes the post-pipeline module so the exact IR
the codegen layer receives can be compiled standalone with `llc`.

Full measurements and method: `benchmarks/oscillator_saturation/findings.md`.

### WebAssembly build support

With `TROPICAL_WASM_EMIT`, the engine lowers the same LLVM module to wasm32
and links it in-process with lld. This is a build-time capability used by
`diffcli compile-wasm`; the browser player lives in `web/runtime/`.

### Metal

With `TROPICAL_METAL`, `MetalKernel` compiles Lean-emitted MSL at runtime and
dispatches one thread per sample. `TROPICAL_BACKEND=metal` selects it for live
audio. Host slots are f64 and narrow to f32 when an immutable render request is
formed; the clock rail stays i64. Hoisted coefficient columns cross in the
packed `buffer(3)` binding.

The JIT always loads alongside Metal for scopes and reference rendering. Tests:

- `tests/web/metal_vs_jit.test.ts`;
- `engine/tests/test_metal_kernel.cpp`;
- MSL and coefficient-column gates in `tropicaltest`.

`MetalKernel` is ObjC++ behind a pure-C++ header. Its sole execution primitive,
`render_tile`, performs a blocking submit/wait into stable caller-owned
storage and permanently fails closed after a terminal command-buffer error.
Only `MetalRenderWorker` may call it. A callback provenance guard rejects and
counts any attempted Metal submission from an audio callback.

Live Metal uses `EpochTileQueue`: two banks of four preallocated tiles, with
worker-owned `Free → Rendering → Ready` transitions and callback-owned
`Ready → Reading → Free` transitions. `Rgpu` is the tile render quantum and
`Bdev` is the negotiated device callback quantum; `Rgpu` must be a positive
multiple of `Bdev`. The callback only performs a bounded activation read,
validates the exact epoch/device/source tag, copies one `Bdev` slice, advances
its cursors, and releases the tile. It never packs slots or coefficient
columns, allocates, waits, retries, or submits GPU work. Missing or mismatched
tiles produce fail-silent output plus sticky diagnostics.

Every raw, glide, anchor, velocity, clock-jump, and hot-swap transition
reserves an exact activation epoch `E`. The old bank remains audible strictly
before `E`; the prepared bank begins at `E`. If preparation misses its target,
the worker retargets and the host recomputes every companion from the new
exact epoch. Physical device frames remain monotonic while the source
coordinate may jump. Activation descriptors are published and acknowledged
in order before the old bank is reused.

Loads remain dual (`load_ir_msl`): the JIT always compiles too and serves
`render_window` scopes and the f64 reference. A hot-swap replaces both
artifacts, but the runtime-global device/source coordinates are independent of
the swapped state.

Enable per session with `TROPICAL_BACKEND=metal` (read at engine boot).
Correctness: `tests/web/metal_vs_jit.test.ts` (SNR vs the f64 JIT — f32
output quantizes at ~-144 dB, so the gate is ~140 dB SNR flat in τ, not
bytes) + `engine/tests/test_metal_kernel.cpp` (ctest).

**Known v1 wart:** kernel-WRITTEN port slots are thread-private locals in the
GPU kernel and never write back to the host slot array, so
`tropical_runtime_get_slot` on such a slot returns the plan default in Metal
mode (the CPU kernel would show the last sample's value). Host-written param
slots are unaffected; the scope reads via `render_window` (JIT) and is exact.

**Coefficient columns (banks-as-data) cross via `buffer(3)`.** A plan that
advertises hoisted columns (`coeff_array_slots`) emits its MSL kernel with a
fourth binding — `constant float* coeff_columns [[buffer(3)]]`, ONE packed
buffer whose per-slot offsets are compile-time literals in plan order —
and reads those slots from it instead of declaring thread-private `arrN`
locals it could never fill (columns-free plans keep the exact 3-binding
ABI, byte-frozen by the msl-golden gates). The control path recomputes
coefficient work at the exact activation epoch and narrows one coherent
immutable slot/column snapshot into the worker request. Thus a banked session
on `TROPICAL_BACKEND=metal` runs the same typed split as the JIT: coefficient
math at knob rate on CPU, the audio loop on GPU reading real columns.
Gated by `msl-column-guard` (tropicaltest), the banked-resonator SNR case
in `metal_vs_jit`, and the column tests in `test_metal_kernel.cpp`.

`TROPICAL_METAL_RENDER_TILE_FRAMES` selects `Rgpu`; absence defaults to 512,
and invalid or non-divisible values refuse before live rendering.
`TROPICAL_BUFFER_LENGTH=16..16384` selects `Bdev` before Runtime/DAC
construction; absence preserves the 512 default. Read-only C diagnostics
expose both quanta, four-tile bank capacity, published/acknowledged activation
epochs, dispatch/starvation/tag/retarget/activation/provenance counters,
worker stage timestamps, activation-latency statistics, and worker CPU/wall
time. The retired pipeline-depth controls and diagnostic are absent.

See [`benchmarks/metal_live/findings.md`](../benchmarks/metal_live/findings.md)
for current qualification measurements.

## Audio output (`dac/TropicalDAC.hpp`)

`TropicalDACImpl<AudioSource>` is the RtAudio device driver. The callback
copies mono output to every device channel and:

- tracks timing stats
  (avg/max callback ms, underrun/overrun counts), and increments a fixed
  preallocated 1 us callback histogram. Qualification reset uses a
  callback-boundary epoch; it never resets counters concurrently mid-callback.
- Qualification output capture uses one construction-time buffer and a
  request/ready sequence. The callback performs only a bounded copy; it does
  not allocate, lock, or perform I/O.
- RtAudio must negotiate the runtime's requested frame count exactly; a
  mismatch closes the stream and refuses before playback.
- A watcher polls for disconnect/default-device changes; recovery and explicit
  device switches use fade-in.

## C API boundary

All handles are opaque `void *`.

- Runtime: create/free, staged artifact loads, process/output, slots,
  random-access render.
- Parameters: the older `ControlParam` pointer API remains for standalone
  bindings; session params are module slots.
- DAC: create/start/stop, stats, device switch/reconnect, playback position.
- Socket: host-side parameter dispatch and telemetry plus queued control-plane
  calls.
- Errors: `tropical_last_error()` is thread-local.

The device boundary clamps non-finite or unbounded values; raw runtime renders
remain the mathematical kernel output for goldens and backend comparisons.

## Tests

See [`tests/CLAUDE.md`](tests/CLAUDE.md). `current_module_process` includes the
serialized-plan rejection boundary alongside current production/ABI checks.

```bash
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

## Editing rules

- Keep public C ABI changes explicit and separately reviewed.
- No interpreter or C++ plan-codegen fallback belongs on the production path.
- Audio-thread code must not allocate, lock, log, or throw.
- Do not add state-shaped compatibility carriers to the Plan-6 runtime.
- Update the compatibility matrix and negative boundary gates when changing
  serialized-plan dispatch.
