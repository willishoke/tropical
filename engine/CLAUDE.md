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
tests/                              current ABI and compatibility CTests
```

## Kernel load path

The production load is:

```text
Lean FlatPlan
  ├─ EmitLlvm → audio LLVM IR
  ├─ Stage0    → optional coefficient LLVM IR
  ├─ EmitMsl   → optional MSL
  └─ toWire    → tropical_plan_5 metadata
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

See [`design/compatibility-matrix.md`](../design/compatibility-matrix.md) for
the bounded plan-4 dispatch accepted by these direct APIs.

## State and publication contract

Production Tropical kernels are closed-form `f(τ, params)`. In the native
runtime:

- `register_count` sizes the SSA temp pool; it is not persistent state;
- the `%registers` kernel-ABI argument is retained but its backing buffer is
  empty;
- plan-4 state keys such as `state_init`, `register_names`,
  `register_types`, and `register_targets` are ignored;
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

### WebAssembly build support

With `TROPICAL_WASM_EMIT`, the engine lowers the same LLVM module to wasm32
and links it in-process with lld. This is a build-time capability used by
`diffcli compile-wasm`; the browser player lives in `web/runtime/`.

### Metal

With `TROPICAL_METAL`, `MetalKernel` compiles Lean-emitted MSL at runtime and
dispatches one thread per sample. `TROPICAL_BACKEND=metal` selects it for
live audio. Host slots are f64 and narrow to f32 at encode; the clock rail
stays i64. Hoisted coefficient columns cross in the packed `buffer(3)` binding.

The JIT always loads alongside Metal for scopes and reference rendering. Tests:

- `tests/web/metal_vs_jit.test.ts`;
- `engine/tests/test_metal_kernel.cpp`;
- MSL and coefficient-column gates in `tropicaltest`.

See [`benchmarks/metal_live/findings.md`](../benchmarks/metal_live/findings.md)
for current qualification measurements.

Qualification controls only: `TROPICAL_METAL_PIPELINE_DEPTH=1..3` selects the
future-block depth and overrides the legacy `TROPICAL_METAL_PIPELINE=1`
(which remains D=3). Invalid explicit depths refuse at kernel construction.
The read-only C diagnostic `tropical_runtime_metal_pipeline_depth` returns 0
for sync/JIT/non-Metal builds. `TROPICAL_BUFFER_LENGTH=16..16384` selects the
live engine block length before Runtime/DAC construction; absence preserves
the 512 default.

## Audio output (`dac/TropicalDAC.hpp`)

`TropicalDACImpl<AudioSource>` is the RtAudio device driver. The callback
copies mono output to every device channel and records callback timing,
underruns, and overruns. A watcher handles device loss/default-device changes;
explicit switches and reconnects use fade-in.

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

See [`tests/CLAUDE.md`](tests/CLAUDE.md). CTest names deliberately distinguish
current production/ABI tests from `compat_legacy_plan4` coverage.

```bash
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

## Editing rules

- Keep public C ABI changes explicit and separately reviewed.
- No interpreter or C++ plan-codegen fallback belongs on the production path.
- Audio-thread code must not allocate, lock, log, or throw.
- Treat state-shaped C++ types as compatibility/dead residue unless the
  compatibility matrix proves a current caller.
- Update the compatibility matrix and its dedicated CTest when changing
  plan-4 dispatch.
