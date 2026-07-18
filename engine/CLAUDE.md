# engine/

C++20 core. Header-heavy by design (templates + inlining for audio-thread performance).

## Layout

```
c_api/    tropical_c.h / .cpp    Stable C API (opaque handles, thread-local errors)
runtime/  FlatRuntime.hpp/.cpp   Plan loading, double-buffered kernel execution
          NumericProgramParser.hpp   tropical_plan_5 JSON → FlatProgram struct (multi-function)
jit/      OrcJitEngine.hpp/.cpp  LLVM ORC JIT (FlatProgram → native kernel)
dac/      TropicalDAC.hpp        Audio output via RtAudio (templated driver)
ControlParam.hpp                 Lock-free atomic parameter struct (shared by runtime + C API)
tests/    test_module_process.cpp  C API + JIT tests (no audio device)
```

## C API boundary (`c_api/tropical_c.h`)

All external access (TypeScript FFI, tests) goes through here. Handles are opaque `void*`.

- **FlatRuntime** — `tropical_runtime_new`, `_load_plan`, `_process`, `_output_buffer`, fade control
- **ControlParam** — `tropical_param_new` (smoothed, one-pole lowpass), `_set`/`_get` (atomic). `Trigger` was retired; `{op:'trigger', name}` refs in plans alias to `{op:'param', name}` at materialization.
- **DAC** — `tropical_dac_new_runtime`, `_start`/`_stop`, `_get_stats`, `_switch_device`, `_is_reconnecting`
- **Device enumeration** — `tropical_audio_device_count`, `_get_device_ids`, `_get_device_info`, `_default_output_device`
- **Errors** — `tropical_last_error()` returns thread-local error string

## Plan loading (`runtime/`)

`FlatRuntime::load_plan()` receives a `tropical_plan_5` JSON string:

1. `NumericProgramParser::parse_plan5()` — thin deserializer, reads the instance functions plus `sinks[]` (outputs) and `sources[]` (inputs; defaults to canonical `[tick, rate]`) into a `FlatProgram` (multi-function) struct. A backcompat lift exists for single-kernel `plan_4` inputs — they parse into a one-instance plan_5 with a top-level temp-mix and the canonical source pair. Legacy `rate`/`tick` operand tags upgrade to `Source{kind:Rate/Tick}` at parse.
2. `OrcJitEngine::compile_flat_program()` — JIT compiles the FlatProgram to a single native kernel function (instances then sinks, per sample). Source operands resolve to `sample_rate_arg` / `current_sample_idx` via `program.sources[i].kind`.
3. State initialization — registers and module slots are type-aware bit-cast (`int64_t[]` backing store, with float/int/bool coercion).
4. Hot-swap state — **CF-only: no by-name state transfer.** Closed-form kernels carry no per-sample state, so on swap registers/arrays/slots zero-init from the fresh kernel and only `sample_index` carries over (`FlatRuntime.cpp` `publish_state`, ~L106). `run_coeff` runs once post-load so coefficient columns aren't read zero. Slot *values* (param/knob positions) therefore do NOT survive a swap; re-applying them by name (`set_slot_by_name`) is the control/session layer's job — nothing clicks because there is no state to carry, not because state is copied kernel-to-kernel. (Whether by-name kernel-to-kernel transfer should return for a future stateful sister runtime is an open design question.)
5. Atomic swap — new kernel published to audio thread via `active_state_` store-release.

## JIT engine (`jit/OrcJitEngine.hpp/.cpp`)

Singleton LLVM ORC engine. `compile_flat_program()`:

1. Build canonical cache key (MD5 of serialized program, param pointers replaced by ordinals)
2. Check in-memory cache and disk cache (`~/.cache/tropical/kernels/<build-id>/`)
3. Generate LLVM IR:
   - Kernel signature: `(inputs, registers, arrays, array_sizes, temps, sample_rate, start_sample_index, param_ptrs, output_buffer, buffer_length) → void`
   - Outer sample loop iterates `buffer_length` times; per sample: each `instance_function` recursively (preamble → per-child {pre_input, child} → body → writebacks; session-level per-wire delays are root-kernel register writebacks here), then each `sink` (`output[target] = gain · Σ slots[inputs]`). No scheduler tier. (Sink-less plan_4 fixtures fall back to the legacy top-level `output_targets` temp-mix ÷20.)
   - Each instruction: resolve typed operands (f64/i64/i1), emit native ops with explicit coercion at type boundaries
   - Array loops: `loop_count > 1` emits elementwise loop, `strides[i]` controls broadcast vs. iterate
4. Add module to LLJIT, look up symbol → `NumericKernelFn`

**No transcendentals in the JIT.** sin / cos / tanh / exp / log / pow are defined as arrow-combinator builders in `Tropical.Stdlib` (`lean/Tropical/Stdlib.lean`) and inlined by the `inline_instances` strata pass from arithmetic primitives plus `Ldexp` / `FloatExponent` (single-instruction IEEE-754 bit ops for 2^n range reduction). Edit the builder (`Tropical.Stdlib.buildSin` / `EmitArrow.Numerics.sinSig` in `lean/Tropical/Stdlib.lean`) to change the approximation.

**Cache invalidation**: build-id subdirectory derived from the binary's LC_UUID (macOS) / ELF build-id. Dylib rebuild auto-invalidates.

## FlatRuntime (`runtime/FlatRuntime.hpp/.cpp`)

Execution container with two `KernelState` slots for lock-free hot-swap.

`KernelState` holds: kernel fn ptr, registers (`int64_t[]`), temps, module slots, array storage/ptrs/sizes, param pointers, sample rate, sample index, register/array/slot names.

**`process()`** (audio thread):
1. Load active state (acquire)
2. Call kernel (single invocation processes entire buffer)
3. Advance sample_index
4. Apply smoothstep fade envelope (Hermite curve, 2048 samples default)

**Fade control**: `begin_fade_in()` / `begin_fade_out()` set atomic counters decremented per sample.

## Metal backend (`metal/`, TROPICAL_METAL builds)

`MetalKernel` (ObjC++ behind a pure-C++ header) executes the fused kernel on
the GPU: MSL emitted by Lean (`EmitMsl`), compiled at runtime
(`newLibraryWithSource`, `MTLMathModeSafe`), one thread per sample,
synchronous per-block dispatch. It rides inside `KernelState` — the existing
double-buffered publish/flip is the hot-swap; `sample_index` carries over as
usual. Loads are DUAL (`load_ir_msl`): the JIT always compiles too and keeps
serving `render_window` (the scope) and the f64 reference; only `process()`
dispatches to Metal. Slots stay f64 host-side, snapshotted to f32 at encode.

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
ABI, byte-frozen by the msl-golden gates). Host side: `process()` packs the
captured generation into f32 staging (`KernelState::metal_column_staging` —
int64 storage bit-punned to f64 like the JIT's array loads, then narrowed
f64→f32 like slots) and `process_block` copies it into the MTLBuffer at
encode (sync) or enqueue (pipelined, per-ring-entry buffers — the
documented D-block param lag, no tear: the pack reads the ONE generation
captured before `audio_processing_` went true). So a banked session on
`TROPICAL_BACKEND=metal` runs the same typed split as the JIT: coefficient
math at knob rate on CPU, the audio loop on GPU reading real columns.
Gated by `msl-column-guard` (tropicaltest), the banked-resonator SNR case
in `metal_vs_jit`, and the column tests in `test_metal_kernel.cpp`.

## Audio output (`dac/TropicalDAC.hpp`)

`TropicalDACImpl<AudioSource>` — templated RtAudio driver. FlatRuntime satisfies the `AudioSource` concept.

- Audio callback copies mono output to all channels, tracks timing stats (avg/max callback ms, underrun/overrun counts)
- Watcher thread polls every 50ms for device disconnect or default device change
- Disconnect recovery: abort stream → 500ms backoff → reopen with fade-in
- `switch_device(id)`: explicit switching with fade-in

## Adding expression ops

To add a new operation to the engine:

1. Add variant to `OpTag` enum in `jit/OrcJitEngine.hpp`
2. Add tag string mapping in `NumericProgramParser.hpp` → `parse_op_tag()`
3. Add LLVM IR emission case in `OrcJitEngine.cpp` → `compile_flat_program()`
