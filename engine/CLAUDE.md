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

1. `NumericProgramParser::parse_plan5()` — thin deserializer, reads the instance functions plus `sinks[]` into a `FlatProgram` (multi-function) struct. A backcompat lift exists for single-kernel `plan_4` inputs (hand-crafted unit tests, legacy saved plans) — they parse into a one-instance plan_5 with a top-level temp-mix.
2. `OrcJitEngine::compile_flat_program()` — JIT compiles the FlatProgram to a single native kernel function (instances then sinks, per sample).
3. State initialization — registers and module slots are type-aware bit-cast (`int64_t[]` backing store, with float/int/bool coercion).
4. Named state transfer — matching registers, arrays, and slots are copied from the active kernel by name for click-free hot-swap.
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

**No transcendentals in the JIT.** sin / cos / tanh / exp / log / pow are defined as `.trop` files in `stdlib/` and inlined by the `inline_instances` strata pass from arithmetic primitives plus `Ldexp` / `FloatExponent` (single-instruction IEEE-754 bit ops for 2^n range reduction). Swap `stdlib/Sin.trop` to change the approximation.

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
