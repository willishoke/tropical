# engine/

C++20 core. Header-heavy by design (templates + inlining for audio-thread performance).

## Layout

```
c_api/    tropical_c.h / .cpp    Stable C API (opaque handles, thread-local errors)
runtime/  FlatRuntime.hpp/.cpp   Plan loading, double-buffered kernel execution
          NumericProgramParser.hpp   tropical_plan_4 JSON → FlatProgram struct
jit/      OrcJitEngine.hpp/.cpp  LLVM ORC JIT (FlatProgram → native kernel)
dac/      TropicalDAC.hpp        Audio output via RtAudio (templated driver)
ControlParam.hpp                 Lock-free atomic parameter struct (shared by runtime + C API)
tests/    test_module_process.cpp  C API + JIT tests (no audio device)
```

## C API boundary (`c_api/tropical_c.h`)

All external access (TypeScript FFI, tests) goes through here. Handles are opaque `void*`.

- **FlatRuntime** — `tropical_runtime_new`, `_load_plan`, `_process`, `_output_buffer`, fade control
- **ControlParam** — `tropical_param_new` (smoothed, one-pole lowpass), `_new_trigger` (fire-once), `_set`/`_get` (atomic)
- **DAC** — `tropical_dac_new_runtime`, `_start`/`_stop`, `_get_stats`, `_switch_device`, `_is_reconnecting`
- **Device enumeration** — `tropical_audio_device_count`, `_get_device_ids`, `_get_device_info`, `_default_output_device`
- **Errors** — `tropical_last_error()` returns thread-local error string

## Plan loading (`runtime/`)

`FlatRuntime::load_plan()` receives a `tropical_plan_4` JSON string:

1. `NumericProgramParser::parse_plan4()` — thin deserializer, reads the pre-compiled instruction stream into a `FlatProgram` struct. No expression walking.
2. `OrcJitEngine::compile_flat_program()` — JIT compiles the FlatProgram to a native kernel function.
3. State initialization — registers are type-aware bit-cast (`int64_t[]` backing store, with float/int/bool coercion).
4. Named state transfer — matching registers and arrays are copied from the active kernel by name for click-free hot-swap.
5. Atomic swap — new kernel published to audio thread via `active_state_` store-release.

## JIT engine (`jit/OrcJitEngine.hpp/.cpp`)

Singleton LLVM ORC engine. `compile_flat_program()`:

1. Build canonical cache key (MD5 of serialized program, param pointers replaced by ordinals)
2. Check in-memory cache and disk cache (`~/.cache/tropical/kernels/<build-id>/`)
3. Generate LLVM IR:
   - Kernel signature: `(inputs, registers, arrays, array_sizes, temps, sample_rate, start_sample_index, param_ptrs, output_buffer, buffer_length) → void`
   - Outer sample loop iterates `buffer_length` times
   - Each instruction: resolve typed operands (f64/i64/i1), emit native ops with explicit coercion at type boundaries
   - Array loops: `loop_count > 1` emits elementwise loop, `strides[i]` controls broadcast vs. iterate
   - Output: sum of mix outputs written to `output_buffer[sample_idx]`
   - State writeback after the loop
4. Add module to LLJIT, look up symbol → `NumericKernelFn`

**No transcendentals in the JIT.** sin / cos / tanh / exp / log / pow are defined as ProgramJSON files in `stdlib/` and inlined at flatten time from arithmetic primitives plus `Ldexp` / `FloatExponent` (single-instruction IEEE-754 bit ops for 2^n range reduction). Swap `stdlib/Sin.json` to change the approximation.

**Cache invalidation**: build-id subdirectory derived from the binary's LC_UUID (macOS) / ELF build-id. Dylib rebuild auto-invalidates.

## FlatRuntime (`runtime/FlatRuntime.hpp/.cpp`)

Execution container with two `KernelState` slots for lock-free hot-swap.

`KernelState` holds: kernel fn ptr, registers (`int64_t[]`), temps, array storage/ptrs/sizes, trigger param pointers, sample rate, sample index, register/array names.

**`process()`** (audio thread):
1. Load active state (acquire)
2. Snapshot trigger params (atomic exchange → frame_value)
3. Call kernel (single invocation processes entire buffer)
4. Advance sample_index
5. Apply smoothstep fade envelope (Hermite curve, 2048 samples default)

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
