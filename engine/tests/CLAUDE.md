# tests/

## Running

```bash
make build                                    # builds with JIT + tests
cmake --build build -j4 && ctest --test-dir build   # run tests
./build/test_module_process                   # run directly for verbose output
```

## What's here

`test_module_process.cpp` — single file, custom harness (`run_test()` / `ASSERT()` / `ASSERT_OK()`). Tests exercise the FlatRuntime C API and JIT code paths **without an audio device**.

### Test cases

1. **scalar accumulator** — sawtooth phase accumulator, 32 process calls
2. **clock: array default input** — `ratios_in=[1.0]`, 32 process calls
3. **clock: multi-ratio array** — `ratios_in=[1,2,4]`, 32 process calls
4. **clock: two-ratio + output tap** — 2 ratios, 64 process calls
5. **intseq with edge detection** — integer sequence stepping, 128 process calls
6. **multi-module fusion** — clock + 4 intseqs + 4 VCOs fused into one kernel, 128 process calls
7. **smoothed param** — SmoothedParam expression with atomic value set
8. **trigger param** — TriggerParam fire-once-per-frame behavior

### Structure

- Tests use `tropical_runtime_new`, `tropical_runtime_load_plan`, `tropical_runtime_process`, `tropical_runtime_output_buffer`
- Each test builds an `tropical_plan_2` JSON string directly (no Graph or Module)
- Plan JSON includes `output_exprs`, `register_exprs`, `state_init`, `register_names`, `outputs`

## CI note

LLVM ORC JIT is always enabled. CI runners need LLVM installed to build. Always run tests locally before pushing JIT-related changes.

## Adding tests

Add a new `run_test(...)` call in `main()`. Build plan JSON strings directly using the `tropical_plan_2` schema, then use `tropical_runtime_*` C API functions to load and process. Assert on output buffer values.
