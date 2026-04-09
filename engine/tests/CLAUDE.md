# tests/

## Running

```bash
make build                                          # builds with JIT + tests
cmake --build build -j4 && ctest --test-dir build   # run tests
./build/test_module_process                         # run directly for verbose output
```

## What's here

`test_module_process.cpp` — single file, custom harness (`run_test()` / `ASSERT()` / `ASSERT_OK()`). Tests exercise the FlatRuntime C API and JIT code paths **without an audio device**.

### Test cases

1. **sawtooth oscillator** — phase accumulator, 32 process calls
2. **two outputs with mix** — multiple output targets summed to buffer
3. **hot-swap preserves state** — load new plan, verify named register state transfer
4. **array literal in expression** — array operands in instructions
5. **integer counter with mod wrap** — typed int registers, modular arithmetic
6. **select/conditional expression** — `Select` instruction with bool condition
7. **multi-register clock** — multiple state registers, clock-style stepping
8. **multiple outputs summed** — output mixing validation
9. **typed int bitwise (LFSR)** — integer bitwise ops (BitXor, RShift, BitAnd, BitOr)
10. **typed bool comparison + select** — bool-typed comparisons and select

### Structure

- Tests use `tropical_runtime_new`, `tropical_runtime_load_plan`, `tropical_runtime_process`, `tropical_runtime_output_buffer`
- Each test builds a `tropical_plan_4` JSON string directly (no TypeScript, no modules)
- Plan JSON includes `instructions`, `state_init`, `register_names`, `register_types`, `register_count`, `output_targets`, `register_targets`, `outputs`

## CI note

LLVM ORC JIT is always enabled. CI runners need LLVM installed to build. Always run tests locally before pushing JIT-related changes.

## Adding tests

Add a new `run_test(...)` call in `main()`. Build plan JSON strings directly using the `tropical_plan_4` schema, then use `tropical_runtime_*` C API functions to load and process. Assert on output buffer values.
