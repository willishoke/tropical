# tests/

## Running

```bash
make build                                    # builds with JIT + tests
cmake --build build -j4 && ctest --test-dir build   # run tests
./build/test_module_process                   # run directly for verbose output
```

## What's here

`test_module_process.cpp` — single file, custom harness (`run_test()` / `ASSERT()` / `ASSERT_OK()`). Tests exercise the C API and JIT code paths **without an audio device**.

### Test cases

1. **scalar module** — construct + 32 `process()` calls
2. **clock: array default input** — `ratios_in=[1.0]` + 32 process calls
3. **clock: multi-ratio array** — `ratios_in=[1,2,4]` + 32 process calls
4. **clock: two-ratio + output tap** — 64 process calls with graph output
5. **intseq: integer-register module** — wired to clock, 128 process calls
6. **multi-module fusion** — clock + 4 intseqs + 4 VCOs, 128 frames (fused graph kernel)

### Structure

- **Helpers** (line ~55) — `wrap01()` and similar expression builders
- **Module spec builders** (line ~73) — `build_clock_spec()`, `build_intseq_spec()`, etc. mirror `module_library.ts` definitions in C
- **Tests** (line ~164) — each test creates a graph, instantiates modules, wires inputs, processes frames, checks outputs
- **Main** (line ~492) — runs all tests, prints pass/fail summary

## CI note

CI builds with `EGRESS_LLVM_ORC_JIT=OFF`, so JIT paths are only tested locally. Always run tests locally before pushing JIT-related changes.

## Adding tests

Add a new `run_test(...)` call in `main()`. Build module specs using the C API functions (`egress_module_spec_create`, `egress_module_spec_add_output`, etc.) following existing patterns. Process with `egress_graph_process()` and assert on output values.
