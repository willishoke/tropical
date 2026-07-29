# engine/tests/

The C++ suites test the execution boundary without opening an audio device.

## Current tests

`test_module_process.cpp` is registered as `current_module_process`. It uses
the public C API plus hand-written LLVM matching the production kernel ABI:

1. a constant closed-form kernel loads and renders;
2. the serialized-plan boundary accepts Plan 5 only and clearly rejects Plan
   4, missing/unknown schemas, retired carriers, missing `dst_kind`, and legacy
   operand kinds;
3. an index ramp proves the sample coordinate advances between buffers without
   per-sample state;
4. the device-boundary clamp is bounded and bit-transparent inside its limit.

This file does not test a plan compiler, plan instructions, named register
transfer, or stateful source semantics. Lean owns LLVM emission and exercises
arithmetic semantics in `tropicaltest`.

`test_metal_kernel.cpp` is registered as `current_metal_kernel` when
`TROPICAL_METAL` is enabled. It tests the current MSL runtime boundary.

## Running

```bash
make build
cmake --build build -j4
ctest --test-dir build --output-on-failure
ctest --test-dir build -R 'current_' --output-on-failure
```

Run the native boundary executable directly for verbose output:

```bash
./build/test_module_process
```

## Adding tests

- Use `current_...` for production language/runtime contracts.
- Unsupported schemas and retired carriers belong in negative boundary tests,
  never in an acceptance fixture.
- Never infer source-language support from arbitrary LLVM supplied through the
  direct C API.
- Update the compatibility matrix when changing a serialized-plan boundary.
