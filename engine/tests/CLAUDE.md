# engine/tests/

The C++ suites test the execution boundary without opening an audio device.
CTest labels and executable names distinguish current Tropical behavior from
legacy-manifest compatibility.

## Current tests

`test_module_process.cpp` is registered as `current_module_process`. It uses
the public C API plus hand-written LLVM matching the production kernel ABI:

1. a constant closed-form kernel loads and renders;
2. an index ramp proves the sample coordinate advances between buffers without
   per-sample state;
3. the device-boundary clamp is bounded and bit-transparent inside its limit.

This file does not test a plan compiler, plan instructions, named register
transfer, or stateful source semantics. Lean owns LLVM emission and exercises
arithmetic semantics in `tropicaltest`.

`test_metal_kernel.cpp` is registered as `current_metal_kernel` when
`TROPICAL_METAL` is enabled. It tests the current MSL runtime boundary.

## Compatibility tests

`test_compat_legacy_plan4.cpp` is registered as
`compat_legacy_plan4_manifest`. It proves one bounded promise: a direct C API
caller can pair a closed-form LLVM kernel with a `tropical_plan_4` metadata
manifest and load it successfully.

The manifest deliberately includes obsolete state keys. Native loading ignores
them and allocates no state-register backing store. This test is not evidence
that Tropical can author registers, delays, updates, feedback, or state
transfer.

The complete classification and proposed review date live in
[`design/compatibility-matrix.md`](../../design/compatibility-matrix.md).

## Running

```bash
make build
cmake --build build -j4
ctest --test-dir build --output-on-failure
ctest --test-dir build -R 'current_|compat_legacy_plan4' --output-on-failure
```

Run an executable directly for its verbose output:

```bash
./build/test_module_process
./build/test_compat_legacy_plan4
```

## Adding tests

- Use `current_...` for production language/runtime contracts.
- Use `compat_legacy_plan4_...` for intentionally retained plan-4 behavior.
- Never infer source-language support from an arbitrary LLVM ABI test.
- Update the compatibility matrix when adding, expanding, or deleting a
  compatibility case.
