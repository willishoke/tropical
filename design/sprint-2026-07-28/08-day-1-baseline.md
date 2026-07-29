# Day-1 release baseline

- **Baseline commit:** `9c492dd52503ee654cb927d9bca67fd97c817cb0`
- **Commit date:** 2026-07-27T07:35:17-07:00
- **Baseline subject:** `refactor(lean): improve code hygiene`
- **Run date:** 2026-07-27 (America/Los_Angeles)
- **Tracked checkout state:** clean before the sprint handoffs were copied in
- **Host:** MacBook Pro (MacBookPro18,1), Apple M1 Pro, 10 cores, 16 GB
- **Operating system:** macOS 26.3 (25D125), arm64
- **CMake:** 4.2.3
- **LLVM:** Homebrew 22.1.7
- **Lean:** 4.29.1 through the repository's `lean/lean-toolchain`
- **Lake:** 5.0.0
- **Bun:** 1.3.12
- **Build type:** `RelWithDebInfo`
- **Configured targets:** JIT, in-process wasm emission, and Metal

## Setup finding

The first `make validate` attempt stopped during CMake configuration because a
new Git worktree does not automatically initialize `lib/json` and
`lib/rtaudio`. Both directories were empty. This was classified as checkout
setup, not a source failure.

The pinned submodules were initialized recursively:

- `lib/json` at `9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03`
- `lib/rtaudio` at `409636b5dcad3054ae5a9e85014bba3861b8edab`

The next attempt built C++ successfully but the sandboxed Lake invocation
could not clone the pinned Turnstile dependency. The validation command was
then repeated with network access; Lake resolved the manifest revision
`a6c2a9774a9b1f6d3946293d3a19291e80c0b8b6`.

Fresh-checkout validation therefore requires both:

```bash
git submodule update --init --recursive
make validate
```

`make validate` remains the source gate. Dependency population is setup, not
an implicit part of that target.

## Gate result

The final baseline result is filled from the successful unrestricted retry:

| Gate | Result | Notes |
|---|---|---|
| CMake configure/build | pass | JIT, wasm emitter, Metal, and C++ tests |
| Lean build | pass | 205 jobs; built binaries only; no `lake exe` |
| `tropicaltest` | pass | 118/118 frozen-golden, proof-surface, and native-equivalence gates |
| Web patch build | pass | 3 wasm patches and manifest generated |
| Bun tests | pass | 110 passed, 1 capability fallback skipped, 0 failed |
| CTest | pass | `module_process` and `metal_kernel`, 2/2 |

The Bun capability-fallback test is registered as skipped even when the
Metal-specific rows execute. Nine Metal/JIT SNR rows passed on this host,
including the long-coordinate, warped-clock, banked-column, and heavy-modal
fixtures. This pre-existing registration detail is not treated as a Metal
qualification result; Lane E supplies the separate pipeline/soak evidence.

Warnings present before sprint work are recorded but are not normalized into
new failures. Any changed warning or red gate after integration is triaged
against this baseline.
