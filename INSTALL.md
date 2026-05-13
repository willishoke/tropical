# Installing tropical

## Homebrew (macOS)

> **Coming soon.** The Homebrew tap is not yet available.

```bash
brew install <tap>/tropical
```

Once installed, `libtropical.dylib` and the MCP server will be
available system-wide.

## From source

### Prerequisites

| Dependency | Version | macOS | Linux |
|-----------|---------|-------|-------|
| LLVM | ≥ 19 | `brew install llvm` | See [LLVM apt](https://apt.llvm.org/) or your distro's package manager |
| CMake | ≥ 3.20 | `brew install cmake` | `apt install cmake` |
| Bun | ≥ 1.3 | `brew install oven-sh/bun/bun` | `curl -fsSL https://bun.sh/install \| bash` |

`CMakeLists.txt` requires LLVM ≥ 19 (`getOrInsertDeclaration` API);
CI builds against LLVM 20. On macOS the build expects LLVM at
`/opt/homebrew/opt/llvm`; if yours is elsewhere, set `LLVM_DIR`:

```bash
make build LLVM_DIR=/path/to/llvm/lib/cmake/llvm
```

### Build

```bash
git clone <repo> && cd tropical
bun install
make build
```

This produces `build/libtropical.dylib` (macOS) or
`build/libtropical.so` (Linux).

### Verifying the install

Four checks; the first three are required, the fourth (stdlib audit)
is run by `make validate`.

```bash
cmake --build build -j4 && ctest --test-dir build   # C++ tests (JIT + C API, no audio device)
bun test                                              # TS tests (compiler + parse + ir + WASM equiv)
bunx tsc --noEmit                                     # type-check the TS pipeline
bun run scripts/validate_stdlib.ts                    # parse, elaborate, lower every stdlib/*.trop
```

`bun test` exercises the cross-backend equivalence gates that fix
the meaning of the strata pipeline: `tests/equiv/jit_vs_interp` (JIT
vs. pure-TS interpreter), `tests/equiv/wasm_vs_jit` (WASM emit vs.
JIT), and `tests/equiv/web_plans_vs_jit` (precompiled web plans vs.
JIT). All load `build/libtropical.dylib` via koffi, so `make build`
must come first.

`make validate` runs the C++ tests, the TS tests, and the stdlib
audit in one go.

### Web demo (optional)

The browser backend has no extra prerequisites — Bun is sufficient.
The build pipeline lives in `web/`:

```bash
bun web/build_patches.ts    # precompile curated patches → web/dist/patches/*.plan.json
bun web/build.ts            # full demo bundle (worklet + main app + index.html → web/dist/)
bun web/dev.ts              # dev server with the COOP/COEP headers SAB requires
```

The browser demo loads the precompiled plans, emits WASM at runtime
(`compiler/emit_wasm.ts`), and runs the kernel inside an
AudioWorklet. See `web/CLAUDE.md` for details.

### Platform notes

**macOS** — Primary platform. Audio output via CoreAudio. Tested on
Apple Silicon and Intel.

**Linux** — Builds and passes tests. Audio output via ALSA (requires
`libasound2-dev` or equivalent). Less tested than macOS.

**Windows** — Not currently supported.
