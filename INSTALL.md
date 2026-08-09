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
| Lean 4 | via elan | `curl … elan … \| sh` (see [leanprover/elan](https://github.com/leanprover/elan)) | same |
| Bun | ≥ 1.3 | `brew install oven-sh/bun/bun` | `curl -fsSL https://bun.sh/install \| bash` |

The Lean toolchain (pinned in `lean/lean-toolchain`) is the production
compiler and MCP server — installing it via [elan](https://github.com/leanprover/elan)
is what `make lean` expects on `$HOME/.elan/bin`. Bun is only needed
for the surviving behavioral test suites and the web demo build; there
is no TypeScript compiler and no koffi FFI subprocess.

`CMakeLists.txt` requires LLVM ≥ 19 (`getOrInsertDeclaration` API);
CI builds against LLVM 20. On macOS the build expects LLVM at
`/opt/homebrew/opt/llvm`; if yours is elsewhere, set `LLVM_DIR`:

```bash
make build LLVM_DIR=/path/to/llvm/lib/cmake/llvm
```

### Build

```bash
git clone <repo> && cd tropical
make build          # C++ core → build/libtropical.dylib
make lean           # Lean frontend + diffcli (the compiler + MCP server)
```

`make build` produces `build/libtropical.dylib` (macOS) or
`build/libtropical.so` (Linux). `make lean` builds the C shim and the
Lean tree, producing `lean/.lake/build/bin/frontend` (the whole stack:
compiler, session, runtime FFI, and MCP server) and
`lean/.lake/build/bin/diffcli`. Launch the MCP server with `make
mcp-lean`.

### Verifying the install

The full gate is `make validate` (it runs everything below in one go).
The individual checks:

```bash
cmake --build build -j4 && ctest --test-dir build         # C++ tests (JIT + C API, no audio device)
./lean/.lake/build/bin/tropicaltest                       # audio + wire/port goldens + native mode-equiv
TROPICAL_ENGINE_CMD="./lean/.lake/build/bin/frontend --rpc" bun test   # WASM≡JIT + MCP behavioral
```

`tropicaltest` is the Lean golden runner: it boots the `Tropical.Stdlib`
builder programs (`lean/Tropical/Stdlib.lean` — 15 programs authored as
arrow-combinator builders), checks the per-program wire+port goldens
(`tests/golden/stdlib/*.hash`, one hash per program), checks the
byte-for-byte audio goldens (`tests/golden/cf`, `tests/golden/msl`), and
runs the native realization-variant equivalence (fused vs. per-instance
microkernel; flat vs. nested) directly through the engine.

The bun suites are the surviving cross-backend gate — `tests/web`
(WASM emitter vs. JIT, off the same `tropical_plan_6`) and the MCP
protocol tests in `mcp/` — both run against the live Lean engine via
`TROPICAL_ENGINE_CMD`. There is no koffi FFI: the bun process talks to
the `frontend` binary over RPC, so `make build` and `make lean` must
come first.

### Web demo (optional)

The browser backend needs Bun plus the Lean `diffcli` (it precompiles
patches through the same engine that serves MCP, so run `make lean`
first). The build pipeline lives in `web/`:

```bash
bun web/build_patches.ts    # precompile curated web/patches/*.json via `diffcli compile` → web/dist/patches/
bun web/build.ts            # full demo bundle (worklet + main app + index.html → web/dist/)
bun web/dev.ts              # dev server with the COOP/COEP headers SAB requires
```

The browser demo loads the precompiled `tropical_plan_6` plans, emits
WASM at runtime (`web/wasm/emit_wasm.ts`), and runs the kernel inside
an AudioWorklet. See `web/CLAUDE.md` for details.

### Platform notes

**macOS** — Primary platform. Audio output via CoreAudio. Tested on
Apple Silicon and Intel.

**Linux** — Builds and passes tests. Audio output via ALSA (requires
`libasound2-dev` or equivalent). Less tested than macOS.

**Windows** — Not currently supported.
