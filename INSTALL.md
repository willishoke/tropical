# Installing tropical

## Homebrew (macOS)

> **Coming soon.** The Homebrew tap is not yet available.

```bash
brew install <tap>/tropical
```

Once installed, the `libtropical.dylib` and MCP server will be available system-wide. No additional setup required.

## From source

### Prerequisites

| Dependency | Version | macOS | Linux |
|-----------|---------|-------|-------|
| LLVM | >= 15 | `brew install llvm` | See [LLVM apt](https://apt.llvm.org/) or your distro's package manager |
| CMake | >= 3.20 | `brew install cmake` | `apt install cmake` |
| Bun | >= 1.0 | `brew install oven-sh/bun/bun` | `curl -fsSL https://bun.sh/install \| bash` |

On macOS, the build expects LLVM at `/opt/homebrew/opt/llvm`. If yours is elsewhere, set `LLVM_DIR`:

```bash
make build LLVM_DIR=/path/to/llvm/lib/cmake/llvm
```

### Build

```bash
git clone <repo> && cd tropical
bun install
make build
```

This produces `build/libtropical.dylib` (macOS) or `build/libtropical.so` (Linux).

### Verify

```bash
cmake --build build -j4 && ctest --test-dir build
```

All tests should pass. Tests exercise the JIT and C API without an audio device.

### Platform notes

**macOS** — Primary platform. Audio output via CoreAudio. Tested on Apple Silicon and Intel.

**Linux** — Builds and passes tests. Audio output via ALSA (requires `libasound2-dev` or equivalent). Less tested than macOS.

**Windows** — Not currently supported.
