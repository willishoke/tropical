#!/usr/bin/env python3
"""The audio runtime carries no toolchain.

libtropical is the runtime a host links; the IR→wasm emitter (LLVM's
WebAssembly backend + lld) is a build-time toolchain and lives in
libtropical_wasm, which only `diffcli` links. This checks the built
libraries say so: libtropical neither loads an lld library nor defines any
`lld::` symbol, and every Lean binary other than diffcli links neither
libtropical_wasm nor lld.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
BIN = ROOT / "lean" / ".lake" / "build" / "bin"
DARWIN = sys.platform == "darwin"

# Lean binaries that link the runtime but must not link the toolchain.
RUNTIME_BINARIES = ["frontend", "tropicaltest"]
OPTIONAL_BINARIES = []


def run(*cmd: str) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def dynamic_deps(lib: pathlib.Path) -> str:
    return run("otool", "-L", str(lib)) if DARWIN else run("ldd", str(lib))


def defined_symbols(lib: pathlib.Path) -> str:
    return run("nm", "-U", str(lib)) if DARWIN else run("nm", "--defined-only", str(lib))


def main() -> int:
    runtime = BUILD / ("libtropical.dylib" if DARWIN else "libtropical.so")
    if not runtime.exists():
        sys.exit(f"check_runtime_deps: {runtime} not built")
    errors = []
    deps = dynamic_deps(runtime)
    if "lld" in deps.lower():
        errors.append(f"{runtime.name} loads an lld library:\n{deps}")
    lld_syms = [l for l in defined_symbols(runtime).splitlines() if "_ZN3lld" in l]
    if lld_syms:
        errors.append(f"{runtime.name} defines {len(lld_syms)} lld:: symbols, e.g. {lld_syms[0]}")
    checked = []
    for name in RUNTIME_BINARIES + OPTIONAL_BINARIES:
        b = BIN / name
        if not b.exists():
            if name in RUNTIME_BINARIES:
                errors.append(f"{b} not built")
            continue
        checked.append(name)
        deps = dynamic_deps(b)
        if "tropical_wasm" in deps or "lld" in deps.lower():
            errors.append(f"{name} links the toolchain:\n{deps}")
    for err in errors:
        print(f"check_runtime_deps: {err}")
    if errors:
        return 1
    print(f"check_runtime_deps: ok ({runtime.name} carries no lld; "
          f"{', '.join(checked)} link no toolchain)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
