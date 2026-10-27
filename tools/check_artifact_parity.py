#!/usr/bin/env python3
"""The offline compiler and the engine agree on every patch's plan.

`tropicalc` compiles a patch with no runtime linked, through
Tropical.Session.compilePatch on a value. `diffcli compile` loads the same
patch into a booted engine and reads the plan back from its session. The
two must print byte-identical plan JSON for every patch in the repo's
corpora; a difference means the pure path and the engine path have
drifted. Also checks that the tropicalc binary links no libtropical.
"""

from __future__ import annotations

import pathlib
import platform
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = ROOT / "lean/.lake/build/bin"
CORPORA = [ROOT / "patches", ROOT / "web/patches"]
SKIP = {"manifest.json"}


def run(binary: pathlib.Path, *args: str) -> str:
    r = subprocess.run([str(binary), *args], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"check_artifact_parity: {binary.name} {' '.join(args)} failed:\n{r.stderr}")
    return r.stdout


def linked_libraries(binary: pathlib.Path) -> str:
    if platform.system() == "Darwin":
        return subprocess.run(["otool", "-L", str(binary)], capture_output=True, text=True).stdout
    return subprocess.run(["ldd", str(binary)], capture_output=True, text=True).stdout


def main() -> int:
    tropicalc = BIN / "tropicalc"
    diffcli = BIN / "diffcli"
    for b in (tropicalc, diffcli):
        if not b.exists():
            sys.exit(f"check_artifact_parity: {b} not built")

    linked = linked_libraries(tropicalc)
    if "libtropical" in linked:
        print("check_artifact_parity: tropicalc links libtropical:")
        print(linked)
        return 1

    patches = sorted(p for c in CORPORA for p in c.glob("*.json") if p.name not in SKIP)
    if not patches:
        sys.exit("check_artifact_parity: no patches found")
    mismatches = []
    for p in patches:
        rel = p.relative_to(ROOT)
        pure = run(tropicalc, str(rel), "--emit=plan")
        engine = run(diffcli, "compile", str(rel))
        if pure != engine:
            mismatches.append(rel)
            print(f"check_artifact_parity: MISMATCH {rel}")
    if mismatches:
        return 1
    print(f"check_artifact_parity: ok ({len(patches)} patches, tropicalc links no runtime)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
