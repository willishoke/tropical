#!/usr/bin/env python3
"""Enforce the Lean library layering declared in lean/lakefile.toml.

Library membership comes from each `[[lean_lib]]`'s `globs`; the allowed
dependency direction between libraries is the table below. A module may
import its own library and the libraries listed for it, nothing else.
Executable roots (`[[lean_exe]]`) are composition roots and may import
anything. Every local module must belong to exactly one library or be an
executable root.

The rules that matter: TropicalCore imports only TropicalCore (plus the
Lean toolchain), so the runtime binding `Tropical.Ffi` can never enter
the compiler core's import closure; TropicalSession imports only the
core, so the session engine reaches the runtime through its host port;
and the `tropicalc` executable's closure stays inside TropicalCore, so
the compiler demonstrably runs with no runtime in the binary.
"""

from __future__ import annotations

import pathlib
import re
import sys
import tomllib


ROOT = pathlib.Path(__file__).resolve().parents[1]
LEAN = ROOT / "lean"

# Library -> the other libraries its modules may import.
ALLOWED = {
    "TropicalCore": set(),
    "TropicalHost": {"TropicalCore"},
    "TropicalSession": {"TropicalCore"},
    "TropicalApp": {"TropicalCore", "TropicalHost", "TropicalSession"},
    "TropicalTest": {"TropicalCore", "TropicalHost", "TropicalSession", "TropicalApp"},
}

# Executable roots whose whole import closure must stay inside the named
# libraries: the compiler runs with no runtime in the binary.
EXE_CLOSURE = {
    "Tropicalc": {"TropicalCore"},
    "TropicaltestCore": {"TropicalCore", "TropicalTest"},
}

# External import roots a library may use. Libraries not listed may use any.
EXTERNAL = {
    "TropicalCore": {"Init", "Lean", "Std"},
    "TropicalHost": {"Init", "Lean", "Std"},
}


def strip_comments(src: str) -> str:
    """Drop `--` line comments and (nested) `/- ... -/` block comments."""
    out, i, depth, n = [], 0, 0, len(src)
    while i < n:
        if src.startswith("/-", i):
            depth += 1
            i += 2
        elif depth and src.startswith("-/", i):
            depth -= 1
            i += 2
        elif depth:
            i += 1
        elif src.startswith("--", i):
            j = src.find("\n", i)
            i = n if j < 0 else j
        else:
            out.append(src[i])
            i += 1
    return "".join(out)


def header_imports(path: pathlib.Path) -> list[str]:
    """The module names in the file's import header."""
    imports = []
    for line in strip_comments(path.read_text()).splitlines():
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] != "import":
            break
        imports.extend(tokens[1:])
    return imports


def glob_matches(glob: str, mod: str) -> bool:
    if glob.endswith(".*"):
        base = glob[:-2]
        return mod == base or mod.startswith(base + ".")
    if glob.endswith(".+"):
        return mod.startswith(glob[:-2] + ".")
    return mod == glob


def main() -> int:
    config = tomllib.loads((LEAN / "lakefile.toml").read_text())
    libs = {lib["name"]: lib.get("globs", []) for lib in config.get("lean_lib", [])}
    exes = {exe.get("root", exe["name"]) for exe in config.get("lean_exe", [])}
    unknown = set(ALLOWED) ^ set(libs)
    if unknown:
        print(f"check_layers: lakefile libs and ALLOWED disagree: {sorted(unknown)}")
        return 1

    mods = {}
    for path in sorted(LEAN.rglob("*.lean")):
        rel = path.relative_to(LEAN)
        if rel.parts[0] == ".lake" or rel.name == "lakefile.lean":
            continue
        mods[".".join(rel.with_suffix("").parts)] = header_imports(path)

    errors = []
    owner = {}
    for mod in mods:
        homes = [name for name, globs in libs.items()
                 if any(glob_matches(g, mod) for g in globs)]
        if mod in exes:
            if homes:
                errors.append(f"{mod}: executable root is also in {homes}")
            continue
        if len(homes) != 1:
            errors.append(f"{mod}: must be in exactly one lean_lib, found {homes}")
            continue
        owner[mod] = homes[0]

    for mod, lib in sorted(owner.items()):
        for imp in mods[mod]:
            if imp in mods:
                target = owner.get(imp)
                if imp in exes:
                    errors.append(f"{mod} ({lib}) imports executable root {imp}")
                elif target is None:
                    continue  # already reported as unowned
                elif target != lib and target not in ALLOWED[lib]:
                    errors.append(f"{mod} ({lib}) imports {imp} ({target})")
            elif lib in EXTERNAL and imp.split(".")[0] not in EXTERNAL[lib]:
                errors.append(f"{mod} ({lib}) imports external {imp}")

    for exe, allowed in EXE_CLOSURE.items():
        if exe not in mods:
            errors.append(f"{exe}: executable root in EXE_CLOSURE not found")
            continue
        seen, stack = set(), [exe]
        while stack:
            mod = stack.pop()
            for imp in mods.get(mod, []):
                if imp in mods and imp not in seen:
                    seen.add(imp)
                    stack.append(imp)
        for mod in sorted(seen):
            if owner.get(mod) not in allowed:
                errors.append(f"{exe}: closure reaches {mod} ({owner.get(mod)}), allowed {sorted(allowed)}")

    for err in errors:
        print(f"check_layers: {err}")
    if errors:
        return 1
    counts = {name: sum(1 for l in owner.values() if l == name) for name in libs}
    summary = ", ".join(f"{name} {n}" for name, n in counts.items())
    print(f"check_layers: ok ({summary}, {len(exes)} executable roots)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
