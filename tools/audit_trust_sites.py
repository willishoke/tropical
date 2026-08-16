#!/usr/bin/env python3
"""Compare tracked Lean trust escapes with Tropical.Trust's typed site list."""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORTER = ROOT / "lean" / ".lake" / "build" / "bin" / "trustreport"


def typed_sites() -> set[tuple[str, str, str, str]]:
    result = subprocess.run(
        [str(REPORTER), "--sites"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    sites: set[tuple[str, str, str, str]] = set()
    for line in result.stdout.splitlines():
        fields = line.split("\t")
        if len(fields) != 4:
            raise ValueError(f"invalid typed trust-site row: {line!r}")
        sites.add(tuple(fields))
    return sites


def tracked_lean_files() -> list[pathlib.Path]:
    result = subprocess.run(
        ["git", "ls-files", "lean/Tropical"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    tracked = [
        ROOT / path
        for path in result.stdout.splitlines()
        if path.endswith(".lean")
    ]
    # `git ls-files` retains paths deleted in the candidate working tree until
    # the cutover commit is staged. Audit the tree being qualified rather than
    # crashing while a tracked module is intentionally being removed.
    return [path for path in tracked if path.exists()]


def source_sites() -> set[tuple[str, str, str, str]]:
    sites: set[tuple[str, str, str, str]] = set()
    unsafe_pattern = re.compile(r"\bunsafe\s+def\s+([A-Za-z_][A-Za-z0-9_']*)")
    implemented_pattern = re.compile(
        r"@\[\s*implemented_by\s+([A-Za-z_][A-Za-z0-9_']*)\s*\]"
        r"\s*def\s+([A-Za-z_][A-Za-z0-9_']*)",
        re.MULTILINE,
    )
    for path in tracked_lean_files():
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(ROOT).as_posix()
        for match in unsafe_pattern.finditer(text):
            sites.add(("unsafe definition", relative, match.group(1), ""))
        for match in implemented_pattern.finditer(text):
            sites.add((
                "implemented_by marker",
                relative,
                match.group(2),
                match.group(1),
            ))
    return sites


def main() -> int:
    if not REPORTER.exists():
        print(f"trust reporter is missing: {REPORTER}", file=sys.stderr)
        return 2
    typed = typed_sites()
    actual = source_sites()
    missing = sorted(actual - typed)
    stale = sorted(typed - actual)
    if missing or stale:
        for site in missing:
            print(f"unledgered production trust site: {site}", file=sys.stderr)
        for site in stale:
            print(f"stale typed trust site: {site}", file=sys.stderr)
        return 1
    print(f"trust source audit passed ({len(actual)} sites)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
