# Current performance baseline

This directory measures the current compiler/runtime without changing plan
schemas or enabling a preview/tiered compiler.

## Build and run

```sh
make build
make lean
benchmarks/current_baseline/run.sh --suite smoke --metal
benchmarks/current_baseline/run.sh --suite full --metal --repeats 3
```

The full matrix is checked in at `fixtures/matrix.json`. It covers the fixed
sine, through-zero flanger, one ring, ring→reverb, four-ring flagship, gong,
plucked string, fixed banks at 16/64/128/256/512, a 512-capacity dynamic bank
at three live counts, and a composed nested-bank fixture.

`runtime_bench.cpp` consumes already-emitted LLVM/MSL/manifest artifacts. This
keeps artifact generation, ORC/MSL load, raw slot writes, and block execution
as separate walls. The orchestrator retains every raw load/write/block sample
in JSONL and adds minimum/median/p95/p99/max summaries.

## Cache safety

The production default remains:

```text
~/.cache/tropical/kernels/<build-id>/
```

Two opt-in benchmark controls are available:

- `TROPICAL_KERNEL_CACHE_ROOT=/owned/path` moves the build-id subtree under a
  caller-owned root.
- `TROPICAL_KERNEL_CACHE_DISABLE=1` disables disk reads and writes.

The harness always uses a fresh temporary root beneath its run directory. A
cold sample is a new process with a fresh root; the paired warm sample is a
second process reusing only that root. It snapshots the ordinary cache before
and after and records whether it changed. It never deletes the ordinary cache.

## Metric boundaries

- Program fixtures report the whole `diffcli compile` wall and separate
  artifact emission subprocess walls.
- Typed playground graphs use `TROPICAL_STAGE0_DUMP` to preserve the exact
  post-split audio LLVM, coefficient LLVM, manifest, and MSL loaded by
  `render-graph`.
- ORC load includes IR parse, optimization/code generation, add-module, and
  lookup because the existing public boundary does not split those phases.
- Metal load includes the mandatory dual JIT load plus MSL library/PSO
  construction.
- `topology_edit_to_publication_ns` is the end-to-end graph compile, split,
  emit, dual-load, and publish subprocess wall.
- Arena node count and a standalone stage-0 split timer remain `null`: exposing
  them would require invasive compiler instrumentation. The report does not
  infer either value.

The JSONL environment record includes the commit/dirty state, whitelisted
hardware facts, OS, Lean/LLVM/build configuration, rate/block, flags, and cache
contract. Stable device identifiers are never collected.
