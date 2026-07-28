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
sine, through-zero flanger, one ring, ring→reverb, the exact four-ring product
circuit, gong, plucked string, fixed banks at 16/64/128/256/512, a
512-capacity dynamic bank at three live counts, and a composed nested-bank
fixture.

The flagship row does not carry a benchmark-owned approximation. It reads the
strict-JSON `GRAPH` declaration between the exact-product markers in
`playground/renderer/app.js`, the same object the renderer passes to
`load_patch_graph`. The schema gate pins its oscillator address path, four
address-driven rings, current parameters and default-six-partial semantics,
modal mix, reverb, filter, output, and taps. Its two structural edits add one
address-driven ring and make one ring's default six partials explicitly seven.

`runtime_bench.cpp` consumes already-emitted LLVM/MSL/manifest artifacts. This
keeps artifact generation, ORC/MSL load, raw slot writes, and block execution
as separate walls. The orchestrator retains every raw load/write/block sample
in JSONL and adds minimum/median/p95/p99/max summaries.

Each compile repeat now performs two complete generations: a cold generation
with a fresh benchmark-owned cache and a warm generation in a new subprocess
reusing that exact cache. Raw walls and minimum/median/variance summaries are
recorded separately. Schema-3 rows retain the byte count and SHA-256 digest of
every manifest, audio IR, coefficient IR, and MSL artifact for every cold and
warm repeat, including both flagship edits. `emitted_bytes_stable` is checked
against that retained digest matrix rather than standing alone.

Run the focused schema/provenance gates with:

```sh
python3 benchmarks/current_baseline/test_run.py
```

## Cache safety

The production default remains:

```text
~/.cache/tropical/kernels/<build-id>/
```

Two opt-in benchmark controls are available:

- `TROPICAL_KERNEL_CACHE_ROOT=/owned/path` moves the build-id subtree under a
  caller-owned root.
- `TROPICAL_KERNEL_CACHE_DISABLE=1` disables disk reads and writes.

The harness removes every inherited `TROPICAL_*` variable, then explicitly
sets and records stage-0 enabled, banked realization, JIT O2, synchronous Metal
for baseline rows, and its cache root. A cold sample is a complete generation
with a fresh root; the paired warm sample is a second complete generation
reusing only that root. It snapshots the ordinary cache before and after and
records a SHA-256 digest over every relative path and file digest. If another
process changes the ordinary cache during the measurement, the trailer retains
the added, removed, and modified paths so a contaminated run cannot masquerade
as cache-safe evidence. The harness never deletes the ordinary cache.

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
