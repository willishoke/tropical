# LLVM benchmarks and spikes

A playground for measuring LLVM-level optimization behavior that
informs tropical's architectural decisions. Each subdirectory or
top-level `.md` file is a focused experiment or recorded result.

## Index

### Completed

- **`active_set_spike/`** — recorded findings from the spike that
  preceded the active-set runtime (PR #146). Verifies that
  `alwaysinline` + conditional dispatch preserves unified-kernel
  optimization, that GVN eliminates inter-instance slot loads across
  function boundaries, and that LLVM loop-unswitching produces
  branchless specialized loops for dynamic alive at O3. Includes
  the source C files (`spike.c`, `current_vs_proposed.c`, `phaser.c`,
  `phaser_compare.c`) and their optimized IR dumps under
  `ir-dumps/`. See `active_set_spike/findings.md` for the writeup.

- **`compile_time/`** — LTO vs no-LTO compile-time scaling for
  monolithic (one TU) vs separate-modules (N TUs) layouts at
  N ∈ {16, 64, 256, 512, 1024, 2048, 4096} stateful kernels. Code-golf
  generator (`gen.py`) emits the C; `bench.py` sweeps and records to
  `data/results.csv`. 15-second hard timeout per compile.

  Findings (2026-05-14): monolithic-nolto scales roughly linearly to
  ~1024 kernels in ~390 ms; LTO cost is superlinear (5.3× at N=1024,
  10.3× at N=2048) and not worth its price for ORC's single-module
  pipeline. Separate-modules is bottlenecked by clang frontend
  overhead (~25 ms/TU), not LTO. The M11 fractal-compilation
  architecture (one LLVM function, N nested basic blocks) maps onto
  the monolithic row — the right choice for compile-time scaling.
  See `compile_time/README.md`.

### In progress

- **`inlining_across_modules/`** — investigates whether the
  active-set findings extend to separately-compiled LLVM modules.
  The active-set spike verified intra-procedural GVN works across
  `alwaysinline` boundaries within one LLVM function. This experiment
  asks whether the same optimization survives when kernels live in
  *separate* modules linked together — relevant to the operadic
  substrate's hot-swap granularity and external-primitive realization
  paths discussed in `design/operadic_ir.md`.
  
  Initial results (2026-05-14): linear-chain tree, 4 variants;
  **LTO dissolves the module boundary** (monolithic and separate-
  modules variants run at identical speed with LTO). Without LTO,
  separate modules pay a ~2.2× cost. See `inlining_across_modules/
  README.md` for the writeup. Variants pending: separate static
  functions (variant 2), external `.so` via dlopen (variant 5),
  LadderFilter-pattern tree, Phaser16-scale tree, conditional-alive
  testing.

## Conventions

- Each experiment is either a single `.md` (for small/contained
  experiments) or a subdirectory containing `findings.md` plus
  source artifacts.
- Source C files are checked in alongside the findings so the spike
  is reproducible.
- IR dumps are checked in under `ir-dumps/` (or similar) when they're
  small enough to be useful as reference. Large dumps should be
  excluded; reproduce them from the source.
- Each experiment's findings doc should state: the question, the
  tropical-context motivation, the methodology, the measured
  results, and what the results imply for architectural decisions.
