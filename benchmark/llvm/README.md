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

### Planned

- **`inlining_across_modules.md`** — investigates whether the
  active-set findings extend to separately-compiled LLVM modules.
  The active-set spike verified intra-procedural GVN works across
  `alwaysinline` boundaries within one LLVM function. This experiment
  asks whether the same optimization survives when kernels live in
  *separate* modules linked together — relevant to the operadic
  substrate's hot-swap granularity and external-primitive realization
  paths discussed in `design/operadic_ir.md`. Doc only at this point;
  harness implementation pending.

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
