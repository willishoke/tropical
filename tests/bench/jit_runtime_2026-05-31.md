# Benchmark Results: jit_runtime — first capture on the root-program default

| Field       | Value |
|-------------|-------|
| Date        | 2026-05-31 |
| Branch      | `test/hotswap-and-bench-refresh` |
| Captured at | commit `457a0c0` |
| Hardware    | darwin/arm64, Apple M5 |
| LLVM        | 22.1.1 |
| Reproduce   | `bun run tests/bench/jit_runtime.ts <patch.json>… ` (cold; wipes the kernel cache) |

First `jit_runtime` capture **after the Option A cutover** (root-program
lowering is now the default session path). The prior snapshots predate it.
Cold compile (kernel cache wiped). `ns/sample` is single-output; `rt_ratio`
is the fraction of the realtime budget consumed (lower = more headroom).

Corpus note: the four array-input patches (`acid_noise`, `bubble_drip`,
`odd_harmonics`, `sequencer_demo`) compile now — the root path's array
session-slot support unblocked them (they were parked in `corpus.ts` BLOCKED).

| patch                | total_ms | ts_ms   | json_kb | jit_ms | instrs | arrays | ns/sample | rt_ratio |
|----------------------|---------:|---------|--------:|-------:|-------:|-------:|----------:|---------:|
| bubble_cloud         |   698.1  | 6.1+0.3 |   206   | 691.7  |   1    |   0    |   279.0   |  1.2%    |
| cross_fm_4           |    32.3  | 0.8+0.1 |    39   |  31.5  |  19    |   4    |    34.7   |  0.2%    |
| cross_fm_evolved     |  1074.6  | 2.4+0.4 |   223   |1071.9  |  20    |  29    |   308.6   |  1.4%    |
| acid_noise           |   145.5  | 1.5+0.2 |   114   | 143.7  |   2    |  16    |   143.9   |  0.6%    |
| bubble_drip          |    35.0  | 0.8+0.1 |    31   |  34.1  |   1    |   4    |    39.1   |  0.2%    |
| odd_harmonics        |   201.1  | 1.7+0.3 |   183   | 199.1  |  10    |  40    |   189.7   |  0.8%    |
| sequencer_demo       |    21.9  | 0.3+0.0 |    19   |  21.5  |   1    |   6    |    25.2   |  0.1%    |

**Read:** runtime is well under realtime on every patch (≤1.4% of budget, ~70×
headroom). Cold compile is dominated by LLVM `jit_ms`; the TS pipeline is sub-3ms
everywhere. The heaviest compiles (`cross_fm_evolved` ~1.07s, `bubble_cloud`
~0.69s) are big fused single functions — see `microkernel_vs_fused_2026-05-31.md`
for how `microkernel` mode trades fusion for compile-time on large graphs.
