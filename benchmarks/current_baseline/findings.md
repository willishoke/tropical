# Current performance baseline findings

**Status:** Harness complete; full frozen row pending after the implementation
commit. The figures below are genuine one-repeat smoke measurements from
2026-07-27 on the canonical Apple M1 Pro (10 CPU cores, 16 GPU cores, 16 GB),
macOS 26.3, LLVM 22.1.7, RelWithDebInfo, 44.1 kHz, B=512.

## Smoke evidence

| fixture | plan instructions | cold ORC load | warm ORC load | JIT median / block | topology→publish |
|---|---:|---:|---:|---:|---:|
| fixed sine | 61 | 17.85 ms | 6.19 ms | 0.0015 ms | — |
| through-zero flanger | 565 | 34.54 ms | 16.52 ms | 0.0318 ms | — |
| four-ring playground | 137 | 46.67 ms | 20.01 ms | 0.2425 ms | 368.25 ms |
| modal bank 16 | 137 | 32.81 ms | 15.38 ms | 0.0602 ms | 312.07 ms |
| modal bank 512 | 137 | 347.27 ms | 77.86 ms | 2.0968 ms | 1093.11 ms |
| dynamic bank, K=512/live=128 | 137 | 348.60 ms | 77.84 ms | 0.5914 ms | 1114.17 ms |

These are smoke numbers, not the frozen product table. They already support
two structural observations:

1. The banked audio plan is flat in capacity at 137 instructions from K=16 to
   K=512. Capacity-dependent coefficient materialization remains visible in
   load time, as designed.
2. The current four-ring flagship meets the kickoff 500 ms warm structural
   target in this run (368 ms end-to-end). The 512-capacity bank remains below
   the 2 s cold hypothesis but is not a proxy for every composition.

## Decision

**Provisional recommendation: no tiered preview implementation yet.** The
flagship smoke clears the interactive budget and the banked plan is
structurally flat. Freeze the full repeated matrix before turning this into a
signed staff decision; if its four-ring median exceeds 500 ms or variance
exceeds 20%, change the decision to “more isolation needed,” not to an
unmeasured optimization.

## Candidate regression budgets

- Four-ring topology→publication: median ≤500 ms, p95 reported (machine-local).
- 512-bank cold topology→publication: median ≤2 s.
- Banked audio plan instruction growth from K=16 to K=512: zero.
- JIT block p99: below 50% of the 11.61 ms B=512 deadline.
- Raw slot-write p99: below one 512-sample block.
- Emitted plan/render bytes must remain unchanged by instrumentation.

## Limitations

- The frontend subprocess wall cannot yet separate authoring/lowering from
  plan serialization.
- ORC load is a single public wall, not separate LLVM emit and object-link
  timers.
- Smoke data has one repeat and was recorded from a dirty implementation
  worktree; it is retained as development evidence, not the frozen row.
- Metal qualification, including live DAC statistics, is reported separately
  in `../metal_live/findings.md`.
