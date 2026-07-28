# Current performance baseline findings

**Status:** Corrected harness complete; the prior full row is retained as
runtime evidence but is not accepted for structural-edit decisions because it
generated each topology only once. A new frozen full row is pending from the
corrected implementation commit. The figures below are genuine one-repeat
development measurements from
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

These are superseded development numbers, not the frozen product table. They
support
two structural observations:

1. The banked audio plan is flat in capacity at 137 instructions from K=16 to
   K=512. Capacity-dependent coefficient materialization remains visible in
   load time, as designed.
2. The current four-ring flagship meets the kickoff 500 ms warm structural
   target in this run (368 ms end-to-end). The 512-capacity bank remains below
   the 2 s cold hypothesis but is not a proxy for every composition.

## Decision

**Decision pending corrected frozen evidence.** A three-repeat corrected smoke
measured the four-ring full-generation wall at 386.10 ms cold and 358.88 ms
warm medians. Its topology-add cold range was 25.2% of the median, above the
handoff's 20% escalation line; the warm range was 3.5%. This is a noise signal,
not permission to tune a threshold. Run the corrected full matrix from a clean
commit and present the raw repeat distribution before a staff engineer signs
“no tiering,” “tiering recommended,” or “more isolation needed.”

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
- The committed `tropical_baseline_row_1` data has valid repeated ORC/runtime
  rows, but only one topology-generation sample. Schema 2 fixes that boundary
  with repeated cold/warm full generations and explicit structural edits.
- Metal qualification, including live DAC statistics, is reported separately
  in `../metal_live/findings.md`.
