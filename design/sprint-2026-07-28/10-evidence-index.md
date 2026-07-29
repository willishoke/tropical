# Consolidation sprint evidence index

- **Baseline:** `9c492dd52503ee654cb927d9bca67fd97c817cb0`
- **Local release candidate:** `2dfab8d0f4c1261d935a222aaf572b9bff84e8fa`
- **Staff acceptance:** local sprint candidate accepted; full Tier 3 and
  external release qualification withheld

This index is the entry point for release review. Evidence is checked in;
terminal-only observations and untracked local handoffs are not release
evidence.

## Outcomes

| Outcome | Primary artifact | Gate or review | Status |
|---|---|---|---|
| Day-1 reality | [baseline and environment](08-day-1-baseline.md) | `make validate` | pass |
| Architecture truth | `design/architecture.md` | stale-token review, source trace, and local links | pass |
| Semantic boundary | `lean/Tropical/Semantics/` | Lean build and independent review | pass — relational capstone; denotational preservation not claimed |
| Trust boundary | `design/trust-boundary.md` and `lean/Tropical/Trust.lean` | generated report and source-discovery audit | pass |
| Current performance | `benchmarks/current_baseline/findings.md` | harness smoke and raw-data validation | pass — fixed renderer measured; no optimization authorized |
| Retired boundaries | `design/compatibility-matrix.md` | current-schema emission and retired-carrier rejection gates | pass |
| Metal correctness/runtime | `benchmarks/metal_live/findings.md` | GPU differentials, native tests, fail-closed review | pass for the stated correctness gates |
| Live Metal qualification | final B512/D3 JSONL row linked below | single actual-DAC long attempt | **fail — release blocking on canonical M1 Pro** |
| Day-10 local candidate | this index | clean macOS Tier-3 components | pass |
| Full Tier 3 / external release | this index | local result plus Linux CI and supported live-Metal row | **incomplete / not accepted** |

## Validation

| Run | Commit | Host | Result | Notes |
|---|---|---|---|---|
| Day 1 | `9c492dd` | Apple M1 Pro / macOS 26.3 | pass | 118/118 Tropical gates, 110 Bun pass/1 capability skip, 2/2 CTest. Fresh worktree required submodule initialization and Lake dependency access. |
| Rolling integration | `bd7c9bf` | Apple M1 Pro / macOS 26.3 | pass | `make validate JOBS=4`: Lean/trust 121/121, Bun 116 pass/1 intentional capability skip, native CTest 2/2 including Metal. |
| Local release candidate | `2dfab8d` | Apple M1 Pro / macOS 26.3 | pass | Clean worktree; timed `make validate JOBS=4`: same gate totals, 144.08 s wall. This is the local Tier-3 subset, not full Tier 3. |
| ThreadSanitizer | `bd7c9bf` | Apple M1 Pro / macOS 26.3 | pass | Fresh `check_runtime_tsan` build with LLVM 22; no race reported. Candidate changes after this SHA are evidence/docs only. |
| Benchmark harnesses | `bd7c9bf` | Apple M1 Pro / macOS 26.3 | pass | Current-baseline self-tests 10/10; Metal harness self-tests 30/30. |
| Velocity discriminator | `bd7c9bf` | Apple M1 Pro / macOS 26.3 | pass | Exact replay 142.543 dB; deliberately stale batch-start oracle 84.759 dB. |
| Linux CI | not run | GitHub-hosted Linux | **not run** | External gate; never inferred from local validation. |

## Metal qualification result

The single final B512/D3 actual-DAC attempt is
[`final-soak-b512-d3-1800s-bd7c9bf-m1pro-20260728.jsonl`](../../benchmarks/metal_live/data/final-soak-b512-d3-1800s-bd7c9bf-m1pro-20260728.jsonl)
(SHA-256
`ab03fbc7ad849799cb0a770c4858d552ccba5d6c505d13792ae8b532e9e8bfd0`).
It aborted after 450.050 measured seconds at the scheduled clock jump: one
21.009750 ms callback exceeded the 11.609977 ms deadline. The measured window
otherwise recorded zero underruns, ownership failures, Metal dispatch
failures, and non-finite samples; callback p99 was 0.203 ms and the start
reference was 144.013 dB. Independent review confirmed the hard miss and the
re-prime-window association. The row was not retried.

## Decision record

See the [staff decision log](09-staff-decision-log.md). S-08 accepts the local
sprint candidate and withholds external release; S-12 records the Metal
qualification failure. No P0 decision remains pending.

## Remaining obligations

| Obligation | Owner | Priority | Target |
|---|---|---|---|
| Redesign or bound synchronous Metal re-prime before proposing another supported live configuration; add stage timing before any new actual-DAC qualification request. | Apple runtime | P0 for live-Metal release | Separately authorized follow-up; no date committed |
| Run GitHub-hosted Linux CI on the exact candidate lineage. | Integration | P0 for external release | Before any external release |
| Decide whether B256 merits a qualification attempt. | Product + Apple runtime | P2 | Optional; no date committed |
| Extend the relational lowering capstone to denotational preservation and backend obligations. | Lean semantics | P2 | Nonblocking follow-up; no date committed |
| Add direct rendered D0 reset/hot-swap and full glide-endpoint probes. | Runtime tests | P2 | Nonblocking hardening; no date committed |

The separately scoped demo/product artifact is not a remaining obligation of
this sprint.
