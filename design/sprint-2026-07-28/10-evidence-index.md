# Consolidation sprint evidence index

- **Baseline:** `9c492dd52503ee654cb927d9bca67fd97c817cb0`
- **Release candidate:** pending
- **Staff acceptance:** pending

This index is the entry point for release review. Evidence is checked in;
terminal-only observations and untracked local handoffs are not release
evidence.

## Outcomes

| Outcome | Primary artifact | Gate or review | Status |
|---|---|---|---|
| Day-1 reality | [baseline and environment](08-day-1-baseline.md) | `make validate` | pass |
| Architecture truth | `design/architecture.md` | stale-token and link audit | pending |
| Semantic boundary | `lean/Tropical/Semantics/` | Lean library build and independent review | pending |
| Trust boundary | `design/trust-boundary.md` and `lean/Tropical/Trust.lean` | ordinary trust audit | pending |
| Current performance | `benchmarks/current_baseline/findings.md` | harness smoke and raw-data validation | pending |
| Metal qualification | `benchmarks/metal_live/findings.md` | automated smoke plus manual evidence | pending |
| Compatibility quarantine | `design/compatibility-matrix.md` | production non-emission gate | pending |
| Day-10 candidate | this index | clean-checkout Tier 3 | pending |

## Validation

| Run | Commit | Host | Result | Notes |
|---|---|---|---|---|
| Day 1 | `9c492dd` | Apple M1 Pro / macOS 26.3 | pass | 118/118 Tropical gates, 110 Bun pass/1 capability skip, 2/2 CTest. Fresh worktree required submodule initialization and Lake dependency access. |
| Rolling integration | pending | Apple M1 Pro / macOS 26.3 | pending | Run after lane merge. |
| Release candidate | pending | Apple M1 Pro / macOS 26.3 | pending | Must be a clean worktree. |
| Linux CI | pending | GitHub-hosted Linux | pending | External gate; do not infer locally. |

## Decision record

See the [staff decision log](09-staff-decision-log.md). No `pending` P0
decision may remain when S-08 is signed.

## Remaining obligations

Unfinished obligations at close must name an owner, priority, and target date.
A hardware qualification failure remains visible here and blocks release; it
is not converted into a documentation footnote.
