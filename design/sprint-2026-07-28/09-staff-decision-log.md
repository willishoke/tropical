# Staff decision log

This is the reconstructable decision record for the consolidation sprint.
Evidence paths are repository-relative and remain valid at the release
candidate.

| ID | Date | Decision | Alternatives rejected | Evidence | Owner |
|---|---|---|---|---|---|
| S-00 | 2026-07-27 | Use isolated worktrees rooted at `9c492dd`; keep the kickoff tree and its unrelated untracked research untouched. | Implement in the dirty kickoff tree; copy unrelated local work into sprint branches. | [Day-1 baseline](08-day-1-baseline.md) | staff |
| S-01 | 2026-07-27 | Freeze the first semantic boundary as a refusal-aware, carrier-parametric denotation over production `Sig`, plus an explicit arena well-formedness contract and an all-constructor `LowersTo` relation over production `lowerSigTree`. | A proof-only shadow syntax; an evaluator that silently defaults refusals; an arena theorem without a sound dedup premise. | `lean/Tropical/Semantics/`, `design/semantics-spine.md` | staff |
| S-02 | 2026-07-27 | Use typed evidence kind, status, priority, owner, maintained gate, and optional theorem-symbol fields; generate the human report from Lean and fail ordinary validation when filesystem-discovered production trust escapes differ from the typed site inventory. | A prose-only ledger; treating executable or inspection evidence as a theorem; a hard-coded audit that cannot discover a new source site. | `lean/Tropical/Trust.lean`, `tools/audit_trust_sites.py`, `design/trust-boundary.md` | staff |
| S-03 | 2026-07-27 | Select handoff fallback 1: `lowerSigTree_lowersTo` is the checked sprint capstone and is explicitly not denotational preservation. The full theorem is blocked on production `DedupSound` preservation and lawful `ENode` equality/hash instances. | A hidden axiom; claiming the relational theorem preserves denotation; delaying all semantic evidence until the stronger proof is available. | `design/semantics-spine.md`, `lean/Tropical/Semantics/LowerSig.lean` | staff |
| S-04 | 2026-07-27 | Quarantine plan 4 as parser/runtime compatibility only, retain it through 2026-10-01 for unknown direct-C callers, and prove current Lean front doors emit state-free plan 5 only. | Treating plan 4 as current source semantics; immediate removal without an external-caller inventory; an unbounded legacy promise. | `design/compatibility-matrix.md`, `engine/tests/test_compat_legacy_plan4.cpp`, `lean/Tropical/Testing/PlanWire.lean` | staff |
| S-05 | pending | P1 scope cuts, if any, pending integration. | — | [Evidence index](10-evidence-index.md) | staff |
| S-06 | pending | Tiered preview recommendation pending measured performance rows. | — | `benchmarks/current_baseline/findings.md` | staff |
| S-07 | pending | Qualified Metal defaults pending hardware evidence. | — | `benchmarks/metal_live/findings.md` | staff |
| S-08 | pending | Release-candidate acceptance pending Tier-3 validation. | — | [Evidence index](10-evidence-index.md) | staff |
