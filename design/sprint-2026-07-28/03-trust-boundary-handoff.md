# Trusted-boundary ledger — sprint handoff

- **Sprint:** 2026-07-28 through 2026-08-10
- **Lane:** C — explicit trust and evidence accounting
- **DRI:** Trust-boundary lane
- **Supervisor:** Staff engineer
- **Status:** Complete — typed ledger, generated report, and audit landed
- **Master:** [Staff engineer sprint handoff](00-staff-engineer-master-handoff.md)
- **Depends on:** Lane B supplies the final semantic capstone/boundary; Lanes D
  and E supply current performance/numeric evidence.
- **Must not overlap:** This lane does not prove Lane B’s theorem and does not
  change backend semantics.

## Mission

Turn Tropical’s dispersed prose assumptions into one checked, queryable ledger.

The ledger must answer, for every claim below:

- What exactly is trusted?
- Is it a theorem, executable check, differential, golden, code inspection,
  numeric tolerance, unsafe optimization, or external dependency?
- Which files implement it?
- Which gate would fail if it drifted?
- Who owns discharging or narrowing it?

The ledger is not a ceremonial list of reassuring words. It must make
unsupported claims visible and must refuse duplicate or ownerless entries.

## Initial obligation inventory

The Day 1 audit starts with at least these entries:

1. **CLOCK_RAIL_IS_EXACT** — runtime integer rail implements the `Int`
   denotation modulo the documented headroom rules.
2. **REDUCE_REGION_EXECUTES_IN_ARRAY_ORDER** — JIT/wasm/MSL execute bank
   bodies for increasing indices with a scalar left fold and clamped prefix.
3. **LOWER_SIG_PTR_REFINES_TREE** — the unsafe pointer-memoized implementation
   returns the same arena/id result as the structural reference for immutable
   `Sig`.
4. **LLVM_TEXT_EXECUTES_PLAN** — generated LLVM implements the plan
   instruction semantics.
5. **MSL_EXECUTES_PLAN_WITHIN_NUMERIC_CONTRACT** — generated MSL implements
   the plan under the stated f32/SNR regime.
6. **WASM_SHARES_LLVM_SEMANTICS** — wasm is produced from the same LLVM
   emitter and remains sample-equivalent to native JIT on the gated surface.
7. **EXACT_FORMULA_APPROXIMATION** — interval arithmetic bounds rounding of
   the implemented formulas, but not yet distance from the true
   transcendental functions.
8. **HOST_PARAM_DISCIPLINES** — Lean and socket hosts apply the same
   re-anchoring formulas and slot order.
9. **CACHE_KEY_COHERENCE** — a cached kernel is invalidated by every semantic
   change that affects execution.
10. **EXTERNAL_TOOLCHAIN** — Lean core dyadics, LLVM/lld, Metal, RtAudio, and
    Turnstile enter the trusted computing base in explicitly different ways.
11. **SERIALIZED_PLAN_SCHEMA_IS_PLAN_5_ONLY** — the original brief proposed a
    scoped Plan-4 compatibility obligation; the staff retirement decision
    superseded it with exact Plan-5 validation and explicit retired-carrier
    rejection.

The audit must add omissions discovered by searching code and docs.

## Artifact design

### Machine-readable Lean ledger

Create:

```text
lean/Tropical/Trust.lean
```

with data shaped approximately as:

```lean
inductive EvidenceKind
  | theorem
  | executableGate
  | differential
  | golden
  | inspection
  | numericTolerance
  | unsafeOptimization
  | externalDependency

structure Obligation where
  id : String
  statement : String
  evidence : Array EvidenceKind
  implementationPaths : Array String
  gateNames : Array String
  owner : String
  status : String
```

Use typed enums for evidence and status. Strings are acceptable for paths,
gate names, and ownership. Do not encode an unproved runtime statement as a
Lean proposition or `axiom` merely to make the ledger look formal.

Expose a deterministic JSON or Markdown rendering through a small pure
function. The lane may add a `diffcli trust-report` verb only if this does not
conflict with Lane D’s CLI timing work; otherwise render it from a dedicated
small executable/test helper. The staff engineer resolves CLI ownership on
Day 1.

### Human-readable report

Generate or maintain:

```text
design/trust-boundary.md
```

from the same entries. It must include:

- the formal statement where one exists;
- the remaining unproved refinement;
- evidence and limitations;
- code/test links;
- follow-up priority.

The generated section must be reproducible. Handwritten explanatory sections
may surround it.

### Audit gate

Add a `tropicaltest` gate or a library-level check that verifies:

- obligation ids are unique;
- every obligation has an owner, implementation path, status, and evidence;
- every production `unsafe` or `implemented_by` occurrence is covered;
- referenced gate names are drawn from a maintained known-gate list or are
  explicitly marked manual;
- no obligation is marked “proved” without a theorem symbol.

The audit does not need filesystem reflection from Lean. A small checked list
plus a shell audit in CI is acceptable if the split is documented.

## Ownership boundaries

This lane owns:

- `lean/Tropical/Trust.lean`
- `design/trust-boundary.md`
- a dedicated trust audit fixture
- the trust-report rendering path

It may make narrow additions to `lean/Tropicaltest.lean` to register the audit.
It must coordinate that edit with Lane G.

It does not own:

- theorem implementations under `Tropical/Semantics`;
- emitters;
- engine runtime behavior;
- Metal numeric thresholds;
- compatibility deletion decisions.

## Work plan

### Day 1: trust-surface audit

- Search production Lean for `unsafe`, `implemented_by`, and any comments using
  “trusted,” “assumption,” or “by inspection.”
- Search C++ and backend docs for contracts not represented in Lean.
- Create the initial obligation table and identify an owner for every row.

### Day 2: schema review

The staff engineer approves:

- evidence taxonomy;
- status taxonomy;
- what “proved” is allowed to mean;
- rendering mechanism;
- audit mechanism.

Freeze the schema after this review.

### Days 3–4: ledger implementation

- Land the typed ledger.
- Add deterministic rendering.
- Add uniqueness/completeness tests.
- Populate all known entries without waiting for other lanes’ final results.

### Days 5–6: unsafe and gate audit

- Cover every production `unsafe` and `implemented_by`.
- Map current theorem modules and executable gates.
- Explicitly separate audio goldens from backend differentials.

### Days 7–8: integrate other lanes

- Link Lane B’s capstone theorem.
- Link Lane D’s reproducibility report and thresholds.
- Link Lane E’s Metal evidence and final qualification result.
- Link Lane F’s retired-schema rejection boundary.

### Day 9: adversarial review

An engineer who did not author the ledger selects five claims from root docs
and attempts to trace each claim to:

```text
statement → implementation → evidence → limitation
```

Any break is a ledger defect.

### Day 10: freeze and render

- Generate the checked-in human report.
- Run the audit on the release candidate.
- Hand all open obligations to the staff engineer with owners and dates.

## Acceptance gates

1. One machine-readable ledger is the source for the report.
2. Every entry has a unique stable id.
3. Every production `unsafe`/`implemented_by` site is listed.
4. `CLOCK_RAIL_IS_EXACT` and
   `REDUCE_REGION_EXECUTES_IN_ARRAY_ORDER` are represented as runtime
   obligations, not falsely as discharged theorems.
5. The Exact layer’s analytic limitation is explicit.
6. JIT-vs-wasm, Metal-vs-JIT, and frozen audio goldens are distinct evidence
   categories.
7. The audit runs in the ordinary validation path.
8. `make validate` is green.

## Non-goals

- No new axioms.
- No full LLVM or Metal verification.
- No proof of transcendental approximation error.
- No CI requirement for a 30-minute hardware soak.
- No generic governance framework outside this repository.

## Stop and escalate

Escalate if:

- two backends implement observably different operation semantics;
- a source claim has no present evidence and cannot honestly be weakened;
- an unsafe site lacks a total reference implementation;
- the audit would require fragile parsing of compiler output;
- an obligation owner refuses or cannot accept the boundary.

The correct sprint outcome may be a visible red/open obligation. Do not hide it
by changing the taxonomy.

## Handoff package

Leave:

- `Tropical.Trust` and its renderer;
- `design/trust-boundary.md`;
- passing trust-audit output;
- an owner/status list of open obligations;
- a short rule for adding a new obligation with future backends or unsafe
  optimizations.
