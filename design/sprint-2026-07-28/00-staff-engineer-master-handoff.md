# Tropical consolidation sprint — staff engineer master handoff

- **Sprint window:** Tuesday 2026-07-28 through Monday 2026-08-10
- **Working days:** 10
- **Supervisor:** Staff engineer
- **Sprint type:** Architecture consolidation and evidence
- **Feature policy:** No new synthesis vocabulary or modal atoms
- **Status:** Integration in progress; release candidate pending final
  validation and B512/D3 long-soak evidence

## Accelerated outcome amendment

The calendar below is a sequencing guide, not elapsed-time evidence. Staff
accelerated the lanes while preserving their review boundaries. The current
close-state is:

- architecture, trust, exact fixed-renderer performance, and the relational
  semantics capstone are integrated;
- `lowerSigTree_lowersTo` reinforces all production constructors but is not
  denotational or backend verification; the stronger semantic theorem is a
  non-blocking follow-up;
- the fixed playground renderer is a separately scoped demo/reference fixture,
  so its measured one-time load does not authorize tiering or compile-time
  optimization;
- Metal support is configuration-specific: B128/D3 is unsupported on the
  canonical M1 Pro, B256 is untested, and B512/D3 is a candidate pending its
  required final 30-minute actual-DAC soak;
- the original Plan-4 quarantine decision has been superseded by an explicit
  aggressive-retirement decision; `tropical_program_2` remains current; and
- demo/product-surface work is out of this sprint and will be scoped
  separately.

## Executive mandate

Tropical has completed a month of architectural compression:

- stateful source semantics were removed;
- time became an exact, navigable coordinate;
- `Sig` became the sole authoring vocabulary and `ENode` the trunk IR;
- the literate surface language, elaborator, rich strata passes, and partial
  walkers retired;
- banks-as-data and stage-0 splitting made heavy modal programs practical;
- Metal became a live heavy-patch backend;
- exact bake arithmetic and the first substantive Lean theorems narrowed the
  trusted base.

The next two weeks convert those discoveries into a platform a team can trust.
The sprint is successful when source, proofs, runtime evidence, tests, and
documentation agree on one current architecture.

The desired close-state is:

> One small closed-form calculus, one explicit semantic/trust boundary, three
> execution targets with an explicit hardware envelope, one current schema
> boundary with legacy removed, and a clean release baseline.

This is deliberately not the sprint that adds the next instrument, restores a
surface language, introduces state, or generalizes to video/control rates.

## Sprint outcome

All seven lanes must compose into these five outcomes:

1. **Truth:** Current architecture docs describe the code at the release
   candidate, including the direct JSON patch-bay path and Metal.
2. **Meaning:** A checked all-constructor Lean theorem reinforces the first
   production lowering boundary relationally; denotational preservation stays
   explicit and non-blocking.
3. **Trust:** Every remaining assumption, unsafe optimization, external
   dependency, and empirical correctness gate is recorded in one queryable
   ledger.
4. **Evidence:** Current post-bank compile/edit performance and Metal’s
   reliability/control-latency envelope are measured reproducibly.
5. **Boundary:** Legacy stateful plan/runtime support is retired; current
   schema names and omission defaults remain explicit.

The staff engineer owns composition and scope, not every implementation.

## Workstream index

| Lane | Handoff | Must-land result | Primary owner profile |
|---|---|---|---|
| A | [Architecture truth](01-architecture-truth-handoff.md) | Current source-to-sound architecture and invariant map | Compiler generalist / technical writer |
| B | [Semantic spine](02-semantics-spine-handoff.md) | `lowerSigTree` denotation-preservation theorem or approved explicit fallback | Lean proof engineer |
| C | [Trust boundary](03-trust-boundary-handoff.md) | Typed obligation ledger, report, and audit | Formal-methods/compiler engineer |
| D | [Performance baseline](04-current-performance-baseline-handoff.md) | Reproducible post-bank compile/edit/runtime data and tiering decision | Compiler performance engineer |
| E | [Metal qualification](05-metal-qualification-handoff.md) | Buffer/depth sweep, soak, parameter-latency measurements, stale-future tests | Apple GPU/runtime engineer |
| F | [Compatibility quarantine](06-compatibility-quarantine-handoff.md) | Legacy matrix, non-emission gate, bounded retain/remove recommendation | Runtime/compiler engineer |
| G | [Integration and release](07-integration-release-handoff.md) | Continuously green integration branch and clean release candidate | Build/release engineer |

Every lane document is a self-contained handoff. This master is authoritative
only for cross-lane priorities, ownership, dependencies, and scope changes.

## Priority tiers

### P0 — sprint cannot close without these

- Clean Day-1 baseline or a staff-owned diagnosis of every red gate.
- Current architecture rewrite.
- Checked semantic theorem or the fallback explicitly selected by Day 4.
- Trust ledger covering all current production `unsafe`/`implemented_by`
  sites and named backend assumptions.
- Production non-emission gate for legacy state.
- Reproducible current performance report.
- Required Metal short smoke and either a passing 30-minute soak or a visible
  release-blocking qualification failure.
- Clean-checkout Day-10 validation.

### P1 — may be cut only at the Day-8 scope freeze

- Generated human trust report from the Lean ledger.
- Empirical parameter-latency matrix for all four write disciplines.
- CTest/test-tree compatibility labeling.
- Current bank-capacity and dynamic-count performance sweep.
- Documentation status headers for historical design notes.

### P2 — stretch; never endanger P0/P1

- Second Apple Silicon machine.
- Desktop-contention comparison beyond one required probe.
- Stronger semantic theorem than the committed first boundary.
- Mechanical separation of compatibility test helpers.
- Stable structural performance proxy in CI beyond existing flatness gates.

## Non-negotiable invariants

All lane decisions preserve:

1. Production kernels are pure `f(τ, params)`.
2. Current front doors cannot spell state or feedback.
3. Cycles are refused at constructing boundaries.
4. `Sig` and `ENode` remain the single authoring/trunk vocabulary pair.
5. Bank folds preserve operand order; no float reassociation is introduced.
6. The fixed-point clock rail stays integer through its declared boundary.
7. JIT remains the CPU correctness reference and random-access scope path.
8. Frozen audio goldens anchor correctness; cross-backend agreement alone does
   not prove it.
9. No new `axiom`, `sorry`, or `partial`.
10. No new `unsafe` without a total reference implementation, ledger entry,
    and staff approval.
11. Built Lean binaries are invoked directly; do not use `lake exe`.
12. Existing user work and unrelated dirty-tree files are not modified.

Any proposed exception is a staff decision and a sprint-scope change.

## Team topology

### Recommended staffing

The plan is sized for:

- one staff supervisor/integration reviewer;
- one Lean proof engineer for Lane B;
- one formal/compiler engineer for Lane C;
- one compiler performance engineer for Lane D;
- one Apple runtime engineer for Lane E;
- one compiler/runtime generalist covering Lane F;
- one documentation/compiler generalist for Lane A;
- Lane G owned by a build engineer or the staff supervisor with support.

People may own more than one lane if staffing is smaller, using only these safe
combinations:

- A + F: architecture truth plus compatibility classification;
- C + B: trust ledger plus semantics, if there is still an independent Lean
  reviewer;
- D + E: general measurements plus Metal qualification;
- G + staff supervision.

Avoid assigning one person both B and G: proof uncertainty must not compete
with release integration. Avoid assigning one person both E and F if both need
`FlatRuntime` changes.

### Required independent reviews

- Lane B theorem: another Lean-capable engineer.
- Lane E runtime changes: a C++/Metal reviewer who did not author them.
- Lane A architecture: one engineer traces the document against source.
- Lane F compatibility: one adversarial front-door reachability review.
- Release candidate: staff engineer plus integration DRI.

## File ownership and collision map

| Surface | Owning lane | Coordination rule |
|---|---|---|
| `README.md`, root/subsystem `CLAUDE.md`, `design/architecture.md` | A | Other lanes provide facts; A writes final wording |
| `lean/Tropical/Semantics*`, semantics fixtures | B | C may link theorem names but not edit proofs |
| `lean/Tropical/Trust.lean`, `design/trust-boundary.md` | C | G registers audit in shared runner |
| `benchmarks/current_baseline/`, opt-in timing instrumentation | D | G owns shared CLI/build registration |
| `engine/metal/*`, Metal tests and findings | E | Shared `FlatRuntime` edit requires F notification |
| compatibility matrix, generic plan-4 parser/tests | F | Shared `FlatRuntime` edit requires E notification |
| `Makefile`, CI, top-level runner/import/build files | G | Other lanes send minimal requested patch |

The staff engineer resolves any unlisted shared file before work begins.
“Small edit” is not an exception to ownership.

## Dependency graph

```text
Day-1 green baseline
    │
    ├── Lane B semantics ──────────────┐
    │                                  │
    ├── Lane D performance ──────┐     │
    │                            │     │
    ├── Lane E Metal ────────────┼─────┤
    │                            │     │
    └── Lane F compatibility ────┤     │
                                 ▼     ▼
                           Lane C trust ledger
                                 │
                                 ▼
                           Lane A final truth pass
                                 │
                                 ▼
                           Lane G release candidate
```

Lane A begins immediately with an audit and draft; it does not freeze final
performance, trust, or compatibility wording until Days 7–8.

Lane C begins with provisional obligations and replaces placeholders as Lanes
B, D, E, and F finish.

## Calendar and checkpoints

### Day 1 — Tuesday, July 28: establish reality

Staff actions:

- Assign lane DRIs and independent reviewers.
- Record the baseline sha, working-tree status, toolchains, and machine
  manifests.
- Run the full ordinary validation and Metal smoke.
- Resolve ownership of `Diffcli.lean`, `Tropicaltest.lean`, `FlatRuntime.*`,
  and any other shared file requested by two lanes.
- Open the sprint decision log.

Lane exit criteria:

- A: mismatch ledger started.
- B: constructor/semantics inventory.
- C: initial obligation inventory.
- D: measurement protocol and three-row smoke.
- E: qualification matrix and July 7 reproduction.
- F: reachability-audit map.
- G: baseline report.

If the baseline is red, the staff engineer assigns diagnosis immediately.
Unrelated sprint work may continue only when the failure is isolated.

### Day 2 — Wednesday, July 29: freeze contracts

Required reviews:

- B: theorem statement, value/refusal model, and well-formedness model.
- C: obligation/evidence schema.
- D: benchmark matrix and candidate product budgets.
- E: Metal thresholds and latency timestamp method.

The staff engineer records each accepted contract. After Day 2, implementation
may simplify internally but may not broaden its promised surface without a
scope decision.

### Day 3 — Thursday, July 30: first useful slices

Expected:

- B scalar/reference denotation compiling;
- C ledger type and rendering skeleton;
- D reproducible harness with cache isolation;
- E block-length/pipeline test seam underway;
- F first production non-emission assertion;
- A authoritative architecture outline.

G performs the first rolling integration and full validation.

### Day 4 — Friday, July 31: risk checkpoint

This is the most important staff review.

Decide:

1. Is Lane B on the full capstone path, or must one documented fallback be
   selected?
2. Does Lane E have a safe block-length control seam and valid latency probe?
3. Has Lane F classified every ambiguous state path?
4. Can Lane D control cold/warm cache state without touching user data?
5. Are any current docs making a claim that no lane can support?

Cut scope now, not on Day 9.

### Day 5 — Monday, August 3: end-of-week integration

Must be independently valuable and reviewable:

- A current architecture draft;
- B proved scalar/reference fragment plus arena plan;
- C populated trust ledger skeleton;
- D harness plus first compile matrix;
- E short Metal sweep and automated smoke;
- F compatibility matrix and non-emission gate;
- G green integrated branch.

No experimental proof or runtime work lands merely to show progress.

### Day 6 — Tuesday, August 4: complete hard middle

- B closes arena extension lemmas.
- D completes compile/bank sweep.
- E starts long soak and parameter-latency runs.
- F applies staff-approved quarantine labels.
- C maps actual gates and unsafe sites.

### Day 7 — Wednesday, August 5: evidence joins

- B adds arrays/banks and targets the capstone.
- D publishes preliminary findings.
- E publishes raw soak/latency data.
- F finalizes compatibility status.
- C replaces provisional entries with evidence links.
- A consumes the first final facts.
- G runs the second rolling full integration.

### Day 8 — Thursday, August 6: scope freeze

At noon:

- no new stretch work;
- no new benchmark dimension;
- no theorem broadening;
- no compatibility deletion newly proposed for this sprint;
- no architectural prose about unlanded behavior.

Staff reviews every P0/P1 row and moves unfinished P2 work to follow-ups.

### Day 9 — Friday, August 7: release candidate

Expected final lane results:

- B theorem/fallback reviewed;
- C trust report and audit;
- D findings and tiering decision;
- E qualification findings and operating envelope;
- F compatibility matrix/recommendation;
- A final architecture truth pass.

G merges in dependency order and produces the release-candidate commit.
Run the full Tier-3 gate set.

### Day 10 — Monday, August 10: close

Only release blockers and factual corrections.

The staff engineer:

- signs the decision log;
- confirms every P0 item;
- accepts or rejects the release candidate;
- assigns every remaining obligation;
- publishes the evidence index and next-sprint recommendations.

## Staff decision log

Maintain this table in the sprint master PR or the final evidence index:

| ID | Date | Decision | Alternatives rejected | Evidence | Owner |
|---|---|---|---|---|---|
| S-01 | Day 2 | Lane B theorem statement | — | link | staff |
| S-02 | Day 2 | Trust evidence taxonomy | — | link | staff |
| S-03 | Day 4 | Semantic full path or fallback | — | link | staff |
| S-04 | Day 4 | Legacy compatibility quarantine (superseded by S-09) | — | Lane F | staff |
| S-05 | Day 8 | P1 scope cuts, if any | — | link | staff |
| S-06 | Day 9 | Fixed-renderer baseline accepted; no tiering/compile optimization | — | Lane D | staff |
| S-07 | Day 9 | Configuration-specific Metal support envelope | — | Lane E | staff |
| S-08 | Day 10 | Release candidate acceptance | — | Tier 3 | staff |
| S-09 | accelerated | Retire Plan-4/runtime/API legacy compatibility immediately | — | Lane F | staff |

Decisions must describe rejected alternatives. “Team agreed” is not enough to
reconstruct why.

## Review protocol

Every substantive PR answers:

### Semantics

- What observable meaning is preserved or intentionally changed?
- Which theorem, golden, or differential supports that statement?
- Did the trusted boundary grow or shrink?

### Realtime behavior

- Can this execute on the audio thread?
- Does it allocate, lock, compile, or touch filesystem/network state there?
- Does it change block, pipeline, or parameter latency?

### Numeric behavior

- Which carrier is used: exact dyadic/interval, i64 rail, f64, or f32?
- Is equality bit-exact, tolerance-based, or structural?
- Is a comparison deciding program shape?

### Compatibility

- Is this reachable from a current front door?
- Does it preserve the CF-only refusal boundary?
- Does it affect plan-4 or direct C API users?

### Operations

- Which focused and full gates ran?
- Are generated files reproducible?
- Is any cache or user state mutated?

## Scope-change policy

The user authorized a full two-week plan, not unlimited architectural
expansion. The staff engineer may trade work within the sprint only when:

- a P0 discovery blocks another P0;
- a correctness defect is found;
- the approved semantic fallback is invoked;
- Metal qualification exposes a product-safety failure;
- compatibility reachability contradicts the documented language.

When scope changes:

1. identify the displaced P1/P2 work;
2. record the decision and evidence;
3. update the affected lane handoff;
4. preserve a coherent close-state.

Do not add:

- a new modal atom;
- a new external language;
- stateful semantics;
- automatic backend selection;
- tiered compilation implementation;
- video/control rate polymorphism;
- full backend verification.

Those are next-sprint candidates after this baseline exists.

## Stop-the-line policy

The following suspend merges until owned:

1. JIT and wasm disagree.
2. Frozen output changes without an approved semantic explanation.
3. Metal plays a stale future block after a write, jump, or swap.
4. A current front door can express state or accept a cycle.
5. The preservation theorem requires an undocumented axiom.
6. A new unsafe optimization has no total reference.
7. The benchmark harness modifies the developer’s ordinary cache or patches.
8. A clean checkout cannot reproduce validation.
9. Documentation and source encode different public contracts at scope freeze.

The staff engineer chooses whether the incident is fixed inside the sprint or
becomes an explicit release blocker. It is never silently downgraded.

## Definition of done

The sprint is done only when all are true:

### Architecture

- `design/architecture.md`, root docs, and subsystem docs describe the current
  direct pipeline and three execution targets.
- Historical/compatibility material is labeled.
- Live parameter writes and structural recompiles are not conflated.

### Formal

- The committed preservation theorem or approved fallback is checked.
- No new `axiom`, `sorry`, `partial`, or untracked unsafe appears.
- The next semantic boundary is stated without claiming it is proved.

### Trust

- The ledger is machine-readable and queryable.
- Every current unsafe and named backend assumption is present.
- Evidence kinds distinguish theorem, golden, differential, tolerance,
  inspection, and external dependency.

### Performance and Metal

- Current post-bank measurements are reproducible.
- The four-ring structural-edit numbers are current.
- The tiered-preview decision is recorded.
- Metal soak/latency evidence and operating envelope are recorded, or a
  qualification failure blocks release visibly.

### Compatibility

- Legacy state/Plan-4/runtime/API aliases are absent from current boundaries.
- Current production paths reject retired schema/state fields.
- `tropical_program_2` and current Plan-5 omission defaults remain supported.

### Release

- Clean-checkout `make validate` passes.
- Linux CI passes.
- Supported macOS Metal smoke passes.
- No unexplained golden changes or generated-file dirt remain.
- The staff engineer signs the decision and evidence index.

## Final evidence package

Lane G assembles, and the staff engineer accepts, one final index linking:

1. baseline and release-candidate shas;
2. Day-1 and Day-10 validation outputs;
3. architecture mismatch resolution;
4. semantic theorem and proof review;
5. trust report and audit result;
6. performance raw data, findings, and tiering decision;
7. Metal raw data, findings, and qualification decision;
8. compatibility matrix and recommendation;
9. all scope decisions;
10. remaining obligations with owner and target date.

Nothing in the evidence package should require reading an untracked local
handoff or reconstructing a result from terminal history.

## Likely next sprint, deliberately not authorized here

If this sprint closes cleanly, the evidence should make one of these the next
focused bet:

- extend the semantic spine through direct lowering or stage-0;
- implement theorem-licensed backend bank-realization policy;
- design a separate flagship instrument without treating the playground
  renderer as its product surface;
- narrow an analytic approximation obligation in the Exact layer;
- begin a second concrete rate consumer before adding rate-indexed types.

The demo/product artifact is deliberately scoped in a separate pass, not
selected by this evidence package. The staff engineer recommends any compiler,
semantics, or runtime follow-up from the final evidence without bending this
sprint’s results toward it.
