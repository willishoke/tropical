# Sprint integration and release baseline — handoff

- **Sprint:** 2026-07-28 through 2026-08-10
- **Lane:** G — integration, gates, and sprint release candidate
- **DRI:** Staff integration
- **Supervisor:** Staff engineer
- **Status:** Local candidate complete; external release blocked by Metal qualification and unrun Linux CI
- **Master:** [Staff engineer sprint handoff](00-staff-engineer-master-handoff.md)
- **Depends on:** All lanes.
- **Must not overlap:** This lane coordinates shared build/test files and does
  not take over another lane’s implementation.

## Mission

Keep `main` continuously releasable while six parallel consolidation lanes
change proofs, docs, benchmarks, tests, and runtime qualification.

The sprint closes on one clean, reproducible baseline:

- current architecture described truthfully;
- approved relational lowering capstone checked;
- trusted boundary queryable;
- current performance measured;
- Metal operating envelope explicitly scoped, with the final B512/D3
  qualification failure retained;
- legacy compatibility removed; retired schemas and carriers reject;
- all ordinary gates green on a clean checkout.

Integration is not a final-day merge event. This lane owns the daily evidence
that the workstreams still compose.

## Branch and merge protocol

Each lane works on one short-lived branch:

```text
sprint/truth
sprint/semantics
sprint/trust
sprint/perf
sprint/metal
sprint/compat
sprint/integration
```

Rules:

1. Rebase or merge current `main` before requesting review.
2. No lane edits another lane’s primary files without both DRIs agreeing.
3. Every PR states:
   - invariant changed or preserved;
   - gates run;
   - generated artifacts;
   - new trust obligations;
   - performance impact, if relevant.
4. Behavior-preserving refactors show the relevant emitted/render equivalence.
5. A red correctness gate stops merges across all lanes until triaged.

The repository’s ordinary commit style remains in force.

## Shared-file ownership

This lane owns conflict coordination for:

- `Makefile`
- `.github/workflows/ci.yml`
- `lean/Tropicaltest.lean`
- `lean/Diffcli.lean`
- `lean/Tropical.lean`
- `CMakeLists.txt`

Owning coordination does not mean authoring every change. A lane that needs one
of these files sends the minimal patch to the integration DRI, who sequences
and lands it.

## Day-1 baseline

Before feature branches diverge:

1. Record `git rev-parse HEAD`.
2. Record toolchain versions.
3. Run, using built binaries rather than `lake exe`:

   ```bash
   make validate
   ```

4. On macOS with Metal:

   ```bash
   cmake --build build -j4
   ctest --test-dir build --output-on-failure
   TROPICAL_ENGINE_CMD="./lean/.lake/build/bin/frontend --rpc" \
     bun test tests/web/metal_vs_jit.test.ts
   ```

5. Record duration and any flaky/retried gate.

If baseline is red, classify the failure before sprint work starts:

- environment/toolchain;
- existing defect;
- nondeterminism;
- stale cache;
- test expectation.

Do not normalize a red baseline into “known flaky” without an owner.

## Gate tiers

### Tier 0 — per commit

- Lean library/executable build for affected modules;
- focused unit/proof gate;
- formatting or lint checks already used by the subtree.

### Tier 1 — per PR

- built `tropicaltest`;
- affected Bun suite;
- affected CTest;
- proof trust audit once Lane C lands.

### Tier 2 — daily integration

```bash
make validate
```

Run from the integration branch after merging current accepted lane heads.

### Tier 3 — sprint release candidate

- clean checkout;
- full `make validate`;
- Linux CI;
- supported macOS Metal smoke;
- Lane E’s manual qualification report;
- benchmark harness smoke;
- documentation link/stale-token audits;
- no uncommitted generated patches or benchmark outputs.

Long performance and soak runs are evidence artifacts, not blocking CI jobs.

## Regression policy

### Correctness

No unexplained changes to:

- audio goldens;
- stdlib wire/port goldens;
- wasm-vs-JIT samples;
- Metal-vs-JIT tolerance rows;
- native realization equivalence;
- refusal behavior at current front doors.

Re-freezing a golden requires staff approval and a written semantic reason.
“Refactor changed bytes” is not sufficient.

### Proof surface

No new production:

- `axiom`;
- `sorry`;
- `partial`;
- `unsafe` without a total reference and trust-ledger entry.

### Performance

Lane D proposes budgets on Day 9. This lane records them but does not make
machine-specific long benchmarks mandatory in Linux CI. Add stable structural
proxies—plan size, instruction count, bank flatness—where they predict the
measured wall.

### Documentation

No merged document may describe planned work as landed. Lane A owns wording;
this lane ensures late code changes are fed back before the freeze.

## Integration schedule

### Day 1 — baseline and ownership lock

- Publish baseline result.
- Resolve shared-file ownership.
- Create the sprint decision log.

### Day 3 — first rolling integration

- Merge harness/scaffolding changes that unblock later work.
- Run Tier 2.
- Surface cross-lane file conflicts before substantive PRs.

### Day 5 — end-of-week checkpoint

Expected:

- architecture draft;
- semantic scalar/reference fragment;
- populated trust ledger skeleton;
- benchmark harness;
- Metal short-sweep harness;
- compatibility matrix draft.

Merge only independently valuable, green slices. Incomplete proof experiments
stay off `main`.

### Day 7 — evidence integration

- Merge current benchmark and Metal harnesses.
- Merge non-emission and trust audits.
- Run Tier 2 on their composition.

### Day 8 — scope freeze

After noon, only must-land deliverables and correctness fixes continue.
Stretch work moves to follow-ups.

### Day 9 — release candidate

- Merge final lane PRs in dependency order.
- Cut an internal sprint release-candidate commit.
- Run Tier 3.

### Day 10 — close

- Fix only release blockers.
- Run clean-checkout Tier 3 again.
- Produce the final sprint evidence index and remaining-work list.

## Merge dependency order

Preferred order:

1. Legacy retirement and Plan-5/current-boundary rejection gates.
2. Semantic modules and relational capstone.
3. Trust ledger and audit, updated with theorem links.
4. Performance harness/instrumentation.
5. Metal harness/runtime qualification changes.
6. Architecture rewrite with final measurements and boundaries.
7. CI/build registration and final report artifacts.

Independent slices may land earlier, but Lane A’s final prose waits for the
facts from other lanes.

## Stop-the-line triggers

Pause integration immediately for:

- a changed frozen render without approved semantics;
- JIT/wasm disagreement;
- Metal stale-future, drift, or dropout failure;
- production emission of legacy state;
- a proof requiring an unreviewed axiom or unsafe implementation;
- inability to run a clean-checkout validation;
- a benchmark harness modifying the user’s ordinary cache or state.

The staff engineer assigns an incident owner and decides whether the sprint
scope is cut to address it.

## Acceptance gates

1. Day-1 and Day-10 full-gate results are archived.
2. Every lane has at least one reviewed, independently useful deliverable.
3. All must-land deliverables are on the release-candidate commit.
4. The release candidate passes clean-checkout validation.
5. Linux CI is green.
6. macOS Metal smoke is green and the required B512/D3 30-minute actual-DAC
   soak passes and is linked.
7. There are no unexplained golden changes.
8. The trust audit and production non-emission gate run in validation.
9. The staff engineer signs the final decision log.

Closeout result: the clean macOS candidate passes the local validation
components, but acceptance gates 5 and 6 remain unsatisfied. GitHub-hosted
Linux CI was not run, and the single final B512/D3 actual-DAC attempt recorded
a hard callback overrun at clock-jump re-prime. The failure is retained as P0
evidence; it closes the measurement obligation but blocks external release
qualification.

## Non-goals

- No public release mechanics or remote deployment.
- No modal feature work.
- No broad dependency upgrades.
- No CI migration.
- No compulsory long-running performance job on every PR.

## Handoff package

Leave one sprint evidence index containing:

- baseline and release-candidate shas;
- validation results and durations;
- merged PR/commit list by lane;
- theorem and trust-report links;
- benchmark and Metal findings;
- compatibility decision;
- unresolved blockers, with owners and next dates.
