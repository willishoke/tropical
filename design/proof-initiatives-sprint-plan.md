# Pipeline proof sprint — five parallel initiatives

- **Ground-truth baseline:** `origin/main`
- **Pinned commit:** `d7d9e1d4b4d161bf8b1e2800497d79c6381a609f`
- **Baseline title:** `Add absolute-time higher-order phaser staging (#239)`
- **Prepared:** 2026-08-21
- **Sprint objective:** establish the semantic interfaces and first concrete
  refinement theorems from authored expression DAGs through Plan-6

## Accepted output-model extension

This sprint also carries the independent-output work required for a usable
audio environment. Plan sinks are pushed logical output channels, while scope
taps remain pull observations over retained slots. Stereo is the first required
case of the general multichannel model. The frozen schema, authoring, kernel,
host, and proof obligations are recorded in
[`stereo-output-contract.md`](stereo-output-contract.md).

## Initiative index

| Worker | Initiative | Priority | Difficulty | Independent start | Integration dependency |
| --- | --- | --- | --- | --- | --- |
| W1 | [`FlatPlan` semantics and well-formedness](proof-initiative-1-flat-plan-semantics.md) | P0 | Large | Yes | Provider for W4/W5 |
| W2 | [Builder and program well-formedness](proof-initiative-2-builder-program-wellformedness.md) | P0 | Medium–Large | Yes | Provider for W3; construction facts for W4 |
| W3 | [`toResolved` denotation preservation](proof-initiative-3-to-resolved-refinement.md) | P0 | Medium–Large | Yes | Adopt W2 predicate names at integration |
| W4 | [Stage-signature and staging refinement](proof-initiative-4-staging-refinement.md) | P0 | Large | Yes, structural work first | W1 Plan semantics; W2 shift certificate |
| W5 | [Clock, bank, and routed capstones](proof-initiative-5-clock-bank-routed-capstones.md) | P0 | Medium–Large | Yes, stream/source work first | W1 Plan semantics |

## What the merged arena cutover unlocked

The five initiatives are now concrete because `Sig` is an `ExprId` in one
append-only `ExprArena`, production semantics already interprets that DAG, and
the former tree-to-ID unsafe boundary is gone. The newest `origin/main` also
adds a real second staging surface—absolute-time tile materialization—so the
sprint must prove or delimit both Stage0 and TileStage rather than documenting
only the pre-#239 pipeline.

The strongest existing anchors are:

- `Semantics.eintern_preserves` and `Semantics.denoteExpr_extends`;
- total child-descending expression and program-pool traversals;
- the canonical modal-universe theorem;
- arena-native clock algebra;
- direct bank order, emitted bank-region shape, and dynamic prefix bounds;
- direct routed-sum authored-order semantics;
- a zero-site production Lean trust-escape inventory.

The missing waist is still `FlatPlan`: it has no executable Lean semantics or
well-formedness predicate. Consequently no current theorem spans source
denotation, staging, and Plan execution.

## Parallelization contract

The tracks are logically related but can avoid edit conflicts. During the
parallel phase, ownership is exclusive:

| Surface | Owner |
| --- | --- |
| `Semantics/Plan*.lean` | W1 |
| `Semantics/WellFormed.lean`, `EmitArrow/BuilderLaws.lean`, `EmitArrow/Sig.lean` | W2 |
| `Semantics/Program.lean`, `Semantics/Strata.lean`, `Ir/Strata/EArena.lean` | W3 |
| `Semantics/Staging.lean`, `Ir/Stage0*.lean`, `Ir/TileStage*.lean` | W4 |
| `ClockPlanLaws.lean`, `BankPlanLaws.lean`, `RoutedSumLaws.lean`, `Ir/Emit.lean`, `Ir/EmitBankLaws.lean` | W5 |
| Aggregate imports, `Tropicaltest.lean`, `Trust.lean`, generated trust report | Integration owner |

Rules:

1. New modules are preferred over edits to shared production definitions.
2. W1 publishes a small Plan semantic interface before W4/W5 integration:
   `PlanState`, `execBlocks`, `evalOperand`, `denoteFlatPlan`, and
   `FlatPlanWellFormed`.
3. W2 publishes predicate names before W3 integration:
   `BuilderWellFormed`, `ProgramWellFormed`, and `CoreProgramWellFormed`.
4. W3 and W4 may begin with explicit equivalent assumptions, but must delete
   compatibility predicates and import the canonical interfaces before merge.
5. W4 owns stage/tile rewrites; W5 owns emitter proof exposure. Neither edits
   the other's production files.
6. Only the integration owner updates aggregate imports, shared runners, or the
   trust ledger, preventing five low-value merge conflicts.

## Dependency graph

```text
W1 Plan semantics ───────────────┬──→ W4 Stage0/Tile semantic capstones
                                └──→ W5 clock/bank/routed Plan capstones

W2 builder/program WF ──────────┬──→ W3 final toResolved theorem vocabulary
                                └──→ W4 shiftSampleIndex construction premise

W3 toResolved refinement ───────────→ later whole compiler theorem
W4 staging refinement ──────────────→ later host publication refinement
W5 fragment capstones ──────────────→ later LLVM/MSL refinement
```

There is no dependency from W3 to W1 in this sprint: W3 stops at
`CoreProgram`. There is no dependency from W5 to W3: W5 proves constructor
fragments against emitted Plan streams.

## Milestones

### M0 — interface freeze

Target: first half-day.

- Every worker verifies the pinned `origin/main` commit.
- W1 publishes provisional Plan semantic names and carrier types.
- W2 publishes provisional well-formedness predicate names.
- W3/W4/W5 confirm their theorem statements can consume those names.
- Exclusive production-file ownership is recorded in each branch.

**Gate:** a short interface note or draft declarations compile, even if proofs
are temporarily `by` placeholders only on worker branches. No placeholder may
enter the integration branch.

### M1 — independent foundations

All five workers proceed in parallel:

- **W1:** scalar/array instruction state, Plan sources, structured regions,
  recursive block order.
- **W2:** addressability, semantic arena WF bridges, qualified intern and smart
  constructor preservation.
- **W3:** proof-visible copy relation, destination arena WF, expression-copy
  denotation.
- **W4:** Stage/StageSig algebra, dependency sorting, resolve monotonicity,
  collect/rebuild invariants.
- **W5:** compile-result relation, clock fragment lemmas, routed stream-shape
  theorem, reuse of bank stream theorem.

**Milepost:** each track builds as an isolated Lean module and has at least one
production-shaped fixture.

### M2 — local capstones

- **W1:** `FlatPlanWellFormed`, determinism, and structural execution safety.
- **W2:** assembly preservation and `shiftSampleIndex` construction safety.
- **W3:** source/Core program denotations and expression-level copy theorem.
- **W4:** StageSig semantic noninterference and Stage0 structural correctness.
- **W5:** clock preservation plus bank/routed source and stream lemmas stated
  against W1's interface.

**Milepost:** all independent theorem statements are complete; remaining work
is explicitly tied to a provider interface rather than an unspecified blocker.

### M3 — semantic integration

Merge providers first:

1. W1 Plan semantics and W2 well-formedness interfaces.
2. W3 program refinement, adopting W2 predicate names.
3. W4 Stage0 semantic refinement, then tile exact-left-endpoint refinement.
4. W5 clock, bank, and routed source-to-Plan capstones.

Resolve aggregate imports only here. Do not merge temporary duplicate
predicates or parallel Plan interpreters.

**Milepost:** theorem symbols exist for:

- `EArena.toResolved` denotation preservation;
- StageSig noninterference;
- typed Stage0 refinement;
- TileStage exact-left-endpoint behavior;
- clock-rail source-to-Plan refinement;
- static/dynamic/nested bank source-to-Plan refinement;
- routed authored-order source-to-Plan refinement.

### M4 — qualification and trust accounting

- Build every new theorem module and the `Tropical.Semantics` aggregate.
- Run focused authoring, routed, phaser, and modal gates.
- Run the full `tropicaltest` binary.
- Run web/backend validation if production code changed beyond proof exposure.
- Update `Tropical.Trust` only with actual theorem symbols.
- Regenerate `design/trust-boundary.md` through `trustreport --write`, then
  verify with `--check`.

The trust statuses for LLVM, MSL, Metal scheduling, host parameter dispatch,
and external toolchains must remain open/evidence-backed/external as
appropriate. Source-to-Plan proofs narrow their limitations; they do not prove
backend execution.

## Recommended merge slices

Keep reviews small and theorem-oriented:

1. Plan semantic state + small instructions.
2. Plan regions + well-formedness.
3. Builder addressability + intern preservation.
4. Assembly and `shiftSampleIndex` construction facts.
5. Expression copy + destination WF.
6. Whole `toResolved` program refinement.
7. StageSig algebra + noninterference.
8. Stage0 structural + semantic refinement.
9. TileStage exact endpoint refinement.
10. Clock Plan capstone.
11. Bank Plan capstones.
12. Routed stream + Plan capstone.
13. Aggregates, gates, and trust-report reconciliation.

Do not combine a proof slice with behavior-changing emitter, staging, or
runtime work unless the theorem exposes an actual bug that requires a separate
fix commit.

## Sprint-wide validation ladder

Fast worker loop:

```text
cd lean && lake build <owned module>
cd lean && lake build <owned test module>
```

Focused integration:

```text
make lean
./lean/.lake/build/bin/tropicaltest --emitarrow-only
./lean/.lake/build/bin/tropicaltest --routed-only
./lean/.lake/build/bin/tropicaltest --phaser-only
./lean/.lake/build/bin/phasercheck
./lean/.lake/build/bin/trustreport --check
```

Final qualification:

```text
./lean/.lake/build/bin/tropicaltest
TROPICAL_ENGINE_CMD="./lean/.lake/build/bin/frontend --rpc" bun test
cmake --build build -j4 && ctest --test-dir build
```

The built Lean binaries must be run directly, not via `lake exe`.

## Definition of done

The sprint is complete when:

- all six documents still describe the code after integration or have been
  updated with explicit deviations;
- every current Plan instruction and source kind has reference semantics;
- builder/program well-formedness is explicit at production assembly seams;
- `toResolved` has whole-program denotation preservation;
- Stage0 has semantic refinement and TileStage has a proved exact endpoint
  contract without overstating interpolation accuracy;
- clock, bank, and routed fragments reach Plan semantics through named
  theorems;
- focused and full qualification gates pass;
- trust-ledger changes cite real theorem symbols and preserve backend limits;
- no worker's temporary compatibility layer, duplicate predicate, or proof
  placeholder reaches the final branch.

## Deferred proof surfaces

These remain important but are intentionally outside the five-worker sprint:

- full `CoreProgram → FlatPlan` compiler refinement for every expression and
  instance feature;
- LLVM-text execution refinement and LLVM optimizer correctness;
- MSL execution within the f32 error contract;
- Metal tile interpolation error bounds over the full admitted domain;
- host atomic publication, callback deadlines, parameter discipline, and
  cache-key coherence;
- the still-open production modal-universe refinement outside the canonical
  theorem's admitted model.

Those are the next layer after this sprint establishes the semantic waist and
fragment capstones.
