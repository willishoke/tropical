# Arena-native `Sig` cutover — sprint completion handoff

- **Starting point:** Phase-5 checkpoint described in
  `design/sig-arena-phase5-handoff.md`
- **Historical baseline:** phase-3 checkpoint `238ef11`
- **Date:** 2026-08-14
- **Status:** Ready to execute
- **Scope:** All remaining work required to delete recursive `Sig` authoring,
  promote the arena-native API to its final names, and close the sprint

## Executive state

Production is already migrated. `Tropical.EmitArrow`, `Tropical.Stdlib`, the
playground compiler, direct expression semantics, and the active clock/bank
proof surface author stable `ExprId`s in one `ExprArena`. Phases 1–5 pass, the
combined Lean build passes, and the full suite is 136/137 on the current host;
the sole failure is the known environment-only Metal-device refusal.

The remaining work is test and deletion work, not a missing production
compiler feature. Recursive authoring survives because several valuable
oracles and fixture builders still construct or inspect recursive `Sig` values.
Those tests must retain their independence while their device-under-test side
moves to the arena-native builder.

This document supersedes the remaining-work section of the Phase-5 checkpoint
handoff and is the execution source of truth for the rest of the sprint.

## Sprint objective

Finish with one authoring representation and one public API:

- `Tropical.EmitArrow.Sig` is the stable expression ID type, not a recursive
  tree;
- all smart constructors and higher-level builders allocate through `BuildM`;
- no production or test module imports the recursive authoring stack;
- recursive `Sig`, term, numeric, modal, patch, gong, and playground builders
  are deleted;
- the three unsafe pointer-lowering trust sites and their scoped obligation are
  gone;
- the temporary `ArenaNative` namespace and `Arena*` module names are promoted
  to the final `Tropical.EmitArrow` layout; and
- all frozen wire, numeric-oracle, semantic, and trust gates retain their
  current meaning.

## Non-negotiable constraints

1. Do not introduce a recursive-tree-to-ID adapter or a compatibility layer
   that recreates the old pure `Sig → Sig` API.
2. Do not use production arena-native modal calculations as their own numeric
   oracle. Keep Float/complex/reference mathematics independent.
3. Preserve authored array, route, declaration, branch, and reduction order.
4. Keep fixture construction atomic: a failed `BuildM` assembly publishes no
   program and no partial expression arena.
5. Do not rewrite audio goldens merely to make the migration pass. Any output
   change is a separate behavior change requiring explicit review.
6. Do not change backend operations, Plan-6 schema, runtime semantics, or the
   public playground JSON contract in this sprint.
7. Treat the missing Metal device as an environment qualification issue, not
   permission to weaken the non-Metal routed-reduction gate.

## Dependency chain

```text
5A native test substrate
  → 5B scalar/clock/ArrowTerm fixtures
  → 5C modal and patch oracle suites
  → 5D playground and exact probes
  → 5E recursive deletion + trust cleanup
  → 5F final namespace/module promotion
  → 5G cutover qualification
```

Deletion and promotion are intentionally separate checkpoints. First prove
that no consumer needs the old declarations; then delete them; only then move
the native declarations into the names they vacated.

## Current remaining inventory

| Inventory | Why it remains | Required disposition |
|---|---|---|
| `Testing/ArrowFixtures.lean` | Central recursive fixture hub used by arrow, slide, stress, bank, and modal gates | Split independent oracles from DUT builders; port DUT builders to `BuildM` |
| `Tropicaltest/Slide.lean`, `Stress.lean`, `Modal.lean` | Consume recursive fixture signatures and expression values, often transitively | Migrate by fixture family and retain existing behavioral assertions |
| `Tropicaltest/Exact.lean` | Imports recursive modal and playground builders to inspect emitted constants | Run equivalent arena builds and inspect frozen IDs through `ArenaInspect` |
| `Tropicaltest/Oriented.lean` | Direct recursive oriented/modal records and constant evaluation | Keep numeric oracle pure; construct only the DUT through arena-native records |
| `Tropicaltest/OrientedPatch.lean` | Direct recursive `PatchGraph`, `Sig`, and playground clock helper | Port graph construction to `ArenaPatch`; port clock helper to `BuildM` |
| `Tropicaltest/GroupedRoomReference.lean` | Direct recursive grouped-room specialization | Use `ArenaModal.GroupedRoomReference` with the existing independent carrier oracle |
| `Tropicaltest/Phaser.lean` | Uses legacy `buildNode`, `rawsOf`, and phaser metadata | Use `Metadata` for pure data and `ArenaCompiler.buildNode` for DUT construction |
| `Tropicaltest/SeamSweep.lean` | Uses legacy baked-reverb probes | Add an arena-native probe returning a frozen arena and mode IDs |
| `Playground/Vocabulary.lean`, `Decode.lean` | Retained solely for the legacy probes above | Delete after all four direct test imports are removed |
| `EmitArrow/Sig.lean`, `Term.lean`, `Numerics.lean`, `Modal/**`, `Patch.lean`, `Gong.lean` | Recursive implementation graph reached by the test inventory | Delete after consumer search is empty |
| Three `LOWER_SIG_PTR_*` trust sites | Unsafe lowering remains physically present in recursive `Sig.lean` | Remove with `Sig.lean`, then remove the typed trust rows and obligation |
| `ArenaNative` and `Arena*` names | Avoided collisions during coexistence | Promote only after recursive declarations are absent |

The affected test/probe surface is broad, but it is not all rewrite work.
Many tests retain their oracle and assertion unchanged; only their DUT builder
and result-inspection boundary moves.

## Test migration decision rule

Classify every legacy test before editing it:

| Existing test shape | Migration action |
|---|---|
| Old/new authoring parity only | Delete the recursive half; retain the already-frozen node/wire observation if it remains useful |
| Independent Float, complex, quadrature, or closed-form oracle | Keep the oracle unchanged; replace only DUT construction with arena-native `BuildM` |
| Recursive tree-shape inspection | Inspect `ENode`s in the frozen `ExprArena`, or use `ArenaInspect` tables |
| Constant folding through `sigConstF?` | Build once, freeze the arena, and read `sigConstTable`/`sigConstDFrom?` at the returned IDs |
| Fixture whose only result is a compiled plan | Port directly to `assemble`/`assembleCompleteWithResult`; compare the same plan, wire, render, or hash |
| Coverage duplicated exactly by the frozen Phase 1–5 gate | Retire it and record why the remaining gate is equivalent |

A test must not be labeled migrated if its “oracle” and its DUT call the same
arena-native implementation for the fact under test.

## Phase 5A — establish the native test substrate

### Goal

Create one reusable, test-only construction boundary so subsequent ports do
not invent local state-running or frozen-arena conventions.

### Work

- Add `Tropical.Testing.ArenaArrowFixtures` as the temporary arena-native DUT
  fixture module.
- Extract any genuinely independent numeric/reference helpers currently mixed
  into `Testing/ArrowFixtures.lean` into `Tropical.Testing.ArrowOracles`.
- Provide small test-only helpers for:
  - running a `BuildM` construction from an existing `Arena`;
  - atomically assembling a program and returning its `ProgramIdx`;
  - freezing the resulting `ExprArena` for classifier inspection;
  - resolving/compiling a returned program through the ordinary production
    boundary; and
  - rendering plans without accepting a recursive `Sig` anywhere in the API.
- Reuse `ArenaNative.ProgramBody`, `CompleteProgramBody`, and production
  assembly rather than defining a second builder state.
- Add a compile-time refusal fixture showing that a failed build leaves the
  source arena unchanged.
- Keep the existing Phase 1–5 evidence untouched while the new fixture module
  is introduced.

### Exit criteria

- The new fixture module imports only arena-native production modules.
- Its public signatures contain `ExprId`/native `Sig`, `BuildM`, `Arena`, and
  `ProgramIdx`; none contain the recursive `Tropical.EmitArrow.Sig`.
- At least one scalar fixture and one multi-declaration fixture compile and
  render through the shared helpers.
- `lake build Tropical.Testing.ArenaArrowFixtures tropicaltest` passes.
- `tropicaltest --arena-native-only` remains green.

## Phase 5B — port scalar, clock, numeric, and ArrowTerm fixtures

### Goal

Remove recursive `Sig`, `Numerics`, and `Term` construction from the non-modal
fixture and test surface.

### Work packages

#### 5B.1 Scalar and clock carriers

- Port literal/expression carriers, clock carriers, fixed-source carriers, and
  long-time/fractional-clock probes.
- Replace pure `Clock → Clock` callbacks with `Clock → BuildM Clock` where they
  allocate nodes.
- Reuse returned IDs instead of reconstructing equal subexpressions; retain
  explicit interning assertions where sharing is the subject of the test.
- Migrate the relevant `Slide.lean` and arrow-law call sites.

#### 5B.2 Numeric bootstrap probes

- Port fixed sine, phasor, exp, log, atan2, and fixed-point probes to
  `ArenaNumerics`.
- Preserve all current independent libm/closed-form comparisons and frozen
  wire/hash assertions.
- Replace recursive constant inspection with frozen-arena classification.

#### 5B.3 ArrowTerm and effect composition

- Port tap banks, shared/independent diagonals, FM, PM-of-PM, flange, reverse,
  product, and `arrN` builders to the monadic native `ArrowTerm` API.
- Migrate `Stress.lean`, the non-modal part of `Slide.lean`, and the arrow-law
  block without changing their render or cost assertions.
- Preserve left-to-right declaration effects and existing instance names.

#### 5B.4 Static and nested bank fixtures

- Port direct, table-backed, dynamic-count, float, and nested bank fixtures.
- Keep the same `ReduceBegin`/`ReduceEnd`, plan-size, typed-split, MSL nesting,
  and authored-order checks.
- Use `ArenaBankOrder` for theorem-facing order facts; do not manufacture an
  unrolled recursive expression as a proof witness.

### Exit criteria

- `Slide.lean` and `Stress.lean` have no recursive `Sig`/`ArrowTerm` dependency.
- The non-modal and bank portions of `Testing/ArrowFixtures.lean` have native
  replacements and no remaining consumers.
- Existing render hashes, exact wire comparisons, instance order, plan sizes,
  and bank region assertions are unchanged.
- Focused builds and the full non-Metal suite remain at least at the current
  136/137 checkpoint.

## Phase 5C — port modal, oriented, grouped-room, and patch oracle suites

### Goal

Move every modal DUT to `ArenaModal`/`ArenaPatch` while preserving the existing
independent mathematics.

### Work packages

#### 5C.1 Modal fixture records

- Port recursive `ModalMode`, bloom pairs, oriented atoms/banks, controls,
  stages, forests, and patch nodes to their arena-native ID-valued records.
- Build coefficient literals and live controls inside one `BuildM` run.
- Keep Float/Cplx mode tables and analytic reference functions outside the
  builder as oracle data.

#### 5C.2 `Tropicaltest/Modal.lean`

- Port modal bank, degree, reverse, sway, pair, integrate, heterodyne, VCO,
  reclock, residue, divided-difference, bloom, forest, and patch DUT builders.
- Retain current quadrature, moment, rational, recurrence, and render
  comparisons unchanged.
- Replace `evalConstSig` with frozen-arena constant-table lookup.
- Preserve refusal tests for unsupported nonterminal DD, bloom, and live-pole
  crossings.

#### 5C.3 Oriented and grouped-room suites

- Port `Oriented.lean` to arena IDs while retaining its independent orientation
  and rational-product oracle.
- Port `GroupedRoomReference.lean` through
  `ArenaModal.GroupedRoomReference`; retain exact-prefix and direct causal/
  reverse oracle checks.
- Port `OrientedPatch.lean` to `ArenaPatch.PatchGraph → BuildM ArrowTerm`.
- Preserve room order, direction locality, repeated-pole limits, row order,
  and fractional Q32.32 control-clock assertions.

#### 5C.4 Production modal paths already under test

- Ensure Phaser, modal-universe history, live-control, gong, forest, and
  playground production arms continue to use the same native compiler path.
- Port Phaser's retained-stage structural checks (`lowerModalRoot`, recursive
  `PatchGraph`, stage controls, generic-filter producer, and modal-mix order)
  to `ArenaModal`/`ArenaPatch` construction and frozen-arena inspection.
- Read Phaser mix and control constants from the frozen arena; do not preserve
  recursive `Sig` construction or `sigConstF?` solely for these assertions.
  Phase 5D owns only Phaser's pure metadata/raw-JSON consumers once these
  structural checks are native.
- Remove any redundant recursive comparison arm whose only purpose was
  coexistence parity, recording the frozen gate that replaces it.

### Exit criteria

- No test imports `EmitArrow.Modal`, `Modal/**`, `Patch`, or `Gong` for DUT
  construction.
- The following gates pass:

```text
tropicaltest --oriented-patch-only
tropicaltest --phaser-only
tropicaltest --modal-universe-history-only
tropicaltest --ecdd-only
```

- The full modal/oracle suite retains its current tolerances, refusal messages,
  route counts, plan footprints, and scratch-policy assertions.

## Phase 5D — retire legacy playground and exact probes

### Goal

Remove the last test imports of `Playground.Vocabulary` and
`Playground.Decode` without weakening exact-bake coverage.

### Work packages

#### 5D.1 Arena-native coefficient probes

- Add test-only probes that execute the actual arena-native
  `defaultStringModes`, resonator, reverb, filter, and gong builders.
- Return the frozen `ExprArena` plus the mode/root IDs needed by the test.
- Fold those IDs through `ArenaInspect.sigConstTable`; do not transcribe the
  coefficient formulas into a second production implementation.
- Migrate `runExactPlayground` and the SeamSweep baked-reverb probe.
- Port Exact's `benchBank`/`runExactRecip10` synthetic all-literal modal bank
  to `BuildM` and the arena-native bank landing/classification path. Preserve
  reciprocal-cache bit identity and keep its wall-clock measurements
  diagnostic rather than gating.

#### 5D.2 Pure metadata consumers

- Change Phaser ratio and raw-node checks to
  `Tropical.Playground.Metadata.modalPhaserRatios` and `Metadata.rawsOf`.
- Where Phaser genuinely tests node construction, call
  `ArenaCompiler.buildNode` inside `BuildM` and inspect the returned native
  node/arena.
- Change all validation, vocabulary, parameter, and raw-graph tests to the
  already-extracted `VocabularyMetadata`/`DecodeMetadata` APIs.

#### 5D.3 Remaining clock/helper probe

- Port `q32DeltaSamples` use in `OrientedPatch.lean` to the arena-native helper
  or express its independent expected arithmetic directly in the oracle.
- Remove the last `Playground.Vocabulary` import.

### Exit criteria

- These searches return no results outside historical design documents:

```text
rg '^import Tropical\.Playground\.(Vocabulary|Decode)$' \
  lean/Tropical/Testing lean/Tropical/Tropicaltest
rg 'Tropical\.Playground\.(defaultStringModes|probeFold|bakedResonatorProbe|bakedReverbProbe|bakedFilterLn80)' \
  lean/Tropical/Testing lean/Tropical/Tropicaltest
```

- `Playground/Vocabulary.lean` and `Playground/Decode.lean` have no consumers.
- Exact-corpse still reports zero production libm call sites.
- Vocabulary fingerprint remains `fnv1a64:5b536cbc16add425`.
- Exact, Phaser, SeamSweep, vocabulary, and malformed-graph gates pass.

## Phase 5E — delete recursive authoring and close the trust inventory

### Goal

Prove the old implementation is unreachable, delete it, and make the trust
ledger describe the candidate tree rather than a quarantine.

### Pre-deletion audit

- Search all Lean sources for direct legacy imports.
- Search public signatures and annotations for recursive `Sig`, `ArrowTerm`,
  modal, patch, and playground builder types.
- Confirm no generated target depends on the legacy modules because of an
  accidental aggregate import.
- Confirm all retained tests have an identified native DUT and an independent
  oracle or frozen observation.

### Delete

- `lean/Tropical/EmitArrow/Sig.lean`
- `lean/Tropical/EmitArrow/Term.lean`
- `lean/Tropical/EmitArrow/Numerics.lean`
- `lean/Tropical/EmitArrow/Modal.lean`
- `lean/Tropical/EmitArrow/Modal/**`
- `lean/Tropical/EmitArrow/Patch.lean`
- `lean/Tropical/EmitArrow/Gong.lean`
- `lean/Tropical/Playground/Vocabulary.lean`
- `lean/Tropical/Playground/Decode.lean`
- the retired recursive `Testing/ArrowFixtures.lean` after its native
  replacement owns the stable fixture surface

### Trust cleanup

- Remove `LOWER_SIG_PTR_GO_UNSAFE`, `LOWER_SIG_PTR_UNSAFE`, and
  `LOWER_SIG_IMPLEMENTED_BY` from `productionTrustSites`.
- Remove the scoped `LOWER_SIG_PTR_REFINES_TREE` obligation.
- Change the ledger invariant from three tracked sites to the actual empty
  Lean trust-escape inventory, unless a separately reviewed site has appeared.
- Regenerate `design/trust-boundary.md` from `Tropical.Trust`.
- Keep the working-tree deletion handling in `tools/audit_trust_sites.py`.

### Exit criteria

- The recursive files are absent.
- The legacy import search is empty.
- `tools/audit_trust_sites.py` passes with zero sites.
- `trustreport --check` passes.
- `lake build tropicaltest` succeeds without any legacy `.olean` being needed.

## Phase 5F — promote final API names and module layout

### Goal

Collapse the temporary parallel API into the single final `Tropical.EmitArrow`
surface. This phase is mechanical: do not mix semantic refactors into it.

### Recommended module mapping

| Temporary module | Final module |
|---|---|
| `ArenaSig.lean` | `Sig.lean` |
| `ArenaInspect.lean` | `Inspect.lean` |
| `ArenaNumerics.lean` | `Numerics.lean` |
| `ArenaTerm.lean` | `Term.lean` |
| `ArenaModal.lean`, `ArenaModal/**` | `Modal.lean`, `Modal/**` |
| `ArenaPatch.lean` | `Patch.lean` |
| `ArenaGong.lean` | `Gong.lean` |
| `ArenaStdlib.lean` | `Stdlib.lean` under `EmitArrow` |
| `ArenaClockAlgebra.lean` | `ClockAlgebra.lean` |
| `ArenaBankOrder.lean` | `BankOrder.lean` |

### Work

- Change `namespace Tropical.EmitArrow.ArenaNative` to
  `namespace Tropical.EmitArrow` throughout the promoted implementation.
- Update all production, semantic, test, trust, and documentation references.
- Remove `.ArenaNative` qualifiers from types and calls.
- Update `Tropical.EmitArrow` and `Tropical.Stdlib` aggregate imports to the
  final module names.
- Rename the temporary native fixture module to the stable
  `Testing.ArrowFixtures` surface after the recursive fixture file is deleted.
- Rename `Testing/ArenaClockLaws.lean` back to the stable
  `Testing/ClockLaws.lean` path.
- Rename the Phase 1–5 checkpoint module from `Testing/ArenaNative.lean` to a
  stable `Testing/EmitArrow.lean` qualification module and update its namespace.
- Decide the final classifier home once: keep it in `EmitArrow.Inspect` rather
  than redistributing private copies into modal modules.
- Rename the focused CLI gate from migration language to a stable authoring
  name, retaining `--arena-native-only` as a temporary alias only if an external
  script is discovered.
- Update active architecture/trust documentation; leave historical phase
  handoffs intact.

### Exit criteria

```text
rg 'Tropical\.EmitArrow\.ArenaNative|namespace Tropical\.EmitArrow\.ArenaNative' \
  lean/Tropical --glob '*.lean'
rg '^import Tropical\.EmitArrow\.Arena' lean --glob '*.lean'
```

Both searches are empty. The final `Tropical.EmitArrow` aggregate exports the
ID-native API directly, with no alias back to a recursive representation.

## Phase 5G — final cutover qualification

### Required build

```text
lake build Tropical.Semantics Tropical.EmitArrow Tropical.Ir.EmitBankLaws \
  Tropical.Playground tropicaltest trustreport phasercheck diffcli
```

### Required focused gates

```text
lean/.lake/build/bin/tropicaltest --arena-native-only
lean/.lake/build/bin/tropicaltest --routed-only
lean/.lake/build/bin/tropicaltest --oriented-patch-only
lean/.lake/build/bin/tropicaltest --phaser-only
lean/.lake/build/bin/tropicaltest --modal-universe-history-only
lean/.lake/build/bin/tropicaltest --ecdd-only
lean/.lake/build/bin/phasercheck
lean/.lake/build/bin/trustreport --check
python3 tools/audit_trust_sites.py
git diff --check
```

### Full suite

```text
lean/.lake/build/bin/tropicaltest
```

On a host without Metal, the only accepted failure is the existing cooperative
Metal load refusal, with the non-Metal routed-sum coverage passing. A Metal
release host must run the same checkpoint to 137/137 before declaring hardware
qualification.

### Frozen observations that must not drift

- Phase 1: 3 programs, 16 authored nodes, 1/9 reachable nodes, 341/1,945 wire
  bytes.
- Phase 2: 15 stdlib programs, 283 stdlib nodes, 268 numeric nodes/46,451 wire
  bytes, 3 carrier instances, 206 carrier nodes/36,103 wire bytes.
- Phase 3: 2,357 authored modal nodes, 2,160 reachable nodes, 24 routed
  reductions, 719,467 wire bytes.
- Phase 4: 7-node native clock-law witness.
- Phase 5: 20 served vocabulary kinds, 3 reserved parameters, fingerprint
  `fnv1a64:5b536cbc16add425`.
- Production Phaser scratch, route counts, reciprocal rows, confluence rows,
  plan footprints, and all existing numeric tolerances remain unchanged.

## Review checkpoints

At the end of every phase:

1. record which legacy imports disappeared;
2. identify every retired test and the surviving equivalent gate;
3. report any assertion/tolerance/golden change explicitly;
4. run the phase's focused targets plus `--arena-native-only`;
5. keep unrelated working-tree changes untouched; and
6. update this handoff only with actual completion evidence.

Do not batch 5E deletion and 5F promotion into an unreviewable semantic diff.
The clean intermediate state—native implementation still under temporary
names, recursive files absent—is the proof that consumer migration is truly
complete.

## Definition of done

The sprint is complete only when all of the following are true:

- no recursive authoring file or import remains;
- no active Lean source mentions `Tropical.EmitArrow.ArenaNative` or imports an
  `Arena*` implementation module;
- all retained tests use the native DUT and retain an independent oracle or a
  documented frozen observation;
- production and test compilation use one `ExprArena` authoring model;
- the Lean trust-site inventory is empty and audited;
- the combined build, focused gates, trust checks, and full non-Metal suite
  meet the qualification rules above; and
- `design/sig-arena-phase5-handoff.md` can be closed with a final checkpoint
  containing the promoted API and deletion evidence.
