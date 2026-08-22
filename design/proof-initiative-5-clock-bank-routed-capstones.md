# Proof initiative 5 — clock, bank, and routed source-to-plan capstones

- **Baseline:** `origin/main` at `d7d9e1d4b4d161bf8b1e2800497d79c6381a609f`
- **Baseline date:** 2026-08-21
- **Priority:** P0
- **Difficulty:** Medium–Large
- **Suggested worker:** W5
- **Dependency:** initiative 1 for final Plan-level statements
- **Can start independently:** yes, through stream-shape and source lemmas

## Outcome

Connect the repository's strongest existing source theorems to emitted Plan
semantics for three high-value fragments:

1. the exact integer clock rail;
2. `bankSum`/`Reduce` regions; and
3. `routedSum` regions.

These are deliberately fragment capstones, not a premature theorem for the
entire compiler.

## Ground truth on the pinned baseline

### Clock

- `EmitArrow.ClockAlgebra` defines the arena-native `OnClockRail` judgment and
  `denoteClock : ... → Int`.
- `warp_inv`, `warp_inv'`, `warp_assoc`, `rev_involution`, `rev_swap`, and
  `rail_split_identity` are proved.
- `Testing.ClockLaws` constructs a production `BuildM` graph and instantiates
  `warp_inv`.
- `CLOCK_RAIL_IS_EXACT` remains open because no theorem connects that `Int`
  denotation to Plan execution or emitted i64 operations.

### Bank

- Direct `ENode.bankSum` semantics is an increasing-index left fold with the
  dynamic count clamped to capacity.
- `EmitArrow.BankOrder` proves authored-order source denotation facts,
  including nested order.
- `Ir.EmitBankLaws.compileBankSum_stream` proves the exact emitted delimiter
  stream under append-only hypotheses.
- `regionTrips_le` and `regionDenotation_static_eq_refFold` are proved.
- `REDUCE_REGION_EXECUTES_IN_ARRAY_ORDER` remains open at the backend boundary.

### Routed sum

- `Semantics.denoteRoutedSum` maps every item once, then folds emits in authored
  order into the fixed output image.
- `Ir.Emit.compileRoutedSum` checks capacity, fanout, routes, targets, nesting,
  scalar values, and effect freedom before emitting
  `RoutedSumBegin/Yield/End`.
- `compileRoutedSum` is private and there is no routed analogue of
  `compileBankSum_stream`.
- `ROUTED_SUM_PRESERVES_AUTHORED_ORDER` is evidence-backed, not a complete
  source-to-Plan theorem.

## Scope and semantic layers

Each capstone should distinguish three claims:

```text
source/direct Expr semantics
  → emitted FlatPlan semantics             mechanized here
  → LLVM/MSL/runtime execution             remains a named boundary
```

Proving the first arrow materially narrows the open trust rows without
claiming anything about LLVM optimizers, Metal compilers, drivers, or hardware.

## Proof surface

### 1. Shared compile relation

Define a small relation between `Ir.Emit.CompileResult` and the Plan semantic
state/operand supplied by initiative 1. Prove reusable lemmas for:

- append-only instruction compilation;
- evaluation of an emitted operand in the final state;
- CSE reuse and zero fallback where those are observable;
- result scalar kind correspondence;
- source operand correspondence for tick/rate and tile tick/phase.

Avoid proving all of `compileNode` in one induction before the fragment
theorems. Establish only the constructor closure needed by clock, bank, and
routed bodies, then expand if useful.

### 2. Clock source-to-Plan capstone

Define the documented i64 image of `Int` explicitly, including modular
two's-complement behavior and shift/mask side conditions. Then prove:

- clock-rail constructors compile to Plan operations with the same image;
- a Plan source of kind `.tick` corresponds to `ClockEnv.tick`;
- literal shifts and low-bit masks satisfy the production headroom rules;
- the existing five clock laws survive compilation to Plan semantics.

Target statement:

```lean
theorem compileClockRail_refines
    (rail : OnClockRail arena root)
    (hcompile : compileNode ... root ... = .ok (...)) :
  evalOperand planState compiledOperand =
    i64Image (denoteClock rail clockEnv)
```

Tile-time source correspondence is useful but subordinate: show how
`.tileTick` supplies the materializer's absolute coordinate without claiming
interpolated Metal equality.

### 3. Bank source-to-Plan capstone

Use the existing stream theorem rather than reproving emission shape. Add the
semantic bridge:

- tables execute before the region;
- dynamic count is evaluated once and clamped;
- binder lookup resolves the matching `loopId` under nesting;
- the body executes once per increasing index;
- the accumulator performs a scalar left fold without reassociation;
- body-local temps do not become post-region observations.

Prove static and dynamic forms, then nested bank composition. The Plan theorem
can discharge order within the reference semantics. Backend execution remains
the `REDUCE_REGION_EXECUTES_IN_ARRAY_ORDER` obligation until LLVM/MSL are
related to Plan semantics.

### 4. Routed source-to-Plan capstone

First add the missing emission-shape theorem, adjacent to the existing bank
law surface:

```lean
theorem compileRoutedSum_stream ... :
  -- invariant prefix ++ RoutedSumBegin ++ mapped body
  -- ++ RoutedSumYield ++ RoutedSumEnd, with returned array operand
```

Then prove:

- the emitted Plan evaluates all mapped values exactly once per active item;
- route application is in `(item, emit)` order;
- inactive routes contribute nothing;
- dynamic count is clamped to capacity;
- the result has exactly `outputCount` floats;
- malformed routes are refused before a region is emitted;
- sequential routed regions do not leak binder or mapped-temp state.

Nested routed sums remain invalid on this baseline and should appear as a
well-formedness/refusal theorem, not be generalized away.

### 5. Representative fragment capstones

Instantiate the generic results on production-authored fixtures:

- the seven-node inverse clock fixture;
- one static and one live-count modal bank;
- one nested bank;
- one static and one live-count routed terminal;
- one routed terminal with inactive routes.

Keep independent numeric or authored-order observations in the tests. Do not
replace them with a test that merely runs the new Plan interpreter twice.

## Work plan

1. Define compile-result/operand correspondence and basic append lemmas.
2. Prove clock constructor compilation and `compileClockRail_refines`.
3. Combine `compileBankSum_stream` with initiative 1's region semantics.
4. Prove static, dynamic, and nested bank capstones.
5. Expose the minimum routed emitter surface and prove
   `compileRoutedSum_stream`.
6. Prove routed semantic preservation and refusal facts.
7. Instantiate all three families on production-authored fixtures.
8. Update trust-ledger limitations only after theorem symbols exist.

## Owned files

- `lean/Tropical/EmitArrow/ClockPlanLaws.lean`
- `lean/Tropical/Ir/BankPlanLaws.lean`
- `lean/Tropical/Ir/RoutedSumLaws.lean`
- `lean/Tropical/Testing/PipelineCapstones.lean`
- Minimal proof-enabling edits to `lean/Tropical/Ir/Emit.lean`
- Extension, not replacement, of `lean/Tropical/Ir/EmitBankLaws.lean`

W5 exclusively owns `Emit.lean` and `EmitBankLaws.lean` during the parallel
phase. It consumes W1's Plan semantics after the shared interface lands.

## Deliverables

- `compileClockRail_refines` and compiled forms of the existing clock laws.
- Static, dynamic-count, and nested bank source-to-Plan theorems.
- A routed-sum emission-shape theorem and authored-order source-to-Plan theorem.
- Production fixture instantiations.
- Precise trust-ledger updates separating proved source-to-Plan facts from
  still-open Plan-to-backend facts.

## Validation

```text
cd lean && lake build Tropical.EmitArrow.ClockPlanLaws Tropical.Ir.BankPlanLaws Tropical.Ir.RoutedSumLaws
cd lean && lake build Tropical.Testing.PipelineCapstones tropicaltest
./lean/.lake/build/bin/tropicaltest --emitarrow-only
./lean/.lake/build/bin/tropicaltest --routed-only
./lean/.lake/build/bin/tropicaltest
./lean/.lake/build/bin/trustreport --check
```

## Exit criteria

- Clock, bank, and routed fragments each have a theorem from direct source
  denotation to Plan denotation.
- Dynamic counts, binder IDs, nesting policy, and authored order appear in the
  statements, not only in tests.
- The routed emitter has a theorem comparable in strength to
  `compileBankSum_stream`.
- Trust rows name actual theorem symbols and retain their backend limitations.
- No source-to-Plan theorem is described as proof of LLVM/MSL execution.

## Risks and non-goals

- **Critical risk:** erasing floating-point order by assuming associativity.
- **High risk:** conflating Plan reference order with backend conformance.
- **High risk:** broadening the routed fragment to nested regions that the
  production emitter explicitly rejects.
- Whole-compiler, LLVM, MSL, runtime, cache, and callback refinement are not in
  scope.

