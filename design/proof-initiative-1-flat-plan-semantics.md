# Proof initiative 1 — `FlatPlan` semantics and well-formedness

- **Baseline:** `origin/main` at `d7d9e1d4b4d161bf8b1e2800497d79c6381a609f`
- **Baseline date:** 2026-08-21
- **Priority:** P0
- **Difficulty:** Large
- **Suggested worker:** W1
- **Primary consumers:** initiatives 4 and 5
- **Can start independently:** yes

## Outcome

Give `Tropical.Plan.FlatPlan` an executable Lean denotation and a structural
well-formedness predicate. The result is the semantic waist needed to state
compiler and staging refinement without treating LLVM, MSL, or the C++ runtime
as the definition of a plan.

This initiative ends at the Plan-6 boundary. It does not prove that generated
LLVM or MSL implements that boundary.

## Ground truth on the pinned baseline

- `lean/Tropical/Plan.lean` defines `NOperand`, `DstSlot`, string-tagged
  `NInstr`, recursive `InstanceFunction`, `SinkSpec`, `SourceKind`, and
  `FlatPlan`.
- `SourceKind` is now
  `.tick | .rate | .tilePhase | .tileTick`; `defaultSources` is
  `[tick, rate]`, while `tileSources` adds the two tile-time rails.
- `FlatPlan` now carries both stage-0 and tile-time metadata:
  `coeffArraySlots`, `tileArraySlots`, `tileIntervalFrames`, and
  `phaserTimeStaging`.
- Instruction execution order is recursive: preamble, then each child's
  pre-input block and child blocks, then the function body. The same order is
  exposed publicly by `Tropical.Ir.Stage0.collectBlocks`; the similarly named
  helper in `Plan.lean` is private.
- `PlanOp` is a typed scalar signature, but an `NInstr.tag` is still a
  `String`. Structural tags include `Pack`, `SetElement`, `Index`,
  `WriteSlot`, `ReduceBegin`/`ReduceEnd`, and the three routed-sum delimiters.
- There is no `FlatPlanWF`, instruction-stream interpreter, or plan denotation
  on `origin/main`.
- Existing executable evidence lives in `Tropicaltest/Synthetic.lean`, the
  native mode-equivalence suite, frozen audio/MSL goldens, and the web
  cross-backend suites. Those are evidence, not a Plan denotation.

## Proof surface

### 1. Semantic state

Introduce a state rich enough to model the observable Plan contract:

```lean
structure PlanInputs (α : Type) where
  inputs : Array (Value α)
  params : String → Option (Value α)
  sources : Array (Value α)
  initialSlots : Array (Value α)
  initialArrays : Array (Array (Value α))

structure PlanState (α : Type) where
  temps : Array (Value α)
  slots : Array (Value α)
  arrays : Array (Array (Value α))
  openLoops : List LoopFrame
```

Names are provisional. Reuse `Tropical.Semantics.Algebra`, `Value`, `Result`,
and `Refusal` where their behavior matches the plan. Keep initialization and
zero-fallback rules explicit; do not silently strengthen runtime behavior.

Define source construction by `SourceKind`, not by hard-coded positions. The
ordinary and tile executions must both be expressible:

- audio: `tick`, `rate`, and, when present, published tile sources;
- coefficient: stable control values and shared coefficient arrays;
- tile materializer: `tilePhase` and `tileTick` supplied by its invocation.

The metadata fields do not execute instructions themselves. Model them as
constraints on how independently executed plans share and publish storage.

### 2. Instruction semantics

Implement a total `Except Refusal` interpreter for:

- scalar `PlanOp`s after parsing `NInstr.tag` with `PlanOp.ofString?`;
- `Pack`, `SetElement`, `Index`, and `WriteSlot`;
- properly nested `ReduceBegin`/`ReduceEnd`, including binder IDs, typed
  accumulators, and dynamic-count clamping;
- `RoutedSumBegin`/`Yield`/`End`, including the fixed output image and route
  order;
- operand namespaces, scalar types, array sizes, and out-of-range behavior.

The reference semantics for both region forms is authored-order scalar
folding. This discharges order questions at the Plan layer while deliberately
leaving backend execution as a separate trust obligation.

Avoid an interpreter that first rewrites regions into an unrolled instruction
array. A direct structured execution relation preserves the delimiter and
binder invariants that later proofs need.

### 3. Recursive function and plan semantics

Define one canonical block order and prove it agrees with
`Stage0.collectBlocks`. Interpret `InstanceFunction` recursively in that
order. Then interpret:

- all `instanceFunctions` in array order;
- sinks as array-order sums of their named module slots followed by `gain`;
- the published output as the sink target image.

Keep a lower-level `execBlocks` theorem surface available. Stage0 and
TileStage proofs need to compare instruction partitions before sink
observation.

### 4. Well-formedness

Define decidable predicates with elimination lemmas. At minimum:

- every scalar and array operand addresses an available namespace;
- every destination fits `registerCount`, `slotCount`, or `arraySlotCount`;
- `arraySlotNames` and `arraySlotSizes` align with `arraySlotCount`;
- `slotNames` and `slotDefaults` align with `slotCount`;
- source operands agree with `sources` and their declared scalar kind;
- no `sessionArray` destination or `sessionArrayReg` operand reaches a wire-ready
  `FlatPlan`;
- scalar op arity/result types are valid;
- array operations respect sizes and element types;
- reduction delimiters are balanced, binder IDs are resolvable, and region
  destinations match;
- routed regions are non-nested, have nonzero capacity/output/fanout, have
  `capacity * fanout` routes, and every target is in range;
- stage metadata slots are in range, deduplicated, and do not claim an
  impossible storage layout;
- recursive instance offsets and counts stay in the global plan bounds;
- sink inputs and targets are in range.

Separate `PlanWellFormed` from “the interpreter succeeds for a particular
environment.” Missing host params or incorrectly shaped supplied storage are
environment failures, not plan-shape failures.

### 5. Initial theorem set

Names may change during implementation, but the interface should provide the
following facts:

```lean
theorem execInstr_preserves_wf ...
theorem execBlocks_deterministic ...
theorem execInstanceFunction_deterministic ...
theorem planWellFormed_no_session_array_leak ...
theorem planWellFormed_regions_balanced ...
theorem planWellFormed_exec_safe ...
theorem collectBlocks_agrees_with_execution_order ...
theorem denoteFlatPlan_deterministic ...
```

`exec_safe` should exclude structural refusals under a well-formed plan and a
well-formed environment. It need not exclude algebraic refusals such as an
operation that is intentionally partial in a carrier.

## Work plan

1. Add the state, source interpretation, scalar-op interpretation, and small
   instruction semantics.
2. Add balanced `Reduce` semantics and prove the clamped-prefix lemma.
3. Add routed-region semantics and authored-order lemmas.
4. Add recursive `InstanceFunction` and sink semantics.
5. Define `NInstrWellFormed`, block/region well-formedness, and
   `FlatPlanWellFormed`; prove elimination lemmas as each clause lands.
6. Prove determinism, well-formed execution safety, and block-order agreement.
7. Add generated and hand-authored fixtures for default, banked, routed,
   nested-instance, stage-0, and tile-source plans.

## Owned files

Prefer new files:

- `lean/Tropical/Semantics/Plan.lean`
- `lean/Tropical/Semantics/PlanWellFormed.lean`
- `lean/Tropical/Testing/PlanSemantics.lean`

W1 should not edit `Ir/Stage0.lean`, `Ir/TileStage.lean`, `Ir/Emit.lean`, or
`Ir/Strata/EArena.lean`. Add aggregate imports only during sprint integration.

## Deliverables

- An executable denotation for every Plan instruction emitted on
  `origin/main`.
- A decidable, theorem-facing `FlatPlanWellFormed` predicate.
- Structural safety and determinism theorems.
- Fixtures that cover ordinary sources and the new tile sources.
- A short mapping from each remaining unsupported behavior to a named
  backend/host obligation rather than an implicit gap.

## Validation

```text
cd lean && lake build Tropical.Semantics.Plan Tropical.Semantics.PlanWellFormed
cd lean && lake build Tropical.Testing.PlanSemantics tropicaltest
./lean/.lake/build/bin/tropicaltest --routed-only
./lean/.lake/build/bin/tropicaltest
```

Run built executables directly, not through `lake exe`, per the repository's
LLVM dynamic-linking constraint.

## Exit criteria

- Every constructor and emitted instruction tag on the pinned baseline is
  handled or deliberately refused by a documented boundary.
- `FlatPlanWellFormed` rules out malformed regions, invalid namespaces, invalid
  metadata indices, and wire-leaking session arrays.
- Execution is deterministic.
- Initiatives 4 and 5 can state refinement against the public `execBlocks` or
  `denoteFlatPlan` interface without importing backend emitters.
- No trust-ledger row is marked proved merely because the interpreter exists.

## Risks and non-goals

- **High risk:** accidentally defining semantics from current emitter code and
  thereby making later “refinement” circular. Use the Plan schema and its
  documented contract as the definition.
- **High risk:** flattening recursive instance functions in an order that
  differs from emission.
- **Medium risk:** treating coefficient and tile plans as one synchronous plan;
  their publication protocol is a host-level relation.
- No LLVM, wasm, MSL, driver, cache, callback, or scheduling proof is in scope.
