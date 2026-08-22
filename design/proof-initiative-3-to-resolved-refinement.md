# Proof initiative 3 — `EArena.toResolved` denotation preservation

- **Baseline:** `origin/main` at `d7d9e1d4b4d161bf8b1e2800497d79c6381a609f`
- **Baseline date:** 2026-08-21
- **Priority:** P0
- **Difficulty:** Medium–Large
- **Suggested worker:** W3
- **Dependency:** initiative 2 for final named well-formedness predicates
- **Can start independently:** yes, with explicit assumptions

## Outcome

Prove that reachability GC at the strata exit preserves expression and nested
program meaning. `EArena.toResolved` should become a verified
representation-changing boundary, not merely a terminating structural copy.

## Ground truth on the pinned baseline

- Source and destination expressions use the same `ENode` vocabulary. The
  conversion is a reachable-only copy into a fresh `ExprArena`, with hash-cons
  memoization.
- `convExprId` and `convProgram` in `Ir/Strata/EArena.lean` are private.
- Both conversions are already total: expressions descend on `ExprId.idx`
  using `ExprArena.wf`; programs descend on `ProgramIdx.idx` using
  `progPoolWf`.
- `convExprId` covers all current constructors, including `tileArray`,
  `tileSampleIndex`, `tilePhase`, `bankSum`, and `routedSum`.
- `convProgram` copies input defaults, instance inputs, and output assigns;
  follows only instance-referenced registry entries in first-use order; and
  turns inert `.prog` declarations into `.progDecl name`.
- `EArena.toResolved` returns `(ExprArena × CoreProgram)` and discards the memo
  that directly records the source/destination root correspondence.
- `Semantics.denoteExpr` and `denoteExpr_extends` exist. There is no denotation
  for `Program` or `CoreProgram` on `origin/main`.

## Semantic boundary

The theorem is about evaluator-reachable behavior:

- source program declarations not referenced by an instance are inert;
- registry insertion order and instance declaration order remain observable
  structural facts, but unreachable pool members have no denotation;
- source and destination expression IDs need not be equal;
- sharing and fresh-arena node counts are not semantic observations;
- refusals should correspond as well as successful values.

## Proof surface

### 1. Proof-visible copy relation

Refactor minimally so the converter exposes a witness without duplicating its
implementation. Suitable options are:

- return the final source-ID to destination-ID memo from a proof-facing helper;
- define a public relational specification and prove the existing private
  converter satisfies it inside `EArena.lean`; or
- add a theorem adjacent to the private definitions and export only the
  theorem.

Define a relation such as:

```lean
def ExprCopyRel (src dst : ExprArena) (srcId dstId : ExprId) : Prop :=
  -- corresponding dereferenced constructors and recursively related children
```

It must cover arrays, tile arrays, both reduction forms, binder IDs, dynamic
counts, and routed metadata exactly.

### 2. Destination well-formedness

Prove that conversion from a semantically well-formed source produces a
semantically well-formed destination:

- children descend;
- dedup is sound;
- stage signatures align with nodes;
- every converted root is addressable.

The current executable checks provide descent, not the complete
`ArenaWellFormed` record. Reuse initiative 2's bridge when available. Until
then, state the stronger source invariant as a theorem parameter and prove the
fresh destination invariant directly from `emptyArena_wellFormed` and
`eintern_preserves`.

### 3. Expression-copy denotation

Prove the constructor-generic capstone:

```lean
theorem denoteExpr_copy_eq
    (hcopy : ExprCopyRel src dst srcId dstId) ... :
  denoteExpr alg env src hSrc srcId =
  denoteExpr alg env dst hDst dstId
```

The proof should be by the copy derivation or source-ID descent. It must retain
the loop environment when descending into `bankSum` and `routedSum`; a theorem
only for the top-level environment is insufficient.

### 4. Source and core program denotations

Add a small, pure denotation for the evaluator-reachable program structure:

- input defaults and supplied inputs;
- params in declaration order;
- instances in declaration order, resolving `typeKey` through the registry;
- instance input expressions in the parent environment;
- `nestedOut` results from already evaluated children;
- output assignments and `.dac` targets in authored order.

Use the same observation type for pooled `Program` and recursive
`CoreProgram`. The source recursion terminates through `progPoolWf`; the core
recursion terminates through `CoreProgram.sizeOf_lt_of_registryGet?`.

Do not include partitioning, slots, or sinks here. Those belong to Plan
semantics and later compiler refinement.

### 5. Program conversion refinement

Prove, for a successful `toResolved`:

```lean
theorem toResolved_preserves_denotation ...
    (hresult : EArena.toResolved ea root = .ok (dst, core)) :
  denoteProgram ea root sourceInputs =
  denoteCoreProgram dst core sourceInputs
```

Also expose useful structural corollaries:

- all destination roots are addressable;
- every destination expression is reachable from the returned core program;
- every instance-referenced registry entry is present exactly once in
  first-use order;
- unreachable source expressions and programs do not affect the result.

The last fact may be relational rather than equality of converted arenas,
because hash-cons layout is intentionally not observable.

## Work plan

1. Define `ExprCopyRel` and make the current copy witness proof-visible.
2. Prove destination arena well-formedness and addressability.
3. Prove expression-copy denotation preservation for every `ENode` case.
4. Define paired source/Core program denotations.
5. Prove `convProgram` preserves the recursive observation.
6. Prove the public `EArena.toResolved` capstone and GC corollaries.
7. Add fixtures with sharing, unreachable nodes, nested registries, banked and
   routed expressions, and tile nodes.

## Owned files

- `lean/Tropical/Semantics/Program.lean`
- `lean/Tropical/Semantics/Strata.lean`
- `lean/Tropical/Testing/StrataSemantics.lean`
- Proof-enabling edits to `lean/Tropical/Ir/Strata/EArena.lean`

W3 exclusively owns `EArena.lean` during parallel work. It should not edit
`Sig.lean`, `Stage0.lean`, `TileStage.lean`, or `Emit.lean`.

## Deliverables

- A proof-visible expression copy relation/witness.
- Source `Program` and destination `CoreProgram` denotations.
- Expression and whole-program preservation theorems.
- Reachability and first-use registry corollaries.
- Regression fixtures covering the new tile constructors.

## Validation

```text
cd lean && lake build Tropical.Semantics.Program Tropical.Semantics.Strata
cd lean && lake build Tropical.Testing.StrataSemantics Tropical.Semantics
./lean/.lake/build/bin/tropicaltest --emitarrow-only
./lean/.lake/build/bin/tropicaltest
```

## Exit criteria

- `EArena.toResolved` has a theorem relating successful output to source
  denotation for arbitrary supported `Algebra` and environments.
- Every current `ENode` constructor is covered.
- Nested instance evaluation and first-use registry GC are included.
- The theorem does not equate source and destination IDs or node arrays.
- Final statements import initiative 2's canonical predicates after
  integration rather than maintaining duplicate well-formedness definitions.

## Risks and non-goals

- **High risk:** proving only a shallow node-copy fact and losing the loop
  environment in reduction bodies.
- **High risk:** making unreachable declarations semantically observable and
  thereby falsifying GC preservation.
- **Medium risk:** exposing converter internals as permanent public API when an
  exported theorem would suffice.
- This initiative does not prove `CoreProgram → FlatPlan`; that requires the
  Plan semantics from initiative 1 and is a later compiler-wide capstone.

