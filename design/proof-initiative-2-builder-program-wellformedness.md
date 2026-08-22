# Proof initiative 2 — builder and program well-formedness preservation

- **Baseline:** `origin/main` at `d7d9e1d4b4d161bf8b1e2800497d79c6381a609f`
- **Baseline date:** 2026-08-21
- **Priority:** P0
- **Difficulty:** Medium–Large
- **Suggested worker:** W2
- **Primary consumers:** initiatives 3 and 4
- **Can start independently:** yes

## Outcome

Replace “IDs are scoped by convention” with explicit predicates and
preservation theorems covering expression construction, declarations,
assembly, and the program pool. A successfully certified production assembly
should yield addressable roots, a semantically well-formed expression arena,
and a child-descending program pool.

## Ground truth on the pinned baseline

- `Tropical.EmitArrow.Sig` and `Clock` are `ExprId` aliases.
- `Builder` publicly contains `exprs : ExprArena` and `decls : Array AInst`;
  its documentation says expression IDs are scoped “by convention.”
- `BuildM` is the public abbreviation `StateT Builder (Except String)`.
  Therefore arbitrary callers can use `get`, `set`, or `modify` and can create
  an invalid builder state. A universal theorem that every `BuildM` action
  preserves well-formedness is false without changing this API.
- `internSig` is private and all ordinary smart constructors call it, but their
  `Sig` arguments are unqualified IDs and may come from a different arena.
- `ExprArena.wf` proves child descent only. `Semantics.ArenaWellFormed` also
  requires a sound dedup map and aligned stage signatures.
- `eintern_preserves` already proves semantic-arena preservation when all
  children are in the current prefix.
- `progPoolWf`, `progPool_children_lt`, and `progPool_registry_lt` already
  provide the program-pool termination spine.
- `ENode` now includes `tileArray`, `tileSampleIndex`, and `tilePhase`, and
  `shiftSampleIndex` rebuilds the frozen arena while substituting the clock
  leaf. These new paths must be included, not treated as future extensions.

## Design decision

Do not claim preservation for arbitrary `BuildM`. Use a qualified action
contract:

```lean
def SigIn (b : Builder) (id : Sig) : Prop := id.idx < b.exprs.nodes.size

def PreservesBuilderWF (action : BuildM α) : Prop :=
  ∀ b, BuilderWellFormed b →
    match action.run b with
    | .error _ => True
    | .ok (_, b') => BuilderWellFormed b' ∧ BuilderExtends b b'
```

The exact shape is provisional. The essential point is that certification is
a property proved for the production smart constructors and combinators, not
an unjustified instance on the raw state monad.

If sealing `BuildM` behind a constructor is chosen, treat it as an API change
requiring a separate review. The proof initiative does not require that
refactor to deliver useful theorems.

## Proof surface

### 1. Expression and builder predicates

Define:

- `ExprIdIn`/`SigIn` for addressability;
- `ENodeChildrenIn` for all child references of a proposed node;
- `BuilderDeclsWellFormed` for every `AInst` input value;
- `BuilderWellFormed`, combining `ArenaWellFormed`, declaration addressability,
  and any declaration-index conditions needed by assembly;
- `BuilderExtends`, including expression-arena extension and preservation of
  existing declarations.

Add bridges between the executable `ExprArena.wf = true` predicate and the
propositional `ChildrenDescend` part of `ArenaWellFormed`. Do not assume
`wf = true` proves dedup soundness or signature alignment by itself.

### 2. Qualified smart-constructor preservation

Prove a common qualified-intern theorem from `eintern_preserves`, then derive
constructor theorems for:

- literals and all scalar unary/binary/ternary constructors;
- input, param, nested-output, sample-rate, and clock leaves;
- `arr`, `tileArray`, `index`, and array mutation where production exposes it;
- `loopIdx`, `bankSum`, and `routedSum`, including all table/value/count roots;
- `tileSampleIndex`, `tilePhase`, and `shiftSampleIndex`.

For constructors that accept existing IDs, the theorem must require those IDs
to be addressable in the input builder. This makes cross-arena misuse visible
in the theorem statement.

For `shiftSampleIndex`, prove at least:

- returned roots are addressable;
- the output builder extends the input builder;
- expression arena well-formedness is preserved;
- the result array has the same length as `roots`;
- every rebuilt node references only the frozen prefix or earlier rebuilt
  nodes.

Initiative 4 owns the semantic substitution theorem; this initiative owns the
construction invariant.

### 3. Program predicates

Define separate layers rather than one opaque `ProgramWF`:

- `ProgramExprRefsIn arena p`: every input default, instance input, and output
  assignment is addressable;
- `ProgramIndicesWellFormed p`: `InputIdx`, `ParamIdx`, `InstanceIdx`,
  `OutputIdx`, and output-target references are in range;
- `ProgramRegistryWellFormed programs i p`: every instance type key resolves,
  pool links descend, and declaration/registry links agree;
- `ProgramWellFormed arena programs i` as the assembled conjunction;
- corresponding `CoreProgramWellFormed` for the materialized recursive form.

Keep type checking separate unless an existing production check already
provides the exact property. Addressability, graph shape, and port typing are
different proof surfaces.

### 4. Assembly preservation

Identify the public `assemble*` functions used by `Stdlib`, playground, and
tests. For each production boundary, prove a theorem of the shape:

```lean
theorem assemble_of_certified_build ... :
  action.run initial = .ok (body, final) →
  PreservesBuilderWF action →
  ProgramBodyWellFormed final body →
  ProgramWellFormed resultArena resultPrograms resultRoot
```

If existing assembly already validates a clause, expose the validation result
as a lemma rather than duplicating the check. If it does not, either add a
boundary check or retain the clause as a precondition; record that choice in
the handoff.

### 5. Failure atomicity

The current `Except` prevents a failed build from publishing a program, but
plain `StateT` does not itself return the intermediate state on error. State
and prove the actual observable guarantee: assembly returns no arena/program
result on failure. Do not claim rollback of an externally visible mutable
store that does not exist.

## Work plan

1. Introduce addressability, node-child, builder, and extension predicates.
2. Bridge executable and propositional arena invariants.
3. Prove the common intern lemma and the scalar/leaf constructors.
4. Cover arrays, reductions, routed sums, and all new tile constructors.
5. Prove `shiftSampleIndex` construction preservation.
6. Add layered `Program`/`CoreProgram` predicates and elimination lemmas.
7. Prove the ordinary and complete assembly boundary theorems.
8. Instantiate the boundary theorem for representative scalar, nested,
   banked, routed, and tile-producing builders.

## Owned files

- `lean/Tropical/Semantics/WellFormed.lean`
- `lean/Tropical/EmitArrow/BuilderLaws.lean`
- `lean/Tropical/Testing/BuilderLaws.lean`
- Minimal proof-enabling edits to `lean/Tropical/EmitArrow/Sig.lean`

W2 owns `Sig.lean` during the parallel phase. Other workers should consume the
published theorem interface instead of editing the builder internals.

## Deliverables

- Layered builder, `Program`, and `CoreProgram` well-formedness predicates.
- Preservation theorems for every production smart constructor, including the
  three tile nodes and `shiftSampleIndex`.
- Certified assembly boundary theorems.
- Negative fixtures for dangling and cross-arena IDs.
- A documented answer to whether raw `BuildM` remains intentionally public.

## Validation

```text
cd lean && lake build Tropical.Semantics.WellFormed Tropical.EmitArrow.BuilderLaws
cd lean && lake build Tropical.Testing.BuilderLaws tropicaltest
./lean/.lake/build/bin/tropicaltest --emitarrow-only
./lean/.lake/build/bin/tropicaltest --phaser-only
```

## Exit criteria

- No theorem relies on the phrase “scoped by convention.”
- Every production smart constructor has a qualified preservation theorem or
  is explicitly classified as an unchecked escape hatch.
- Successful certified assembly produces addressable program roots and both
  expression- and program-pool well-formedness.
- `shiftSampleIndex` is construction-safe on the new absolute-time path.
- Initiative 3 can import program predicates without redefining them.

## Risks and non-goals

- **Critical risk:** stating the false theorem that arbitrary public `BuildM`
  actions preserve invariants.
- **High risk:** using `ExprArena.wf` as if it also proved dedup and stage
  signature correctness.
- **Medium risk:** growing `ProgramWellFormed` into a full source type system.
- This initiative does not prove expression or program denotational
  preservation; initiative 3 owns that layer.

