# Semantic spine: first checked slice

Decision date: 2026-07-27
Owner: Lean semantics
Follow-up date: 2026-07-29
Decision: full first lowering seam proved

## What landed

`Tropical.Semantics.denoteSig` is a carrier-parametric denotation of the
production `Tropical.EmitArrow.Sig` type. It covers all fourteen constructors,
uses an explicit `Except Refusal` result, resolves loop indices by unique binder
id, converts dynamic bank counts to signed integers and clamps them to
`[0, capacity]` exactly as production `regionTrips` does, and folds bank
contributions from index zero upward with a scalar left fold. It assumes no
associativity or commutativity.

`Tropical.Semantics.lowerSigTree_preserves` is the checked capstone. For every
carrier algebra, environment, production `Sig`, and well-formed initial
`ExprArena`, it proves that structural `lowerSigTree` returns a well-formed
arena extension and that the returned `ExprId` has exactly the direct
`denoteSig` result. The equality includes explicit refusals and therefore does
not depend on a source well-formedness premise.

The supporting proof establishes the previously missing production facts:

1. empty-arena well-formedness;
2. lawful executable equality for node keys and indices, exposing the standard
   hash-map insertion laws to Lean;
3. qualified `eintern` preservation under `ChildrenInPrefix`;
4. returned-node dereference and stability of every prior dereference;
5. a total, child-descending `denoteExpr`;
6. denotation stability across arena extension;
7. mutual preservation for `LowersTo` and ordered `LowersMany` traces.

`lowerSigTree_lowersTo` remains as the useful operational trace theorem. The
unsafe pointer implementation remains the separate
`LOWER_SIG_PTR_REFINES_TREE` obligation.

## Modeled refusals and evaluation order

- missing input, parameter, nested-instance, nested-output, or loop-binder
  lookup;
- any unary, binary, clamp, select, or index refusal supplied by the algebra;
- any failure converting a dynamic count or loop index into the carrier;
- a failure while evaluating a literal array or explicit bank table.

The semantics evaluates both operands and ternary branches structurally from
left to right before applying the algebra. In particular, `select` is modeled
as an eager primitive because lowering preserves both child nodes; a later
backend theorem must document whether runtime branch laziness can observably
change refusals. Bank table values are eagerly checked even though body
subexpressions perform the indexed reads. This is the conservative production
node contract; a later arena evaluator must retain the same order or explicitly
narrow the admitted fragment.

## Production fixtures

`Tropical.Testing.Semantics` instantiates the preservation theorem against the
root Q32.32 clock,
a `bankFold` modal-column expression, a nested bank with distinct binder ids,
and a parameterized `select`. The same corpus differentially compares compiled
`lowerSig` with `lowerSigTree`; that differential is evidence for the unsafe
optimization, not a pointer-identity theorem. After each lowering it also
re-interns every emitted node and checks that the observed dedup hit returns
the original id without changing the node/signature arrays. This is finite
fixture evidence for the unsafe pointer implementation. `DedupSound` itself is
now maintained by the proved `eintern_preserves` theorem.

## Remaining spine

With the full first seam closed, the remaining boundaries are:

1. expression arena denotation to lowered whole-program denotation;
2. lowered program denotation to staged `tropical_plan_6`;
3. plan instruction semantics to LLVM/JIT and wasm execution;
4. plan semantics to MSL under its documented f32 numeric contract.
