# Semantic spine: first checked slice

Decision date: 2026-07-27
Owner: Lean semantics
Decision: approved handoff fallback 1

## What landed

`Tropical.Semantics.denoteSig` is a carrier-parametric denotation of the
production `Tropical.EmitArrow.Sig` type. It covers all fourteen constructors,
uses an explicit `Except Refusal` result, resolves loop indices by unique binder
id, converts dynamic bank counts to signed integers and clamps them to
`[0, capacity]` exactly as production `regionTrips` does, and folds bank
contributions from index zero upward with a scalar left fold. It assumes no
associativity or commutativity.

`Tropical.Semantics.lowerSigTree_lowersTo` is the checked sprint capstone. It
proves that production `lowerSigTree` produces the mutual `LowersTo` /
`LowersMany` relation for every constructor and every initial `ExprArena`.
Arrays and bank tables lower from left to right; bank bodies lower after every
table; a present dynamic count lowers after the body; the final production
`eintern` step retains the static capacity and binder id.

This is the authorized relational fallback. It is not the proposed
denotational theorem and must not be cited as one.

## Why the full theorem stopped here

Production exposes `ExprArena.wf`, which checks that child ids descend, but it
does not expose or maintain a proposition connecting a `dedup.get? node = some
id` hit to `arena.deref id = some node`. The semantics names that missing
condition `DedupSound`. Without it, an arbitrary admitted arena can contain a
malformed dedup map and `eintern` can return an unrelated existing id, so a
total arena evaluator cannot soundly prove returned-node correctness.

The custom `ENode` `BEq` and hash also do not currently provide the lawful
instances used by the standard hash-map insertion lemmas. Adding those
production invariants is a prerequisite, not something this proof may assume
silently.

## Exact next theorem boundary

First prove, in production terms:

1. an empty expression arena satisfies `ArenaWellFormed`;
2. every final node assembled by a `LowersTo` constructor satisfies
   `ChildrenInPrefix arena node`, meaning all of its children belong to the
   frozen prefix produced by the preceding child lowerings;
3. `eintern node` preserves `ArenaWellFormed` only under that
   `ChildrenInPrefix` premise (the statement for an arbitrary node is false);
4. under `DedupSound`, the id returned by that qualified `eintern node`
   dereferences to `node`;
5. existing prefix dereferences and denotations are stable across the
   qualified `eintern`;
6. a total, well-founded `denoteExpr` agrees with `denoteSig` along `LowersTo`.

Only then state `lowerSigTree_preserves`. The pointer implementation remains a
separate `LOWER_SIG_PTR_REFINES_TREE` runtime/unsafe obligation.

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

`Tropical.Testing.Semantics` checks the capstone against the root Q32.32 clock,
a `bankFold` modal-column expression, a nested bank with distinct binder ids,
and a parameterized `select`. The same corpus differentially compares compiled
`lowerSig` with `lowerSigTree`; that differential is evidence for the unsafe
optimization, not a pointer-identity theorem. After each lowering it also
re-interns every emitted node and checks that the observed dedup hit returns
the original id without changing the node/signature arrays. This is finite
fixture evidence for the derived index, not a proof of `DedupSound`.

## Remaining spine

After the full first seam closes, the remaining boundaries are:

1. expression arena denotation to lowered whole-program denotation;
2. lowered program denotation to staged `tropical_plan_5`;
3. plan instruction semantics to LLVM/JIT and wasm execution;
4. plan semantics to MSL under its documented f32 numeric contract.
