# Semantic spine, first vertical slice — sprint handoff

- **Sprint:** 2026-07-28 through 2026-08-10
- **Lane:** B — Lean semantics and preservation proof
- **DRI:** Assign a Lean proof owner at kickoff
- **Supervisor:** Staff engineer
- **Status:** Planned
- **Master:** [Staff engineer sprint handoff](00-staff-engineer-master-handoff.md)
- **Depends on:** None for implementation; Lane C consumes its theorem and
  assumption names.
- **Must not overlap:** No backend, runtime, benchmark, or architecture-doc
  edits.

## Outcome — fallback 1 landed

The original denotational target below is retained as the lane's historical
brief. The approved fallback supersedes it for this sprint:
`Tropical.Semantics.lowerSigTree_lowersTo` is checked for every production
`Sig` constructor against the production `lowerSigTree`. This is a relational
lowering theorem, not denotational preservation and not LLVM, wasm, or Metal
verification. The stronger theorem is a non-blocking follow-up requiring
production `DedupSound` preservation and lawful `ENode` equality/hash
behavior.

## Mission

Land the first end-to-end semantic preservation theorem over a production
compiler boundary:

> Lowering an authoring `Sig` through the structural reference
> `lowerSigTree` into the resolved arena preserves its denotation.

This is the first segment of the intended spine:

```text
Sig denotation
    ≃ ExprArena/ENode denotation
    ≃ lowered program denotation
    ≃ staged tropical_plan_5 denotation
```

Only the first equivalence is committed for this sprint. The work must leave a
clean interface for the later boundaries without pretending they are already
proved.

## Why this boundary

`Sig` is now the sole authoring vocabulary and `ENode` is the trunk IR.
`lowerSigTree` structurally mirrors the fourteen `Sig` constructors and is the
reference implementation behind the pointer-memoized production lowering:

```lean
@[implemented_by lowerSigPtr]
def lowerSig (s : Sig) : EArenaM ExprId := lowerSigTree s
```

Proving this seam gives the project:

- a denotation shared by future transformation proofs;
- a precise statement of well-formed environment and index obligations;
- a checked reference against which the narrow unsafe optimization is judged;
- a useful proof, rather than another isolated algebra identity.

## Committed theorem

The exact names may change during the Day 2 design review, but the statement
must have this content:

```lean
theorem lowerSigTree_preserves
    (s : Sig)
    (env : SigEnv α)
    (arena : ExprArena)
    (hArena : ArenaWellFormed arena)
    (hEnv : EnvWellFormed env)
    (hSig : SigWellFormed s env) :
    let (id, arena') := (lowerSigTree s).run arena
    denoteExpr arena' env id = denoteSig env s
```

Requirements:

- quantified over a carrier/algebra where practical;
- total on the admitted fragment;
- explicit about bounds or failed scalar operations;
- no hidden “default zero” for invalid indices;
- no `axiom`, `sorry`, `partial`, or new `unsafe`;
- exercises the production `Sig` and `ENode` types, not proof-only twins.

The DRI may use `Except`/`Option` equality if invalid scalar operations are
part of the existing semantics. Refusal must be modeled, not erased.

## Denotational scope

The must-land target covers all fourteen current `Sig` constructors:

- numbers;
- unary and binary scalar operations;
- `clamp` and `select`;
- input, parameter, and nested-output references;
- sample rate and sample index;
- literal arrays and indexing;
- loop indices;
- `bankSum`, including dynamic count clamping and binder identity.

This does **not** mean proving the analytic truth of `sin`, `exp`, `log`, or
gamma. Scalar primitive interpretation is an algebra parameter or an explicit
operation semantics. This theorem says lowering preserves the operations and
their order.

## Proposed module split

The lane owns new files under:

```text
lean/Tropical/Semantics/
  Value.lean       scalar/array values and refusal result
  Algebra.lean     primitive operation interface or executable semantics
  Sig.lean         SigEnv, well-formedness, denoteSig
  Arena.lean       arena well-formedness and denoteExpr
  LowerSig.lean    intern-extension lemmas and capstone theorem
```

Add a small import façade only if needed:

```text
lean/Tropical/Semantics.lean
```

Test/proof fixtures belong under:

```text
lean/Tropical/Testing/Semantics.lean
```

Do not put the proof into `Tropicaltest`; the theorem must compile with the
library. A fixture may instantiate it in `Tropical.Testing`.

## Proof design constraints

### One value model

Do not create a second compiler IR. The semantics may introduce a value domain,
environment, algebra, and propositions, but the syntax in the theorem must be
the production syntax.

### Arena extension

`eintern` may return an existing id or append a node. The proof needs lemmas
that:

- existing denotations are stable under an intern;
- the returned id denotes the interned node;
- references in an appended node point into the prior/frozen prefix;
- recursive lowering preserves arena well-formedness.

Name these lemmas for reuse by future transformation proofs.

### Binders

`bankSum` uses unique binder ids, not accidental array positions. Its
denotation must use an explicit loop-index environment. Nested banks must
shadow or distinguish binders according to the production contract.

### Order

The bank denotation is the left fold in increasing array order. Do not use
associativity or commutativity. This should line up with
`EmitArrow.BankOrder` and `Ir.EmitBankLaws`.

### Unsafe production implementation

Do not attempt to reason directly about `ptrAddrUnsafe` in this sprint.
`lowerSigTree` remains the definitional/reference implementation and
`lowerSigPtr` remains the compiled implementation selected by
`implemented_by`.

Lane C records “the implementation refines the reference” as an explicit
optimization obligation. This lane may add differential fixtures between the
two, but must not represent that as a theorem about pointer identity.

## Work plan

### Day 1: inventory and theorem sketch

- Enumerate every `Sig` and `ENode` constructor.
- Inventory existing scalar semantics in `ConstFold`, emitters, and tests.
- Write the proposed denotation and theorem statement in a design note inside
  the lane PR.
- Identify whether `ENode`’s frozen-prefix property already has a reusable
  proposition.

### Day 2: design review and statement freeze

The staff engineer and one independent Lean reviewer approve:

- value/refusal model;
- environment shape;
- arena well-formedness statement;
- bank binder semantics;
- exact capstone theorem.

After this review, do not broaden the theorem. Any new semantic question goes
in “follow-ups.”

### Days 3–4: leaf and scalar fragment

- Land `Value`, `Algebra`, and `Sig` semantics.
- Cover leaves, unary/binary, clamp, and select.
- Prove environment lookup behavior with no silent defaults.

### Days 5–6: arena evaluator and intern lemmas

- Define total well-founded arena evaluation.
- Prove extension stability and returned-node correctness.
- Prove preservation for the scalar/reference fragment.

### Days 7–8: arrays and banks

- Add array/index and loop environment semantics.
- Add static and dynamic bank folds.
- Reuse the existing order and dynamic-prefix lemmas rather than duplicating
  them.

### Day 9: capstone and production fixture

- Close `lowerSigTree_preserves`.
- Instantiate it on at least:
  - the root clock;
  - a small modal bank;
  - a nested-bank fixture;
  - a parameterized expression with `select`.

### Day 10: review and handoff

- Run a proof-surface audit.
- Document the next compiler boundary and all assumptions not discharged.
- Do not begin the next preservation theorem.

## Scope fallback

If the arena-intern proof is not converging by the end of Day 4, the staff
engineer chooses exactly one fallback:

1. land a relational `LowersTo` semantics with the same all-constructor
   coverage and prove `lowerSigTree` produces that relation; or
2. land the capstone for the bank-free fragment and a separate, fully proved
   bank denotation theorem, with a dated follow-up for composition.

It is not acceptable to land:

- a theorem with `sorry`;
- a theorem over proof-only syntax;
- a scalar-only theorem described as covering production banks;
- an undocumented axiom about intern behavior.

## Acceptance gates

1. The library builds with the new semantics modules.
2. The capstone theorem or approved fallback is checked by Lean.
3. All fourteen constructors are either covered or listed in the fallback’s
   explicit remaining set.
4. Production fixtures typecheck against the theorem.
5. This audit returns no new trust escape:

   ```bash
   rg -n '\b(axiom|sorry|partial|unsafe)\b' lean/Tropical/Semantics \
     lean/Tropical/Testing/Semantics.lean
   ```

6. Existing proof modules still build.
7. `make validate` is green.

## Non-goals

- No proof of LLVM, WebAssembly, or MSL correctness.
- No real-analysis proof for transcendental approximations.
- No redesign of `Sig`, `ENode`, or the plan.
- No replacement of the pointer memo.
- No type-indexed `Mor` migration.
- No second preservation boundary.

## Risks and stop conditions

Stop and escalate if:

- the production semantics differ between JIT and MSL at the operation level;
- an invalid index currently reads a default value in one path and refuses in
  another;
- a `bankSum` order or dynamic-count rule differs between syntax and emit;
- proving arena well-formedness requires changing the production intern order;
- the theorem statement changes after Day 4 without staff approval.

Any such discovery is a sprint result. Record it precisely; do not patch around
it inside the proof.

## Handoff package

Leave:

- the checked theorem and supporting modules;
- a one-page semantics map;
- a list of modeled refusals;
- a list of remaining semantic/compiler boundaries;
- the exact obligation passed to Lane C for the unsafe implementation;
- no unused experimental definitions.
