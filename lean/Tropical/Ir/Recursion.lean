import Tropical.Ir.Nodes

/-!
# Structural recursion over `Expr` — port of compiler/ir/recursion.ts

One generic walker (`mapExpr`) owns the structural traversal across
every `Expr` constructor. Each pass supplies a small hook set
describing only the cases it cares about; everything else recurses
structurally for free.

Hook contract (the TS `MapHooks`, with `Option` playing `NoRewrite`):
  - `expr e = some r` replaces the node (short-circuits recursion);
    `none` recurses structurally.
  - `binder` transforms a `Binder` at every binding site
    (combinators, let, match arms). Default: identity.

The TS walker preserves object identity on no-change as a sharing
optimization; the codec is structural, so a plain rebuild is
output-faithful. (Expr-level sharing is invisible to
`tropical_resolved_1` — only the three arena pools carry identity.)
-/

namespace Tropical.Ir

structure MapHooks where
  expr : Expr → Option Expr := fun _ => none
  binder : Binder → Binder := id

mutual

partial def mapExpr (h : MapHooks) (e : Expr) : Expr :=
  match h.expr e with
  | some r => r
  | none =>
    match e with
    -- Leaves — no children to walk.
    | .num _ | .bool _
    | .inputRef _ | .regRef _ | .paramRef _ | .typeParamRef _ | .bindingRef _
    | .nestedOut _ _ | .sampleRate | .sampleIndex => e
    | .arr items => .arr (items.map (mapExpr h))
    -- ADT.
    | .tag def_ variant payload =>
      .tag def_ variant (payload.map (mapTagPayload h))
    | .match_ def_ scrutinee arms =>
      .match_ def_ (mapExpr h scrutinee) (arms.map (mapMatchArm h))
    -- Let.
    | .letIn binders body =>
      .letIn (binders.map (mapLetBinder h)) (mapExpr h body)
    -- Combinators with one binder.
    | .generate count iter body =>
      .generate (mapExpr h count) (h.binder iter) (mapExpr h body)
    | .iterate count init iter body =>
      .iterate (mapExpr h count) (mapExpr h init) (h.binder iter) (mapExpr h body)
    | .chain count init iter body =>
      .chain (mapExpr h count) (mapExpr h init) (h.binder iter) (mapExpr h body)
    | .map2 over elem body =>
      .map2 (mapExpr h over) (h.binder elem) (mapExpr h body)
    -- Combinators with two binders.
    | .fold over init acc elem body =>
      .fold (mapExpr h over) (mapExpr h init) (h.binder acc) (h.binder elem)
        (mapExpr h body)
    | .scan over init acc elem body =>
      .scan (mapExpr h over) (mapExpr h init) (h.binder acc) (h.binder elem)
        (mapExpr h body)
    | .zipWith a b x y body =>
      .zipWith (mapExpr h a) (mapExpr h b) (h.binder x) (h.binder y)
        (mapExpr h body)
    -- Operators.
    | .binary tag lhs rhs => .binary tag (mapExpr h lhs) (mapExpr h rhs)
    | .unary tag arg => .unary tag (mapExpr h arg)
    | .clamp v lo hi => .clamp (mapExpr h v) (mapExpr h lo) (mapExpr h hi)
    | .select c t f => .select (mapExpr h c) (mapExpr h t) (mapExpr h f)
    | .arraySet a i v => .arraySet (mapExpr h a) (mapExpr h i) (mapExpr h v)
    | .index a i => .index (mapExpr h a) (mapExpr h i)
    | .zeros count => .zeros (mapExpr h count)

partial def mapLetBinder (h : MapHooks) : LetBinder → LetBinder
  | .mk binder value => .mk (h.binder binder) (mapExpr h value)

partial def mapTagPayload (h : MapHooks) : TagPayload → TagPayload
  | .mk field value => .mk field (mapExpr h value)

partial def mapMatchArm (h : MapHooks) : MatchArm → MatchArm
  | .mk variant binders body =>
    .mk variant (binders.map h.binder) (mapExpr h body)

end

/-- Port-type / shape-dim walker (used by specialize). -/
def mapPortType (shapeDim : ShapeDim → ShapeDim) : PortType → PortType
  | .array element shape => .array element (shape.map shapeDim)
  | pt => pt

end Tropical.Ir
