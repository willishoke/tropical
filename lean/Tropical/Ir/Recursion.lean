import Tropical.Ir.Nodes

/-!
# Port-type structural helper — port of compiler/ir/recursion.ts

The generic `Expr` walker (`mapExpr` + the `MapHooks` binder-transform) that
this module once owned is gone: the strata passes operate on the hash-consed
id-form (`mapExprId` over `ENode` in `Strata/EArena.lean`), not the tree, so the
tree walker had no callers. Only the port-type / shape-dim walker survives —
`specialize` uses it to substitute type-param shape dimensions.
-/

namespace Tropical.Ir

/-- Port-type / shape-dim walker (used by specialize). -/
def mapPortType (shapeDim : ShapeDim → ShapeDim) : PortType → PortType
  | .array element shape => .array element (shape.map shapeDim)
  | pt => pt

end Tropical.Ir
