import Std.Data.HashMap
import Tropical.Ir.Nodes
import Tropical.Ir.CoreArena

/-!
# `E*` compatibility aliases — the id-form IS the IR now

`ExprArena`/`ENode` (the hash-consed resolved-expression DAG) and the id-valued
`Program` live in `Nodes.lean`. Before the `Expr`-deletion flip, the strata
pipeline worked on a *parallel* id-form (`EProgram`, `EInputDecl`, … built from
the tree `Program` by `toEProgram`/`toExprArena`). Now that `Program` itself is
id-valued, those `E*` types are exactly the base types — kept here as `abbrev`s
so the pass modules read unchanged. The tree bridges (`toExprArena`,
`ExprArena.toExpr`, `toEProgram`) are gone: there is no tree to convert.
-/

namespace Tropical.Ir

/-- The pipeline's program form is now just `Program` (id-valued). -/
abbrev EProgram := Program
abbrev EInputDecl := InputDecl
abbrev EInstanceInput := InstanceInput
abbrev EBodyDecl := BodyDecl
abbrev EOutputAssign := OutputAssign

end Tropical.Ir
