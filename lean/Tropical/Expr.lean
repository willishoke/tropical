import Lean.Data.Json

/-!
# Json field helpers

The small accessor kit the engine layer uses on tool-argument and
document Json (`getField?` / `getStrField?` / `opOf?`).

The wire-expression walkers that used to live here (`validateExpr`,
`exprDependencies`, `prettyExpr`) are gone: wire expressions are TYPED
now (`Tropical.WireExpr`) — the decoder is the validator, `deps` is
the dependency walk, and `pretty` is the renderer. Raw Json past the
tool boundary no longer carries expression structure.
-/

namespace Tropical.Expr

open Lean (Json)

def getField? (j : Json) (k : String) : Option Json :=
  (j.getObjVal? k).toOption

def getStrField? (j : Json) (k : String) : Option String :=
  match getField? j k with
  | some (.str s) => some s
  | _ => none

def opOf? (j : Json) : Option String :=
  match j with
  | .obj _ => getStrField? j "op"
  | _ => none

end Tropical.Expr
