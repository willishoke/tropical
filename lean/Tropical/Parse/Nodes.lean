import Lean.Data.Json
import Tropical.Parse.OrderedJson

/-!
# Parse.Nodes — the scalar-kind vocabulary

What survives of the ParsedProgram AST: `ScalarKind`, the value-kind
vocabulary (`float`/`int`/`bool`) shared by the IR, the plan, and the
wiring layer.

The rest of the parser AST — `ParsedExpr`, the decl/assign/program
node types, the parse-level op tags, the JSON encoder — died with the
elaborator (2026-07-26): program bodies no longer cross the wire, so
there is nothing left to parse them into. The JSON front door is the
patch-bay subset (`Parse/Raise.lean` `normalizeProgramFile`); wire
expressions live in the session grammar (`Tropical.WireExpr`'s decoder)
and lower directly (`Engine.wireExprToResolved`).
-/

namespace Tropical.Parse

inductive ScalarKind where
  | float | int | bool
deriving BEq, Inhabited, Repr

def ScalarKind.wire : ScalarKind → String
  | .float => "float" | .int => "int" | .bool => "bool"

def ScalarKind.ofWire? : String → Option ScalarKind
  | "float" => some .float | "int" => some .int | "bool" => some .bool
  | _ => none

theorem ScalarKind.ofWire_wire (k : ScalarKind) :
    ScalarKind.ofWire? k.wire = some k := by
  cases k <;> rfl

end Tropical.Parse
