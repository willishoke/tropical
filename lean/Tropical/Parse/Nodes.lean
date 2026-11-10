import Lean.Data.Json
import Tropical.Parse.OrderedJson

/-!
# Parse.Nodes — the scalar-kind vocabulary

What survives of the ParsedProgram AST: `ScalarKind`, the value-kind
vocabulary (`float`/`int`/`int32`/`bool`) shared by the IR, the plan, and the
wiring layer.

The rest of the parser AST — `ParsedExpr`, the decl/assign/program
node types, the parse-level op tags, the JSON encoder — died with the
elaborator (2026-07-26): program bodies no longer cross the wire, so
there is nothing left to parse them into. The JSON front door is the
patch-bay subset (`Parse/Raise.lean` `normalizeProgramFile`); wire
expressions live in the session grammar (`Tropical.WireExpr`'s decoder)
and lower directly (`Session.wireExprToResolved`).
-/

namespace Tropical.Parse

/-- `float` is an approximate real, realized per target (f64 on the LLVM
    path, f32 on Metal). The integer kinds are exact and their widths are
    meaning: `int` is two's complement on ℤ/2⁶⁴ and `int32` on ℤ/2³² — add,
    sub and mul wrap, on every target. Neither integer kind converts to the
    other, or to `float`, implicitly; the casts (`toInt`, `toInt32`,
    `toFloat`) are the only crossings. -/
inductive ScalarKind where
  | float | int | bool
  | int32
deriving BEq, Inhabited, Repr

def ScalarKind.wire : ScalarKind → String
  | .float => "float" | .int => "int" | .bool => "bool" | .int32 => "int32"

def ScalarKind.ofWire? : String → Option ScalarKind
  | "float" => some .float | "int" => some .int | "bool" => some .bool
  | "int32" => some .int32
  | _ => none

/-- An exact integer kind (`int` or `int32`). -/
def ScalarKind.isInteger : ScalarKind → Bool
  | .int | .int32 => true
  | _ => false

theorem ScalarKind.ofWire_wire (k : ScalarKind) :
    ScalarKind.ofWire? k.wire = some k := by
  cases k <;> rfl

end Tropical.Parse
