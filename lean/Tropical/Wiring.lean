import Lean.Data.Json
import Tropical.Expr
import Tropical.WireExpr
import Tropical.Parse.Nodes

/-!
# Port types and connection checking

Port of the post-strata `PortType` view (compiler/ir/port_type.ts) and
`checkArrayConnection` (compiler/array_wiring.ts): scalar widening
(bool → int → float), scalar→array broadcast insertion, array→array
shape broadcasting. Error strings mirror the TS source exactly — they
appear verbatim in `type_mismatch` envelopes.

The Lean engine never *constructs* port types; they arrive as structured
JSON in catalog entries from the compiler service and are parsed here
only for checking. Display strings also travel precomputed in the
catalog; `portTypeToString` exists for the connection-error messages.
-/

namespace Tropical.Wiring

open Lean (Json)
open Tropical.Expr (getField? getStrField?)

/-- The parse-layer enum is canonical (`Ir` and `Plan` alias it too);
    the widening lattice below is what belongs to the wiring layer. -/
abbrev ScalarKind := Tropical.Parse.ScalarKind

/-- Widening lattice rank: bool → int → float. -/
def rank : ScalarKind → Nat
  | .bool => 0 | .int => 1 | .float => 2

def widens (src dst : ScalarKind) : Bool := rank src ≤ rank dst

def narrowingHint : ScalarKind → String
  | .int   => "to_int()"
  | .bool  => "to_bool()"
  | .float => "to_float()"

/-- An array element: display name plus the underlying scalar kind
    (aliases carry their base). -/
structure Elem where
  display : String
  kind    : Option ScalarKind

inductive PortType where
  | scalar (k : ScalarKind)
  | alias  (name : String)
  | array  (elem : Elem) (shape : Array Nat)

def parsePortType? (j : Json) : Option PortType := do
  match getStrField? j "kind" with
  | some "scalar" =>
    let k ← Parse.ScalarKind.ofWire? (← getStrField? j "scalar")
    return .scalar k
  | some "alias" =>
    let a ← getField? j "alias"
    return .alias ((getStrField? a "name").getD "?")
  | some "array" =>
    let elem : Elem ← match getField? j "element" with
      | some (.str s) => pure { display := s, kind := Parse.ScalarKind.ofWire? s }
      | some a => pure { display := (getStrField? a "name").getD "?",
                         kind := (getStrField? a "base").bind Parse.ScalarKind.ofWire? }
      | none => none
    let shape ← match getField? j "shape" with
      | some (.arr dims) => dims.mapM fun d => match d with
          | .num n => some n.toFloat.toUInt64.toNat
          | _ => none
      | _ => none
    return .array elem shape
  | _ => none

def portTypeToString : PortType → String
  | .scalar k => k.wire
  | .alias n  => n
  | .array e shape =>
    s!"{e.display}[{String.intercalate "," (shape.map toString).toList}]"

def portTypeEqual : PortType → PortType → Bool
  | .scalar a, .scalar b => a == b
  | .alias a, .alias b   => a == b   -- TS compares alias decls by object identity; by name here
  | .array ea sa, .array eb sb =>
    sa == sb && ea.display == eb.display
  | _, _ => false

/-- NumPy-style shape broadcasting (compiler/term.ts broadcastShapes). -/
def broadcastShapes (a b : Array Nat) : Option (Array Nat) := Id.run do
  let rank := Nat.max a.size b.size
  let mut result : Array Nat := Array.replicate rank 0
  for i in [0:rank] do
    let da := if i < a.size then a[a.size - 1 - i]! else 1
    let db := if i < b.size then b[b.size - 1 - i]! else 1
    if da == db then result := result.set! (rank - 1 - i) da
    else if da == 1 then result := result.set! (rank - 1 - i) db
    else if db == 1 then result := result.set! (rank - 1 - i) da
    else return none
  return some result

structure ConnectionCheck where
  compatible    : Bool
  broadcastExpr : Option Tropical.WireExpr := none
  error         : Option String := none
  resultShape   : Option (Array Nat) := none

private def broadcastTo (refExpr : Tropical.WireExpr) (shape : Array Nat) : Tropical.WireExpr :=
  .broadcastTo refExpr shape

private def floatT : PortType := .scalar .float

/-- Port of compiler/array_wiring.ts `checkArrayConnection` — same
    decision tree, same error strings. -/
def checkArrayConnection (srcIn dstIn : Option PortType) (refExpr : Tropical.WireExpr) : ConnectionCheck := Id.run do
  let src := srcIn.getD floatT
  let dst := dstIn.getD floatT

  if portTypeEqual src dst then
    return { compatible := true }

  match src, dst with
  | .scalar sk, .scalar dk =>
    if sk == dk || widens sk dk then return { compatible := true }
    return { compatible := false,
             error := some s!"Lossy conversion: cannot narrow {portTypeToString src} to {portTypeToString dst} — wrap source in {narrowingHint dk} to narrow explicitly" }
  | .scalar _, .array _ dstShape =>
    return { compatible := true,
             broadcastExpr := some (broadcastTo refExpr dstShape),
             resultShape := some dstShape }
  | .array _ _, .scalar _ =>
    return { compatible := false,
             error := some s!"Cannot connect {portTypeToString src} to {portTypeToString dst} — reduce or index the array first" }
  | .array se srcShape, .array de dstShape =>
    let sk := se.kind.getD .float
    let dk := de.kind.getD .float
    if sk != dk && !widens sk dk then
      return { compatible := false,
               error := some s!"Lossy conversion: cannot narrow {portTypeToString src} to {portTypeToString dst} — wrap source in {narrowingHint dk} to narrow explicitly" }
    match broadcastShapes srcShape dstShape with
    | none =>
      return { compatible := false,
               error := some s!"Shape mismatch: source is {portTypeToString src} but destination expects {portTypeToString dst} (shapes not broadcast-compatible)" }
    | some resultShape =>
      if srcShape == dstShape then
        return { compatible := true, resultShape := some resultShape }
      return { compatible := true,
               broadcastExpr := some (broadcastTo refExpr dstShape),
               resultShape := some resultShape }
  | _, _ =>
    -- Alias / mixed cases: structural equality already failed above.
    return { compatible := false,
             error := some s!"Type mismatch: source is {portTypeToString src} but destination expects {portTypeToString dst}" }

end Tropical.Wiring
