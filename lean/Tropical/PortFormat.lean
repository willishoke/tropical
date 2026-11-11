import Lean.Data.Json
import Tropical.Parse.Nodes

/-!
# PortFormat — the number format of a value at a port

A value that crosses a port (a param a host writes, a program output a host
reads) has a scalar kind and, when the kind is an integer, a fixed-point
format: `fracBits` fraction bits, so the word `w` means `w · 2^-fracBits`.
That is the whole vocabulary. Inside the kernel an integer is an integer:
the IR, its passes and its semantics never see a format. A format is checked
where a value crosses a port without arithmetic, and it is carried into the
plan's `slot_types` and the kernel manifest so a host knows how to read and
write the word.

The labels use the Qm.n notation with the sign bit counted in m, so m + n is
the word width: `Q16.16` and `Q0.32` are 32-bit words, `Q48.16` is τ's
64-bit rail, and a Q1.15 value held in a 32-bit word is `Q17.15`.

Nothing here is tropical-specific: it is the scalar and manifest level, the
part a different front end with the same kernel shape would share.
-/

namespace Tropical

open Lean (Json)

abbrev FormatKind := Tropical.Parse.ScalarKind

/-- A scalar kind plus, for an integer kind, its fraction bits. -/
structure PortFormat where
  kind : FormatKind := .float
  fracBits : Nat := 0
deriving BEq, Repr, Inhabited

namespace PortFormat

/-- The word width of an integer kind; 0 for `float` and `bool`. -/
def wordBits : FormatKind → Nat
  | .int32 => 32
  | .int => 64
  | _ => 0

/-- `float` with no fraction bits: the default, and what every port was
    before ports had formats. -/
def isDefault (f : PortFormat) : Bool := f.kind == .float && f.fracBits == 0

/-- Why a declared format is not one, if it is not: a fraction needs an
    integer kind, and at most the whole word may be fraction (`Q0.32`). -/
def error? (f : PortFormat) : Option String :=
  match f.kind with
  | .int32 | .int =>
    if f.fracBits ≤ wordBits f.kind then none
    else some s!"{f.kind.wire} has {wordBits f.kind} bits; {f.fracBits} fraction bits do not fit"
  | .float | .bool =>
    if f.fracBits == 0 then none
    else some s!"fraction bits need an integer kind (int32 or int), not {f.kind.wire}"

/-- `Q16.16`, `Q0.32`, `Q17.15`; the kind's name for `float` and `bool`. -/
def label (f : PortFormat) : String :=
  match f.kind with
  | .int32 | .int => s!"Q{wordBits f.kind - f.fracBits}.{f.fracBits}"
  | k => k.wire

/-- `int32 Q16.16`, `float`: how a refusal names a format. -/
def describe (f : PortFormat) : String :=
  match f.kind with
  | .int32 | .int => s!"{f.kind.wire} {f.label}"
  | k => k.wire

/-- The format a `type_obj`/declaration JSON object carries: `scalar` (the
    kind, default `float`) and `frac_bits` (default 0). -/
def ofJson? (j : Json) : Except String PortFormat := do
  let kind ← match j.getObjVal? "scalar" with
    | .ok (.str s) =>
      match Tropical.Parse.ScalarKind.ofWire? s with
      | some k => pure k
      | none => throw s!"unknown scalar kind '{s}'"
    | .ok _ => throw "'scalar' must be a string"
    | .error _ => pure .float
  let fracBits ← match j.getObjVal? "frac_bits" with
    | .ok (.num n) =>
      if n.exponent == 0 && n.mantissa ≥ 0 then pure n.mantissa.toNat
      else throw "'frac_bits' must be a non-negative integer"
    | .ok _ => throw "'frac_bits' must be a non-negative integer"
    | .error _ => pure 0
  let f : PortFormat := { kind, fracBits }
  if let some e := f.error? then throw e
  return f

end PortFormat
end Tropical
