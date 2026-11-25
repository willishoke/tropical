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

/-- The words of an integer format: `[-2^(w-1), 2^(w-1))`. -/
def wordRange (f : PortFormat) : Int × Int :=
  let w := wordBits f.kind
  if w == 0 then (0, 0) else (-((2 : Int) ^ (w - 1)), (2 : Int) ^ (w - 1))

/-- `2^k` as an exact decimal (`2^-1` is `0.5`, `2^-16` is
    `0.0000152587890625`): the ends of a fixed-point range, printed exactly. -/
private def pow2Decimal (k : Int) : String :=
  if k ≥ 0 then toString ((2 : Nat) ^ k.toNat)
  else
    let e := (-k).toNat
    let digits := toString ((5 : Nat) ^ e)
    let padded := "".pushn '0' (e - digits.length) ++ digits
    "0." ++ padded

/-- The real values an integer format holds, `[-2^(w-1-n), 2^(w-1-n))`, as a
    label: `[-32768, 32768)` for `Q16.16`, `[-0.5, 0.5)` for `Q0.32`. -/
def rangeLabel (f : PortFormat) : String :=
  let w := wordBits f.kind
  if w == 0 then "" else
    let k : Int := (w : Int) - 1 - f.fracBits
    s!"[-{pow2Decimal k}, {pow2Decimal k})"

/-- Why real value `x` (shown as `shown`) does not fit an integer format, if
    it does not: its word `x · 2^fracBits` must lie in the word's range (it is
    then truncated toward zero, `Kernel.encodeWord`). A value past the range
    would saturate, so it is refused instead, at ingest (`paramDecl.value`)
    and at `set_param` alike. `none` for `float`. -/
def rangeError? (f : PortFormat) (shown : String) (x : Float) : Option String :=
  if !f.kind.isInteger then none else
    let w := wordBits f.kind
    let scaled := x * (2.0 : Float) ^ f.fracBits.toFloat
    let lo := -((2.0 : Float) ^ (w - 1).toFloat)
    let hi := (2.0 : Float) ^ (w - 1).toFloat
    if lo ≤ scaled && scaled < hi then none
    else some s!"value {shown} is outside the range of {f.describe}, {f.rangeLabel}"

/-- The word a bare literal stands for at an integer port of format `f`: the
    literal is the real value, exactly as `paramDecl.value` is, so `440` at a
    Q16.16 port is the word `440 · 2^16`. Exact decimal arithmetic: a literal
    whose word is not an integer, or lies outside the word's range, is
    refused. -/
def encodeLiteral? (f : PortFormat) (n : Lean.JsonNumber) : Except String Int := do
  let num : Int := n.mantissa * (2 : Int) ^ f.fracBits
  let den : Int := (10 : Int) ^ n.exponent
  if num % den != 0 then
    throw s!"literal {n} is not exactly representable in {f.describe} (steps of 2^-{f.fracBits})"
  let w := num / den
  let (lo, hi) := f.wordRange
  unless lo ≤ w && w < hi do
    throw s!"literal {n} is outside the range of {f.describe}, {f.rangeLabel}"
  return w

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

/-- The real interval an input's value is defined on, with open or closed
    ends: `(0, 1]` for a pulse width. Like a format it is checked where a
    literal crosses the port bare (`Session.bareLiteralAtPort`): a literal
    outside it is refused there. A value that moves (a param, another
    instance's output, arithmetic) is the program's to clamp. -/
structure PortInterval where
  lo : Lean.JsonNumber
  loOpen : Bool
  hi : Lean.JsonNumber
  hiOpen : Bool
deriving Repr, Inhabited

namespace PortInterval

/-- `a < b` and `a = b` on exact decimals, by cross-multiplying. -/
private def scaled (a b : Lean.JsonNumber) : Int × Int :=
  (a.mantissa * (10 : Int) ^ b.exponent, b.mantissa * (10 : Int) ^ a.exponent)

def contains (i : PortInterval) (n : Lean.JsonNumber) : Bool :=
  let (x, lo) := scaled n i.lo
  let (x', hi) := scaled n i.hi
  (if i.loOpen then lo < x else lo ≤ x) && (if i.hiOpen then x' < hi else x' ≤ hi)

/-- `(0, 1]`. -/
def label (i : PortInterval) : String :=
  s!"{if i.loOpen then "(" else "["}{i.lo}, {i.hi}{if i.hiOpen then ")" else "]"}"

end PortInterval

/-- A phasor-frequency input's anchoring, declared by the type on the input
    (`design/host-param-dispatch.md`, "Patch-bay params"). The type promises
    that its phase is `Φ(inc, C) + off` with `inc = trunc(f·2³²/SR)` read from
    this input, `off = trunc(φ·2³²)` from the input named `phase`, and `C` the
    Q48.16 clock read from the input named `clock`, so a host can retune the
    frequency without a phase jump by moving `phase` (the `anchor` discipline).
    The session checks a param's declared discipline against it at compile. -/
structure PortAnchor where
  phase : String
  clock : String
deriving Repr, Inhabited, BEq

end Tropical
