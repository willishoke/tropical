import Tropical.Plan

/-!
# ConstFold — the compile-time constant evaluator (the value algebra)

The fourth interpretation of the scalar-op signature `PlanOp`, alongside the
LLVM and MSL text emitters and the type-inference: evaluation into VALUES. Given
constant arguments, `foldOp` computes the constant result with the kernel's
exact semantics — f64 float math, i64 two's-complement wraparound, int-if-either
comparison dispatch, div/mod zero-guards, `fptosi` truncation. It is
PARTIAL by design (`none` = "don't fold": an unfoldable op like Ldexp/
FloatExponent, or a value that refuses like a non-finite float→int) — the caller
then emits the op at runtime.

This is backend-neutral: it produces `CVal` values, not target text (the MSL
renderer `cvalText` that materializes a folded constant stays with the MSL
emitter). It lives apart from any text backend because it IS a distinct algebra
over the signature — the denotation, not a rendering.
-/

namespace Tropical.Ir

open Tropical.Plan (ScalarType PlanOp)

/-- A value known at emit time. Floats carry full f64 precision through
    the fold; they round to f32 only if/when they materialize as kernel
    text. Ints are i64 two's-complement (wrapped after every op). -/
inductive CVal where
  | f (x : Float)
  | i (n : Int)
  | b (v : Bool)
deriving Inhabited

/-- Wrap an arbitrary-precision Int to i64 two's-complement — the fold
    must reproduce the kernel's wraparound exactly. -/
def wrap64 (x : Int) : Int :=
  ((x + 9223372036854775808) % 18446744073709551616) - 9223372036854775808

/-- Two's-complement u64 image of an i64-range Int (for bitwise ops —
    core has no `HAnd Int`). -/
def toU64 (x : Int) : Nat :=
  (((x % 18446744073709551616) + 18446744073709551616) % 18446744073709551616).toNat

/-- i64 bitwise op via the u64 image. -/
def bit64 (f : Nat → Nat → Nat) (a b : Int) : Int :=
  wrap64 (Int.ofNat (f (toU64 a) (toU64 b)))

/-- f64 → i64 truncation toward zero (`fptosi` / MSL `long(f)`). Refuses
    non-finite / out-of-range (the caller then declines to fold). -/
def truncToInt? (v : Float) : Option Int :=
  if v.isNaN || v.isInf || v >= 9.223372036854776e18 || v <= -9.223372036854776e18 then none
  else if v >= 0.0 then some (Int.ofNat v.toUInt64.toNat)
  else some (-(Int.ofNat (-v).toUInt64.toNat))

def CVal.ty : CVal → ScalarType
  | .f _ => .float | .i _ => .int | .b _ => .bool

/-- Coerce a constant between scalar types with the kernel's exact
    semantics (fptosi truncation, sitofp nearest, truthiness). `none`
    means "don't fold" (e.g. non-finite float→int). -/
def CVal.coerce? : CVal → ScalarType → Option CVal
  | v, t =>
    if v.ty == t then some v else
    match v, t with
    | .f x, .int  => (truncToInt? x).map .i
    | .f x, .bool => some (.b (x != 0.0))
    | .i n, .float => some (.f (Float.ofInt n))
    | .i n, .bool => some (.b (n != 0))
    | .b v, .float => some (.f (if v then 1.0 else 0.0))
    | .b v, .int  => some (.i (if v then 1 else 0))
    | v, _ => some v

/-- Fold one op over constant args, mirroring `emitOp`'s runtime
    semantics exactly: float math in f64 (what the CPU kernel computes),
    int math wrapped to i64, comparisons dispatched int-if-either-int,
    div/mod zero-guards, fptosi truncation. `none` = "don't fold" —
    either the tag is unfoldable (Ldexp/FloatExponent bit tricks stay
    runtime) or a value refuses (non-finite → int). -/
def foldOp (tag : String) (resultType : ScalarType) (args : Array CVal) : Option CVal := do
  let aF (i : Nat) : Option Float := do
    match (← args[i]?).coerce? .float with | some (.f x) => some x | _ => none
  let aI (i : Nat) : Option Int := do
    match (← args[i]?).coerce? .int with | some (.i n) => some n | _ => none
  let aB (i : Nat) : Option Bool := do
    match (← args[i]?).coerce? .bool with | some (.b v) => some v | _ => none
  let isInt := resultType == .int
  let eitherInt :=
    (((args[0]?).map (·.ty == .int)).getD false ||
     ((args[1]?).map (·.ty == .int)).getD false)
  -- int compare path coerces BOTH to int (mirrors argAs .int)
  let cmpInt (p : Int → Int → Bool) : Option CVal := do pure (.b (p (← aI 0) (← aI 1)))
  let cmpFlt (p : Float → Float → Bool) : Option CVal := do pure (.b (p (← aF 0) (← aF 1)))
  let some op := Tropical.Plan.PlanOp.ofString? tag | none
  match op with
  | .add =>
    if isInt then do let a ← aI 0; let b ← aI 1; pure (.i (wrap64 (a + b)))
    else do let a ← aF 0; let b ← aF 1; pure (.f (a + b))
  | .sub =>
    if isInt then do let a ← aI 0; let b ← aI 1; pure (.i (wrap64 (a - b)))
    else do let a ← aF 0; let b ← aF 1; pure (.f (a - b))
  | .mul =>
    if isInt then do let a ← aI 0; let b ← aI 1; pure (.i (wrap64 (a * b)))
    else do let a ← aF 0; let b ← aF 1; pure (.f (a * b))
  | .div =>
    if isInt then do
      let a ← aI 0; let b ← aI 1
      pure (.i (if b == 0 then 0 else wrap64 (Int.tdiv a b)))
    else do
      let a ← aF 0; let b ← aF 1
      pure (.f (if b == 0.0 then 0.0 else a / b))
  | .mod =>
    if isInt then do
      let a ← aI 0; let b ← aI 1
      pure (.i (if b == 0 then 0 else wrap64 (Int.tmod a b)))
    else do
      let a ← aF 0; let b ← aF 1
      pure (.f (if b == 0.0 then 0.0 else a - (a / b).floor * b))
  | .floorDiv =>
    if isInt then do
      let a ← aI 0; let b ← aI 1
      pure (.i (if b == 0 then 0 else wrap64 (Int.tdiv a b)))
    else do
      let a ← aF 0; let b ← aF 1
      pure (.f (if b == 0.0 then 0.0 else a / b).floor)
  | .less      => if eitherInt then cmpInt (fun a b => decide (a < b)) else cmpFlt (fun a b => decide (a < b))
  | .lessEq    => if eitherInt then cmpInt (fun a b => decide (a ≤ b)) else cmpFlt (fun a b => decide (a ≤ b))
  | .greater   => if eitherInt then cmpInt (fun a b => decide (a > b)) else cmpFlt (fun a b => decide (a > b))
  | .greaterEq => if eitherInt then cmpInt (fun a b => decide (a ≥ b)) else cmpFlt (fun a b => decide (a ≥ b))
  | .equal     => if eitherInt then cmpInt (fun a b => a == b) else cmpFlt (fun a b => a == b)
  | .notEqual  => if eitherInt then cmpInt (fun a b => a != b) else cmpFlt (fun a b => a != b)
  | .and => do let a ← aB 0; let b ← aB 1; pure (.b (a && b))
  | .or  => do let a ← aB 0; let b ← aB 1; pure (.b (a || b))
  | .bitAnd => do let a ← aI 0; let b ← aI 1; pure (.i (bit64 (· &&& ·) a b))
  | .bitOr  => do let a ← aI 0; let b ← aI 1; pure (.i (bit64 (· ||| ·) a b))
  | .bitXor => do let a ← aI 0; let b ← aI 1; pure (.i (bit64 (· ^^^ ·) a b))
  | .lshift => do
    let a ← aI 0; let sh ← aI 1
    if sh < 0 || sh > 63 then none else pure (.i (wrap64 (a <<< sh.toNat)))
  | .rshift => do
    let a ← aI 0; let sh ← aI 1
    if sh < 0 || sh > 63 then none else pure (.i (a >>> sh.toNat))  -- Int >>> is arithmetic (floor)
  | .neg =>
    if isInt then do let v ← aI 0; pure (.i (wrap64 (-v)))
    else do let v ← aF 0; pure (.f (-v))
  | .abs =>
    if isInt then do
      let v ← aI 0; pure (.i (if v < 0 then wrap64 (-v) else v))
    else do let v ← aF 0; pure (.f v.abs)
  | .sqrt  => do let v ← aF 0; pure (.f v.sqrt)
  | .floor => do let v ← aF 0; pure (.f v.floor)
  | .ceil  => do let v ← aF 0; pure (.f v.ceil)
  | .round => do let v ← aF 0; pure (.f v.round)   -- half-away, = llvm.round = metal::round
  | .not => do
    match ← args[0]? with
    | .b v => pure (.b (!v))
    | .i n => pure (.b (n == 0))
    | .f x => pure (.b (x == 0.0))
  | .bitNot => do let v ← aI 0; pure (.i (bit64 (· ^^^ ·) v (-1)))
  | .toInt   => do (← args[0]?).coerce? .int
  | .toBool  => do (← args[0]?).coerce? .bool
  | .toFloat => do (← args[0]?).coerce? .float
  | .clamp =>
    if isInt then do
      let v ← aI 0; let lo ← aI 1; let hi ← aI 2
      let lc := if v > lo then v else lo
      pure (.i (if lc < hi then lc else hi))
    else do
      let v ← aF 0; let lo ← aF 1; let hi ← aF 2
      let lc := if v > lo then v else lo
      pure (.f (if lc < hi then lc else hi))
  | .select => do
    let cond ← match ← args[0]? with
      | .b v => some v
      | .i n => some (n != 0)
      | .f x => some (x != 0.0)
    let v ← if cond then args[1]? else args[2]?
    v.coerce? resultType
  | _ => none  -- Ldexp / FloatExponent / anything exotic: stay runtime

end Tropical.Ir
