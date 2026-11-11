import Tropical.Plan

/-!
# ConstFold — the compile-time constant evaluator (the value algebra)

The fourth interpretation of the scalar-op signature `PlanOp`, alongside the
LLVM and MSL text emitters and the type-inference: evaluation into VALUES. Given
constant arguments, `foldOp` computes the constant result with the kernel's
exact semantics — f64 float math, i64 and i32 two's-complement wraparound,
int-if-either comparison dispatch, and the TOTAL integer semantics every
backend implements: division and remainder by zero are 0, `MIN / -1` wraps to
`MIN` (and `MIN % -1` is 0), a shift amount is taken modulo the word width, and
a float→int conversion saturates (NaN → 0). Every integer op therefore folds on
every input. `none` remains only for an unfoldable tag (the Ldexp/
FloatExponent bit tricks stay runtime) or malformed args — the caller then
emits the op at runtime.

This is backend-neutral: it produces `CVal` values, not target text (the MSL
renderer `cvalText` that materializes a folded constant stays with the MSL
emitter). It lives apart from any text backend because it IS a distinct algebra
over the signature — the denotation, not a rendering.
-/

namespace Tropical.Ir

open Tropical.Plan (ScalarType PlanOp)

/-- A value known at emit time. Floats carry full f64 precision through
    the fold; they round to f32 only if/when they materialize as kernel
    text. `i` is an `int` (i64 two's-complement, wrapped after every op),
    `i32` an `int32` (i32 two's-complement, its value in `[-2³¹, 2³¹)`). -/
inductive CVal where
  | f (x : Float)
  | i (n : Int)
  | b (v : Bool)
  | i32 (n : Int)
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

/-- Wrap an arbitrary-precision Int to i32 two's-complement: the canonical
    representative of its residue modulo 2³² in `[-2³¹, 2³¹)`. -/
def wrap32 (x : Int) : Int :=
  ((x + 2147483648) % 4294967296) - 2147483648

/-- Two's-complement u32 image of an Int (its least residue modulo 2³²). -/
def toU32 (x : Int) : Nat :=
  (((x % 4294967296) + 4294967296) % 4294967296).toNat

/-- i32 bitwise op via the u32 image. -/
def bit32 (f : Nat → Nat → Nat) (a b : Int) : Int :=
  wrap32 (Int.ofNat (f (toU32 a) (toU32 b)))

/-- Clamp an Int into `[lo, hi]`. -/
def clampInt (lo hi x : Int) : Int := max lo (min hi x)

/-- f64 → i64, saturating: truncation toward zero, clamped to
    `[-2⁶³, 2⁶³ − 1]`, NaN → 0. This is Rust's `as` and wasm's
    `i64.trunc_sat_f64_s`; the backends spell it `llvm.fptosi.sat.i64.f64`
    and an explicit clamp in MSL. The final clamp is a no-op on the
    truncated branch (|v| < 2⁶³ there); it makes the range a theorem
    (`ConstFoldLaws.truncSatToInt_isI64`) without reasoning about `Float`. -/
def truncSatToInt (v : Float) : Int :=
  clampInt (-9223372036854775808) 9223372036854775807 <|
    if v.isNaN then 0
    else if v >= 9.223372036854775808e18 then 9223372036854775807
    else if v <= -9.223372036854775808e18 then -9223372036854775808
    else if v >= 0.0 then Int.ofNat v.toUInt64.toNat
    else -(Int.ofNat (-v).toUInt64.toNat)

/-- f64 → i32, saturating (`llvm.fptosi.sat.i32.f64`, wasm
    `i32.trunc_sat_f64_s`): the i64 saturation clamped again to
    `[-2³¹, 2³¹ − 1]`. Truncation is monotone, so saturating twice is
    saturating once at the narrower width. -/
def truncSatToInt32 (v : Float) : Int :=
  clampInt (-2147483648) 2147483647 (truncSatToInt v)

def CVal.ty : CVal → ScalarType
  | .f _ => .float | .i _ => .int | .b _ => .bool | .i32 _ => .int32

/-- Coerce a constant between scalar types with the kernel's exact
    semantics (saturating float → int truncation, sitofp nearest,
    truthiness, `int → int32` keeps the low 32 bits, `int32 → int`
    sign-extends). Total: every pair converts (the `Option` is the
    signature `foldOp`'s argument readers share). -/
def CVal.coerce? : CVal → ScalarType → Option CVal
  | v, t =>
    if v.ty == t then some v else
    match v, t with
    | .f x, .int  => some (.i (truncSatToInt x))
    | .f x, .int32 => some (.i32 (truncSatToInt32 x))
    | .f x, .bool => some (.b (x != 0.0))
    | .i n, .float => some (.f (Float.ofInt n))
    | .i n, .int32 => some (.i32 (wrap32 n))
    | .i n, .bool => some (.b (n != 0))
    | .i32 n, .float => some (.f (Float.ofInt n))
    | .i32 n, .int => some (.i n)
    | .i32 n, .bool => some (.b (n != 0))
    | .b v, .float => some (.f (if v then 1.0 else 0.0))
    | .b v, .int  => some (.i (if v then 1 else 0))
    | .b v, .int32 => some (.i32 (if v then 1 else 0))
    | v, _ => some v

/-- Fold one op over constant args, mirroring `emitOp`'s runtime
    semantics exactly: float math in f64 (what the CPU kernel computes),
    int math wrapped to i64 and int32 math to i32, comparisons dispatched
    int-if-either-integer, and the total integer edges (a zero divisor
    gives 0, `MIN / -1` wraps to `MIN`, a shift amount is reduced modulo
    the width, float → int saturates). `none` = "don't fold": only an
    unfoldable tag (the Ldexp/FloatExponent bit tricks stay runtime) or
    missing args. -/
def foldOp (tag : String) (resultType : ScalarType) (args : Array CVal) : Option CVal := do
  let aF (i : Nat) : Option Float := do
    match (← args[i]?).coerce? .float with | some (.f x) => some x | _ => none
  let aI (i : Nat) : Option Int := do
    match (← args[i]?).coerce? .int with | some (.i n) => some n | _ => none
  let aI32 (i : Nat) : Option Int := do
    match (← args[i]?).coerce? .int32 with | some (.i32 n) => some n | _ => none
  let aB (i : Nat) : Option Bool := do
    match (← args[i]?).coerce? .bool with | some (.b v) => some v | _ => none
  let isInt := resultType == .int
  let isI32 := resultType == .int32
  let isIntegerTy (v : CVal) : Bool := v.ty == .int || v.ty == .int32
  let eitherInt :=
    (((args[0]?).map isIntegerTy).getD false ||
     ((args[1]?).map isIntegerTy).getD false)
  -- int compare path coerces BOTH to int (mirrors argAs .int); an int32
  -- operand sign-extends, so its order is its value's
  let cmpInt (p : Int → Int → Bool) : Option CVal := do pure (.b (p (← aI 0) (← aI 1)))
  let cmpFlt (p : Float → Float → Bool) : Option CVal := do pure (.b (p (← aF 0) (← aF 1)))
  -- an i32 quotient or remainder: 0 for a zero divisor, and otherwise the
  -- truncated quotient wrapped to the word, so `INT32_MIN / -1` is
  -- `INT32_MIN` (2³¹ wraps) and `INT32_MIN % -1` is 0 — the i64 rule.
  let quot32 (q : Int → Int → Int) : Option CVal := do
    let a ← aI32 0; let b ← aI32 1
    pure (.i32 (if b == 0 then 0 else wrap32 (q a b)))
  let some op := Tropical.Plan.PlanOp.ofString? tag | none
  match op with
  | .add =>
    if isI32 then do let a ← aI32 0; let b ← aI32 1; pure (.i32 (wrap32 (a + b)))
    else if isInt then do let a ← aI 0; let b ← aI 1; pure (.i (wrap64 (a + b)))
    else do let a ← aF 0; let b ← aF 1; pure (.f (a + b))
  | .sub =>
    if isI32 then do let a ← aI32 0; let b ← aI32 1; pure (.i32 (wrap32 (a - b)))
    else if isInt then do let a ← aI 0; let b ← aI 1; pure (.i (wrap64 (a - b)))
    else do let a ← aF 0; let b ← aF 1; pure (.f (a - b))
  | .mul =>
    if isI32 then do let a ← aI32 0; let b ← aI32 1; pure (.i32 (wrap32 (a * b)))
    else if isInt then do let a ← aI 0; let b ← aI 1; pure (.i (wrap64 (a * b)))
    else do let a ← aF 0; let b ← aF 1; pure (.f (a * b))
  | .div =>
    if isI32 then quot32 Int.tdiv
    else if isInt then do
      let a ← aI 0; let b ← aI 1
      pure (.i (if b == 0 then 0 else wrap64 (Int.tdiv a b)))
    else do
      let a ← aF 0; let b ← aF 1
      pure (.f (if b == 0.0 then 0.0 else a / b))
  | .mod =>
    if isI32 then quot32 Int.tmod
    else if isInt then do
      let a ← aI 0; let b ← aI 1
      pure (.i (if b == 0 then 0 else wrap64 (Int.tmod a b)))
    else do
      let a ← aF 0; let b ← aF 1
      pure (.f (if b == 0.0 then 0.0 else a - (a / b).floor * b))
  | .floorDiv =>
    if isI32 then quot32 Int.tdiv
    else if isInt then do
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
  | .bitAnd =>
    if isI32 then do let a ← aI32 0; let b ← aI32 1; pure (.i32 (bit32 (· &&& ·) a b))
    else do let a ← aI 0; let b ← aI 1; pure (.i (bit64 (· &&& ·) a b))
  | .bitOr =>
    if isI32 then do let a ← aI32 0; let b ← aI32 1; pure (.i32 (bit32 (· ||| ·) a b))
    else do let a ← aI 0; let b ← aI 1; pure (.i (bit64 (· ||| ·) a b))
  | .bitXor =>
    if isI32 then do let a ← aI32 0; let b ← aI32 1; pure (.i32 (bit32 (· ^^^ ·) a b))
    else do let a ← aI 0; let b ← aI 1; pure (.i (bit64 (· ^^^ ·) a b))
  -- the shift amount is taken modulo the width (`%` on Int is the least
  -- nonnegative residue, so -1 shifts by 31 or 63), as wasm and Java do
  | .lshift =>
    if isI32 then do
      let a ← aI32 0; let sh ← aI32 1
      pure (.i32 (wrap32 (a <<< (sh % 32).toNat)))
    else do
      let a ← aI 0; let sh ← aI 1
      pure (.i (wrap64 (a <<< (sh % 64).toNat)))
  | .rshift =>
    if isI32 then do
      let a ← aI32 0; let sh ← aI32 1
      pure (.i32 (a >>> (sh % 32).toNat))
    else do
      let a ← aI 0; let sh ← aI 1
      pure (.i (a >>> (sh % 64).toNat))  -- Int >>> is arithmetic (floor)
  | .neg =>
    if isI32 then do let v ← aI32 0; pure (.i32 (wrap32 (-v)))
    else if isInt then do let v ← aI 0; pure (.i (wrap64 (-v)))
    else do let v ← aF 0; pure (.f (-v))
  | .abs =>
    if isI32 then do
      let v ← aI32 0; pure (.i32 (if v < 0 then wrap32 (-v) else v))
    else if isInt then do
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
    | .i32 n => pure (.b (n == 0))
    | .f x => pure (.b (x == 0.0))
  | .bitNot =>
    if isI32 then do let v ← aI32 0; pure (.i32 (bit32 (· ^^^ ·) v (-1)))
    else do let v ← aI 0; pure (.i (bit64 (· ^^^ ·) v (-1)))
  | .toInt   => do (← args[0]?).coerce? .int
  | .toInt32 => do (← args[0]?).coerce? .int32
  | .toBool  => do (← args[0]?).coerce? .bool
  | .toFloat => do (← args[0]?).coerce? .float
  | .clamp =>
    if isI32 then do
      let v ← aI32 0; let lo ← aI32 1; let hi ← aI32 2
      let lc := if v > lo then v else lo
      pure (.i32 (if lc < hi then lc else hi))
    else if isInt then do
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
      | .i32 n => some (n != 0)
      | .f x => some (x != 0.0)
    let v ← if cond then args[1]? else args[2]?
    v.coerce? resultType
  | _ => none  -- Ldexp / FloatExponent / anything exotic: stay runtime

end Tropical.Ir
