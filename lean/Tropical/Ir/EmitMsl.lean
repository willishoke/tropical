import Std.Data.HashMap
import Std.Data.HashSet
import Lean.Data.Json
import Tropical.Plan

/-!
# EmitMsl — `FlatPlan → Metal Shading Language source` (the Metal backend)

The MSL sibling of `EmitLlvm`: the same fractal walk over the same
`tropical_plan_5`, emitting a compute kernel instead of a CPU loop. The
per-sample loop collapses — one thread per sample
(`thread_position_in_grid`), legal because kernels are closed-form
`f(τ, params)` with no per-sample state.

## Kernel ABI (must match `engine/metal/MetalKernel.mm`)

```
kernel void tropical_kernel(
    device float*                  output_buffer [[buffer(0)]],
    constant float*                slots         [[buffer(1)]],
    constant TropicalKernelConsts& k             [[buffer(2)]],
    uint s [[thread_position_in_grid]])
```

with `TropicalKernelConsts = { ulong start_sample_index; float
sample_rate; uint buffer_length; }` (8+4+4, no padding).

## Divergences from EmitLlvm, all deliberate

* **f32 values, i64 clocks.** `.float` is `float` (Apple GPUs have no
  f64); `.int` is native `long` — the Q32.32/Q0.32/Q2.30 integer
  datapath is bit-identical to the JIT (design/fixed-carrier.md). The
  f32 surface is the enumerated residue (envelope exp, param landings,
  the DAC scale), tolerance-gated by `metal_vs_jit`.
* **Thread-private TYPED slots.** Module slots are same-sample dataflow
  only (producers topo-sorted before consumers; CF-only — nothing reads
  a previous sample's slot). Slots the kernel WRITES become thread-
  private locals at their NATIVE type — the Q32.32 clock crosses a wire
  as a `long`, never an f32 (an f32 slot would shed sub-sample bits
  immediately and whole samples past ~2²⁴). Reads before any write hit
  the host snapshot in the `constant` buffer (host-written param slots
  always do). Only `output_buffer[s]` is written per-thread — no
  cross-thread races by construction. Constants fold THROUGH kernel-
  written slots (safe: the CPU kernel rewrites them every sample, so
  host writes there are clobbered on CPU too — a folded slot cannot be
  a live knob).
* **Typed locals, no i64 temp pool.** The bitcast pool exists so one C
  array can back three types; MSL locals are just typed variables
  `tf{n}`/`ti{n}`/`tb{n}` with the same last-write-type discipline
  (`tempTypes`).
* **Emit-time f64 constant folding.** Instruction chains whose inputs
  are compile-time constants (literals, and `rate` = the plan's
  sampleRate; `tick` never) are evaluated HERE in f64/Int — mirroring
  `EmitLlvm`'s runtime semantics op for op (Int ops wrap to i64
  two's-complement) — and land as single literals. This is what makes
  literal-frequency patches carry byte-exact i64 phase increments onto
  the GPU instead of f32-rounded ones. Slot-driven landings still
  compute in f32 in-kernel (tolerance-gated). The LLVM path is
  deliberately untouched — it stays byte-frozen.
* **f32 literals are exact bits** (`as_type<float>(0x…u)`), the same
  philosophy as `f64Lit`'s hex bits: never trust a decimal round-trip.

Compile with `MTLMathModeSafe` (no fast-math, no fma contraction) so
the f32 arithmetic is IEEE-ordered and the SNR gates are stable.
-/

namespace Tropical.Ir.EmitMsl

open Lean (Json JsonNumber)
open Tropical.Plan

abbrev ScalarType := Tropical.Plan.ScalarType

-- ─────────────────────────────────────────────────────────────
-- Formatting helpers
-- ─────────────────────────────────────────────────────────────

/-- 8-digit zero-padded hex of a UInt32. -/
def hex8 (u : UInt32) : String :=
  let s := String.ofList (Nat.toDigits 16 u.toNat)
  String.ofList (List.replicate (8 - s.length) '0') ++ s

/-- An f32 MSL literal as raw IEEE-754 bits — the f64 value rounds to f32
    exactly once, here, and the text reproduces those bits exactly. -/
def f32Lit (x : Float) : String :=
  s!"as_type<float>(0x{hex8 x.toFloat32.toBits}u)"

/-- An i64 MSL literal. `long` in MSL; INT64_MIN needs the two's-complement
    dodge (the bare literal would overflow before negation). -/
def i64Lit (n : Int) : String :=
  if n == -9223372036854775808 then "(-9223372036854775807L - 1L)"
  else s!"{n}L"

/-- Integer value of a JsonNumber const (mirrors EmitLlvm.jnToInt). -/
def jnToInt (n : JsonNumber) : Int :=
  let rec divPow (m : Int) : Nat → Int
    | 0 => m
    | k+1 => divPow (m / 10) k
  divPow n.mantissa n.exponent

def mslTy : ScalarType → String
  | .float => "float" | .int => "long" | .bool => "bool"

-- ─────────────────────────────────────────────────────────────
-- Compile-time constant values (the fold domain)
-- ─────────────────────────────────────────────────────────────

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

/-- Materialize a constant as MSL text (floats round to f32 HERE). -/
def CVal.text : CVal → String
  | .f x => f32Lit x
  | .i n => i64Lit n
  | .b v => if v then "true" else "false"

-- ─────────────────────────────────────────────────────────────
-- Emitter state + monad
-- ─────────────────────────────────────────────────────────────

/-- A typed value: MSL operand text, scalar type, and — when known at
    emit time — its constant value (the fold rides along in `cv`). -/
structure TVal where
  ref : String
  ty  : ScalarType
  cv  : Option CVal := none
deriving Inhabited

/-- An open `ReduceBegin`/`ReduceEnd` region: the accumulator temp (a
    mutable typed local declared before the loop), the loop-index
    variable, and the temp-state snapshots to restore at close (body
    locals are scoped to the loop braces in MSL, so their names must
    not survive as "declared"). -/
structure ReduceCtx where
  accTemp : Nat
  accTy : ScalarType
  idxVar : String
  savedTempTypes : Std.HashMap Nat ScalarType
  savedConstTemps : Std.HashMap Nat CVal
  savedDeclared : Std.HashSet String

structure St where
  next : Nat := 0
  lines : Array String := #[]
  indent : Nat := 1
  /-- Temp slot → scalar type last written (same discipline as EmitLlvm). -/
  tempTypes : Std.HashMap Nat ScalarType := {}
  /-- Temp slot → constant value, when the last write was folded. Killed
      by any non-const write to the slot. -/
  constTemps : Std.HashMap Nat CVal := {}
  /-- Declared local variable names (`tf3`, `ti7`, …) — first write
      declares, later writes assign. -/
  declared : Std.HashSet String := {}
  sources : Array SourceKind := #[]
  /-- The plan's sample rate — a per-plan CONSTANT (the manifest seeds the
      runtime with it), so `rate` source reads carry it as a fold value:
      param landings `toInt(freq·2³²/SR)` collapse to exact i64 literals. -/
  sampleRate : Float := 44100.0
  /-- Module slot → scalar type last written THIS kernel (thread-private
      typed locals `slf{i}`/`sli{i}`/`slb{i}` — kernel-written slots are
      same-sample dataflow, so they keep their NATIVE type: the Q32.32
      clock rides a `long` local through a wire, never an f32). Absent =
      not written yet → reads hit the host snapshot `slots[i]`. -/
  slotTypes : Std.HashMap Nat ScalarType := {}
  /-- Module slot → constant value when the last kernel write was folded.
      Safe: a kernel-written slot is rewritten EVERY sample on the CPU,
      so host writes to it are clobbered there too — folding cannot kill
      a live knob (param slots are host-written only, never folded). -/
  constSlots : Std.HashMap Nat CVal := {}
  /-- Hoisted coefficient columns (banks-as-data): array slot → element
      offset into the packed `coeff_columns` device buffer (`buffer(3)`).
      These slots are filled HOST-SIDE by the stage-0 coefficient kernel
      (generation-buffered; `process()` uploads the captured generation
      per dispatch, f64→f32 like slots), so the kernel reads them from
      the binding instead of declaring thread-private locals it could
      never fill. Offsets are compile-time constants — plan order of
      `coeff_array_slots`, the SAME order the runtime packs the upload
      staging. Empty for ordinary plans (the byte-frozen 3-binding ABI). -/
  coeffOffsets : Std.HashMap Nat Nat := {}
  /-- Innermost open reduction region (v1: nesting rejected). -/
  reduce? : Option ReduceCtx := none

abbrev M := EStateM String St

def fresh : M String := do
  let s ← get
  set { s with next := s.next + 1 }
  pure s!"t{s.next}"

def line (l : String) : M Unit :=
  modify fun s => { s with lines := s.lines.push (String.ofList (List.replicate (s.indent * 4) ' ') ++ l) }

def pushIndent : M Unit := modify fun s => { s with indent := s.indent + 1 }
def popIndent : M Unit := modify fun s => { s with indent := s.indent - 1 }

def fail {α} (msg : String) : M α := throw msg

/-- Bind an expression to a fresh immutable local of the given type. -/
def bindVal (ty : ScalarType) (expr : String) : M TVal := do
  let r ← fresh
  line s!"const {mslTy ty} {r} = {expr};"
  pure ⟨r, ty, none⟩

-- ─────────────────────────────────────────────────────────────
-- Coercion (mirrors EmitLlvm.coerce; conversions are MSL casts)
-- ─────────────────────────────────────────────────────────────

def coerce (v : TVal) (to : ScalarType) : M TVal := do
  if v.ty == to then pure v else
  -- constant? coerce at emit time (keeps the fold alive through casts).
  match v.cv.bind (·.coerce? to) with
  | some c => pure ⟨c.text, to, some c⟩
  | none =>
    match v.ty, to with
    | .float, .int  => bindVal .int s!"long({v.ref})"
    | .float, .bool => bindVal .bool s!"({v.ref} != 0.0f)"
    | .int,   .float => bindVal .float s!"float({v.ref})"
    | .int,   .bool => bindVal .bool s!"({v.ref} != 0L)"
    | .bool,  .float => bindVal .float s!"float({v.ref})"
    | .bool,  .int  => bindVal .int s!"long({v.ref})"
    | _, _ => pure v

def coerceF32 (v : TVal) : M String := do pure (← coerce v .float).ref

-- ─────────────────────────────────────────────────────────────
-- Temps (typed locals) and slots (thread-private when written)
-- ─────────────────────────────────────────────────────────────

def tempVarName (slot : Nat) (ty : ScalarType) : String :=
  match ty with
  | .float => s!"tf{slot}" | .int => s!"ti{slot}" | .bool => s!"tb{slot}"

def loadTempTyped (slot : Nat) (ty : ScalarType) : M TVal := do
  match (← get).constTemps.get? slot with
  | some c => pure ⟨c.text, c.ty, some c⟩
  | none => pure ⟨tempVarName slot ty, ty, none⟩

/-- Store a typed value into temp `slot`: declare-or-assign the typed
    local, record the type, kill any stale const. Constant stores don't
    emit at all — they just record the fold — EXCEPT the accumulator of
    an open reduction region, which must always materialize (its value
    varies across iterations even when one write folds). -/
def storeTempTyped (slot : Nat) (v : TVal) : M Unit := do
  if let some rc := (← get).reduce? then
    if slot == rc.accTemp then
      let cv ← coerce v rc.accTy
      line s!"{tempVarName slot rc.accTy} = {cv.ref};"
      modify fun s => { s with constTemps := s.constTemps.erase slot }
      return
  match v.cv with
  | some c =>
    modify fun s => { s with
      tempTypes := s.tempTypes.insert slot v.ty
      constTemps := s.constTemps.insert slot c }
  | none =>
    let name := tempVarName slot v.ty
    if (← get).declared.contains name then
      line s!"{name} = {v.ref};"
    else
      line s!"{mslTy v.ty} {name} = {v.ref};"
      modify fun s => { s with declared := s.declared.insert name }
    modify fun s => { s with
      tempTypes := s.tempTypes.insert slot v.ty
      constTemps := s.constTemps.erase slot }

def slotVarName (idx : Nat) (ty : ScalarType) : String :=
  match ty with
  | .float => s!"slf{idx}" | .int => s!"sli{idx}" | .bool => s!"slb{idx}"

/-- A slot read, as a typed TVal: constant if the last write folded;
    the typed thread-private local if written earlier in this kernel;
    else the host snapshot (float). -/
def loadSlot (idx : Nat) : M TVal := do
  match (← get).constSlots.get? idx with
  | some c => pure ⟨c.text, c.ty, some c⟩
  | none =>
    match (← get).slotTypes.get? idx with
    | some ty => pure ⟨slotVarName idx ty, ty, none⟩
    | none => pure ⟨s!"slots[{idx}]", .float, none⟩

/-- Store a typed value to a module slot: declare-or-assign the typed
    local at the value's NATIVE type (no f32 detour — this is what keeps
    integer clocks exact through wires), track the const fold. -/
def storeSlot (idx : Nat) (v : TVal) : M Unit := do
  match v.cv with
  | some c =>
    modify fun st => { st with
      slotTypes := st.slotTypes.insert idx v.ty
      constSlots := st.constSlots.insert idx c }
  | none =>
    let name := slotVarName idx v.ty
    if (← get).declared.contains name then
      line s!"{name} = {v.ref};"
    else
      line s!"{mslTy v.ty} {name} = {v.ref};"
      modify fun st => { st with declared := st.declared.insert name }
    modify fun st => { st with
      slotTypes := st.slotTypes.insert idx v.ty
      constSlots := st.constSlots.erase idx }

-- ─────────────────────────────────────────────────────────────
-- Operand resolution (mirrors EmitLlvm.resolveOperand)
-- ─────────────────────────────────────────────────────────────

def resolveOperand : NOperand → M TVal
  | .const v t => match t with
    | .int  => let n := jnToInt v; pure ⟨i64Lit n, .int, some (.i (wrap64 n))⟩
    | .bool => let b := jnToInt v != 0
               pure ⟨if b then "true" else "false", .bool, some (.b b)⟩
    | .float => let x := v.toFloat; pure ⟨f32Lit x, .float, some (.f x)⟩
  | .reg slot _ => do
    let ty := (← get).tempTypes.getD slot .float
    loadTempTyped slot ty
  | .source idx _ => do
    let srcs := (← get).sources
    match srcs[idx]? with
    | some .tick => pure ⟨"current_idx", .int, none⟩
    | some .rate => do
      let sr := (← get).sampleRate
      pure ⟨"k.sample_rate", .float, some (.f sr)⟩
    | none => pure ⟨f32Lit 0.0, .float, some (.f 0.0)⟩
  | .slot idx t => do
    coerce (← loadSlot idx) t
  | .input _ _ => fail "EmitMsl: 'input' operand — the runtime passes no inputs (embedded in expressions)"
  | .param _ _ => fail "EmitMsl: 'param' operand is dead on the session path (use slots)"
  | .arrayReg _ => fail "EmitMsl: bare 'arrayReg' operand outside an array op"
  | .sessionArrayReg _ => fail "EmitMsl: 'sessionArrayReg' must be remapped before emit"
  | .loopIdx => do
    let some rc := (← get).reduce?
      | fail "EmitMsl: loop_idx operand outside a ReduceBegin/ReduceEnd region"
    pure ⟨rc.idxVar, .int, none⟩

-- ─────────────────────────────────────────────────────────────
-- The constant fold — f64/i64 evaluation mirroring emitOp
-- ─────────────────────────────────────────────────────────────

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
  match tag with
  | "Add" =>
    if isInt then do let a ← aI 0; let b ← aI 1; pure (.i (wrap64 (a + b)))
    else do let a ← aF 0; let b ← aF 1; pure (.f (a + b))
  | "Sub" =>
    if isInt then do let a ← aI 0; let b ← aI 1; pure (.i (wrap64 (a - b)))
    else do let a ← aF 0; let b ← aF 1; pure (.f (a - b))
  | "Mul" =>
    if isInt then do let a ← aI 0; let b ← aI 1; pure (.i (wrap64 (a * b)))
    else do let a ← aF 0; let b ← aF 1; pure (.f (a * b))
  | "Div" =>
    if isInt then do
      let a ← aI 0; let b ← aI 1
      pure (.i (if b == 0 then 0 else wrap64 (Int.tdiv a b)))
    else do
      let a ← aF 0; let b ← aF 1
      pure (.f (if b == 0.0 then 0.0 else a / b))
  | "Mod" =>
    if isInt then do
      let a ← aI 0; let b ← aI 1
      pure (.i (if b == 0 then 0 else wrap64 (Int.tmod a b)))
    else do
      let a ← aF 0; let b ← aF 1
      pure (.f (if b == 0.0 then 0.0 else a - (a / b).floor * b))
  | "FloorDiv" =>
    if isInt then do
      let a ← aI 0; let b ← aI 1
      pure (.i (if b == 0 then 0 else wrap64 (Int.tdiv a b)))
    else do
      let a ← aF 0; let b ← aF 1
      pure (.f (if b == 0.0 then 0.0 else a / b).floor)
  | "Less"      => if eitherInt then cmpInt (fun a b => decide (a < b)) else cmpFlt (fun a b => decide (a < b))
  | "LessEq"    => if eitherInt then cmpInt (fun a b => decide (a ≤ b)) else cmpFlt (fun a b => decide (a ≤ b))
  | "Greater"   => if eitherInt then cmpInt (fun a b => decide (a > b)) else cmpFlt (fun a b => decide (a > b))
  | "GreaterEq" => if eitherInt then cmpInt (fun a b => decide (a ≥ b)) else cmpFlt (fun a b => decide (a ≥ b))
  | "Equal"     => if eitherInt then cmpInt (fun a b => a == b) else cmpFlt (fun a b => a == b)
  | "NotEqual"  => if eitherInt then cmpInt (fun a b => a != b) else cmpFlt (fun a b => a != b)
  | "And" => do let a ← aB 0; let b ← aB 1; pure (.b (a && b))
  | "Or"  => do let a ← aB 0; let b ← aB 1; pure (.b (a || b))
  | "BitAnd" => do let a ← aI 0; let b ← aI 1; pure (.i (bit64 (· &&& ·) a b))
  | "BitOr"  => do let a ← aI 0; let b ← aI 1; pure (.i (bit64 (· ||| ·) a b))
  | "BitXor" => do let a ← aI 0; let b ← aI 1; pure (.i (bit64 (· ^^^ ·) a b))
  | "LShift" => do
    let a ← aI 0; let sh ← aI 1
    if sh < 0 || sh > 63 then none else pure (.i (wrap64 (a <<< sh.toNat)))
  | "RShift" => do
    let a ← aI 0; let sh ← aI 1
    if sh < 0 || sh > 63 then none else pure (.i (a >>> sh.toNat))  -- Int >>> is arithmetic (floor)
  | "Neg" =>
    if isInt then do let v ← aI 0; pure (.i (wrap64 (-v)))
    else do let v ← aF 0; pure (.f (-v))
  | "Abs" =>
    if isInt then do
      let v ← aI 0; pure (.i (if v < 0 then wrap64 (-v) else v))
    else do let v ← aF 0; pure (.f v.abs)
  | "Sqrt"  => do let v ← aF 0; pure (.f v.sqrt)
  | "Floor" => do let v ← aF 0; pure (.f v.floor)
  | "Ceil"  => do let v ← aF 0; pure (.f v.ceil)
  | "Round" => do let v ← aF 0; pure (.f v.round)   -- half-away, = llvm.round = metal::round
  | "Not" => do
    match ← args[0]? with
    | .b v => pure (.b (!v))
    | .i n => pure (.b (n == 0))
    | .f x => pure (.b (x == 0.0))
  | "BitNot" => do let v ← aI 0; pure (.i (bit64 (· ^^^ ·) v (-1)))
  | "ToInt"   => do (← args[0]?).coerce? .int
  | "ToBool"  => do (← args[0]?).coerce? .bool
  | "ToFloat" => do (← args[0]?).coerce? .float
  | "Clamp" =>
    if isInt then do
      let v ← aI 0; let lo ← aI 1; let hi ← aI 2
      let lc := if v > lo then v else lo
      pure (.i (if lc < hi then lc else hi))
    else do
      let v ← aF 0; let lo ← aF 1; let hi ← aF 2
      let lc := if v > lo then v else lo
      pure (.f (if lc < hi then lc else hi))
  | "Select" => do
    let cond ← match ← args[0]? with
      | .b v => some v
      | .i n => some (n != 0)
      | .f x => some (x != 0.0)
    let v ← if cond then args[1]? else args[2]?
    v.coerce? resultType
  | _ => none  -- Ldexp / FloatExponent / anything exotic: stay runtime

-- ─────────────────────────────────────────────────────────────
-- Operation emit (mirrors EmitLlvm.emitOp, expression-style)
-- ─────────────────────────────────────────────────────────────

private def argAs (args : Array TVal) (i : Nat) (target : ScalarType) : M String := do
  match args[i]? with
  | some v => pure (← coerce v target).ref
  | none => fail s!"EmitMsl: missing operand {i}"

private def binF (op : String) (a b : String) : M TVal :=
  bindVal .float s!"({a} {op} {b})"
private def binI (op : String) (a b : String) : M TVal :=
  bindVal .int s!"({a} {op} {b})"
private def cmpF (op : String) (a b : String) : M TVal :=
  bindVal .bool s!"({a} {op} {b})"
private def cmpI (op : String) (a b : String) : M TVal :=
  bindVal .bool s!"({a} {op} {b})"
private def callF (fn : String) (a : String) : M TVal :=
  bindVal .float s!"{fn}({a})"

/-- The runtime (non-folded) op emission. Zero-guards mirror EmitLlvm's
    selects as ternaries; every result is a fresh `const` local. -/
def emitOpRuntime (tag : String) (resultType : ScalarType) (args : Array TVal) : M TVal := do
  let aF (i : Nat) := argAs args i .float
  let aI (i : Nat) := argAs args i .int
  let aB (i : Nat) := argAs args i .bool
  let isInt := resultType == .int
  let eitherInt : M Bool := do
    pure (((args[0]?).map (·.ty == .int)).getD false ||
          ((args[1]?).map (·.ty == .int)).getD false)
  match tag with
  | "Add" => if isInt then binI "+" (← aI 0) (← aI 1) else binF "+" (← aF 0) (← aF 1)
  | "Sub" => if isInt then binI "-" (← aI 0) (← aI 1) else binF "-" (← aF 0) (← aF 1)
  | "Mul" => if isInt then binI "*" (← aI 0) (← aI 1) else binF "*" (← aF 0) (← aF 1)
  | "Div" =>
    if isInt then
      let a ← aI 0; let b ← aI 1
      bindVal .int s!"({b} == 0L ? 0L : ({a} / {b}))"
    else
      let a ← aF 0; let b ← aF 1
      bindVal .float s!"({b} == 0.0f ? 0.0f : ({a} / {b}))"
  | "Mod" =>
    if isInt then
      let a ← aI 0; let b ← aI 1
      bindVal .int s!"({b} == 0L ? 0L : ({a} % {b}))"
    else
      let a ← aF 0; let b ← aF 1
      let q ← binF "/" a b
      let fq ← callF "floor" q.ref
      bindVal .float s!"({b} == 0.0f ? 0.0f : ({a} - {fq.ref} * {b}))"
  | "FloorDiv" =>
    if isInt then
      let a ← aI 0; let b ← aI 1
      bindVal .int s!"({b} == 0L ? 0L : ({a} / {b}))"
    else
      let a ← aF 0; let b ← aF 1
      let dv ← bindVal .float s!"({b} == 0.0f ? 0.0f : ({a} / {b}))"
      callF "floor" dv.ref
  | "Less"      => if (← eitherInt) then cmpI "<"  (← aI 0) (← aI 1) else cmpF "<"  (← aF 0) (← aF 1)
  | "LessEq"    => if (← eitherInt) then cmpI "<=" (← aI 0) (← aI 1) else cmpF "<=" (← aF 0) (← aF 1)
  | "Greater"   => if (← eitherInt) then cmpI ">"  (← aI 0) (← aI 1) else cmpF ">"  (← aF 0) (← aF 1)
  | "GreaterEq" => if (← eitherInt) then cmpI ">=" (← aI 0) (← aI 1) else cmpF ">=" (← aF 0) (← aF 1)
  | "Equal"     => if (← eitherInt) then cmpI "==" (← aI 0) (← aI 1) else cmpF "==" (← aF 0) (← aF 1)
  | "NotEqual"  => if (← eitherInt) then cmpI "!=" (← aI 0) (← aI 1) else cmpF "!=" (← aF 0) (← aF 1)
  | "And" => bindVal .bool s!"({← aB 0} && {← aB 1})"
  | "Or"  => bindVal .bool s!"({← aB 0} || {← aB 1})"
  | "BitAnd" => binI "&" (← aI 0) (← aI 1)
  | "BitOr"  => binI "|" (← aI 0) (← aI 1)
  | "BitXor" => binI "^" (← aI 0) (← aI 1)
  | "LShift" => binI "<<" (← aI 0) (← aI 1)
  | "RShift" => binI ">>" (← aI 0) (← aI 1)   -- long >> long is arithmetic in MSL
  | "Neg" =>
    if isInt then bindVal .int s!"(0L - {← aI 0})"
    else bindVal .float s!"(-{← aF 0})"
  | "Abs" =>
    if isInt then
      let v ← aI 0
      bindVal .int s!"({v} < 0L ? (0L - {v}) : {v})"
    else callF "fabs" (← aF 0)
  | "Sqrt"  => callF "sqrt"  (← aF 0)
  | "Floor" => callF "floor" (← aF 0)
  | "Ceil"  => callF "ceil"  (← aF 0)
  | "Round" => callF "round" (← aF 0)   -- MSL round() is half-away, = llvm.round
  | "Ldexp" =>
    -- native ldexp(float, int); the exponent arg truncates like fptosi.
    let x ← aF 0
    let n ← aF 1
    bindVal .float s!"ldexp({x}, int({n}))"
  | "FloatExponent" =>
    -- raw exponent field, sign-extended shift like the f64 bit trick
    -- (bits >> 52 − 1023), transposed to f32 (bits >> 23 − 127).
    let x ← aF 0
    bindVal .float s!"float((as_type<int>({x}) >> 23) - 127)"
  | "Not" =>
    match args[0]? with
    | none => fail "EmitMsl: Not missing operand"
    | some a =>
      match a.ty with
      | .bool  => bindVal .bool s!"(!{a.ref})"
      | .int   => bindVal .bool s!"({a.ref} == 0L)"
      | .float => bindVal .bool s!"({a.ref} == 0.0f)"
  | "BitNot" => bindVal .int s!"(~{← aI 0})"
  | "ToInt"   => match args[0]? with | some a => coerce a .int   | none => fail "ToInt missing operand"
  | "ToBool"  => match args[0]? with | some a => coerce a .bool  | none => fail "ToBool missing operand"
  | "ToFloat" => match args[0]? with | some a => coerce a .float | none => fail "ToFloat missing operand"
  | "Clamp" =>
    if isInt then
      let v ← aI 0; let lo ← aI 1; let hi ← aI 2
      let lc ← bindVal .int s!"({v} > {lo} ? {v} : {lo})"
      bindVal .int s!"({lc.ref} < {hi} ? {lc.ref} : {hi})"
    else
      let v ← aF 0; let lo ← aF 1; let hi ← aF 2
      let lc ← bindVal .float s!"({v} > {lo} ? {v} : {lo})"
      bindVal .float s!"({lc.ref} < {hi} ? {lc.ref} : {hi})"
  | "Select" =>
    let cond ← match args[0]? with
      | none => fail "Select missing cond"
      | some a =>
        match a.ty with
        | .bool  => pure a.ref
        | .int   => pure s!"({a.ref} != 0L)"
        | .float => pure s!"({a.ref} != 0.0f)"
    let t ← argAs args 1 resultType
    let f ← argAs args 2 resultType
    bindVal resultType s!"({cond} ? {t} : {f})"
  | other => fail s!"EmitMsl: unsupported op '{other}'"

/-- Fold when every arg is a compile-time constant; else emit. -/
def emitOp (tag : String) (resultType : ScalarType) (args : Array TVal) : M TVal := do
  let cvs := args.map (·.cv)
  if cvs.all (·.isSome) then
    match foldOp tag resultType (cvs.map (·.get!)) with
    | some c => pure ⟨c.text, c.ty, some c⟩
    | none => emitOpRuntime tag resultType args
  else emitOpRuntime tag resultType args

-- ─────────────────────────────────────────────────────────────
-- Instruction emit (mirrors EmitLlvm.emitInstr)
-- ─────────────────────────────────────────────────────────────

private def resolveF32 (o : NOperand) : M String := do
  coerceF32 (← resolveOperand o)

/-- The read text for element `idx` of array `slot`: the thread-private
    local for in-kernel arrays, the packed device buffer at a
    compile-time offset for hoisted coefficient columns. -/
private def arrayElem (slot : Nat) (idx : String) : M String := do
  match (← get).coeffOffsets.get? slot with
  | some off => pure s!"coeff_columns[{off} + {idx}]"
  | none => pure s!"arr{slot}[{idx}]"

/-- Hoisted columns are GPU-read-only (`constant` address space; the
    stage-0 kernel fills them host-side). A plan that still writes one
    in-kernel is a split bug — fail loudly, before the load. -/
private def guardColumnWrite (slot : Nat) (what : String) : M Unit := do
  if (← get).coeffOffsets.contains slot then
    fail s!"EmitMsl: {what} writes hoisted coefficient column {slot} (columns are filled host-side by the stage-0 kernel and read-only on the GPU)"

/-- `Pack` — fill the thread-private array with the resolved args. -/
private def emitPack (dst : Nat) (args : Array NOperand) : M Unit := do
  guardColumnWrite dst "Pack"
  for i in [0:args.size] do
    let v ← resolveF32 args[i]!
    line s!"arr{dst}[{i}] = {v};"

/-- Array sizes are plan constants — carried in the emitter, not loaded. -/
private def arraySizeLit (sizes : Array Nat) (slot : Nat) : M Nat := do
  match sizes[slot]? with
  | some n => pure n
  | none => fail s!"EmitMsl: array slot {slot} out of range"

def arrayRegSlot : NOperand → M Nat
  | .arrayReg s => pure s
  | _ => fail "EmitMsl: expected an arrayReg operand"

/-- `Index` — bounds-checked read (0.0 out of range), safe clamped access. -/
private def emitIndex (sizes : Array Nat) (dst : Nat) (args : Array NOperand) : M Unit := do
  let arrSlot ← arrayRegSlot args[0]!
  let size ← arraySizeLit sizes arrSlot
  let idxV ← match args[1]? with | some o => resolveOperand o | none => fail "Index missing idx"
  let rawIdx := (← coerce idxV .int).ref
  let inR ← bindVal .bool s!"({rawIdx} >= 0L && {rawIdx} < {size}L)"
  let safe ← bindVal .int s!"({inR.ref} ? {rawIdx} : 0L)"
  let elem ← arrayElem arrSlot safe.ref
  let v ← bindVal .float s!"({inR.ref} ? {elem} : 0.0f)"
  storeTempTyped dst v

/-- `SetElement` — bounds-checked in-place write. -/
private def emitSetElement (sizes : Array Nat) (args : Array NOperand) : M Unit := do
  let arrSlot ← arrayRegSlot args[0]!
  guardColumnWrite arrSlot "SetElement"
  let size ← arraySizeLit sizes arrSlot
  let idxV ← match args[1]? with | some o => resolveOperand o | none => fail "SetElement missing idx"
  let rawIdx := (← coerce idxV .int).ref
  let v ← match args[2]? with | some o => resolveF32 o | none => fail "SetElement missing value"
  line s!"if ({rawIdx} >= 0L && {rawIdx} < {size}L) \{ arr{arrSlot}[{rawIdx}] = {v}; }"

/-- Elementwise array op: a real `for` loop over `loopCount` elements. -/
private def emitElementwise (sizes : Array Nat) (instr : NInstr) (dstSlot : Nat) : M Unit := do
  guardColumnWrite dstSlot "elementwise op"
  let nargs := instr.args.size
  -- Pre-resolve scalar operands outside the loop (mirrors EmitLlvm).
  let mut argArrs : Array (Option Nat) := #[]
  let mut argScalars : Array (Option TVal) := #[]
  for i in [0:nargs] do
    if (instr.strides[i]?.getD 0) == 1 then
      argArrs := argArrs.push (some (← arrayRegSlot instr.args[i]!))
      argScalars := argScalars.push none
    else
      argArrs := argArrs.push none
      argScalars := argScalars.push (some (← resolveOperand instr.args[i]!))
  let _ ← arraySizeLit sizes dstSlot
  let n ← fresh
  line s!"for (uint ew{n} = 0; ew{n} < {instr.loopCount}u; ++ew{n}) \{"
  pushIndent
  let mut iterArgs : Array TVal := #[]
  for i in [0:nargs] do
    match argArrs[i]! with
    | some slot =>
      let elem ← arrayElem slot s!"ew{n}"
      let v ← bindVal .float elem
      iterArgs := iterArgs.push v
    | none => iterArgs := iterArgs.push (argScalars[i]!).get!
  let res ← emitOpRuntime instr.tag instr.resultType iterArgs
  let f32 ← coerceF32 res
  line s!"arr{dstSlot}[ew{n}] = {f32};"
  popIndent
  line "}"

def emitInstr (sizes : Array Nat) (instr : NInstr) : M Unit := do
  match instr.tag, instr.dst with
  | "ReduceBegin", .temp accTemp =>
    if (← get).reduce?.isSome then
      fail "EmitMsl: nested ReduceBegin regions are not supported (v1)"
    let init ← match instr.args[0]? with
      | some o => resolveOperand o
      | none => fail "ReduceBegin missing init operand"
    let initV ← coerce init instr.resultType
    -- Trip-count-as-data: an optional args[1] is the RUNTIME effective count —
    -- a `long` local clamped to [0, loopCount] BEFORE the loop (loop-invariant;
    -- `min`/`max` are metal:: via the preamble's using-directive). No args[1]
    -- emits the static literal bound, byte-identical to the pre-dynamic form.
    let bound ← match instr.args[1]? with
      | none => pure s!"{instr.loopCount}L"
      | some o =>
        let v ← resolveOperand o
        let iv ← coerce v .int
        let b ← bindVal .int s!"min(max({iv.ref}, 0L), {instr.loopCount}L)"
        pure b.ref
    -- The accumulator is a mutable typed local declared BEFORE the loop
    -- (never const-folded — its value varies across iterations).
    let accName := tempVarName accTemp instr.resultType
    if (← get).declared.contains accName then
      line s!"{accName} = {initV.ref};"
    else
      line s!"{mslTy instr.resultType} {accName} = {initV.ref};"
    let n ← fresh
    let idxVar := s!"rd{n}"
    line s!"for (long {idxVar} = 0; {idxVar} < {bound}; ++{idxVar}) \{"
    let st ← get
    modify fun s => { s with
      indent := s.indent + 1
      declared := s.declared.insert accName
      tempTypes := s.tempTypes.insert accTemp instr.resultType
      constTemps := s.constTemps.erase accTemp
      reduce? := some {
        accTemp, accTy := instr.resultType, idxVar
        savedTempTypes := st.tempTypes.insert accTemp instr.resultType
        savedConstTemps := st.constTemps.erase accTemp
        savedDeclared := st.declared.insert accName } }
  | "ReduceEnd", .temp accTemp =>
    let some rc := (← get).reduce?
      | fail "EmitMsl: ReduceEnd without an open ReduceBegin"
    if rc.accTemp != accTemp then
      fail "EmitMsl: ReduceEnd accumulator does not match the open ReduceBegin"
    modify fun s => { s with indent := s.indent - 1 }
    line "}"
    -- Body locals were scoped to the loop braces; only the accumulator
    -- survives (its declaration precedes the loop).
    modify fun s => { s with
      reduce? := none
      tempTypes := rc.savedTempTypes
      constTemps := rc.savedConstTemps
      declared := rc.savedDeclared }
  | "WriteSlot", .moduleSlot idx =>
    let v ← match instr.args[0]? with | some o => resolveOperand o | none => fail "WriteSlot missing arg"
    storeSlot idx v
  | "SmoothParam", _ =>
    fail "EmitMsl: SmoothParam is dead on the session path (params are slots)"
  | "Pack", .array dst => emitPack dst instr.args
  | "Index", .temp dst => emitIndex sizes dst instr.args
  | "SetElement", .array _ => emitSetElement sizes instr.args
  | _, .temp slot =>
    let args ← instr.args.mapM resolveOperand
    let result ← emitOp instr.tag instr.resultType args
    let coerced ← coerce result instr.resultType
    storeTempTyped slot coerced
  | _, .moduleSlot idx =>
    let args ← instr.args.mapM resolveOperand
    let result ← emitOp instr.tag instr.resultType args
    storeSlot idx result
  | _, .array dst => emitElementwise sizes instr dst
  | tag, .sessionArray _ =>
    fail s!"EmitMsl: sessionArray dst for '{tag}' must be remapped before emit"

-- ─────────────────────────────────────────────────────────────
-- Fractal per-instance walk + sinks (mirror EmitLlvm)
-- ─────────────────────────────────────────────────────────────

partial def emitKernelBlock (sizes : Array Nat) (inst : InstanceFunction) : M Unit := do
  for i in inst.preambleInstructions do emitInstr sizes i
  for child in inst.children do
    for i in child.preInputInstructions do emitInstr sizes i
    emitKernelBlock sizes child
  for i in inst.instructions do emitInstr sizes i

def emitSinks (sinks : Array SinkSpec) : M Unit := do
  for sink in sinks do
    let mut acc := f32Lit 0.0
    for slotIdx in sink.inputs do
      let v ← coerceF32 (← loadSlot slotIdx)
      let r ← bindVal .float s!"({acc} + {v})"
      acc := r.ref
    line s!"output_buffer[s] = {acc} * {f32Lit sink.gain.toFloat};"

-- ─────────────────────────────────────────────────────────────
-- Module entry point
-- ─────────────────────────────────────────────────────────────

private def header : String :=
  "#include <metal_stdlib>\n" ++
  "using namespace metal;\n\n" ++
  "struct TropicalKernelConsts {\n" ++
  "    ulong start_sample_index;\n" ++
  "    float sample_rate;\n" ++
  "    uint  buffer_length;\n" ++
  "};\n\n" ++
  "kernel void tropical_kernel(\n" ++
  "    device float*                  output_buffer [[buffer(0)]],\n" ++
  "    constant float*                slots         [[buffer(1)]],\n" ++
  "    constant TropicalKernelConsts& k             [[buffer(2)]],\n" ++
  "    uint s [[thread_position_in_grid]])\n" ++
  "{\n" ++
  "    if (s >= k.buffer_length) { return; }\n" ++
  "    const long current_idx = long(k.start_sample_index) + long(s);\n"

/-- The column-binding header: `header` plus `buffer(3)` — the packed
    coefficient-column buffer, uploaded by the host from the generation
    `process()` captures. Emitted ONLY when the plan advertises hoisted
    columns; ordinary plans keep the exact 3-binding header above (the
    msl-golden gates freeze that text). -/
private def headerColumns : String :=
  "#include <metal_stdlib>\n" ++
  "using namespace metal;\n\n" ++
  "struct TropicalKernelConsts {\n" ++
  "    ulong start_sample_index;\n" ++
  "    float sample_rate;\n" ++
  "    uint  buffer_length;\n" ++
  "};\n\n" ++
  "kernel void tropical_kernel(\n" ++
  "    device float*                  output_buffer [[buffer(0)]],\n" ++
  "    constant float*                slots         [[buffer(1)]],\n" ++
  "    constant TropicalKernelConsts& k             [[buffer(2)]],\n" ++
  "    constant float*                coeff_columns [[buffer(3)]],\n" ++
  "    uint s [[thread_position_in_grid]])\n" ++
  "{\n" ++
  "    if (s >= k.buffer_length) { return; }\n" ++
  "    const long current_idx = long(k.start_sample_index) + long(s);\n"

/-- Emit the full MSL source for a FlatPlan's fused kernel. One thread
    per sample; the host dispatches `buffer_length` threads.

    A plan that advertises hoisted coefficient columns
    (`coeff_array_slots` — banks-as-data) emits in COLUMN-BINDING mode:
    the kernel signature gains `constant float* coeff_columns
    [[buffer(3)]]` (one packed buffer; per-slot offsets are compile-time
    literals in plan order — the same order the runtime packs the upload
    staging), reads of those slots index the binding, and no
    thread-private `arrN` local is declared for them (the stage-0
    coefficient kernel fills them host-side; an in-kernel local could
    never be filled). All OTHER array slots keep thread-private locals
    and in-kernel fills. Columns-free plans keep the exact 3-binding
    header, byte-frozen by the msl-golden gates. -/
def emitKernel (plan : FlatPlan) : Except String String := do
  let sizes := plan.arraySlotSizes
  -- Packed offsets for the hoisted columns, in the plan's advertised
  -- order — MUST match FlatRuntime's upload packing (it walks
  -- coeff_array_slots in the same order).
  let mut coeffOffsets : Std.HashMap Nat Nat := {}
  let mut colOff := 0
  for s in plan.coeffArraySlots do
    let some sz := sizes[s]?
      | throw s!"EmitMsl: coeff_array_slots entry {s} out of range (plan has {sizes.size} array slots)"
    coeffOffsets := coeffOffsets.insert s colOff
    colOff := colOff + max sz 1
  let body : M Unit := do
    -- thread-private array scratch (per-sample on CPU too); hoisted
    -- coefficient columns are read from the buffer(3) binding instead.
    for i in [0:plan.arraySlotCount] do
      unless coeffOffsets.contains i do
        let sz := sizes[i]?.getD 1
        line s!"float arr{i}[{max sz 1}];"
    for inst in plan.instanceFunctions do
      emitKernelBlock sizes inst
    emitSinks plan.sinks
  let init : St := { sources := plan.sources, sampleRate := plan.sampleRate.toFloat,
                     coeffOffsets }
  match body.run init with
  | .error e _ => .error e
  | .ok _ st =>
    let bodyText := String.intercalate "\n" st.lines.toList
    -- `current_idx` may be unused after full folding; cast to void to
    -- keep -Wunused clean would misfire on use, so leave it — Metal's
    -- compiler doesn't warn on unused const locals by default.
    let hdr := if plan.coeffArraySlots.isEmpty then header else headerColumns
    .ok (hdr ++ bodyText ++ "\n}\n")

end Tropical.Ir.EmitMsl
