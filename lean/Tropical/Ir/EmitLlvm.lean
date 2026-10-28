import Std.Data.HashMap
import Lean.Data.Json
import Tropical.Plan
import Tropical.Kernel
import Tropical.Ir.Stage0
import Tropical.Ir.EmitCommon

/-!
# EmitLlvm — `FlatPlan → textual LLVM IR`

A faithful port of `engine/jit/OrcJitEngine.cpp`'s fused-mode codegen
(`compile_flat_program` + `EmitCtx`), emitting textual LLVM IR instead of
driving the C++ `IRBuilder`. The engine's `compile_ir_text` consumes this
text; the gate is **audio equality** with `load_plan` (not IR-text
equality), so this emits uniform, un-folded IR and lets the JIT's
optimizer normalize (e.g. constant div-by-zero guards fold away).

## The port (`design/typed-kernel.md`)

```
define void @tropical_kernel(ptr %tau, ptr %in_1, …, ptr %in_k, ptr %out)
```

One pointer per input of the plan's kernel type (`Tropical.Kernel.ofPlan`:
`tau` first — `i64` Q32.32 clock values, shape `[frames]` — then one
one-element `double` per `param:` slot in slot order) and one for the
output (`double`, shape `[frames, C]`, frame-major: sink target `c` writes
`s * C + c`). `frames` is static: the loop bound is a literal, and a
different buffer length is a different kernel. Nothing else crosses: the
module slot vector, the array backing stores and the SSA scratch are
`alloca`s inside the function, seeded per call from the plan's slot
defaults, the param inputs and the stage-0 prologue (the coefficient plan's
instructions, emitted before the sample loop — currying, not a second
kernel). The rate is a literal. `%current_idx` is `tau[s] >> 32`.

A second exported function, `@tropical_kernel_argv(ptr %argv)`, unpacks a
pointer table and calls the port — the host-side thunk a C caller needs,
not part of the type. `compile_ir_text` renames every definition to a
content-addressed symbol and looks the thunk up.

`emitStage0` is the Metal adapter's CPU pre-pass: the coefficient plan as
`@tropical_stage0(ptr %slots, ptr %arrays, ptr %array_sizes, ptr %temps)`
over host storage, so the slot vector and the coefficient columns the GPU
kernel reads are computed in f64 on the host. Internal to that realization;
a JIT artifact carries none.

Registers/temps/arrays are `i64`-backed (float=bitcast, int=raw,
bool=zext); slots are direct `double`. Instruction operand/dst slots are
already global (the partitioner pre-shifts).

f64 constants are emitted as LLVM hex-bits (`0x…`, raw IEEE-754) so the
parser reproduces them exactly rather than risking decimal round-trip.
-/

namespace Tropical.Ir.EmitLlvm

open Lean (Json)
open Tropical.Plan
open Tropical.Ir.EmitCommon

abbrev ScalarType := Tropical.Plan.ScalarType

-- ─────────────────────────────────────────────────────────────
-- Formatting helpers
-- ─────────────────────────────────────────────────────────────

/-- 16-digit zero-padded hex of a UInt64 (LLVM `double 0x…` literal). -/
def hex16 (u : UInt64) : String :=
  let s := String.ofList (Nat.toDigits 16 u.toNat)
  "0x" ++ String.ofList (List.replicate (16 - s.length) '0') ++ s

/-- An f64 LLVM literal as raw IEEE-754 bits — exact, parser-safe. -/
def f64Lit (x : Float) : String := hex16 x.toBits

def llTy : ScalarType → String
  | .float => "double" | .int => "i64" | .bool => "i1"

-- ─────────────────────────────────────────────────────────────
-- Emitter state + monad
-- ─────────────────────────────────────────────────────────────

/-- A typed SSA value: the LLVM operand text plus its scalar type. -/
structure TVal where
  ref : String
  ty  : ScalarType
deriving Inhabited

/-- An open `ReduceBegin`/`ReduceEnd` region: the region's binder id (what a
    `loopIdx id` operand resolves against), the accumulator temp and its
    alloca, the iteration-counter alloca, the loop labels, and the `tempVals`
    snapshot to restore at close (body temps don't escape). -/
structure ReduceCtx where
  id : Nat
  accTemp : Nat
  accPtr : String
  accTy : ScalarType
  idxPtr : String
  condLabel : String
  endLabel : String
  savedTempVals : Std.HashMap Nat TVal
deriving Inhabited

structure RoutedCtx where
  id : Nat
  dst : Nat
  dstPtr : String
  routeGlobal : String
  routeCount : Nat
  fanout : Nat
  idxPtr : String
  condLabel : String
  endLabel : String
  savedTempVals : Std.HashMap Nat TVal
deriving Inhabited

structure St where
  next : Nat := 0
  lines : Array String := #[]
  globals : Array String := #[]
  /-- Global temp slot → the SSA value last written there. Temps are pure
      intra-call scratch in the fused kernel (sinks read module slots; the
      runtime never reads `%temps` back), so instruction results stay SSA
      values instead of round-tripping through the `%temps` array — the
      same shape EmitMsl uses (typed locals). This is what keeps the
      one-block kernel free of tens of thousands of escaping stores that
      LLVM's machine scheduler and regalloc otherwise choke on. A `reg`
      read with no prior write falls back to loading `%temps` (the
      runtime zero-initializes it), preserving graceful-exclusion
      semantics. -/
  tempVals : Std.HashMap Nat TVal := {}
  sources : Array SourceKind := #[]
  /-- The rate the plan was compiled for: a literal in the kernel. -/
  sampleRate : Float := 44100.0
  /-- Array slot sizes — plan constants, so bounds checks are literals. -/
  arraySizes : Array Nat := #[]
  /-- Port kernel: every array slot is an `alloca` named `%arr<slot>`.
      Stage-0 pre-pass: arrays come through the `%arrays` pointer table. -/
  arraysInline : Bool := true
  /-- Label of the basic block currently being emitted into. Starts at
      `entry`; array ops (SetElement / elementwise) and zeroing loops open
      new blocks, so the loop header's phi and the back-edge must reference
      the *actual* predecessor, never a hardcoded label. -/
  curBlock : String := "entry"
  /-- The stack of open reduction regions, outermost first (`ReduceBegin`
      pushes, `ReduceEnd` pops). Nested banks: `loopIdx id` resolves by
      searching this stack for its binder id; accumulator temp reads/writes
      search it innermost-first, so an inner-region instruction touching an
      OUTER accumulator finds the outer alloca. -/
  reduces : Array ReduceCtx := #[]
  routeds : Array RoutedCtx := #[]

abbrev M := EStateM String St

/-- The open region owning accumulator temp `slot`, innermost first. -/
def findAccCtx (slot : Nat) : M (Option ReduceCtx) := do
  let rs := (← get).reduces
  for i in [0:rs.size] do
    let rc := rs[rs.size - 1 - i]!
    if rc.accTemp == slot then return some rc
  return none

def fresh : M String := do
  let s ← get
  set { s with next := s.next + 1 }
  pure s!"%t{s.next}"

def line (l : String) : M Unit :=
  modify fun s => { s with lines := s.lines.push ("  " ++ l) }

def fail {α} (msg : String) : M α := throw msg

-- ─────────────────────────────────────────────────────────────
-- Coercion (mirrors EmitCtx::coerce_val)
-- ─────────────────────────────────────────────────────────────

def coerce (v : TVal) (to : ScalarType) : M TVal := do
  if v.ty == to then pure v else
  let r ← fresh
  match v.ty, to with
  | .float, .int  => line s!"{r} = fptosi double {v.ref} to i64"; pure ⟨r, .int⟩
  | .float, .bool => line s!"{r} = fcmp une double {v.ref}, 0x0000000000000000"; pure ⟨r, .bool⟩
  | .int,   .float => line s!"{r} = sitofp i64 {v.ref} to double"; pure ⟨r, .float⟩
  | .int,   .bool => line s!"{r} = icmp ne i64 {v.ref}, 0"; pure ⟨r, .bool⟩
  | .bool,  .float => line s!"{r} = uitofp i1 {v.ref} to double"; pure ⟨r, .float⟩
  | .bool,  .int  => line s!"{r} = zext i1 {v.ref} to i64"; pure ⟨r, .int⟩
  | _, _ => pure v

def coerceF64 (v : TVal) : M String := do pure (← coerce v .float).ref

-- ─────────────────────────────────────────────────────────────
-- Loads / stores (mirror the gep_temp / load_*_typed / slot helpers)
-- ─────────────────────────────────────────────────────────────

/-- Load `temps[slot]` (i64-backed) as the given type — the fallback for a
    `reg` read with no prior write this kernel (reads the runtime's
    zero-initialized scratch, preserving graceful-exclusion semantics). -/
def loadTempTyped (slot : Nat) (ty : ScalarType) : M TVal := do
  let g ← fresh; line s!"{g} = getelementptr inbounds i64, ptr %temps, i64 {slot}"
  let raw ← fresh; line s!"{raw} = load i64, ptr {g}, align 8"
  match ty with
  | .int => pure ⟨raw, .int⟩
  | .float => let b ← fresh; line s!"{b} = bitcast i64 {raw} to double"; pure ⟨b, .float⟩
  | .bool => let b ← fresh; line s!"{b} = trunc i64 {raw} to i1"; pure ⟨b, .bool⟩

/-- Record a temp write as an SSA value for later `reg` reads. Emits no
    store: temps are intra-call scratch in the fused kernel, and every
    block the emitter opens falls through, so an earlier def dominates
    every later use. (The old store/load round-trip was bitcast-exact, so
    handing back the same TVal is bit-identical.) Inside a reduction
    region, a write to the accumulator temp stores through its alloca
    instead — the running value crosses the loop back-edge in memory. -/
def storeTempTyped (slot : Nat) (v : TVal) : M Unit := do
  if let some rc ← findAccCtx slot then
    let cv ← coerce v rc.accTy
    line s!"store {llTy rc.accTy} {cv.ref}, ptr {rc.accPtr}, align 8"
    return
  modify fun s => { s with tempVals := s.tempVals.insert slot v }

def loadSlotF64 (idx : Nat) : M String := do
  let g ← fresh; line s!"{g} = getelementptr inbounds double, ptr %slots, i64 {idx}"
  let v ← fresh; line s!"{v} = load double, ptr {g}, align 8"
  pure v

def storeSlotF64 (idx : Nat) (v : String) : M Unit := do
  let g ← fresh; line s!"{g} = getelementptr inbounds double, ptr %slots, i64 {idx}"
  line s!"store double {v}, ptr {g}, align 8"

-- Arrays: an alloca per slot in the port kernel; the `%arrays` pointer
-- table in the stage-0 pre-pass. Sizes are plan constants either way.
def loadArrayPtr (slot : Nat) : M String := do
  if (← get).arraysInline then return s!"%arr{slot}"
  let g ← fresh; line s!"{g} = getelementptr inbounds ptr, ptr %arrays, i64 {slot}"
  let p ← fresh; line s!"{p} = load ptr, ptr {g}, align 8"
  pure p

def loadArraySize (slot : Nat) : M String := do
  pure (toString ((← get).arraySizes[slot]?.getD 0))

-- Fresh integer id (for unique basic-block labels).
def freshId : M Nat := do
  let s ← get
  set { s with next := s.next + 1 }
  pure s.next

/-- Emit a basic-block label (no indentation) and make it current. -/
def labelLine (l : String) : M Unit :=
  modify fun s => { s with lines := s.lines.push (l ++ ":"), curBlock := l }

/-- The arrayReg slot of an operand (for Index/SetElement/elementwise). -/
def arrayRegSlot (operand : NOperand) : M Nat :=
  match expectArrayReg "EmitLlvm" operand with
  | .ok slot => pure slot
  | .error e => fail e

-- ─────────────────────────────────────────────────────────────
-- Operand resolution (mirrors EmitCtx::resolve_typed)
-- ─────────────────────────────────────────────────────────────

def resolveOperand : NOperand → M TVal
  | .const v t => match t with
    | .int  => pure ⟨toString (jsonNumberToInt v), .int⟩
    | .bool => pure ⟨if jsonNumberToInt v != 0 then "true" else "false", .bool⟩
    | .float => pure ⟨f64Lit v.toFloat, .float⟩
  | .reg slot _ => do
    -- Inside a reduction region, an accumulator temp reads the running
    -- value from its alloca (an SSA value can't cross the back-edge).
    -- Searched through the whole region stack: an inner-region read of
    -- an OUTER accumulator must find the outer alloca.
    if let some rc ← findAccCtx slot then
      let v ← fresh
      line s!"{v} = load {llTy rc.accTy}, ptr {rc.accPtr}, align 8"
      return ⟨v, rc.accTy⟩
    match (← get).tempVals.get? slot with
    | some v => pure v
    | none => loadTempTyped slot .float
  | .loopIdx id => do
    -- Resolve the binder id against the stack of open regions (unique along
    -- a nesting chain, so first match is THE match).
    if let some rc := (← get).reduces.find? (fun rc => rc.id == id) then
      let v ← fresh
      line s!"{v} = load i64, ptr {rc.idxPtr}, align 8"
      return ⟨v, .int⟩
    if let some rc := (← get).routeds.find? (fun rc => rc.id == id) then
      let v ← fresh
      line s!"{v} = load i64, ptr {rc.idxPtr}, align 8"
      return ⟨v, .int⟩
    fail s!"EmitLlvm: loop_idx id={id} matches no open reduction region"
  | .source idx _ => do
    let srcs := (← get).sources
    match srcs[idx]? with
    | some .tick => pure ⟨"%current_idx", .int⟩
    | some .rate => pure ⟨f64Lit (← get).sampleRate, .float⟩
    | none => pure ⟨f64Lit 0.0, .float⟩
  | .slot idx t => do
    let v ← loadSlotF64 idx
    match t with
    | .float => pure ⟨v, .float⟩
    | .int  => let r ← fresh; line s!"{r} = fptosi double {v} to i64"; pure ⟨r, .int⟩
    | .bool => let r ← fresh; line s!"{r} = fcmp one double {v}, 0x0000000000000000"; pure ⟨r, .bool⟩
  -- External inputs never reached the session path (the compile maps them
  -- to constant zero); the port has no such region, so the operand IS zero.
  | .input _ _ => pure ⟨f64Lit 0.0, .float⟩
  | .param _ _ => fail "EmitLlvm: 'param' operand is dead on the session path (use slots)"
  | .arrayReg _ => fail "EmitLlvm: bare 'arrayReg' operand outside an array op"
  | .sessionArrayReg _ => fail "EmitLlvm: 'sessionArrayReg' must be remapped before emit"

-- ─────────────────────────────────────────────────────────────
-- Operation emit (mirrors EmitCtx::emit_typed_op)
-- ─────────────────────────────────────────────────────────────

/-- Coerce arg `i` to `target`, returning its ref text. -/
private def argAs (args : Array TVal) (i : Nat) (target : ScalarType) : M String := do
  match args[i]? with
  | some v => pure (← coerce v target).ref
  | none => fail s!"EmitLlvm: missing operand {i}"

private def binF (op : String) (a b : String) : M TVal := do
  let r ← fresh; line s!"{r} = {op} double {a}, {b}"; pure ⟨r, .float⟩
private def binI (op : String) (a b : String) : M TVal := do
  let r ← fresh; line s!"{r} = {op} i64 {a}, {b}"; pure ⟨r, .int⟩
private def cmpF (pred : String) (a b : String) : M TVal := do
  let r ← fresh; line s!"{r} = fcmp {pred} double {a}, {b}"; pure ⟨r, .bool⟩
private def cmpI (pred : String) (a b : String) : M TVal := do
  let r ← fresh; line s!"{r} = icmp {pred} i64 {a}, {b}"; pure ⟨r, .bool⟩
private def callF (fn : String) (a : String) : M TVal := do
  let r ← fresh; line s!"{r} = call double {fn}(double {a})"; pure ⟨r, .float⟩

private def zeroF : String := f64Lit 0.0

def emitOp (tag : String) (resultType : ScalarType) (args : Array TVal) : M TVal := do
  let aF (i : Nat) := argAs args i .float
  let aI (i : Nat) := argAs args i .int
  let aB (i : Nat) := argAs args i .bool
  let isInt := resultType == .int
  -- comparison dispatch: int compare if either operand is natively int
  let eitherInt : M Bool := do
    pure (((args[0]?).map (·.ty == .int)).getD false ||
          ((args[1]?).map (·.ty == .int)).getD false)
  let op ← match validateScalarOp "EmitLlvm" tag args.size with
    | .ok op => pure op
    | .error e => fail e
  match op with
  | .add => if isInt then binI "add" (← aI 0) (← aI 1) else binF "fadd" (← aF 0) (← aF 1)
  | .sub => if isInt then binI "sub" (← aI 0) (← aI 1) else binF "fsub" (← aF 0) (← aF 1)
  | .mul => if isInt then binI "mul" (← aI 0) (← aI 1) else binF "fmul" (← aF 0) (← aF 1)
  | .div =>
    if isInt then
      let b ← aI 1; let q ← binI "sdiv" (← aI 0) b
      let z ← cmpI "eq" b "0"; let r ← fresh
      line s!"{r} = select i1 {z.ref}, i64 0, i64 {q.ref}"; pure ⟨r, .int⟩
    else
      let b ← aF 1; let q ← binF "fdiv" (← aF 0) b
      let z ← cmpF "oeq" b zeroF; let r ← fresh
      line s!"{r} = select i1 {z.ref}, double {zeroF}, double {q.ref}"; pure ⟨r, .float⟩
  | .mod =>
    if isInt then
      let b ← aI 1; let q ← binI "srem" (← aI 0) b
      let z ← cmpI "eq" b "0"; let r ← fresh
      line s!"{r} = select i1 {z.ref}, i64 0, i64 {q.ref}"; pure ⟨r, .int⟩
    else
      let a ← aF 0; let b ← aF 1
      let z ← cmpF "oeq" b zeroF
      let q ← binF "fdiv" a b
      let fq ← callF "@llvm.floor.f64" q.ref
      let prod ← binF "fmul" fq.ref b
      let m ← binF "fsub" a prod.ref
      let r ← fresh
      line s!"{r} = select i1 {z.ref}, double {zeroF}, double {m.ref}"; pure ⟨r, .float⟩
  | .floorDiv =>
    if isInt then
      let b ← aI 1; let q ← binI "sdiv" (← aI 0) b
      let z ← cmpI "eq" b "0"; let r ← fresh
      line s!"{r} = select i1 {z.ref}, i64 0, i64 {q.ref}"; pure ⟨r, .int⟩
    else
      let b ← aF 1; let q ← binF "fdiv" (← aF 0) b
      let z ← cmpF "oeq" b zeroF; let dv ← fresh
      line s!"{dv} = select i1 {z.ref}, double {zeroF}, double {q.ref}"
      callF "@llvm.floor.f64" dv
  | .less      => if (← eitherInt) then cmpI "slt" (← aI 0) (← aI 1) else cmpF "olt" (← aF 0) (← aF 1)
  | .lessEq    => if (← eitherInt) then cmpI "sle" (← aI 0) (← aI 1) else cmpF "ole" (← aF 0) (← aF 1)
  | .greater   => if (← eitherInt) then cmpI "sgt" (← aI 0) (← aI 1) else cmpF "ogt" (← aF 0) (← aF 1)
  | .greaterEq => if (← eitherInt) then cmpI "sge" (← aI 0) (← aI 1) else cmpF "oge" (← aF 0) (← aF 1)
  | .equal     => if (← eitherInt) then cmpI "eq"  (← aI 0) (← aI 1) else cmpF "oeq" (← aF 0) (← aF 1)
  | .notEqual  => if (← eitherInt) then cmpI "ne"  (← aI 0) (← aI 1) else cmpF "une" (← aF 0) (← aF 1)
  | .and => let r ← fresh; line s!"{r} = and i1 {← aB 0}, {← aB 1}"; pure ⟨r, .bool⟩
  | .or  => let r ← fresh; line s!"{r} = or i1 {← aB 0}, {← aB 1}"; pure ⟨r, .bool⟩
  | .bitAnd => binI "and" (← aI 0) (← aI 1)
  | .bitOr  => binI "or"  (← aI 0) (← aI 1)
  | .bitXor => binI "xor" (← aI 0) (← aI 1)
  | .lshift => binI "shl" (← aI 0) (← aI 1)
  | .rshift => binI "ashr" (← aI 0) (← aI 1)
  | .neg =>
    if isInt then binI "sub" "0" (← aI 0) else
    let r ← fresh; line s!"{r} = fneg double {← aF 0}"; pure ⟨r, .float⟩
  | .abs =>
    if isInt then
      let v ← aI 0; let neg ← binI "sub" "0" v
      let lt ← cmpI "slt" v "0"; let r ← fresh
      line s!"{r} = select i1 {lt.ref}, i64 {neg.ref}, i64 {v}"; pure ⟨r, .int⟩
    else callF "@llvm.fabs.f64" (← aF 0)
  | .sqrt  => callF "@llvm.sqrt.f64"  (← aF 0)
  | .floor => callF "@llvm.floor.f64" (← aF 0)
  | .ceil  => callF "@llvm.ceil.f64"  (← aF 0)
  | .round => callF "@llvm.round.f64" (← aF 0)
  | .ldexp =>
    let x ← aF 0
    let ni ← fresh; line s!"{ni} = fptosi double {← aF 1} to i64"
    let add ← fresh; line s!"{add} = add i64 {ni}, 1023"
    let bias ← fresh; line s!"{bias} = shl i64 {add}, 52"
    let scale ← fresh; line s!"{scale} = bitcast i64 {bias} to double"
    binF "fmul" x scale
  | .floatExponent =>
    let bits ← fresh; line s!"{bits} = bitcast double {← aF 0} to i64"
    let sh ← fresh; line s!"{sh} = ashr i64 {bits}, 52"
    let e ← fresh; line s!"{e} = sub i64 {sh}, 1023"
    let r ← fresh; line s!"{r} = sitofp i64 {e} to double"; pure ⟨r, .float⟩
  | .not =>
    match args[0]? with
    | none => fail "EmitLlvm: Not missing operand"
    | some a =>
      match a.ty with
      | .bool  => let r ← fresh; line s!"{r} = xor i1 {a.ref}, true"; pure ⟨r, .bool⟩
      | .int   => let r ← fresh; line s!"{r} = icmp eq i64 {a.ref}, 0"; pure ⟨r, .bool⟩
      | .float => let r ← fresh; line s!"{r} = fcmp oeq double {a.ref}, {zeroF}"; pure ⟨r, .bool⟩
  | .bitNot => let r ← fresh; line s!"{r} = xor i64 {← aI 0}, -1"; pure ⟨r, .int⟩
  | .toInt   => match args[0]? with | some a => coerce a .int   | none => fail "ToInt missing operand"
  | .toBool  => match args[0]? with | some a => coerce a .bool  | none => fail "ToBool missing operand"
  | .toFloat => match args[0]? with | some a => coerce a .float | none => fail "ToFloat missing operand"
  | .clamp =>
    if isInt then
      let val ← aI 0; let lo ← aI 1; let hi ← aI 2
      let gt ← cmpI "sgt" val lo; let loc ← fresh
      line s!"{loc} = select i1 {gt.ref}, i64 {val}, i64 {lo}"
      let lt ← cmpI "slt" loc hi; let r ← fresh
      line s!"{r} = select i1 {lt.ref}, i64 {loc}, i64 {hi}"; pure ⟨r, .int⟩
    else
      let val ← aF 0; let lo ← aF 1; let hi ← aF 2
      let gt ← cmpF "ogt" val lo; let loc ← fresh
      line s!"{loc} = select i1 {gt.ref}, double {val}, double {lo}"
      let lt ← cmpF "olt" loc hi; let r ← fresh
      line s!"{r} = select i1 {lt.ref}, double {loc}, double {hi}"; pure ⟨r, .float⟩
  | .select =>
    let cond ← match args[0]? with
      | none => fail "Select missing cond"
      | some a =>
        match a.ty with
        | .bool  => pure a.ref
        | .int   => do let r ← fresh; line s!"{r} = icmp ne i64 {a.ref}, 0"; pure r
        | .float => do let r ← fresh; line s!"{r} = fcmp une double {a.ref}, {zeroF}"; pure r
    let t ← argAs args 1 resultType
    let f ← argAs args 2 resultType
    let r ← fresh
    line s!"{r} = select i1 {cond}, {llTy resultType} {t}, {llTy resultType} {f}"
    pure ⟨r, resultType⟩

-- ─────────────────────────────────────────────────────────────
-- Instruction emit (mirrors EmitCtx::emit_instr)
-- ─────────────────────────────────────────────────────────────

/-- Resolve an operand and coerce to f64 (mirrors `resolve_as_f64`). -/
private def resolveF64 (o : NOperand) : M String := do
  coerceF64 (← resolveOperand o)

/-- `Pack` — fill `arrays[dst]` with the resolved-as-f64 args. -/
private def emitPack (dst : Nat) (args : Array NOperand) : M Unit := do
  let dstPtr ← loadArrayPtr dst
  for i in [0:args.size] do
    let v ← resolveF64 args[i]!
    let bc ← fresh; line s!"{bc} = bitcast double {v} to i64"
    let ep ← fresh; line s!"{ep} = getelementptr inbounds i64, ptr {dstPtr}, i64 {i}"
    line s!"store i64 {bc}, ptr {ep}, align 8"

/-- `Index` — bounds-checked `arrays[arr][idx]` (0.0 out of range) → temp. -/
private def emitIndex (dst : Nat) (args : Array NOperand) : M Unit := do
  let arrSlot ← arrayRegSlot args[0]!
  let size ← loadArraySize arrSlot
  let arrPtr ← loadArrayPtr arrSlot
  let idxV ← match args[1]? with | some o => resolveOperand o | none => fail "Index missing idx"
  let rawIdx := (← coerce idxV .int).ref
  let neg ← fresh; line s!"{neg} = icmp slt i64 {rawIdx}, 0"
  let notNeg ← fresh; line s!"{notNeg} = xor i1 {neg}, true"
  let lt ← fresh; line s!"{lt} = icmp ult i64 {rawIdx}, {size}"
  let inRange ← fresh; line s!"{inRange} = and i1 {notNeg}, {lt}"
  let ep ← fresh; line s!"{ep} = getelementptr inbounds i64, ptr {arrPtr}, i64 {rawIdx}"
  let raw ← fresh; line s!"{raw} = load i64, ptr {ep}, align 8"
  let val ← fresh; line s!"{val} = bitcast i64 {raw} to double"
  let sel ← fresh; line s!"{sel} = select i1 {inRange}, double {val}, double {zeroF}"
  storeTempTyped dst ⟨sel, .float⟩

/-- `SetElement` — bounds-checked in-place write to `arrays[arr][idx]`. -/
private def emitSetElement (args : Array NOperand) : M Unit := do
  let arrSlot ← arrayRegSlot args[0]!
  let size ← loadArraySize arrSlot
  let arrPtr ← loadArrayPtr arrSlot
  let idxV ← match args[1]? with | some o => resolveOperand o | none => fail "SetElement missing idx"
  let rawIdx := (← coerce idxV .int).ref
  let neg ← fresh; line s!"{neg} = icmp slt i64 {rawIdx}, 0"
  let notNeg ← fresh; line s!"{notNeg} = xor i1 {neg}, true"
  let lt ← fresh; line s!"{lt} = icmp ult i64 {rawIdx}, {size}"
  let inRange ← fresh; line s!"{inRange} = and i1 {notNeg}, {lt}"
  let n ← freshId
  let wbb := s!"set_write_{n}"; let mbb := s!"set_merge_{n}"
  line s!"br i1 {inRange}, label %{wbb}, label %{mbb}"
  labelLine wbb
  let v ← match args[2]? with | some o => resolveF64 o | none => fail "SetElement missing value"
  let bc ← fresh; line s!"{bc} = bitcast double {v} to i64"
  let ep ← fresh; line s!"{ep} = getelementptr inbounds i64, ptr {arrPtr}, i64 {rawIdx}"
  line s!"store i64 {bc}, ptr {ep}, align 8"
  line s!"br label %{mbb}"
  labelLine mbb

/-- Elementwise array op: a scalar loop over `loopCount` elements. Each
    arg with `strides[i]==1` is an array (load element `idx`); others are
    scalar (broadcast). Bit-identical to the C++ SIMD path (elementwise,
    no fast-math). -/
private def emitElementwise (instr : NInstr) (dstSlot : Nat) : M Unit := do
  let nargs := instr.args.size
  -- Pre-resolve: array operands → base ptr; scalar operands → TVal.
  let mut argPtrs : Array (Option String) := #[]
  let mut argScalars : Array (Option TVal) := #[]
  for i in [0:nargs] do
    if (instr.strides[i]?.getD 0) == 1 then
      argPtrs := argPtrs.push (some (← loadArrayPtr (← arrayRegSlot instr.args[i]!)))
      argScalars := argScalars.push none
    else
      argPtrs := argPtrs.push none
      argScalars := argScalars.push (some (← resolveOperand instr.args[i]!))
  let dstPtr ← loadArrayPtr dstSlot
  let n ← freshId
  let condbb := s!"ew_cond_{n}"; let bodybb := s!"ew_body_{n}"; let endbb := s!"ew_end_{n}"
  let idxAlloca ← fresh; line s!"{idxAlloca} = alloca i64, align 8"
  line s!"store i64 0, ptr {idxAlloca}, align 8"
  line s!"br label %{condbb}"
  labelLine condbb
  let idx ← fresh; line s!"{idx} = load i64, ptr {idxAlloca}, align 8"
  let c ← fresh; line s!"{c} = icmp ult i64 {idx}, {instr.loopCount}"
  line s!"br i1 {c}, label %{bodybb}, label %{endbb}"
  labelLine bodybb
  let mut iterArgs : Array TVal := #[]
  for i in [0:nargs] do
    match argPtrs[i]! with
    | some p =>
      let ep ← fresh; line s!"{ep} = getelementptr inbounds i64, ptr {p}, i64 {idx}"
      let raw ← fresh; line s!"{raw} = load i64, ptr {ep}, align 8"
      let v ← fresh; line s!"{v} = bitcast i64 {raw} to double"
      iterArgs := iterArgs.push ⟨v, .float⟩
    | none => iterArgs := iterArgs.push (argScalars[i]!).get!
  let res ← emitOp instr.tag instr.resultType iterArgs
  let f64 ← coerceF64 res
  let bc ← fresh; line s!"{bc} = bitcast double {f64} to i64"
  let dep ← fresh; line s!"{dep} = getelementptr inbounds i64, ptr {dstPtr}, i64 {idx}"
  line s!"store i64 {bc}, ptr {dep}, align 8"
  let nidx ← fresh; line s!"{nidx} = add i64 {idx}, 1"
  line s!"store i64 {nidx}, ptr {idxAlloca}, align 8"
  line s!"br label %{condbb}"
  labelLine endbb

def emitInstr (instr : NInstr) : M Unit := do
  match instr.tag, instr.dst with
  | "ReduceBegin", .temp accTemp =>
    -- Nested regions are supported; an ancestor with the SAME binder id is a
    -- construction bug (ids must be unique along a nesting chain).
    if (← get).reduces.any (fun rc => rc.id == instr.loopId) then
      fail s!"EmitLlvm: nested ReduceBegin id={instr.loopId} collides with an open ancestor region"
    let init ← match instr.args[0]? with
      | some o => resolveOperand o
      | none => fail "ReduceBegin missing init operand"
    let initV ← coerce init instr.resultType
    -- Trip-count-as-data: an optional args[1] is the RUNTIME effective count.
    -- Resolve it HERE, before the loop blocks open (loop-invariant), coerce to
    -- i64 (a slot read arrives f64 → fptosi), and clamp to [0, loopCount] via
    -- icmp+select (the emitIndex bounds-check style). No args[1] emits the
    -- static literal — byte-identical to the pre-dynamic form.
    let bound ← match instr.args[1]? with
      | none => pure (toString instr.loopCount)
      | some o =>
        let v ← resolveOperand o
        let iv := (← coerce v .int).ref
        let neg ← fresh; line s!"{neg} = icmp slt i64 {iv}, 0"
        let lo ← fresh; line s!"{lo} = select i1 {neg}, i64 0, i64 {iv}"
        let over ← fresh; line s!"{over} = icmp sgt i64 {lo}, {instr.loopCount}"
        let cl ← fresh; line s!"{cl} = select i1 {over}, i64 {instr.loopCount}, i64 {lo}"
        pure cl
    let accPtr ← fresh
    line s!"{accPtr} = alloca {llTy instr.resultType}, align 8"
    line s!"store {llTy instr.resultType} {initV.ref}, ptr {accPtr}, align 8"
    let idxPtr ← fresh
    line s!"{idxPtr} = alloca i64, align 8"
    line s!"store i64 0, ptr {idxPtr}, align 8"
    let n ← freshId
    let condbb := s!"rd_cond_{n}"; let bodybb := s!"rd_body_{n}"; let endbb := s!"rd_end_{n}"
    line s!"br label %{condbb}"
    labelLine condbb
    let k ← fresh; line s!"{k} = load i64, ptr {idxPtr}, align 8"
    let c ← fresh; line s!"{c} = icmp ult i64 {k}, {bound}"
    line s!"br i1 {c}, label %{bodybb}, label %{endbb}"
    labelLine bodybb
    modify fun s => { s with reduces := s.reduces.push {
      id := instr.loopId, accTemp, accPtr, accTy := instr.resultType, idxPtr,
      condLabel := condbb, endLabel := endbb, savedTempVals := s.tempVals } }
  | "ReduceEnd", .temp accTemp =>
    let some rc := (← get).reduces.back?
      | fail "EmitLlvm: ReduceEnd without an open ReduceBegin"
    if rc.accTemp != accTemp then
      fail "EmitLlvm: ReduceEnd accumulator does not match the innermost open ReduceBegin"
    let k ← fresh; line s!"{k} = load i64, ptr {rc.idxPtr}, align 8"
    let k1 ← fresh; line s!"{k1} = add i64 {k}, 1"
    line s!"store i64 {k1}, ptr {rc.idxPtr}, align 8"
    line s!"br label %{rc.condLabel}"
    labelLine rc.endLabel
    -- Body temps don't escape; the accumulator's final value does. Restore
    -- is LIFO: each region carries its own entry snapshot.
    let acc ← fresh
    line s!"{acc} = load {llTy rc.accTy}, ptr {rc.accPtr}, align 8"
    let newVals := rc.savedTempVals.insert accTemp ⟨acc, rc.accTy⟩
    modify fun s => { s with reduces := s.reduces.pop, tempVals := newVals }
  | "RoutedSumBegin", .array dst =>
    if ( ← get).reduces.any (fun rc => rc.id == instr.loopId) ||
        ( ← get).routeds.any (fun rc => rc.id == instr.loopId) then
      fail s!"EmitLlvm: RoutedSumBegin id={instr.loopId} collides with an open ancestor region"
    if instr.loopCount == 0 || instr.routedOutputCount == 0 ||
        instr.routedRoutes.isEmpty then
      fail "EmitLlvm: RoutedSumBegin has empty structural metadata"
    if instr.routedRoutes.size % instr.loopCount != 0 then
      fail "EmitLlvm: RoutedSumBegin route count is not divisible by capacity"
    let fanout := instr.routedRoutes.size / instr.loopCount
    let bound ← match instr.args[0]? with
      | none => pure (toString instr.loopCount)
      | some o =>
        let v ← resolveOperand o
        let iv := (← coerce v .int).ref
        let neg ← fresh; line s!"{neg} = icmp slt i64 {iv}, 0"
        let lo ← fresh; line s!"{lo} = select i1 {neg}, i64 0, i64 {iv}"
        let over ← fresh; line s!"{over} = icmp sgt i64 {lo}, {instr.loopCount}"
        let cl ← fresh; line s!"{cl} = select i1 {over}, i64 {instr.loopCount}, i64 {lo}"
        pure cl
    let dstPtr ← loadArrayPtr dst
    for output in [0:instr.routedOutputCount] do
      let ep ← fresh
      line s!"{ep} = getelementptr inbounds i64, ptr {dstPtr}, i64 {output}"
      line s!"store i64 0, ptr {ep}, align 8"
    let n ← freshId
    let routeGlobal := s!"@routed_routes_{n}"
    let routeValues := String.intercalate ", " <|
      instr.routedRoutes.toList.map fun route =>
        s!"i64 {match route with | none => -1 | some output => Int.ofNat output}"
    let globalLine :=
      s!"{routeGlobal} = private constant [{instr.routedRoutes.size} x i64] [{routeValues}]"
    modify fun s => { s with globals := s.globals.push globalLine }
    let idxPtr ← fresh
    line s!"{idxPtr} = alloca i64, align 8"
    line s!"store i64 0, ptr {idxPtr}, align 8"
    let condbb := s!"rs_cond_{n}"; let bodybb := s!"rs_body_{n}";
    let endbb := s!"rs_end_{n}"
    line s!"br label %{condbb}"
    labelLine condbb
    let k ← fresh; line s!"{k} = load i64, ptr {idxPtr}, align 8"
    let c ← fresh; line s!"{c} = icmp ult i64 {k}, {bound}"
    line s!"br i1 {c}, label %{bodybb}, label %{endbb}"
    labelLine bodybb
    modify fun s => { s with routeds := s.routeds.push {
      id := instr.loopId, dst, dstPtr, routeGlobal,
      routeCount := instr.routedRoutes.size, fanout, idxPtr,
      condLabel := condbb, endLabel := endbb, savedTempVals := s.tempVals } }
  | "RoutedSumYield", .array dst =>
    let some rc := (← get).routeds.back?
      | fail "EmitLlvm: RoutedSumYield without an open routed region"
    if rc.dst != dst || instr.args.size != rc.fanout then
      fail "EmitLlvm: RoutedSumYield shape does not match its open region"
    let item ← fresh; line s!"{item} = load i64, ptr {rc.idxPtr}, align 8"
    for emitPos in [0:instr.args.size] do
      let recordBase ← fresh; line s!"{recordBase} = mul i64 {item}, {rc.fanout}"
      let record ← fresh; line s!"{record} = add i64 {recordBase}, {emitPos}"
      let routePtr ← fresh
      line s!"{routePtr} = getelementptr inbounds [{rc.routeCount} x i64], ptr {rc.routeGlobal}, i64 0, i64 {record}"
      let target ← fresh; line s!"{target} = load i64, ptr {routePtr}, align 8"
      let active ← fresh; line s!"{active} = icmp sge i64 {target}, 0"
      let branchId ← freshId
      let applybb := s!"rs_apply_{branchId}"; let contbb := s!"rs_cont_{branchId}"
      line s!"br i1 {active}, label %{applybb}, label %{contbb}"
      labelLine applybb
      let contribution ← resolveF64 instr.args[emitPos]!
      let ep ← fresh; line s!"{ep} = getelementptr inbounds i64, ptr {rc.dstPtr}, i64 {target}"
      let raw ← fresh; line s!"{raw} = load i64, ptr {ep}, align 8"
      let current ← fresh; line s!"{current} = bitcast i64 {raw} to double"
      let next ← fresh; line s!"{next} = fadd double {current}, {contribution}"
      let bits ← fresh; line s!"{bits} = bitcast double {next} to i64"
      line s!"store i64 {bits}, ptr {ep}, align 8"
      line s!"br label %{contbb}"
      labelLine contbb
  | "RoutedSumEnd", .array dst =>
    let some rc := (← get).routeds.back?
      | fail "EmitLlvm: RoutedSumEnd without an open routed region"
    if rc.dst != dst then
      fail "EmitLlvm: RoutedSumEnd destination does not match its open region"
    let k ← fresh; line s!"{k} = load i64, ptr {rc.idxPtr}, align 8"
    let k1 ← fresh; line s!"{k1} = add i64 {k}, 1"
    line s!"store i64 {k1}, ptr {rc.idxPtr}, align 8"
    line s!"br label %{rc.condLabel}"
    labelLine rc.endLabel
    modify fun s => { s with routeds := s.routeds.pop, tempVals := rc.savedTempVals }
  | "WriteSlot", .moduleSlot idx =>
    let v ← match instr.args[0]? with | some o => resolveOperand o | none => fail "WriteSlot missing arg"
    storeSlotF64 idx (← coerceF64 v)
  | "SmoothParam", _ =>
    fail "EmitLlvm: SmoothParam is dead on the session path (params are slots)"
  | "Pack", .array dst => emitPack dst instr.args
  | "Index", .temp dst => emitIndex dst instr.args
  | "SetElement", .array _ => emitSetElement instr.args
  | _, .temp slot =>
    -- scalar op → temp. Coerce to the declared result type before
    -- storing and record THAT as the temp's type (mirrors the C++ tail).
    let args ← instr.args.mapM resolveOperand
    let result ← emitOp instr.tag instr.resultType args
    let coerced ← coerce result instr.resultType
    storeTempTyped slot coerced
  | _, .moduleSlot idx =>
    let args ← instr.args.mapM resolveOperand
    let result ← emitOp instr.tag instr.resultType args
    storeSlotF64 idx (← coerceF64 result)
  | _, .array dst => emitElementwise instr dst
  | tag, .sessionArray _ =>
    fail s!"EmitLlvm: sessionArray dst for '{tag}' must be remapped before emit"

-- ─────────────────────────────────────────────────────────────
-- Fractal per-instance walk (mirrors EmitCtx::emit_kernel_block)
-- ─────────────────────────────────────────────────────────────

-- CF-only: no register writebacks — there is no per-sample state.
def emitKernelBlock (inst : InstanceFunction) : M Unit := do
  -- preamble → per-child {pre_input, child} → body
  for i in inst.preambleInstructions do emitInstr i
  for _h : child in inst.children do
    for i in child.preInputInstructions do emitInstr i
    emitKernelBlock child
  for i in inst.instructions do emitInstr i
termination_by sizeOf inst
decreasing_by exact Tropical.Plan.InstanceFunction.sizeOf_lt_of_mem_children _h

-- ─────────────────────────────────────────────────────────────
-- Sinks (mirrors the loop-body sink mix)
-- ─────────────────────────────────────────────────────────────

def emitSinks (sinks : Array SinkSpec) (outputChannelCount : Nat) : M Unit := do
  -- A sparse target image is semantically zero in every missing channel.  Do
  -- not rely on a particular host to clear reused wasm/native output storage.
  for target in [0:outputChannelCount] do
    unless sinks.any (fun sink => sink.target == target) do
      let outputIdx ←
        if outputChannelCount == 1 then
          pure "%s"
        else do
          let frameBase ← fresh
          line s!"{frameBase} = mul i64 %s, {outputChannelCount}"
          let targetIdx ← fresh
          line s!"{targetIdx} = add i64 {frameBase}, {target}"
          pure targetIdx
      let g ← fresh
      line s!"{g} = getelementptr inbounds double, ptr %out, i64 {outputIdx}"
      line s!"store double {zeroF}, ptr {g}, align 8"
  for sink in sinks do
    let mut acc := zeroF
    for slotIdx in sink.inputs do
      let v ← loadSlotF64 slotIdx
      let r ← fresh; line s!"{r} = fadd double {acc}, {v}"; acc := r
    let scaled ← fresh
    line s!"{scaled} = fmul double {acc}, {f64Lit sink.gain.toFloat}"
    let outputIdx ←
      if outputChannelCount == 1 && sink.target == 0 then
        pure "%s"
      else do
        let frameBase ← fresh
        line s!"{frameBase} = mul i64 %s, {outputChannelCount}"
        let targetIdx ← fresh
        line s!"{targetIdx} = add i64 {frameBase}, {sink.target}"
        pure targetIdx
    let g ← fresh
    line s!"{g} = getelementptr inbounds double, ptr %out, i64 {outputIdx}"
    line s!"store double {scaled}, ptr {g}, align 8"

private def validateSinkTargets (sinks : Array SinkSpec)
    (outputChannelCount : Nat) : Except String Unit := do
  if outputChannelCount == 0 || outputChannelCount > 64 then
    throw s!"EmitLlvm: output channel count {outputChannelCount} is outside [1,64]"
  for i in [0:sinks.size] do
    let target := sinks[i]!.target
    if target >= outputChannelCount then
      throw s!"EmitLlvm: sink target {target} is outside output channel count {outputChannelCount}"
    for j in [0:i] do
      if sinks[j]!.target == target then
        throw s!"EmitLlvm: duplicate sink target {target}"

-- ─────────────────────────────────────────────────────────────
-- Module entry point
-- ─────────────────────────────────────────────────────────────

private def intrinsicDecls : String :=
  "declare double @llvm.sqrt.f64(double)\n" ++
  "declare double @llvm.floor.f64(double)\n" ++
  "declare double @llvm.ceil.f64(double)\n" ++
  "declare double @llvm.round.f64(double)\n" ++
  "declare double @llvm.fabs.f64(double)\n"

/-- A counted loop storing zero into `n` i64 cells at `ptr`. Explicit IR
    rather than `llvm.memset`, so the wasm32 lowering never needs a libc
    import. -/
private def zeroCells (ptr : String) (n : Nat) : M Unit := do
  if n == 0 then return
  let k ← freshId
  let condbb := s!"zero_cond_{k}"; let bodybb := s!"zero_body_{k}"; let endbb := s!"zero_end_{k}"
  let pred := (← get).curBlock
  line s!"br label %{condbb}"
  labelLine condbb
  let i ← fresh
  let inext := s!"{i}_next"
  line s!"{i} = phi i64 [ 0, %{pred} ], [ {inext}, %{bodybb} ]"
  let c ← fresh; line s!"{c} = icmp ult i64 {i}, {n}"
  line s!"br i1 {c}, label %{bodybb}, label %{endbb}"
  labelLine bodybb
  let g ← fresh; line s!"{g} = getelementptr inbounds i64, ptr {ptr}, i64 {i}"
  line s!"store i64 0, ptr {g}, align 8"
  line s!"{inext} = add i64 {i}, 1"
  line s!"br label %{condbb}"
  labelLine endbb

private def slotDefaultFloat : Json → Float
  | .num n => n.toFloat
  | _ => 0.0

/-- Emit the full LLVM module text for a plan's kernel at the port: the
    port function, the argv thunk, and the intrinsic declarations. The
    stage-0 split's coefficient plan, when present, becomes the prologue. -/
def emitKernel (split : Tropical.Ir.Stage0.Split) (frames : Nat) :
    Except String String := do
  let plan := split.audio
  let outputChannelCount := plan.outputChannelCount
  validateSinkTargets plan.sinks outputChannelCount
  if frames == 0 then throw "EmitLlvm: frames must be positive"
  let paramSlots := Tropical.Kernel.paramSlotIndices plan
  let slotCount := max plan.slotCount 1
  let registerCount := max plan.registerCount 1
  let sizes := plan.arraySlotSizes
  let body : M Unit := do
    -- Internal storage: the module slot vector, the SSA scratch and one
    -- backing store per array slot, all per call.
    line s!"%slots = alloca double, i64 {slotCount}, align 8"
    line s!"%temps = alloca i64, i64 {registerCount}, align 8"
    for i in [0:sizes.size] do
      line s!"%arr{i} = alloca i64, i64 {max (sizes[i]?.getD 0) 1}, align 8"
    zeroCells "%temps" registerCount
    zeroCells "%slots" slotCount
    for i in [0:sizes.size] do
      zeroCells s!"%arr{i}" (sizes[i]?.getD 0)
    -- Slots start at zero; a param's value is its input, never a literal, so
    -- the IR is knob-invariant (one cached kernel serves every default). A
    -- non-param slot with a nonzero default — none on the session path, but
    -- a plan from a file could carry one — is seeded as the host used to.
    for i in [0:plan.slotCount] do
      if !(paramSlots.contains i) then
        let v := slotDefaultFloat (plan.slotDefaults[i]?.getD (Json.num 0))
        if v != 0.0 then storeSlotF64 i (f64Lit v)
    for j in [0:paramSlots.size] do
      let v ← fresh; line s!"{v} = load double, ptr %in_{j + 1}, align 8"
      storeSlotF64 paramSlots[j]! v
    -- Stage-0 prologue: the coefficient plan's instructions, once per call.
    -- `%current_idx` is unreachable from τ-independent code; defined so a
    -- stray tick read is zero, as the one-sample host call made it.
    line "%cidx0 = add i64 0, 0"
    if let some coeff := split.coeff? then
      for inst in coeff.instanceFunctions do
        emitKernelBlock inst
    -- Temps are per-kernel SSA scratch; the audio plan's reads see only its
    -- own definitions (boundary values crossed through slots at the split).
    modify fun st => { st with tempVals := {}, reduces := #[], routeds := #[] }
    let pred := (← get).curBlock
    line "br label %loop_cond"
    labelLine "loop_cond"
    line s!"%s = phi i64 [ 0, %{pred} ], [ %s_next, %LOOP_LAST ]"
    line s!"%loopcond = icmp ult i64 %s, {frames}"
    line "br i1 %loopcond, label %loop_body, label %loop_end"
    labelLine "loop_body"
    line "%tau_ptr = getelementptr inbounds i64, ptr %tau, i64 %s"
    line "%tau_s = load i64, ptr %tau_ptr, align 8"
    line "%current_idx = ashr i64 %tau_s, 32"
    for inst in plan.instanceFunctions do
      emitKernelBlock inst
    emitSinks plan.sinks outputChannelCount
  let init : St := { sources := plan.sources, sampleRate := plan.sampleRate.toFloat,
                     arraySizes := sizes, arraysInline := true }
  -- Inside the prologue `%current_idx` is not yet defined (the loop defines
  -- it); the prologue's stray tick reads are renamed to `%cidx0`.
  match body.run init with
  | .error e _ => .error e
  | .ok _ st =>
    let lastBlock := st.curBlock
    let lines := st.lines.map fun l => l.replace "%LOOP_LAST" s!"%{lastBlock}"
    -- Prologue lines precede `loop_cond:`; rewrite their tick reads.
    let (prologue, rest) := Id.run do
      let mut pro : Array String := #[]
      let mut rest : Array String := #[]
      let mut inLoop := false
      for l in lines do
        if l == "loop_cond:" then inLoop := true
        if inLoop then rest := rest.push l
        else pro := pro.push (l.replace "%current_idx" "%cidx0")
      return (pro, rest)
    let bodyText := String.intercalate "\n" (prologue ++ rest).toList
    let globalsText := String.intercalate "\n" st.globals.toList
    let inputArgs := String.intercalate ", "
      ((List.range paramSlots.size).map fun j => s!"ptr noalias nocapture readonly %in_{j + 1}")
    let portArgs := "ptr noalias nocapture readonly %tau" ++
      (if paramSlots.isEmpty then "" else ", " ++ inputArgs) ++
      ", ptr noalias nocapture %out"
    let header :=
      s!"define void @tropical_kernel({portArgs}) \{\n" ++
      "entry:\n"
    let tail :=
      "\n  %s_next = add i64 %s, 1\n" ++
      "  br label %loop_cond\n\n" ++
      "loop_end:\n" ++
      "  ret void\n" ++
      "}\n\n"
    -- The argv thunk: argv[0] = tau, argv[1..k] = params, argv[k+1] = out.
    let argc := paramSlots.size + 2
    let thunkLoads := String.intercalate "" ((List.range argc).map fun i =>
      s!"  %ap{i} = getelementptr inbounds ptr, ptr %argv, i64 {i}\n" ++
      s!"  %a{i} = load ptr, ptr %ap{i}, align 8\n")
    let callArgs := String.intercalate ", " ((List.range argc).map fun i => s!"ptr %a{i}")
    let thunk :=
      "define void @tropical_kernel_argv(ptr %argv) {\n" ++
      "entry:\n" ++ thunkLoads ++
      s!"  call void @tropical_kernel({callArgs})\n" ++
      "  ret void\n" ++
      "}\n\n"
    .ok (globalsText ++ (if globalsText.isEmpty then "" else "\n\n") ++
      header ++ bodyText ++ tail ++ thunk ++ intrinsicDecls)

/-- The Metal adapter's CPU pre-pass: the coefficient plan over host
    storage, one evaluation per call. -/
def emitStage0 (coeff : FlatPlan) : Except String String := do
  let body : M Unit := do
    line "%current_idx = add i64 0, 0"
    for inst in coeff.instanceFunctions do
      emitKernelBlock inst
  let init : St := { sources := coeff.sources, sampleRate := coeff.sampleRate.toFloat,
                     arraySizes := coeff.arraySlotSizes, arraysInline := false }
  match body.run init with
  | .error e _ => .error e
  | .ok _ st =>
    let bodyText := String.intercalate "\n" st.lines.toList
    let globalsText := String.intercalate "\n" st.globals.toList
    .ok (globalsText ++ (if globalsText.isEmpty then "" else "\n\n") ++
      "define void @tropical_stage0(ptr %slots, ptr %arrays, ptr %array_sizes, ptr %temps) {\n" ++
      "entry:\n" ++ bodyText ++ "\n  ret void\n}\n\n" ++ intrinsicDecls)

end Tropical.Ir.EmitLlvm
