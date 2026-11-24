import Tropical.Semantics.Plan
import Tropical.Semantics.Scalar

/-!
# A symbolic carrier for the Plan reference semantics (experimental)

`Algebra SVal` whose values are either a constant of the reference algebra
(`CVal`) or a bit-vector term over free inputs. Running the existing Plan
semantics (`initialPlanState`, `execPlanFunctions`) over it with the param
slots and the clock left free produces, for an integer kernel, the kernel's
specification as a term: the plan's own meaning, not a transcription.

* An operation whose arguments are all constants is `Scalar.applyOp`, i.e.
  `ConstFold.foldOp` itself, so constant work (the float chain that computes a
  rate-dependent constant, say) is the reference semantics verbatim.
* An operation with a symbolic argument becomes an SMT-LIB bit-vector term.
  The claim that each such term means what `foldOp` computes on the
  corresponding `Int` values (wrap32/wrap64 of the ring ops, `>>>` floor
  shift by the amount modulo the width, truncated division with the total
  edges, signed comparison after sign extension, the low-word and
  sign-extending conversions) is the one per-op correspondence this carrier
  adds to the trusted base (`design/` has no ledger entry: this is a
  research carrier).
* Anything else with a symbolic argument (float arithmetic, transcendental
  ops, symbolic array indices, symbolic bank counts) is refused, by name.

Terms are trees; the printer shares structurally equal subterms with one
`define-fun` each. Printing visits a shared subterm once per occurrence, which
is fine for the stdlib kernels and not a general solution.
-/

namespace Tropical.Semantics.Symbolic

open Tropical.Ir
open Tropical.Plan (PlanOp ScalarType FlatPlan SourceKind)

/-- A bit-vector or boolean term. `sort` is `int32` (32 bits), `int` (64 bits)
    or `bool`; a float term is never built. `head` is the SMT-LIB head as
    printed, e.g. `bvadd` or `(_ extract 31 0)`. -/
inductive Term where
  | lit (sort : ScalarType) (bits : Nat)
  | var (name : String) (sort : ScalarType)
  | app (head : String) (sort : ScalarType) (args : Array Term)
deriving Inhabited, Repr

def Term.sort : Term → ScalarType
  | .lit s _ | .var _ s | .app _ s _ => s

inductive SVal where
  | c (v : CVal)
  | s (t : Term)
deriving Inhabited

def SVal.ty : SVal → ScalarType
  | .c v => v.ty
  | .s t => t.sort

private def width : ScalarType → Option Nat
  | .int32 => some 32 | .int => some 64 | _ => none

private def refuse (op detail : String) : Outcome β := .error { operation := s!"symbolic.{op}", detail }

def constTerm : CVal → Option Term
  | .i32 n => some (.lit .int32 (toU32 n))
  | .i n => some (.lit .int (toU64 n))
  | .b v => some (.lit .bool (if v then 1 else 0))
  | .f _ => none

private def zeroOf (s : ScalarType) : Term := .lit s 0

/-- `CVal.coerce?` lifted to terms (total integer/bool conversions only). -/
def toSortT (target : ScalarType) : SVal → Option Term
  | .c v => (v.coerce? target).bind constTerm
  | .s t =>
    if t.sort == target then some t else
    match t.sort, target with
    | .int, .int32 => some (.app "(_ extract 31 0)" .int32 #[t])
    | .int32, .int => some (.app "(_ sign_extend 32)" .int #[t])
    | .bool, .int32 | .bool, .int =>
        some (.app "ite" target #[t, .lit target 1, .lit target 0])
    | .int32, .bool | .int, .bool =>
        some (.app "not" .bool #[.app "=" .bool #[t, zeroOf t.sort]])
    | _, _ => none

private def args2 (op : String) (s : ScalarType) (a : Array SVal) : Outcome (Term × Term) :=
  match a[0]?.bind (toSortT s), a[1]?.bind (toSortT s) with
  | some x, some y => .ok (x, y)
  | _, _ => refuse op s!"operands have no {s.wire} view"

private def intSortOf (sort : ScalarType) : ScalarType := if sort == .int32 then .int32 else .int

/-- The shift amount `sh % w` (least nonnegative residue, as `foldOp`). -/
private def shiftAmount (op : String) (s : ScalarType) (w : Nat) (a : SVal) : Outcome Term :=
  match a with
  | .c v =>
    match v.coerce? s with
    | some (.i32 n) | some (.i n) => .ok (.lit s (n % (w : Int)).toNat)
    | _ => refuse op "shift amount has no integer view"
  | .s _ =>
    match toSortT s a with
    | some t => .ok (.app "bvand" s #[t, .lit s (w - 1)])
    | none => refuse op "shift amount has no integer view"

/-- `op` at `sort` with at least one symbolic argument. -/
def symOp (op : PlanOp) (sort : ScalarType) (a : Array SVal) : Outcome SVal := do
  let name := op.name
  let ring (head : String) : Outcome SVal := do
    let some _ := width sort | refuse name s!"{sort.wire} arithmetic on a symbolic value"
    let (x, y) ← args2 name sort a
    pure (.s (.app head sort #[x, y]))
  let quot (head : String) : Outcome SVal := do
    let some _ := width sort | refuse name s!"{sort.wire} division on a symbolic value"
    let (x, y) ← args2 name sort a
    pure (.s (.app "ite" sort #[.app "=" .bool #[y, zeroOf sort], zeroOf sort, .app head sort #[x, y]]))
  let eitherInt := a.any fun v => v.ty == .int || v.ty == .int32
  let cmp (head : String) : Outcome SVal := do
    if !eitherInt then refuse name "symbolic float comparison"
    let (x, y) ← args2 name .int a
    pure (.s (.app head .bool #[x, y]))
  let bitw (head : String) : Outcome SVal := do
    let s := intSortOf sort
    let (x, y) ← args2 name s a
    pure (.s (.app head s #[x, y]))
  let shift (head : String) : Outcome SVal := do
    let s := intSortOf sort
    let w := if s == .int32 then 32 else 64
    let some x := a[0]?.bind (toSortT s) | refuse name "shiftee has no integer view"
    let some amt := a[1]? | refuse name "missing shift amount"
    pure (.s (.app head s #[x, ← shiftAmount name s w amt]))
  let unaryInt (f : Term → Term) : Outcome SVal := do
    let some _ := width sort | refuse name s!"{sort.wire} on a symbolic value"
    let some x := a[0]?.bind (toSortT sort) | refuse name "operand has no integer view"
    pure (.s (f x))
  let conv (target : ScalarType) : Outcome SVal :=
    match a[0]?.bind (toSortT target) with
    | some t => .ok (.s t)
    | none => refuse name s!"no symbolic conversion to {target.wire}"
  let truth (v : SVal) : Outcome Term :=
    match v with
    | .s t =>
      if t.sort == .bool then .ok t
      else if (width t.sort).isSome then .ok (.app "not" .bool #[.app "=" .bool #[t, zeroOf t.sort]])
      else refuse name "symbolic float truthiness"
    | .c _ => refuse name "unreachable: constant truth"
  match op with
  | .add => ring "bvadd"
  | .sub => ring "bvsub"
  | .mul => ring "bvmul"
  | .div | .floorDiv => quot "bvsdiv"
  | .mod => quot "bvsrem"
  | .less => cmp "bvslt"
  | .lessEq => cmp "bvsle"
  | .greater => cmp "bvsgt"
  | .greaterEq => cmp "bvsge"
  | .equal => cmp "="
  | .notEqual => cmp "distinct"
  | .and | .or =>
    let (x, y) ← args2 name .bool a
    pure (.s (.app (if op == .and then "and" else "or") .bool #[x, y]))
  | .bitAnd => bitw "bvand"
  | .bitOr => bitw "bvor"
  | .bitXor => bitw "bvxor"
  | .lshift => shift "bvshl"
  | .rshift => shift "bvashr"
  | .neg => unaryInt fun x => .app "bvneg" sort #[x]
  | .abs => unaryInt fun x =>
      Term.app "ite" sort #[Term.app "bvslt" .bool #[x, zeroOf sort], Term.app "bvneg" sort #[x], x]
  | .bitNot =>
    let s := intSortOf sort
    let some x := a[0]?.bind (toSortT s) | refuse name "operand has no integer view"
    pure (.s (.app "bvnot" s #[x]))
  | .not =>
    let some v := a[0]? | refuse name "missing operand"
    pure (.s (.app "not" .bool #[← truth v]))
  | .toInt => conv .int
  | .toInt32 => conv .int32
  | .toBool => conv .bool
  | .clamp =>
    let some _ := width sort | refuse name s!"{sort.wire} clamp on a symbolic value"
    match a[0]?.bind (toSortT sort), a[1]?.bind (toSortT sort), a[2]?.bind (toSortT sort) with
    | some v, some lo, some hi =>
      let lc := Term.app "ite" sort #[.app "bvsgt" .bool #[v, lo], v, lo]
      pure (.s (.app "ite" sort #[.app "bvslt" .bool #[lc, hi], lc, hi]))
    | _, _, _ => refuse name "operands have no integer view"
  | .select =>
    let some cnd := a[0]? | refuse name "missing condition"
    match cnd with
    | .c v =>
      -- a constant condition picks its branch exactly as `foldOp` does
      let pick := match v with
        | .b b => b | .i n => n != 0 | .i32 n => n != 0 | .f x => x != 0.0
      let some branch := (if pick then a[1]? else a[2]?) | refuse name "missing branch"
      match branch with
      | .c w => match w.coerce? sort with
        | some w => pure (.c w)
        | none => refuse name "no conversion"
      | .s _ =>
          match toSortT sort branch with
          | some t => .ok (.s t)
          | none => refuse name "no conversion"
    | .s _ =>
      let c ← truth cnd
      match a[1]?.bind (toSortT sort), a[2]?.bind (toSortT sort) with
      | some x, some y => pure (.s (.app "ite" sort #[c, x, y]))
      | _, _ => refuse name s!"{sort.wire} select branches on a symbolic condition"
  | _ => refuse name "no symbolic semantics for this op"

private def scalars (op : String) (a : Array (Value SVal)) : Outcome (Array SVal) :=
  a.mapM fun
    | .scalar v => .ok v
    | .array _ => refuse op "array operand to a scalar op"

/-- `op` at `sort`: all-constant arguments go through the reference algebra
    (`Scalar.applyOp` = `foldOp`), otherwise `symOp`. -/
def applyOp (op : PlanOp) (sort : ScalarType) (args : Array (Value SVal)) :
    Outcome (Value SVal) := do
  let xs ← scalars op.name args
  let consts := xs.filterMap fun | .c v => some v | .s _ => none
  if consts.size == xs.size then
    match Scalar.applyOp op sort (consts.map .scalar) with
    | .ok (.scalar v) => pure (.scalar (.c v))
    | .ok (.array _) => refuse op.name "array result"
    | .error e => .error e
  else
    pure (.scalar (← symOp op sort xs))

private def planOpOfBinaryTag : BinaryOpTag → PlanOp
  | .add => .add | .sub => .sub | .mul => .mul | .div => .div | .mod => .mod
  | .lt => .less | .lte => .lessEq | .gt => .greater | .gte => .greaterEq
  | .eq => .equal | .neq => .notEqual | .and => .and | .or => .or
  | .bitAnd => .bitAnd | .bitOr => .bitOr | .bitXor => .bitXor
  | .lshift => .lshift | .rshift => .rshift
  | .floorDiv => .floorDiv | .ldexp => .ldexp

private def planOpOfUnaryTag : UnaryOpTag → PlanOp
  | .neg => .neg | .not => .not | .bitNot => .bitNot
  | .sqrt => .sqrt | .abs => .abs | .floor => .floor | .ceil => .ceil
  | .round => .round | .floatExponent => .floatExponent
  | .toInt => .toInt | .toBool => .toBool | .toFloat => .toFloat
  | .toInt32 => .toInt32

private def sortOf : Value SVal → ScalarType
  | .scalar v => v.ty
  | .array _ => .float

private def inferSort (op : PlanOp) (args : Array (Value SVal)) : ScalarType :=
  op.resultType (args.map sortOf)

private def constInt (operation : String) : Value SVal → Except Refusal Int
  | .scalar (.c v) =>
    match v.coerce? .int with
    | some (.i n) => .ok n
    | _ => .error { operation, detail := "no integer view" }
  | _ => .error { operation, detail := "symbolic or array value where a constant integer is required" }

def toSort (sort : ScalarType) : Value SVal → Result SVal
  | .scalar (.c v) =>
    match v.coerce? sort with
    | some w => .ok (.scalar (.c w))
    | none => refuse "atSort" s!"no conversion to {sort.wire}"
  | .scalar v@(.s _) =>
    match toSortT sort v with
    | some t => .ok (.scalar (.s t))
    | none => refuse "atSort" s!"no symbolic conversion to {sort.wire}"
  | .array _ => refuse "atSort" "an array has no scalar sort"

def algebra : Algebra SVal where
  literal := fun n => .ok (.scalar (.c (.f n.toFloat)))
  unary := fun tag v => let op := planOpOfUnaryTag tag; applyOp op (inferSort op #[v]) #[v]
  binary := fun tag a b =>
    let op := planOpOfBinaryTag tag; applyOp op (inferSort op #[a, b]) #[a, b]
  clamp := fun v lo hi => applyOp .clamp (inferSort .clamp #[v, lo, hi]) #[v, lo, hi]
  select := fun c t e => applyOp .select (inferSort .select #[c, t, e]) #[c, t, e]
  index := fun array index => (do
    let .array items := array | refuse "index" "not an array"
    let i : Int ← constInt "symbolic.index" index
    if 0 ≤ i then
      match items[i.toNat]? with
      | some v => .ok v
      | none => .ok (.scalar (.c (.f 0.0)))
    else .ok (.scalar (.c (.f 0.0))) : Outcome (Value SVal))
  loopIndex := fun n => .ok (.scalar (.c (.i n)))
  dynamicCount := constInt "symbolic.count"
  zero := .ok (.scalar (.c (.f 0.0)))
  sortedLiteral := fun sort n => .ok (.scalar (.c (Scalar.literalAt sort n)))
  sortedUnary := fun sort tag v => applyOp (planOpOfUnaryTag tag) sort #[v]
  sortedBinary := fun sort tag a b => applyOp (planOpOfBinaryTag tag) sort #[a, b]
  sortedClamp := fun sort v lo hi => applyOp .clamp sort #[v, lo, hi]
  sortedSelect := fun sort c t e => applyOp .select sort #[c, t, e]
  atSort := toSort

/-! ## Running a plan symbolically and printing SMT-LIB -/

def smtSort : ScalarType → String
  | .int32 => "(_ BitVec 32)" | .int => "(_ BitVec 64)" | .bool => "Bool" | .float => "Float64"

def smtName (n : String) : String := s!"|{n}|"

private def litText : ScalarType → Nat → String
  | .bool, b => if b == 0 then "false" else "true"
  | s, n => s!"(_ bv{n} {if s == .int32 then 32 else 64})"

structure Printer where
  defs : Array String := #[]
  memo : Std.HashMap String String := {}
  vars : Array (String × ScalarType) := #[]

/-- Print a term, one `define-fun` per distinct application. -/
partial def Printer.emit (t : Term) : StateM Printer String := do
  match t with
  | .lit s n => pure (litText s n)
  | .var n s =>
    modify fun p => if p.vars.any (·.1 == n) then p else { p with vars := p.vars.push (n, s) }
    pure (smtName n)
  | .app head s args =>
    let names ← args.mapM Printer.emit
    let body := s!"({head} {" ".intercalate names.toList})"
    match (← get).memo[body]? with
    | some n => pure n
    | none =>
      let n := s!"e{(← get).defs.size}"
      modify fun p => { p with
        defs := p.defs.push s!"(define-fun {n} () {smtSort s} {body})"
        memo := p.memo.insert body n }
      pure n

/-- The specification of one frame: run the plan with `param:` slots free
    (named after the slot, at its declared kind), the clock free (`tau`, the
    int rail), the rate at the plan's rate, other slots at their defaults
    (as the reference render does), and read each sink's single input slot.
    Returns `(output channel, term)` pairs. -/
def specOfPlan (plan : FlatPlan) : Except String (Array (Nat × Term)) := do
  let slotSort (i : Nat) : ScalarType := ((plan.slotType? i).map (·.kind)).getD .float
  let slots : Array (Value SVal) := (List.range plan.slotCount).toArray.map fun i =>
    let name := plan.slotNames[i]?.getD s!"slot{i}"
    if name.startsWith "param:" then .scalar (.s (.var name (slotSort i)))
    else match plan.slotDefaults[i]? with
      | some (.num v) => .scalar (.c (.f v.toFloat))
      | _ => .scalar (.c (.f 0.0))
  let arrays : Array (Array (Value SVal)) :=
    plan.arraySlotSizes.map fun size => Array.replicate size (.scalar (.c (.f 0.0)))
  let sources : Array (Value SVal) := plan.sources.map fun
    | .tau => .scalar (.s (.var "tau" .int))
    | .rate => .scalar (.c (.f plan.sampleRate.toFloat))
  let inputs : PlanInputs SVal := { sources, initialSlots := slots, initialArrays := arrays }
  let run : Outcome (PlanState SVal) := do
    let initial ← initialPlanState algebra inputs plan
    execPlanFunctions algebra inputs initial plan
  let state ← match run with
    | .ok s => pure s
    | .error e => throw s!"{e.operation}: {e.detail}"
  plan.sinks.mapM fun sink => do
    let #[slot] := sink.inputs
      | throw s!"sink {sink.target}: {sink.inputs.size} inputs (only a single-slot sink is modeled)"
    if sink.gain.toFloat != 1.0 then throw s!"sink {sink.target}: gain is not 1"
    match state.slots[slot]? with
    | some (.scalar (.s t)) => pure (sink.target, t)
    | some (.scalar (.c v)) =>
      match constTerm v with
      | some t => pure (sink.target, t)
      | none => throw s!"sink {sink.target}: a constant float output"
    | _ => throw s!"sink {sink.target}: slot {slot} is not a scalar"

/-- SMT-LIB text: free inputs declared, each output asserted equal to its
    term (`|out.N|`). -/
def specSmt (plan : FlatPlan) (header : String) : Except String String := do
  let outs ← specOfPlan plan
  let (asserts, p) := (outs.mapM fun ((ch, t) : Nat × Term) => do
      let n ← Printer.emit t
      pure s!"(declare-const |out.{ch}| {smtSort t.sort})\n(assert (= |out.{ch}| {n}))" :
        StateM Printer (Array String)).run {}
  let decls := p.vars.map fun (n, s) => s!"(declare-const {smtName n} {smtSort s})"
  pure ("\n".intercalate (#[s!"; {header}"] ++ decls ++ p.defs ++ asserts).toList ++ "\n")

end Tropical.Semantics.Symbolic
