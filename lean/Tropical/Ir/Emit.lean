import Std.Data.HashMap
import Lean.Data.Json
import Tropical.Ir.Core
import Tropical.Ir.CoreArena
import Tropical.Ir.Staging
import Tropical.Plan

/-!
# Emit — `ExprId → FlatProgram`

Line-faithful port of `compiler/ir/emit_resolved.ts`'s `Emitter` over
the Core sub-IR (total matches — the strata-dropped constructors are
unrepresentable here, where the TS emitter threw on them).

## CSE identity discipline

Expressions are lowered into a hash-consed DAG (`CoreArena`) before they
are compiled: each node is interned to an `ExprId`, so equal subtrees are
one node and CSE keys directly on `(ExprId, expectedKey)` — O(1) per node,
no per-node structural-id recomputation (issue #190). Interning is sound for
the pure ops here (merging same-structure nodes never changes a value), and
because the goldens hash rendered audio (not plan bytes) the DAG walk's
register relabeling is free. This replaced the TS port's recursive
`structuralId` (`op:add|args=i:7`-shaped string keys), whose per-node
O(subtree) recompute blew up on the duplicated subtrees that inlining a
modulated clock produces.

Number rendering for error messages mirrors JS toString for the
integer/plain-decimal range (no 1e±21 exponent forms — out of corpus).
-/

namespace Tropical.Ir.Emit

open Lean (Json JsonNumber)
open Tropical.Ir
open Tropical.Ir.Core
open Tropical.Plan (NOperand DstSlot NInstr)

abbrev ScalarType := Tropical.Plan.ScalarType

-- ─────────────────────────────────────────────────────────────
-- JS-number helpers
-- ─────────────────────────────────────────────────────────────

/-- Strip trailing zeros from the mantissa: the canonical form under
    which two `JsonNumber`s are equal iff their rationals are. -/
def normNum (n : JsonNumber) : Int × Nat :=
  go n.exponent n.mantissa n.exponent
where
  go : Nat → Int → Nat → Int × Nat
    | 0, m, e => (m, e)
    | fuel + 1, m, e =>
      if e == 0 || m % 10 != 0 then (m, e) else go fuel (m / 10) (e - 1)

/-- JS `String(n)` for the integer/plain-decimal range. -/
def jsNumString (n : JsonNumber) : String :=
  let (m, e) := normNum n
  if e == 0 then toString m
  else
    let sign := if m < 0 then "-" else ""
    let digits := toString m.natAbs
    let digits := if digits.length ≤ e
      then String.ofList (List.replicate (e + 1 - digits.length) '0') ++ digits
      else digits
    let point := digits.length - e
    sign ++ digits.take point ++ "." ++ digits.drop point

/-- `Number.isInteger` on the parsed value. -/
def jsIsInteger (n : JsonNumber) : Bool :=
  (normNum n).2 == 0

/-- `val !== 0 && val !== 1` (the bool-narrowing check). -/
def jsIsZeroOrOne (n : JsonNumber) : Bool :=
  let (m, e) := normNum n
  e == 0 && (m == 0 || m == 1)

-- ─────────────────────────────────────────────────────────────
-- Op-tag mappings (BINARY_TAG / UNARY_TAG / TERNARY_TAG values)
-- ─────────────────────────────────────────────────────────────

open Tropical.Plan (PlanOp promoteTypes)

/-- Lift the IR binary tag into the plan-op signature. -/
def planOpOfBinary : BinaryOpTag → PlanOp
  | .add => .add | .sub => .sub | .mul => .mul | .div => .div | .mod => .mod
  | .floorDiv => .floorDiv
  | .lt => .less | .lte => .lessEq | .gt => .greater | .gte => .greaterEq
  | .eq => .equal | .neq => .notEqual
  | .bitAnd => .bitAnd | .bitOr => .bitOr | .bitXor => .bitXor
  | .lshift => .lshift | .rshift => .rshift
  | .and => .and | .or => .or
  | .ldexp => .ldexp

/-- Lift the IR unary tag into the plan-op signature. -/
def planOpOfUnary : UnaryOpTag → PlanOp
  | .neg => .neg | .abs => .abs | .sqrt => .sqrt
  | .floor => .floor | .ceil => .ceil | .round => .round
  | .not => .not | .bitNot => .bitNot
  | .floatExponent => .floatExponent
  | .toInt => .toInt | .toBool => .toBool | .toFloat => .toFloat

-- ─────────────────────────────────────────────────────────────
-- Slot tables
-- ─────────────────────────────────────────────────────────────

structure ArraySlotInfo where
  slot : Nat
  size : Nat
deriving Repr, Inhabited

/-- Port of `EmitSlots`. Index-keyed association arrays stand in for
    the TS Maps (all keys are decl-table positions). `nestedOutputSlots?`
    keeps the TS undefined/empty distinction — the two produce different
    NestedOut error messages. -/
structure EmitSlots where
  paramHandles : Array (Nat × String) := #[]
  paramSlots : Array (Nat × Nat) := #[]
  nestedOutputSlots? : Option (Array (Nat × Array (Nat × Nat))) := none
  nestedInputSlots : Array (Nat × Array (Nat × Nat)) := #[]
  inputSlotOverride : Array (Nat × Nat) := #[]
  inputArraySlots : Array (Nat × ArraySlotInfo) := #[]
  nestedInputArraySlots : Array (Nat × Array (Nat × ArraySlotInfo)) := #[]
  nestedOutputArraySlots : Array (Nat × Array (Nat × ArraySlotInfo)) := #[]
deriving Inhabited

private def lookup {α} (m : Array (Nat × α)) (k : Nat) : Option α :=
  (m.find? (·.1 == k)).map (·.2)

-- ─────────────────────────────────────────────────────────────
-- Emitter state
-- ─────────────────────────────────────────────────────────────

inductive CompileResult where
  | scalar (op : NOperand) (scalarType : ScalarType)
  | array (op : NOperand) (size : Nat) (scalarType : ScalarType)
deriving Repr, Inhabited

def CompileResult.isArray : CompileResult → Bool
  | .scalar .. => false | .array .. => true

def CompileResult.op : CompileResult → NOperand
  | .scalar op _ => op | .array op _ _ => op

def CompileResult.scalarType : CompileResult → ScalarType
  | .scalar _ t => t | .array _ _ t => t

structure EmitSt where
  slots : EmitSlots
  inputPortTypes : Array ScalarType
  nextReg : Nat := 0
  nextArraySlot : Nat := 0
  arraySizes : Array Nat := #[]
  instrs : Array NInstr := #[]
  memo : Std.HashMap String CompileResult := {}
  -- Hash-consed (DAG) form of the expressions emit walks. Equal subtrees are
  -- one `ExprId`, so CSE keys on the id (O(1)) instead of recomputing a
  -- structural id per node (O(subtree)) — see issue #190.
  arena : CoreArena := {}
  -- ── Staging (Phase 1 of the typed stage-0 refactor) ──
  -- The binding context the current block resolves under (rebuilt by
  -- `emitProgram` per pre-input block / body section), the stage of the
  -- node currently being compiled, and the per-instruction stage record
  -- (parallel to `instrs`: every `emit` pushes both).
  stageCtx : Staging.StageCtx := {}
  curStage : Option Stage := none
  instrStages : Array (Option Stage) := #[]
deriving Inhabited

abbrev EmitM := StateT EmitSt (Except String)

/-- Run `act` with `curStage` set (restored after) — instructions it
    emits are attributed to that stage. -/
def withStage {α} (s : Option Stage) (act : EmitM α) : EmitM α := do
  let prev := (← get).curStage
  modify fun st => { st with curStage := s }
  let r ← act
  modify fun st => { st with curStage := prev }
  pure r

/-- Intern a flat `CNode` into the emit arena, returning its (shared) id. The
    post-strata leaves arrive pre-interned (Phase B); this covers only the
    emit's own synthesized nodes (the port-default `0`). -/
def internNode (n : CNode) : EmitM ExprId :=
  modifyGet fun st => let (id, a) := (intern n).run st.arena; (id, { st with arena := a })

/-- The TS operand `kind` strings (error-message rendering). -/
private def operandKind : NOperand → String
  | .const .. => "const" | .input .. => "input" | .reg .. => "reg"
  | .arrayReg _ => "array_reg" | .sessionArrayReg _ => "session_array_reg"
  | .param .. => "param"
  | .source .. => "source" | .slot .. => "slot"
  | .loopIdx _ => "loop_idx"

private def allocReg : EmitM Nat :=
  modifyGet fun s => (s.nextReg, { s with nextReg := s.nextReg + 1 })

private def allocArraySlot (size : Nat) : EmitM Nat :=
  modifyGet fun s => (s.nextArraySlot,
    { s with nextArraySlot := s.nextArraySlot + 1, arraySizes := s.arraySizes.push size })

private def emit (i : NInstr) : EmitM Unit :=
  modify fun s => { s with instrs := s.instrs.push i,
                           instrStages := s.instrStages.push s.curStage }

private def expectedKey : Option ScalarType → String
  | none => ""
  | some t => t.wire

-- ─────────────────────────────────────────────────────────────
-- Terminal check
-- ─────────────────────────────────────────────────────────────

private def resolveNumericLiteralType (n : JsonNumber) (expected : Option ScalarType) :
    EmitM ScalarType := do
  match expected with
  | some .int =>
    unless jsIsInteger n do
      throw s!"Lossy conversion: literal {jsNumString n} cannot narrow to int. Wrap the source in to_int() to narrow explicitly."
    pure .int
  | some .bool =>
    unless jsIsZeroOrOne n do
      throw s!"Lossy conversion: literal {jsNumString n} cannot narrow to bool. Wrap the source in to_bool() to narrow explicitly."
    pure .bool
  | _ => pure .float

private def tryTerminal (e : CNode) (expected : Option ScalarType) :
    EmitM (Option (NOperand × ScalarType)) := do
  let st ← get
  match e with
  | .num n =>
    let t ← resolveNumericLiteralType n expected
    pure (some (.const n t, t))
  | .bool b =>
    pure (some (.const (if b then (1 : Nat) else 0) .bool, .bool))
  | .inputRef i =>
    -- Array-typed input port defers to the uncached path.
    if (lookup st.slots.inputArraySlots i.idx).isSome then
      pure none
    else
      let portT := st.inputPortTypes[i.idx]?.getD .float
      match lookup st.slots.inputSlotOverride i.idx with
      | some slot => pure (some (.slot slot portT, portT))
      | none => pure (some (.input i.idx portT, portT))
  | .paramRef i =>
    match lookup st.slots.paramSlots i.idx with
    | some slot => pure (some (.slot slot .float, .float))
    | none =>
      match lookup st.slots.paramHandles i.idx with
      | some ptr => pure (some (.param ptr .float, .float))
      | none => pure (some (.const (0 : Nat) .float, .float))
  | .sampleRate => pure (some (Tropical.Plan.opRate, .float))
  | .sampleIndex => pure (some (Tropical.Plan.opTick, .int))
  | .loopIdx id => pure (some (Tropical.Plan.NOperand.loopIdx id, .int))
  | .nestedOut inst out => do
    if let some perInstArr := lookup st.slots.nestedOutputArraySlots inst.idx then
      if (lookup perInstArr out.idx).isSome then
        return none
    match st.slots.nestedOutputSlots? with
    | none =>
      throw s!"emit_resolved: NestedOut to instance idx={inst.idx} output idx={out.idx} requires nestedOutputSlots — pass them via EmitSlots when invoking emit on a fractal kernel."
    | some maps =>
      match lookup maps inst.idx with
      | none =>
        throw s!"emit_resolved: NestedOut to instance idx={inst.idx} — no slot map entry. partition_recursive should record every child instance before emitting the parent kernel."
      | some perInst =>
        match lookup perInst out.idx with
        | none =>
          throw s!"emit_resolved: NestedOut to instance idx={inst.idx} output idx={out.idx} — output port not in slot map."
        | some slotRaw => pure (some (.slot slotRaw .float, .float))
  | _ => pure none

-- ─────────────────────────────────────────────────────────────
-- Compile
-- ─────────────────────────────────────────────────────────────

mutual

partial def compileNode (id : ExprId) (expected : Option ScalarType := none) :
    EmitM CompileResult := do
  let some node := (← get).arena.deref id
    | throw s!"emit_resolved: dangling ExprId {id.idx}"
  if let some (op, t) ← tryTerminal node expected then
    return .scalar op t
  let key := s!"{id.idx}:{expectedKey expected}"
  if let some r := (← get).memo.get? key then
    return r
  let st ← get
  let r ← withStage (some (Staging.stageOf st.arena st.stageCtx id))
    (compileNodeUncached node expected)
  modify fun s => { s with memo := s.memo.insert key r }
  return r

private partial def compileNodeUncached (e : CNode) (expected : Option ScalarType) :
    EmitM CompileResult := do
  match e with
  | .arr items => compilePack items expected
  | .inputRef i =>
    match lookup (← get).slots.inputArraySlots i.idx with
    | some info => pure (.array (.sessionArrayReg info.slot) info.size .float)
    | none =>
      throw s!"emit_resolved: inputRef to non-array port idx={i.idx} reached compileNodeUncached unexpectedly"
  | .nestedOut inst out =>
    let perInst := lookup (← get).slots.nestedOutputArraySlots inst.idx
    match perInst.bind (lookup · out.idx) with
    | some info => pure (.array (.sessionArrayReg info.slot) info.size .float)
    | none =>
      throw "emit_resolved: nestedOut to non-array sub-instance output reached compileNodeUncached unexpectedly"
  | .binary tag a b => compileBinary (planOpOfBinary tag) a b expected
  | .unary tag a => compileUnary (planOpOfUnary tag) a expected
  | .clamp a b c => compileTernary .clamp a b c expected
  | .select a b c => compileTernary .select a b c expected
  | .arraySet a b c => compileSetElement a b c
  | .index a b => compileIndex a b
  | .bankSum count tables body dynCount? idxId => compileBankSum count tables body dynCount? idxId
  | .num _ | .bool _ | .paramRef _ | .sampleRate | .sampleIndex | .loopIdx _ =>
    throw "emit_resolved: terminal node reached compileNodeUncached (port bug)"

/-- Unbox a size-1 array to a scalar via Index[0]. -/
private partial def unboxIfUnit (r : CompileResult) : EmitM CompileResult := do
  match r with
  | .array op 1 rt =>
    let dst ← allocReg
    emit (Tropical.Plan.instrIndex dst #[op, .const (0 : Nat) .int] rt)
    pure (.scalar (.reg dst rt) rt)
  | r => pure r

private partial def compilePack (elements : Array ExprId) (expected : Option ScalarType) :
    EmitM CompileResult := do
  let size := elements.size
  let slot ← allocArraySlot size
  let args ← elements.mapM fun el => do
    let r ← compileNode el expected
    pure (if r.isArray then NOperand.const (0 : Nat) .float else r.op)
  emit (Tropical.Plan.instrPack slot args)
  pure (.array (.arrayReg slot) size .float)

private partial def compileBinary (op : PlanOp) (lhs rhs : ExprId)
    (expected : Option ScalarType) : EmitM CompileResult := do
  let propagated := if expected == some .bool then none else expected
  let argExpected : Option ScalarType :=
    if op.isBitwise then some .int
    else if op.isComparison then none
    else propagated
  let l ← compileNode lhs argExpected
  let secondExpected : Option ScalarType :=
    if op.isComparison then
      if l.isArray then some .float
      else if l.scalarType == .bool then none
      else some l.scalarType
    else argExpected
  let r ← compileNode rhs secondExpected
  let l ← unboxIfUnit l
  let r ← unboxIfUnit r
  let rt := op.resultType #[l.scalarType, r.scalarType]
  match l, r with
  | .scalar lop _, .scalar rop _ =>
    let dst ← allocReg
    emit (Tropical.Plan.instrScalar op.name dst #[lop, rop] rt)
    pure (.scalar (.reg dst rt) rt)
  | _, _ =>
    let size := match l with
      | .array _ s _ => s
      | _ => match r with | .array _ s _ => s | _ => 0
    let slot ← allocArraySlot size
    let strides := #[if l.isArray then 1 else 0, if r.isArray then 1 else 0]
    emit (Tropical.Plan.instrArray op.name slot #[l.op, r.op] size strides rt)
    pure (.array (.arrayReg slot) size rt)

private partial def compileUnary (op : PlanOp) (arg : ExprId)
    (expected : Option ScalarType) : EmitM CompileResult := do
  let argExpected : Option ScalarType :=
    if op.isTranscendental then none
    else if op.isComparison then none
    else if op == .bitNot then some .int
    -- `ToInt` narrows its argument *from* float; it must not propagate an
    -- outer `int` expectation onto a float source (e.g. `toInt(f0 * 2.414…)`
    -- where the result is used in an int context). Leave the arg unconstrained.
    else if op == .toInt then none
    else expected
  let a ← compileNode arg argExpected
  let a ← unboxIfUnit a
  let rt := op.resultType #[a.scalarType]
  match a with
  | .scalar aop _ =>
    let dst ← allocReg
    emit (Tropical.Plan.instrScalar op.name dst #[aop] rt)
    pure (.scalar (.reg dst rt) rt)
  | .array aop size _ =>
    let slot ← allocArraySlot size
    emit (Tropical.Plan.instrArray op.name slot #[aop] size #[1] rt)
    pure (.array (.arrayReg slot) size rt)

private partial def compileTernary (op : PlanOp) (n1 n2 n3 : ExprId)
    (expected : Option ScalarType) : EmitM CompileResult := do
  let condExpected : Option ScalarType := if op == .select then some .bool else expected
  let armExpected := if expected == some .bool then none else expected
  let a ← compileNode n1 condExpected
  let b ← compileNode n2 armExpected
  let c ← compileNode n3 armExpected
  let a ← unboxIfUnit a
  let b ← unboxIfUnit b
  let c ← unboxIfUnit c
  let rt := op.resultType #[a.scalarType, b.scalarType, c.scalarType]
  if !a.isArray && !b.isArray && !c.isArray then
    let dst ← allocReg
    emit (Tropical.Plan.instrScalar op.name dst #[a.op, b.op, c.op] rt)
    pure (.scalar (.reg dst rt) rt)
  else
    let size := match a with
      | .array _ s _ => s
      | _ => match b with
        | .array _ s _ => s
        | _ => match c with | .array _ s _ => s | _ => 0
    let slot ← allocArraySlot size
    let strides := #[if a.isArray then 1 else 0, if b.isArray then 1 else 0,
      if c.isArray then 1 else 0]
    emit (Tropical.Plan.instrArray op.name slot #[a.op, b.op, c.op] size strides rt)
    pure (.array (.arrayReg slot) size rt)

private partial def compileIndex (arrNode idxNode : ExprId) : EmitM CompileResult := do
  let arr ← compileNode arrNode
  let idx ← compileNode idxNode (some .int)
  match arr with
  | .scalar .. =>
    throw ("emit_resolved: 'index' op has non-array operand. This usually means an "
      ++ "array-typed input port (e.g. `sequence: int[N]`) is being indexed inside "
      ++ "the program body — the per-instance compile path doesn't yet materialize "
      ++ "array input operands.")
  | .array arrOp _ rt =>
    let dst ← allocReg
    let idxOp := if idx.isArray then NOperand.const (0 : Nat) .int else idx.op
    emit (Tropical.Plan.instrIndex dst #[arrOp, idxOp] rt)
    pure (.scalar (.reg dst rt) rt)

/-- Emit an indexed reduction (`CNode.bankSum`) as a `ReduceBegin`/body/`ReduceEnd`
    region — the banks-as-data lowering (slice 3b). The loop visits elements in
    array order — the same order the unrolled fold nests its adds — so the region
    renders bit-identical to the unroll for ANY scalar element type (order
    preservation needs no associativity; the earlier i64-only restriction was the
    scaffolding's, not the fold's).

    - The coefficient columns (`tables`) are materialized ONCE *before* the
      region: compiling them here (not lazily inside the body) keeps their `Pack`
      loop-invariant, and the CSE memo makes the body's `index` reuse the slot.
    - The accumulator's type FOLLOWS THE BODY (bool promotes to int): the body
      compiles first with no imposed expectation, and the region opener is then
      spliced in front of the body's instructions (instr + stage arrays insert in
      lockstep). The accumulator temp lives *outside* the region (seeded to a
      typed 0 by `ReduceBegin`); the body computes one element's contribution and
      an explicit `Add acc acc contrib` accumulates — the exact slice-3a contract.
    - The CSE memo is snapshotted across the region: body-local temps do not
      escape (the runtime zero-scratches post-region reads), so a later sequential
      bank must not reuse them. Table entries (memoized before the snapshot) and
      any subterm compiled outside the region survive.
    - NESTED banks: an inner `bankSum` in the body recurses through this same
      function. The splice point is safe — `regionStart` is recorded before the
      body compiles, and an inner region splices at a strictly later index, so
      the outer's insertion point never moves — and the memo snapshot/restore
      nests via the call frames (`memo0` is a local, not emitter state). `idxId`
      is the region's binder id: it rides `ReduceBegin` as `loop_id`, and the
      body's `loopIdx idxId` operands resolve against the emitters' stack of
      open regions. -/
private partial def compileBankSum (count : Nat) (tables : Array ExprId) (body : ExprId)
    (dynCount? : Option ExprId := none) (idxId : Nat := 0) : EmitM CompileResult := do
  for t in tables do
    let _ ← compileNode t
  -- The optional runtime effective count (trip-count-as-data) compiles BEFORE
  -- the region, alongside the tables, so any instructions it needs are
  -- loop-invariant. Its operand rides `ReduceBegin` as args[1]; `loopCount`
  -- stays the static capacity and the emitters clamp at the loop head.
  let countOp? ← dynCount?.mapM fun dc => do
    let r ← compileNode dc (some .int)
    if r.isArray then
      throw "emit_resolved: bankSum dynamic count is array-valued; the effective trip count must be a scalar"
    pure r.op
  let acc ← allocReg
  let memo0 := (← get).memo
  let regionStart := (← get).instrs.size
  let contrib ← compileNode body
  if contrib.isArray then
    throw "emit_resolved: bankSum body is array-valued; the mode contribution must be a scalar"
  let ty := if contrib.scalarType == .bool then .int else contrib.scalarType
  modify fun s => { s with
    instrs := s.instrs.insertIdx! regionStart
      (Tropical.Plan.instrReduceBegin acc (.const (0 : Nat) ty) count ty countOp? idxId)
    instrStages := s.instrStages.insertIdx! regionStart s.curStage }
  emit (Tropical.Plan.instrScalar "Add" acc #[.reg acc ty, contrib.op] ty)
  emit (Tropical.Plan.instrReduceEnd acc ty)
  modify fun s => { s with memo := memo0 }
  pure (.scalar (.reg acc ty) ty)

private partial def compileSetElement (arrNode idxNode valNode : ExprId) :
    EmitM CompileResult := do
  let arr ← compileNode arrNode
  let idx ← compileNode idxNode
  let val ← compileNode valNode
  match arr with
  | .scalar .. =>
    let size := 1
    let slot ← allocArraySlot size
    pure (.array (.arrayReg slot) size .float)
  | .array arrOp size _ =>
    let idxOp := if idx.isArray then NOperand.const (0 : Nat) .float else idx.op
    let valOp := if val.isArray then NOperand.const (0 : Nat) .float else val.op
    match arrOp with
    | .arrayReg slot =>
      emit (Tropical.Plan.instrSetElement slot #[arrOp, idxOp, valOp])
    | .sessionArrayReg slot =>
      emit (Tropical.Plan.instrSessionSetElement slot #[arrOp, idxOp, valOp])
    | _ =>
      throw s!"emit_resolved: compileSetElement expected array_reg or session_array_reg operand, got {operandKind arrOp}"
    pure (.array arrOp size .float)

end

-- ─────────────────────────────────────────────────────────────
-- Top-level emit driver
-- ─────────────────────────────────────────────────────────────

/-- Internal emit result (TS `FlatProgram`). `instrStages` /
    `perChildPreInputStages` are parallel to the instruction arrays: the
    binding-time stage of the node each instruction was emitted for
    (staging metadata — absent from every wire form). -/
structure FlatProgram where
  registerCount : Nat
  arraySlotCount : Nat
  arraySlotSizes : Array Nat
  instructions : Array NInstr
  perChildPreInput : Array (Array NInstr)
  outputTargets : Array Nat
  instrStages : Array (Option Stage) := #[]
  perChildPreInputStages : Array (Array (Option Stage)) := #[]
deriving Inhabited

/-- Per-kernel staging input: the stage each input port's wire binds at
    (in the parent's context) and each child's per-output stages, both
    supplied by the partitioner (which recursed into children first).
    Defaults resolve every parametric dep to `s1` — callers that don't
    stage (the raw per-program paths) lose nothing. -/
structure StagingInfo where
  inputStages : Array Stage := #[]
  childOutStages : Array (Array Stage) := #[]
deriving Inhabited

/-- The binding context for pre-input block `k` (only already-run
    siblings `j < k` are resolvable) or, with `k = childCount`, for the
    parent body (every child has run). -/
def StagingInfo.ctxAt (info : StagingInfo) (k : Nat) : Staging.StageCtx :=
  { inputStages := info.inputStages
    childOut := (Array.range info.childOutStages.size).map fun j =>
      if j < k then info.childOutStages[j]? else none }

/-- Fractal context: the sub-instance decls of the program being
    emitted, plus the enclosing program (registry lookups for the
    children's port tables). -/
structure NestedContext where
  instances : Array CoreBodyDecl := #[]
  enclosing : CoreProgram

/-- TS `inputDeclScalarType`. -/
def inputDeclScalarType (t? : Option CorePortType) : ScalarType :=
  match t? with
  | none => .float
  | some (.scalar k) => k
  | some (.alias base) => base
  | some (.array ..) => .float

private def instName : CoreBodyDecl → String
  | .inst n .. => n
  | .param n _ => n
  | .progDecl n => n

private def instTypeKey : CoreBodyDecl → String
  | .inst _ k .. => k
  | _ => ""

private def instInputs : CoreBodyDecl → Array CoreInstanceInput
  | .inst _ _ _ i => i
  | _ => #[]

def emitProgram (outputExprs : Array ExprId)
    (outputPortScalarCounts : Array Nat)
    (nested : NestedContext)
    (staging : StagingInfo := {}) : EmitM FlatProgram := do
  let mut outputTargets : Array Nat := #[]
  -- Port-default fallback (an unwired child port with no declared default):
  -- one shared `0` id in the arena. Interned once; equal to any existing `.num 0`.
  let zeroId ← internNode (.num (0 : Nat))

  -- ── Fractal: per-child pre-input blocks ──
  let mut perChildPreInput : Array (Array NInstr) := #[]
  let mut perChildPreInputStages : Array (Array (Option Stage)) := #[]
  let preChildBaseline := (← get).instrs.size
  let scalarSlotMaps := (← get).slots.nestedInputSlots
  let arraySlotMaps := (← get).slots.nestedInputArraySlots
  for k in [0:nested.instances.size] do
    let decl := nested.instances[k]!
    let childStart := (← get).instrs.size
    let childScalarMap := lookup scalarSlotMaps k
    let childArrayMap := lookup arraySlotMaps k
    let wiredByPort := (instInputs decl).map fun i => (i.port.idx, i.value)
    let some childType := nested.enclosing.registryGet? (instTypeKey decl)
      | throw s!"emit_resolved: instance '{instName decl}' typeKey '{instTypeKey decl}' missing from enclosing registry"
    -- Pre-input block k resolves under "siblings j < k have run".
    modify fun s => { s with stageCtx := staging.ctxAt k }
    let ports := childType.inputs
    for i in [0:ports.size] do
      let portDecl := ports[i]!
      let wireExpr := match lookup wiredByPort i with
        | some v => v
        | none => portDecl.default?.getD zeroId
      let scalarSlot := childScalarMap.bind (lookup · i)
      let arrayInfo := childArrayMap.bind (lookup · i)
      let wireStage := some (Staging.stageOf (← get).arena (← get).stageCtx wireExpr)
      match scalarSlot with
      | some slot =>
        let portT := inputDeclScalarType portDecl.type?
        let r ← compileNode wireExpr (some portT)
        withStage wireStage do
          let valOp ← match r with
            | .array op _ rt =>
              let dst ← allocReg
              emit (Tropical.Plan.instrIndex dst #[op, .const (0 : Nat) .int] rt)
              pure (NOperand.reg dst rt)
            | .scalar op _ => pure op
          emit (Tropical.Plan.instrWriteSlot slot valOp portT)
      | none =>
        match arrayInfo with
        | some info =>
          let r ← compileNode wireExpr (some .float)
          match r with
          | .scalar .. =>
            throw s!"emit_resolved: array-typed child input port at idx={i} of '{instName decl}' received scalar-shaped wire expression; expected array of size {info.size}"
          | .array op size _ =>
            if size != info.size then
              throw s!"emit_resolved: array-typed child input port at idx={i} of '{instName decl}' has size {info.size}, wire expression evaluates to array of size {size}"
            withStage wireStage do
              emit (Tropical.Plan.instrSessionArray "Add" info.slot
                #[op, .const (0 : Nat) .float] info.size #[1, 0] .float)
        | none => pure ()  -- no allocated parent-side slot — nothing to emit
    let st ← get
    perChildPreInput := perChildPreInput.push (st.instrs.extract childStart st.instrs.size)
    perChildPreInputStages := perChildPreInputStages.push
      (st.instrStages.extract childStart st.instrStages.size)
  -- Truncate the running list back to the pre-child boundary; the
  -- blocks live in perChildPreInput, the temps stay reserved.
  modify fun s => { s with instrs := s.instrs.extract 0 preChildBaseline,
                           instrStages := s.instrStages.extract 0 preChildBaseline }
  -- The body (and the outputs below) resolve with every child run.
  modify fun s => { s with stageCtx := staging.ctxAt nested.instances.size }

  -- ── Output targets: one entry per scalar slot of every port ──
  if outputPortScalarCounts.size != outputExprs.size then
    throw s!"emitProgram: outputPortScalarCounts.length ({outputPortScalarCounts.size}) must match outputExprs.length ({outputExprs.size})"
  for portI in [0:outputExprs.size] do
    let expr := outputExprs[portI]!
    let declaredCount := outputPortScalarCounts[portI]!
    let r ← compileNode expr (some .float)
    let outStage := some (Staging.stageOf (← get).arena (← get).stageCtx expr)
    modify fun s => { s with curStage := outStage }
    if declaredCount == 1 then
      let dst ← allocReg
      match r with
      | .array op _ rt =>
        emit (Tropical.Plan.instrIndex dst #[op, .const (0 : Nat) .int] rt)
      | .scalar op rt =>
        emit (Tropical.Plan.instrScalar "Add" dst #[op, .const (0 : Nat) rt] rt)
      outputTargets := outputTargets.push dst
    else
      match r with
      | .scalar .. =>
        throw s!"emitProgram: output port {portI} declared as array of scalar count {declaredCount}, but expression is scalar"
      | .array op size rt =>
        if size != declaredCount then
          throw s!"emitProgram: output port {portI} declared as array of scalar count {declaredCount}, but expression has size {size}"
        for elemI in [0:declaredCount] do
          let dst ← allocReg
          emit (Tropical.Plan.instrIndex dst #[op, .const elemI .int] rt)
          outputTargets := outputTargets.push dst
    modify fun s => { s with curStage := none }

  let st ← get
  return {
    registerCount := st.nextReg
    arraySlotCount := st.nextArraySlot
    arraySlotSizes := st.arraySizes
    instructions := st.instrs
    perChildPreInput
    outputTargets
    instrStages := st.instrStages
    perChildPreInputStages }

/-- Public entry: construct the emitter state and run the driver.
    CF-only has no per-sample state, so there are no register backing
    slots to seed. -/
def emitResolvedProgram
    (outputExprs : Array ExprId)
    (outputPortScalarCounts : Array Nat)
    (inputPortTypes : Array ScalarType)
    (slots : EmitSlots)
    (arena : CoreArena)
    (nested : NestedContext)
    (staging : StagingInfo := {}) : Except String FlatProgram := do
  let st : EmitSt := { slots, inputPortTypes, arena }
  let (prog, _) ← (emitProgram outputExprs outputPortScalarCounts nested staging).run st
  return prog

end Tropical.Ir.Emit
