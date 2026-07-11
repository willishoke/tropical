import Tropical.Lowering
import Tropical.Ir.CompileResolved
import Tropical.Plan

/-!
# Session compile — partition + plan assembly (Phase 6 stage 6c)

Port of `compiler/ir/partition_recursive.ts` and
`compiler/ir/compile_session_slotted*.ts` over the Core sub-IR: the
recursive box-closed lowering (one kernel per `InstanceDecl` at every
nesting depth), the per-instance remap into the unified namespaces,
sinks, and slot metadata. The output is a `Plan.FlatPlan`.

What did NOT need porting, and why:

- `translateNode` / `InputBinding`: under the root-program lowering the
  session's wires are body expressions of the synthetic root — they
  lower through the *emitter*, in the root kernel's scope. No session
  ExprNode ever needs translating at partition time, every child input
  port has a slot override, and the root has no ports — so `opInput`
  operands cannot survive into remap, preambles are always empty, and
  `tempsConsumed` is always 0. The remap's `input` case keeps the TS
  fallback behavior (a literal-0 const) rather than throwing, mirroring
  the `inputBindingFor → defaults → 0` chain it would take there.
- the legacy per-instance scheduler tier: retired in TS already.

Slot-allocation parity: the engine's `Lowering.allocate` produces
[params, top-level outputs]; this module continues with [nested outputs (depth-first,
instance order), then ALL inputs (top-level + nested, depth-first)] —
the exact two-phase `preallocateOutputsRecursive` /
`preallocateInputsRecursive` order, including the array-input alias
quotient (a single-`ref` array wire binds the consumer port to the
producer's slot; a `sessionArraySlot` wire binds to the delay's slot).

Errors are `Except String` with byte-exact TS messages; the engine
maps them to `internal_error` (the envelope every TS compile-path
throw produced via `toEnvelope`).
-/

namespace Tropical.Compile

open Lean (Json JsonNumber toJson)
open Tropical.Ir.Core
open Tropical.Ir.CompileResolved (compileResolved Context)
open Tropical.Ir.Emit (ArraySlotInfo)
open Tropical.Plan (NOperand DstSlot NInstr InstanceFunction)
open Tropical.Expr (getField? getStrField? opOf?)

abbrev ScalarType := Tropical.Plan.ScalarType

-- ─────────────────────────────────────────────────────────────
-- Port metadata (the WirePortMeta image)
-- ─────────────────────────────────────────────────────────────

structure PortMeta where
  scalarSlotNames : Array String := #[]
  scalarTypes : Array ScalarType := #[]
  arraySlot? : Option Nat := none
  arraySize? : Option Nat := none
deriving Repr, Inhabited

/-- Parse a `Lowering.Alloc` meta Json (the shape `allocOutputPort` and
    the delay extractor write) into the typed form. -/
def PortMeta.ofJson (j : Json) : PortMeta :=
  let strArr (k : String) : Array String :=
    match getField? j k with
    | some (.arr a) => a.filterMap fun | .str s => some s | _ => none
    | _ => #[]
  let natField (k : String) : Option Nat :=
    match getField? j k with
    | some (.num n) => some n.toFloat.toUInt64.toNat
    | _ => none
  { scalarSlotNames := strArr "scalarSlotNames"
    scalarTypes := (strArr "scalarTypes").map fun s =>
      (Tropical.Parse.ScalarKind.ofWire? s).getD .float
    arraySlot? := natField "arraySlot"
    arraySize? := natField "arraySize" }

-- ─────────────────────────────────────────────────────────────
-- Compile-time session allocation state
-- ─────────────────────────────────────────────────────────────

/-- A string-keyed association with O(1) lookup *and* preserved insertion order:
    `entries` is the ordered list (the iteration order → the plan's slot layout),
    `index` maps a key to its first position for O(1) `get?` (vs the old linear
    `Array.find?`). First-match semantics match the old `find?`; iterating
    `entries` reproduces the old `Array` order, so the emitted slot layout is
    byte-for-byte unchanged. -/
structure OrderedAssoc (α : Type) where
  entries : Array (String × α) := #[]
  index : Std.HashMap String Nat := {}
deriving Inhabited

namespace OrderedAssoc
variable {α : Type}
def get? (m : OrderedAssoc α) (k : String) : Option α :=
  (m.index.get? k).bind fun i => (m.entries[i]?).map (·.2)
/-- Append, preserving first-match: a duplicate key keeps the first index. -/
def push (m : OrderedAssoc α) (kv : String × α) : OrderedAssoc α :=
  if m.index.contains kv.1 then { m with entries := m.entries.push kv }
  else { entries := m.entries.push kv, index := m.index.insert kv.1 m.entries.size }
def ofArray (a : Array (String × α)) : OrderedAssoc α :=
  a.foldl (fun m kv => m.push kv) {}
end OrderedAssoc

structure SessionAlloc where
  slotCount : Nat
  paramSlots : OrderedAssoc Nat
  outputSlotRegistry : OrderedAssoc Nat
  inputSlotRegistry : OrderedAssoc Nat := {}
  outputPortMeta : OrderedAssoc PortMeta
  inputPortMeta : OrderedAssoc PortMeta := {}
  ioCount : Nat
  ioSizes : Array Nat
  ioNames : Array String
deriving Inhabited

def SessionAlloc.ofAlloc (a : Tropical.Lowering.Alloc) : SessionAlloc :=
  { slotCount := a.slotCount
    paramSlots := OrderedAssoc.ofArray a.paramSlots
    outputSlotRegistry := OrderedAssoc.ofArray a.outputSlots
    outputPortMeta := OrderedAssoc.ofArray (a.outputMeta.map fun (k, j) => (k, PortMeta.ofJson j))
    ioCount := a.ioCount
    ioSizes := a.ioSizes
    ioNames := a.ioNames }

private def assocGet? {α} (m : Array (String × α)) (k : String) : Option α :=
  (m.find? (·.1 == k)).map (·.2)

-- ─────────────────────────────────────────────────────────────
-- expandPortToSlots over Core port types
-- ─────────────────────────────────────────────────────────────

structure SlotExpansion where
  names : Array String := #[]
  types : Array ScalarType := #[]
  arraySize? : Option Nat := none
deriving Inhabited

/-- Port of `expandPortToSlots`: scalar → one slot of its kind; alias →
    one opaque float slot; array → one array slot of size ∏shape. -/
def expandPortToSlots (baseName : String) (t? : Option CorePortType) :
    Except String SlotExpansion := do
  match t? with
  | none | some (.scalar _) =>
    let k := match t? with | some (.scalar k) => k | _ => .float
    return { names := #[baseName], types := #[k] }
  | some (.alias _) =>
    return { names := #[baseName], types := #[.float] }
  | some (.array _ shape) =>
    let mut total := 1
    for dim in shape do
      match dim with
      | .lit n => total := total * n.toFloat.toUInt64.toNat
      | .unresolved =>
        throw (s!"expandPortToSlots: array port '{baseName}' has unresolved "
          ++ "type-param dimension; ensure specialize ran on the owning program")
    return { arraySize? := some total }

private def slotKey (instPath portName : String) : String :=
  s!"{instPath}.{portName}"

/-- Port of `allocateOutputSlots` (IR-typed form). Idempotent. -/
def allocateOutputSlots (s : SessionAlloc) (instPath : String) (prog : CoreProgram) :
    Except String SessionAlloc := do
  let mut s := s
  for port in prog.outputs do
    let portKey := slotKey instPath port.name
    if (s.outputPortMeta.get? portKey).isSome then
      continue
    let exp ← expandPortToSlots portKey port.type?
    match exp.arraySize? with
    | some arraySize =>
      let arraySlot := s.ioCount
      s := { s with
        ioCount := s.ioCount + 1
        ioSizes := s.ioSizes.push arraySize
        ioNames := s.ioNames.push portKey
        outputPortMeta := s.outputPortMeta.push (portKey,
          { arraySlot? := some arraySlot, arraySize? := some arraySize }) }
    | none =>
      let mut reg := s.outputSlotRegistry
      for i in [0:exp.names.size] do
        reg := reg.push (exp.names[i]!, s.slotCount + i)
      s := { s with
        outputSlotRegistry := reg
        outputPortMeta := s.outputPortMeta.push (portKey,
          { scalarSlotNames := exp.names, scalarTypes := exp.types })
        slotCount := s.slotCount + exp.names.size }
  return s

/-- The array-input alias quotient (`tryAliasInputArrayWire`). -/
private def tryAliasInputArrayWire (s : SessionAlloc) (wires : Array Tropical.Wire)
    (instPath portName : String) : Option ArraySlotInfo := do
  let w ← wires.find? fun w => w.instName == instPath && w.portName == portName
  match w.expr with
  | .obj _ =>
    if opOf? w.expr == some "ref" then do
      let srcInst ← getStrField? w.expr "instance"
      let srcOut ← getStrField? w.expr "output"
      let pmeta ← s.outputPortMeta.get? (slotKey srcInst srcOut)
      let slot ← pmeta.arraySlot?
      let size ← pmeta.arraySize?
      pure { slot, size }
    else if opOf? w.expr == some "sessionArraySlot" then do
      let idx ← match getField? w.expr "index" with
        | some (.num n) => some n.toFloat.toUInt64.toNat
        | _ => none
      let size ← match getField? w.expr "size" with
        | some (.num n) => some n.toFloat.toUInt64.toNat
        | _ => none
      pure { slot := idx, size }
    else none
  | _ => none

/-- Port of `allocateInputSlots`. Idempotent; the alias check binds a
    single-`ref` (or extracted array-delay) wire's consumer port to the
    producer slot instead of allocating + copying. -/
def allocateInputSlots (s : SessionAlloc) (wires : Array Tropical.Wire)
    (instPath : String) (prog : CoreProgram) : Except String SessionAlloc := do
  let mut s := s
  for port in prog.inputs do
    let portKey := slotKey instPath port.name
    if (s.inputPortMeta.get? portKey).isSome then
      continue
    let exp ← expandPortToSlots portKey port.type?
    match exp.arraySize? with
    | some arraySize =>
      match tryAliasInputArrayWire s wires instPath port.name with
      | some aliased =>
        if aliased.size != arraySize then
          throw (s!"allocateInputSlots: array-input alias size mismatch for '{portKey}': "
            ++ s!"consumer expects size {arraySize}, producer slot has size {aliased.size}")
        s := { s with inputPortMeta := s.inputPortMeta.push (portKey,
          { arraySlot? := some aliased.slot, arraySize? := some arraySize }) }
      | none =>
        let arraySlot := s.ioCount
        s := { s with
          ioCount := s.ioCount + 1
          ioSizes := s.ioSizes.push arraySize
          ioNames := s.ioNames.push portKey
          inputPortMeta := s.inputPortMeta.push (portKey,
            { arraySlot? := some arraySlot, arraySize? := some arraySize }) }
    | none =>
      let mut reg := s.inputSlotRegistry
      for i in [0:exp.names.size] do
        reg := reg.push (exp.names[i]!, s.slotCount + i)
      s := { s with
        inputSlotRegistry := reg
        inputPortMeta := s.inputPortMeta.push (portKey,
          { scalarSlotNames := exp.names, scalarTypes := exp.types })
        slotCount := s.slotCount + exp.names.size }
  return s

-- ─────────────────────────────────────────────────────────────
-- Slot lookups (partition_recursive helpers)
-- ─────────────────────────────────────────────────────────────

private def lookupOutputSlot (s : SessionAlloc) (instPath portName : String) : Option Nat := do
  let pmeta ← s.outputPortMeta.get? (slotKey instPath portName)
  let first ← pmeta.scalarSlotNames[0]?
  s.outputSlotRegistry.get? first

private def lookupInputSlot (s : SessionAlloc) (instPath portName : String) : Option Nat := do
  let pmeta ← s.inputPortMeta.get? (slotKey instPath portName)
  let first ← pmeta.scalarSlotNames[0]?
  s.inputSlotRegistry.get? first

private def lookupOutputArraySlot (s : SessionAlloc) (instPath portName : String) :
    Option ArraySlotInfo := do
  let pmeta ← s.outputPortMeta.get? (slotKey instPath portName)
  pure { slot := ← pmeta.arraySlot?, size := ← pmeta.arraySize? }

private def lookupInputArraySlot (s : SessionAlloc) (instPath portName : String) :
    Option ArraySlotInfo := do
  let pmeta ← s.inputPortMeta.get? (slotKey instPath portName)
  pure { slot := ← pmeta.arraySlot?, size := ← pmeta.arraySize? }

/-- Naming-transparent synthetic session root. -/
def rootInstancePath : String := "__root__"

private def joinInstancePath (parent child : String) : String :=
  if parent == rootInstancePath then child else s!"{parent}.{child}"

-- ─────────────────────────────────────────────────────────────
-- Accumulators
-- ─────────────────────────────────────────────────────────────

structure Accumulators where
  nextRegRaw : Nat := 0
  nextArrayRaw : Nat := 0
  arraySlotSizes : Array Nat := #[]
  arraySlotNames : Array String := #[]
deriving Inhabited

-- ─────────────────────────────────────────────────────────────
-- Per-instance plan remap (compile_session_slotted_helpers.ts)
-- ─────────────────────────────────────────────────────────────

private def shiftDst (regOffset arrayOffset : Nat) : DstSlot → DstSlot
  | .temp slot => .temp (slot + regOffset)
  | .array slot => .array (slot + arrayOffset)
  -- Session-absolute already; collapse the namespace tag.
  | .sessionArray slot => .array slot
  | .moduleSlot i => .moduleSlot i

private def remapOperand (instanceName : String) (regOffset arrayOffset : Nat) :
    NOperand → Except String NOperand
  | .const v t => .ok (.const v t)
  | .source i t => .ok (.source i t)
  | .slot i t => .ok (.slot i t)
  -- Unreachable under the root lowering (every child input port has a
  -- slot override); TS would resolve through inputBindingFor → the
  -- defaults chain → literal 0. Mirror the terminal value.
  | .input _ t => .ok (.const 0 t)
  | .reg slot t => .ok (.reg (slot + regOffset) t)
  | .arrayReg slot => .ok (.arrayReg (slot + arrayOffset))
  | .sessionArrayReg slot => .ok (.arrayReg slot)
  | .loopIdx id => .ok (.loopIdx id)
  | .param _ _ =>
    .error (s!"compileSessionSlotted: legacy 'param' operand encountered "
      ++ s!"in '{instanceName}'. Session-level params should resolve "
      ++ "to slot operands before this point.")

private def remapInstr (instanceName : String) (regOffset arrayOffset : Nat)
    (i : NInstr) : Except String NInstr := do
  return { i with
    dst := shiftDst regOffset arrayOffset i.dst
    args := ← i.args.mapM (remapOperand instanceName regOffset arrayOffset) }

/-- Output writebacks per declared port (the remap's `writeSlots`).
    `portStages` (parallel to `outputPortNames`; empty = unstaged) tags
    each emitted write with its port's binding-time stage. -/
private def emitWriteSlots (s : SessionAlloc) (instanceName : String)
    (outputPortNames : Array String) (outputTargets : Array Nat) (regOffset : Nat)
    (portStages : Array Tropical.Ir.Stage := #[]) :
    Except String (Array NInstr × Array (Option Tropical.Ir.Stage)) := do
  let mut writeSlots : Array NInstr := #[]
  let mut writeStages : Array (Option Tropical.Ir.Stage) := #[]
  let mut targetIdx := 0
  for portI in [0:outputPortNames.size] do
    let portName := outputPortNames[portI]!
    let stage := portStages[portI]?
    let portKey := slotKey instanceName portName
    let some pmeta := s.outputPortMeta.get? portKey
      | throw (s!"compileSessionSlotted: instance '{instanceName}' port '{portName}' "
          ++ "missing outputPortMeta entry (allocateOutputSlots should have run).")
    match pmeta.arraySlot?, pmeta.arraySize? with
    | some arrSlot, some arrSize =>
      let arrOp := NOperand.arrayReg arrSlot
      for elemI in [0:arrSize] do
        let some localTemp := outputTargets[targetIdx]?
          | throw (s!"compileSessionSlotted: instance '{instanceName}' missing output_targets[{targetIdx}] "
              ++ s!"for array port '{portName}' element {elemI}.")
        let absTemp := localTemp + regOffset
        writeSlots := writeSlots.push (Tropical.Plan.instrSetElement arrSlot
          #[arrOp, .const elemI .int, .reg absTemp .float])
        writeStages := writeStages.push stage
        targetIdx := targetIdx + 1
    | _, _ =>
      for scalarI in [0:pmeta.scalarSlotNames.size] do
        let scalarSlotName := pmeta.scalarSlotNames[scalarI]!
        let some slotIdx := s.outputSlotRegistry.get? scalarSlotName
          | throw s!"compileSessionSlotted: scalar slot '{scalarSlotName}' not in outputSlotRegistry."
        let some localTemp := outputTargets[targetIdx]?
          | throw (s!"compileSessionSlotted: instance '{instanceName}' missing output_targets[{targetIdx}] "
              ++ s!"for scalar slot '{scalarSlotName}' (port '{portName}', element {scalarI}).")
        let scalarType := pmeta.scalarTypes[scalarI]?.getD .float
        let absTemp := localTemp + regOffset
        writeSlots := writeSlots.push (Tropical.Plan.instrWriteSlot slotIdx
          (.reg absTemp scalarType) scalarType)
        writeStages := writeStages.push stage
        targetIdx := targetIdx + 1
  if targetIdx != outputTargets.size then
    throw (s!"compileSessionSlotted: instance '{instanceName}' has {outputTargets.size} "
      ++ s!"output_targets but only {targetIdx} were consumed by slot expansion. "
      ++ "This indicates a port-shape / emit mismatch.")
  return (writeSlots, writeStages)

-- ─────────────────────────────────────────────────────────────
-- partitionKernel
-- ─────────────────────────────────────────────────────────────

private def instParts : CoreBodyDecl → Option (String × String)
  | .inst name typeKey _ _ => some (name, typeKey)
  | _ => none

/-- The wire expression bound to a child input port, mirroring
    `emitProgram`'s selection: explicit wire, else the port's declared
    default, else `none` (the shared literal 0 — `fold`). -/
private def childWireExpr (decl : CoreBodyDecl) (portIdx : Nat)
    (portDecl : CoreInputDecl) : Option Tropical.Ir.ExprId :=
  let wired := match decl with
    | .inst _ _ _ inputs => (inputs.find? (·.port.idx == portIdx)).map (·.value)
    | _ => none
  wired <|> portDecl.default?

/-- Partition a single kernel recursively. Returns the kernel tree, the
    advanced allocation + accumulators, this kernel's instruction-stage
    blocks in emit order (preamble; per child: pre-input, its blocks;
    body — the exact linearization `EmitLlvm.emitKernelBlock` walks),
    and its per-output binding-time stages. `inputStages` is the stage
    each of this program's input ports binds at (from the parent). -/
partial def partitionKernel (instancePath : String) (prog : CoreProgram)
    (arena : Tropical.Ir.CoreArena)
    (wires : Array Tropical.Wire) (s : SessionAlloc) (acc : Accumulators)
    (inputSlotOverride : Array (Nat × Nat) := #[])
    (inputArraySlots : Array (Nat × ArraySlotInfo) := #[])
    (paramSlots : Array (Nat × Nat) := #[])
    (inputStages : Array Tropical.Ir.Stage := #[]) :
    Except String (InstanceFunction × SessionAlloc × Accumulators
      × Array (Array (Option Tropical.Ir.Stage)) × Array Tropical.Ir.Stage) := do
  let mut s := s
  let mut acc := acc

  -- ── 1. Recurse into sub-InstanceDecls. ──
  let mut children : Array InstanceFunction := #[]
  let mut childStageBlocks : Array (Array (Array (Option Tropical.Ir.Stage))) := #[]
  let mut childOutStages : Array (Array Tropical.Ir.Stage) := #[]
  let mut nestedOutputSlots : Array (Nat × Array (Nat × Nat)) := #[]
  let mut nestedInputSlots : Array (Nat × Array (Nat × Nat)) := #[]
  let mut nestedOutputArraySlots : Array (Nat × Array (Nat × ArraySlotInfo)) := #[]
  let mut nestedInputArraySlots : Array (Nat × Array (Nat × ArraySlotInfo)) := #[]

  let instDecls := prog.instances
  for k in [0:instDecls.size] do
    let some (childName, typeKey) := instParts instDecls[k]!
      | throw "partitionKernel: non-instance decl in instance table (port bug)"
    let childPath := joinInstancePath instancePath childName
    let some declType := prog.registryGet? typeKey
      | throw s!"partitionKernel: instance '{childPath}' typeKey '{typeKey}' missing from registry"

    s ← allocateOutputSlots s childPath declType
    s ← allocateInputSlots s wires childPath declType

    let mut childOutputMap : Array (Nat × Nat) := #[]
    let mut childOutputArrayMap : Array (Nat × ArraySlotInfo) := #[]
    for i in [0:declType.outputs.size] do
      let outDecl := declType.outputs[i]!
      match lookupOutputSlot s childPath outDecl.name with
      | some slot => childOutputMap := childOutputMap.push (i, slot)
      | none =>
        if let some info := lookupOutputArraySlot s childPath outDecl.name then
          childOutputArrayMap := childOutputArrayMap.push (i, info)
    nestedOutputSlots := nestedOutputSlots.push (k, childOutputMap)
    nestedOutputArraySlots := nestedOutputArraySlots.push (k, childOutputArrayMap)

    let mut childInputMap : Array (Nat × Nat) := #[]
    let mut childInputArrayMap : Array (Nat × ArraySlotInfo) := #[]
    for i in [0:declType.inputs.size] do
      let inDecl := declType.inputs[i]!
      match lookupInputSlot s childPath inDecl.name with
      | some slot => childInputMap := childInputMap.push (i, slot)
      | none =>
        if let some info := lookupInputArraySlot s childPath inDecl.name then
          childInputArrayMap := childInputArrayMap.push (i, info)
    nestedInputSlots := nestedInputSlots.push (k, childInputMap)
    nestedInputArraySlots := nestedInputArraySlots.push (k, childInputArrayMap)

    -- Child input stages: each port's wire expression resolved under
    -- THIS kernel's context at pre-input block k (siblings j < k have
    -- run). Mirrors emitProgram's wire selection exactly.
    let ctxK : Tropical.Ir.Staging.StageCtx :=
      { inputStages
        childOut := (Array.range instDecls.size).map fun j =>
          if j < k then childOutStages[j]? else none }
    let mut childInputStages : Array Tropical.Ir.Stage := #[]
    for i in [0:declType.inputs.size] do
      let stage := match childWireExpr instDecls[k]! i declType.inputs[i]! with
        | some expr => Tropical.Ir.Staging.stageOf arena ctxK expr
        | none => .fold   -- the shared literal-0 default
      childInputStages := childInputStages.push stage

    let (childFn, s', acc', childBlocks, childOuts) ←
      partitionKernel childPath declType arena wires s acc
        childInputMap childInputArrayMap #[] childInputStages
    s := s'
    acc := acc'
    children := children.push childFn
    childStageBlocks := childStageBlocks.push childBlocks
    childOutStages := childOutStages.push childOuts

  -- ── 2. Compile this kernel's body. ──
  let ctx : Context := {
    paramSlots
    nestedOutputSlots? := some nestedOutputSlots
    nestedInputSlots
    inputSlotOverride
    inputArraySlots
    nestedInputArraySlots
    nestedOutputArraySlots
    staging := { inputStages, childOutStages } }
  let plan ← compileResolved prog arena ctx

  -- ── 3. Remap into the unified slot/temp space. ──
  let regOffset := acc.nextRegRaw
  let arrayOffset := acc.nextArrayRaw

  -- This kernel's per-output stages, under the full context (every
  -- child has run by the time the body executes).
  let fullCtx : Tropical.Ir.Staging.StageCtx :=
    { inputStages, childOut := childOutStages.map some }
  let outStages := (Array.range prog.outputs.size).map fun i =>
    match prog.assigns.find? (fun a => match a.target with
      | .port idx => idx.idx == i
      | .dac => false) with
    | some a => Tropical.Ir.Staging.stageOf arena fullCtx a.expr
    | none => .s1

  let body ← plan.instructions.mapM (remapInstr instancePath regOffset arrayOffset)
  let perChildPreInput ← plan.perChildPreInput.mapM
    (·.mapM (remapInstr instancePath regOffset arrayOffset))
  let (writeSlots, writeStages) ← emitWriteSlots s instancePath (prog.outputs.map (·.name))
    plan.outputTargets regOffset outStages
  let instanceInstructions := body ++ writeSlots

  -- Attach each per-child pre-input block to its child.
  if perChildPreInput.size != children.size then
    throw (s!"partitionKernel: instance '{instancePath}': perChildPreInput length "
      ++ s!"({perChildPreInput.size}) does not match children length "
      ++ s!"({children.size}). emit_resolved + compileResolved must produce "
      ++ "one block per nested InstanceDecl in body order.")
  children := children.mapIdx fun i c => c.withPreInput perChildPreInput[i]!

  -- Stage blocks in emit order (the Stage0.collectBlocks linearization):
  -- preamble (empty), per child its pre-input block then its own
  -- blocks, then this body (+ output writebacks).
  let mut stageBlocks : Array (Array (Option Tropical.Ir.Stage)) := #[#[]]
  for k in [0:children.size] do
    stageBlocks := stageBlocks.push (plan.perChildPreInputStages[k]?.getD #[])
    stageBlocks := stageBlocks ++ (childStageBlocks[k]?.getD #[])
  stageBlocks := stageBlocks.push (plan.instrStages ++ writeStages)

  let fn : InstanceFunction := .mk
    (s!"instance_" ++ (instancePath.replace "." "_"))
    instancePath
    #[]                     -- preamble (always empty under the root lowering)
    instanceInstructions
    #[]                     -- pre_input (parent attaches a copy on its own pass)
    regOffset arrayOffset
    plan.registerCount      -- + tempsConsumed (always 0; no preamble emitter)
    children

  -- ── 4. Accumulator updates (this kernel's own contribution). ──
  acc := { acc with
    arraySlotSizes := acc.arraySlotSizes ++ plan.arraySlotSizes
    arraySlotNames := acc.arraySlotNames
      ++ plan.arraySlotNames.map (joinInstancePath instancePath ·)
    nextRegRaw := acc.nextRegRaw + plan.registerCount
    nextArrayRaw := acc.nextArrayRaw + plan.arraySlotCount }

  return (fn, s, acc, stageBlocks, outStages)

-- ─────────────────────────────────────────────────────────────
-- Session compile (compile_session_slotted.ts)
-- ─────────────────────────────────────────────────────────────

structure SessionInput where
  /-- Session instances in registry order: (name, the type's Core form). -/
  instances : Array (String × CoreProgram)
  /-- Post-extraction wires (alias checks only — not re-lowered). -/
  wiresPost : Array Tropical.Wire
  graphOutputs : Array (String × String)
  /-- Param mirror: name → raw value Json (slot_defaults echo). -/
  params : Array (String × Json)
  /-- Allocation (params, top-level outputs). -/
  alloc : Tropical.Lowering.Alloc
  /-- The elaborated session root, downcast to Core. -/
  root : CoreProgram
  /-- The shared hash-consed expression DAG that every instance's (and the
      root's) leaf `ExprId`s index into — one arena for root + registry
      (Phase B). -/
  arena : Tropical.Ir.CoreArena
  mode : Tropical.Plan.CompilationMode := .fused

private def rootParamName : CoreBodyDecl → Option String
  | .param name _ => some name
  | _ => none

/-- Build slot metadata (`buildSlotMetadata`). -/
private def slotMetadata (s : SessionAlloc) (params : Array (String × Json))
    (paramSlots : OrderedAssoc Nat) :
    Nat × Array String × Array Json := Id.run do
  let slotCount := s.slotCount
  let mut names := Array.replicate slotCount ""
  let mut defaults : Array Json := Array.replicate slotCount (toJson (0 : Nat))
  for (name, idx) in s.outputSlotRegistry.entries do
    if idx < slotCount then names := names.set! idx name
  for (name, idx) in paramSlots.entries do
    if idx < slotCount then
      names := names.set! idx s!"param:{name}"
      if let some v := assocGet? params name then
        defaults := defaults.set! idx v
  for (name, idx) in s.inputSlotRegistry.entries do
    if idx < slotCount then names := names.set! idx s!"input:{name}"
  return (slotCount, names, defaults)

/-- Materialize the audible outputs as device-bound sinks (`emitSinks`). -/
private def emitSinks (s : SessionAlloc) (graphOutputs : Array (String × String)) :
    Except String (Array Tropical.Plan.SinkSpec) := do
  let mut inputs : Array Nat := #[]
  for (inst, output) in graphOutputs do
    let key := slotKey inst output
    let some idx := s.outputSlotRegistry.get? key
      | throw s!"compileSessionSlotted: dac wire '{key}' has no allocated output slot."
    inputs := inputs.push idx
  return #[{ inputs, gain := Tropical.Plan.defaultSinkGain, target := 0 }]

/-- `preallocateOutputsRecursive`: parent before children, body order. -/
private partial def preallocOutputs (s : SessionAlloc) (path : String)
    (prog : CoreProgram) : Except String SessionAlloc := do
  let mut s ← allocateOutputSlots s path prog
  for d in prog.instances do
    if let some (childName, typeKey) := instParts d then
      let some childType := prog.registryGet? typeKey
        | throw s!"compileSession: instance '{path}.{childName}' typeKey '{typeKey}' missing from registry"
      s ← preallocOutputs s (joinInstancePath path childName) childType
  return s

/-- `preallocateInputsRecursive`: runs AFTER all outputs so the alias
    check can see every producer's meta. -/
private partial def preallocInputs (s : SessionAlloc) (wires : Array Tropical.Wire)
    (path : String) (prog : CoreProgram) : Except String SessionAlloc := do
  let mut s ← allocateInputSlots s wires path prog
  for d in prog.instances do
    if let some (childName, typeKey) := instParts d then
      let some childType := prog.registryGet? typeKey
        | throw s!"compileSession: instance '{path}.{childName}' typeKey '{typeKey}' missing from registry"
      s ← preallocInputs s wires (joinInstancePath path childName) childType
  return s

/-- The session → `tropical_plan_5` lowering: two-phase slot
    pre-allocation, accumulator seeding from the session I/O array
    space, one `partitionKernel` over the synthetic root, sinks, and
    slot metadata. The staged variant also returns the per-instruction
    binding-time stages in emit order (the `Stage0.collectBlocks`
    linearization) — the typed side of the stage differential. -/
def compileSessionStaged (input : SessionInput) :
    Except String (Tropical.Plan.FlatPlan × Array (Array (Option Tropical.Ir.Stage))) := do
  let mut s := SessionAlloc.ofAlloc input.alloc

  -- Two-phase pre-allocation: all outputs first (the input alias check
  -- needs producers' meta), then all inputs; both walks recurse into
  -- nested instance decls in body order.
  for (name, prog) in input.instances do
    s ← preallocOutputs s name prog
  for (name, prog) in input.instances do
    s ← preallocInputs s input.wiresPost name prog

  -- Seed the array-slot accumulator with the session-level I/O slots.
  let acc : Accumulators := {
    arraySlotSizes := s.ioSizes
    arraySlotNames := s.ioNames
    nextArrayRaw := s.ioCount }

  -- Root param module slots, keyed by the root program's ParamIdx.
  let mut paramSlots : Array (Nat × Nat) := #[]
  let rootParams := input.root.params.filterMap rootParamName
  for i in [0:rootParams.size] do
    if let some slot := s.paramSlots.get? rootParams[i]! then
      paramSlots := paramSlots.push (i, slot)

  let (fn, s', acc, stageBlocks, _) ← partitionKernel rootInstancePath input.root input.arena
    input.wiresPost s acc #[] #[] paramSlots #[]
  s := s'

  let sinks ← emitSinks s input.graphOutputs
  let (slotCount, slotNames, slotDefaults) :=
    slotMetadata s input.params s.paramSlots

  return ({
    compilationMode := input.mode
    arraySlotNames := acc.arraySlotNames
    registerCount := acc.nextRegRaw
    arraySlotCount := acc.nextArrayRaw
    arraySlotSizes := acc.arraySlotSizes
    instanceFunctions := #[fn]
    sinks
    slotCount
    slotNames
    slotDefaults }, stageBlocks)

def compileSession (input : SessionInput) : Except String Tropical.Plan.FlatPlan :=
  (compileSessionStaged input).map (·.1)

end Tropical.Compile
