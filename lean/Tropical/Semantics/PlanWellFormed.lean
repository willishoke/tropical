import Tropical.Semantics.Plan

/-!
# Structural well-formedness for Plan-6

This module validates the Plan semantic waist without importing a backend.
It checks the namespaces and structured-region invariants needed by
`execBlocks`; environment shape remains a separate predicate because host
inputs and decoded initial storage are not part of `FlatPlan` itself.
-/

namespace Tropical.Semantics

open Tropical.Plan

private inductive RegionFrame where
  | reduce (binderId accTemp : Nat) (resultType : ScalarType)
  | routed (binderId dst capacity outputCount : Nat)
      (routes : Array (Option Nat)) (fanout : Option Nat)

private def RegionFrame.binderId : RegionFrame → Nat
  | .reduce binderId .. | .routed binderId .. => binderId

private def RegionFrame.isRouted : RegionFrame → Bool
  | .routed .. => true
  | .reduce .. => false

private def operandScalarType? : NOperand → Option ScalarType
  | .const _ ty | .input _ ty | .reg _ ty | .param _ ty
  | .source _ ty | .slot _ ty => some ty
  | .loopIdx _ => some .int
  | .arrayReg _ | .sessionArrayReg _ => none

private def sourceType : SourceKind → ScalarType
  | .tick | .tileTick => .int
  | .rate | .tilePhase => .float

private def operandWellFormed (plan : FlatPlan) (frames : List RegionFrame) :
    NOperand → Bool
  | .const _ _ => true
  | .input _ _ => false
  | .reg slot _ => slot < plan.registerCount
  | .arrayReg slot => slot < plan.arraySlotCount
  | .sessionArrayReg _ => false
  | .param _ _ => false
  | .source index ty =>
      match plan.sources[index]? with
      | some kind => ty == sourceType kind
      | none => false
  | .slot index _ => index < plan.slotCount
  | .loopIdx id => frames.any (fun frame => frame.binderId == id)

private def scalarOperandWellFormed (plan : FlatPlan)
    (frames : List RegionFrame) (operand : NOperand) : Bool :=
  operandWellFormed plan frames operand &&
    match operandScalarType? operand with | some _ => true | none => false

private def scalarDstWellFormed (plan : FlatPlan) : DstSlot → Bool
  | .temp slot => slot < plan.registerCount
  | .moduleSlot slot => slot < plan.slotCount
  | .array _ | .sessionArray _ => false

private def arrayDst? : DstSlot → Option Nat
  | .array slot => some slot
  | _ => none

private def planOpArity : PlanOp → Nat
  | .select | .clamp => 3
  | .neg | .abs | .sqrt | .floor | .ceil | .round
  | .not | .bitNot | .floatExponent | .toInt | .toBool | .toFloat => 1
  | _ => 2

private def scalarOpWellFormed (plan : FlatPlan) (frames : List RegionFrame)
    (instr : NInstr) (op : PlanOp) : Bool :=
  scalarDstWellFormed plan instr.dst
    && instr.args.size == planOpArity op
    && instr.args.all (scalarOperandWellFormed plan frames)
    && instr.resultType == op.resultType (instr.args.filterMap operandScalarType?)

private def elementwiseOpWellFormed (plan : FlatPlan)
    (frames : List RegionFrame) (instr : NInstr) (op : PlanOp)
    (dst : Nat) : Bool :=
  dst < plan.arraySlotCount
    && instr.loopCount <= (plan.arraySlotSizes[dst]?.getD 0)
    && instr.args.size == planOpArity op
    && instr.strides.size == instr.args.size
    && (List.range instr.args.size).all fun index =>
      let stride := instr.strides[index]!
      let operand := instr.args[index]!
      if stride == 1 then
        match operand with
        | .arrayReg slot =>
            slot < plan.arraySlotCount
              && instr.loopCount <= (plan.arraySlotSizes[slot]?.getD 0)
        | _ => false
      else stride == 0 && scalarOperandWellFormed plan frames operand
    && instr.resultType == op.resultType (instr.args.filterMap operandScalarType?)

private def smallInstrWellFormed (plan : FlatPlan)
    (frames : List RegionFrame) (instr : NInstr) : Bool :=
  match PlanOp.ofString? instr.tag with
  | some op =>
      match arrayDst? instr.dst with
      | some dst => elementwiseOpWellFormed plan frames instr op dst
      | none => scalarOpWellFormed plan frames instr op
  | none =>
      match instr.tag, instr.dst with
      | "Pack", .array dst =>
          dst < plan.arraySlotCount
            && instr.args.size == (plan.arraySlotSizes[dst]?.getD 0)
            && instr.args.all (scalarOperandWellFormed plan frames)
      | "Index", .temp dst =>
          dst < plan.registerCount && instr.args.size == 2
            && (match instr.args[0]? with
                | some (NOperand.arrayReg slot) => slot < plan.arraySlotCount
                | _ => false)
            && (match instr.args[1]? with
                | some operand => scalarOperandWellFormed plan frames operand
                | none => false)
      | "SetElement", .array dst =>
          dst < plan.arraySlotCount && instr.args.size == 3
            && (match instr.args[0]? with
                | some (NOperand.arrayReg source) => source == dst
                | _ => false)
            && (match instr.args[1]? with
                | some operand => scalarOperandWellFormed plan frames operand
                | none => false)
            && (match instr.args[2]? with
                | some operand => scalarOperandWellFormed plan frames operand
                | none => false)
      | "WriteSlot", .moduleSlot dst =>
          dst < plan.slotCount && instr.args.size == 1
            && (match instr.args[0]? with
                | some operand => scalarOperandWellFormed plan frames operand
                | none => false)
      | _, _ => false

private def scanBlocks (plan : FlatPlan) :
    List RegionFrame → List NInstr → Option (List RegionFrame)
  | frames, [] => some frames
  | frames, instr :: rest =>
      if instr.tag == "ReduceBegin" then
        match instr.dst with
        | .temp acc =>
            if acc < plan.registerCount
                && (instr.args.size == 1 || instr.args.size == 2)
                && instr.args.all (scalarOperandWellFormed plan frames)
                && !frames.any (fun frame => frame.binderId == instr.loopId) then
              scanBlocks plan
                (.reduce instr.loopId acc instr.resultType :: frames) rest
            else none
        | _ => none
      else if instr.tag == "ReduceEnd" then
        match frames, instr.dst with
        | .reduce _ acc ty :: outer, .temp endAcc =>
            if acc == endAcc && ty == instr.resultType && instr.args.isEmpty then
              scanBlocks plan outer rest
            else none
        | _, _ => none
      else if instr.tag == "RoutedSumBegin" then
        match instr.dst with
        | .array dst =>
            if dst < plan.arraySlotCount && instr.args.size <= 1
                && instr.args.all (scalarOperandWellFormed plan frames)
                && instr.loopCount > 0 && instr.routedOutputCount > 0
                && instr.routedOutputCount == (plan.arraySlotSizes[dst]?.getD 0)
                && !instr.routedRoutes.isEmpty
                && !frames.any RegionFrame.isRouted
                && !frames.any (fun frame => frame.binderId == instr.loopId) then
              scanBlocks plan
                (.routed instr.loopId dst instr.loopCount instr.routedOutputCount
                  instr.routedRoutes none :: frames) rest
            else none
        | _ => none
      else if instr.tag == "RoutedSumYield" then
        match frames, instr.dst with
        | .routed id dst capacity outputCount routes none :: outer, .array yieldDst =>
            let fanout := instr.args.size
            if yieldDst == dst && fanout > 0
                && routes.size == capacity * fanout
                && instr.args.all (scalarOperandWellFormed plan frames)
                && routes.all (fun route => route.all (· < outputCount)) then
              scanBlocks plan
                (.routed id dst capacity outputCount routes (some fanout) :: outer) rest
            else none
        | _, _ => none
      else if instr.tag == "RoutedSumEnd" then
        match frames, instr.dst with
        | .routed _ dst _ _ _ (some _) :: outer, .array endDst =>
            if dst == endDst && instr.args.isEmpty then scanBlocks plan outer rest
            else none
        | _, _ => none
      else if smallInstrWellFormed plan frames instr then
        scanBlocks plan frames rest
      else none

/-- A block is independently closed and every operand/destination is
addressable under its structured binder stack. -/
private def blocksWellFormed (plan : FlatPlan) (instrs : Array NInstr) : Bool :=
  match scanBlocks plan [] instrs.toList with
  | some [] => true
  | _ => false

def BlocksWellFormed (plan : FlatPlan) (instrs : Array NInstr) : Prop :=
  blocksWellFormed plan instrs = true

instance (plan : FlatPlan) (instrs : Array NInstr) :
    Decidable (BlocksWellFormed plan instrs) := by
  unfold BlocksWellFormed
  infer_instance

/-- Instruction-local well-formedness outside a structured region. -/
def NInstrWellFormed (plan : FlatPlan) (instr : NInstr) : Prop :=
  smallInstrWellFormed plan [] instr = true

instance (plan : FlatPlan) (instr : NInstr) :
    Decidable (NInstrWellFormed plan instr) := by
  unfold NInstrWellFormed
  infer_instance

private def instanceWellFormed (plan : FlatPlan) (inst : InstanceFunction) : Bool :=
  if !(inst.registerOffset + inst.registerCount <= plan.registerCount
      && inst.arraySlotOffset <= plan.arraySlotCount
      && decide (BlocksWellFormed plan inst.preambleInstructions)
      && decide (BlocksWellFormed plan inst.preInputInstructions)
      && decide (BlocksWellFormed plan inst.instructions)) then
    false
  else Id.run do
    for _h : child in inst.children do
      if !instanceWellFormed plan child then return false
    return true
termination_by sizeOf inst
decreasing_by
  exact Tropical.Plan.InstanceFunction.sizeOf_lt_of_mem_children _h

private def indicesUniqueAndInRange (limit : Nat) (indices : Array Nat) : Bool :=
  indices.all (· < limit) && indices.toList.Nodup

private def publicationRolesUnique (plan : FlatPlan) : Bool :=
  plan.coeffArraySlots.all (fun slot => !plan.tileArraySlots.contains slot)

private def sinkInputsInRange (plan : FlatPlan) : Bool :=
  plan.sinks.all (fun sink => sink.inputs.all (· < plan.slotCount))

/-- Host/environment shape needed to initialize a well-formed plan. -/
structure PlanEnvironmentWellFormed (plan : FlatPlan) (inputs : PlanInputs α) : Prop where
  slots : inputs.initialSlots.size = plan.slotCount
  arrays : inputs.initialArrays.size = plan.arraySlotCount
  arraySizes : ∀ index, index < plan.arraySlotCount →
    (inputs.initialArrays[index]?).map Array.size = plan.arraySlotSizes[index]?
  sources : inputs.sources.size = plan.sources.size

/-- Executable structural validator for a wire-ready Plan. Backend limits (for
example the native/Metal maximum channel width) remain separate obligations. -/
def flatPlanWellFormed (plan : FlatPlan) : Bool :=
  plan.outputLayoutWellFormed &&
  (plan.arraySlotNames.size == plan.arraySlotCount &&
  (plan.arraySlotSizes.size == plan.arraySlotCount &&
  (plan.slotNames.size == plan.slotCount &&
  (plan.slotDefaults.size == plan.slotCount &&
  ((plan.sources == defaultSources || plan.sources == tileSources) &&
  (indicesUniqueAndInRange plan.arraySlotCount plan.coeffArraySlots &&
  (indicesUniqueAndInRange plan.arraySlotCount plan.tileArraySlots &&
  (publicationRolesUnique plan &&
  ((plan.tileArraySlots.isEmpty || plan.tileIntervalFrames > 0) &&
  (plan.instanceFunctions.all (instanceWellFormed plan) &&
   sinkInputsInRange plan))))))))))

/-- Canonical theorem-facing structural predicate. -/
def FlatPlanWellFormed (plan : FlatPlan) : Prop :=
  flatPlanWellFormed plan = true

instance (plan : FlatPlan) : Decidable (FlatPlanWellFormed plan) := by
  unfold FlatPlanWellFormed
  infer_instance

theorem FlatPlanWellFormed.outputLayout {plan : FlatPlan}
    (hwf : FlatPlanWellFormed plan) : plan.outputLayoutWellFormed = true := by
  simp [FlatPlanWellFormed, flatPlanWellFormed] at hwf
  exact hwf.1

theorem planWellFormed_no_session_array_leak {plan : FlatPlan}
    (_hwf : FlatPlanWellFormed plan) {instr : NInstr}
    (hinstr : NInstrWellFormed plan instr) :
    (match instr.dst with | .sessionArray _ => False | _ => True) := by
  cases hdst : instr.dst <;> simp
  unfold NInstrWellFormed at hinstr
  cases hop : PlanOp.ofString? instr.tag <;>
    simp [smallInstrWellFormed, hop, hdst, scalarOpWellFormed,
      scalarDstWellFormed, arrayDst?] at hinstr

theorem planWellFormed_output_nonempty {plan : FlatPlan}
    (hwf : FlatPlanWellFormed plan) : plan.outputChannelCount > 0 := by
  cases Nat.eq_zero_or_pos plan.outputChannelCount with
  | inr hpositive => exact hpositive
  | inl hzero =>
      have hlayout := hwf.outputLayout
      have hfalse : plan.outputLayoutWellFormed = false := by
        simp [FlatPlan.outputLayoutWellFormed, hzero]
      rw [hfalse] at hlayout
      contradiction

end Tropical.Semantics
