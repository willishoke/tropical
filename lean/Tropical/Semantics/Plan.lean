import Tropical.Plan
import Tropical.Semantics.Environment

/-!
# Reference semantics for Plan-6

This module defines the carrier-facing state and the small-instruction waist of
the Plan interpreter.  Region execution is deliberately layered on top of this
surface: consumers such as Stage0 need a stable `PlanState`/`evalOperand`
interface without depending on a backend emitter.

Scalar kinds are validated by `PlanWellFormed`; the denotation keeps values in
the carrier-parametric `Value` type and never invents a value for a malformed
namespace lookup.
-/

namespace Tropical.Semantics

open Tropical.Plan
open Tropical.Ir

/-- Runtime values supplied from outside a Plan execution. Source positions are
resolved from the plan's `SourceKind` array before constructing this record. -/
structure PlanInputs (α : Type) where
  inputs : Array (Value α) := #[]
  params : String → Option (Value α) := fun _ => none
  sources : Array (Value α) := #[]
  initialSlots : Array (Value α) := #[]
  initialArrays : Array (Array (Value α)) := #[]

/-- Named source rails from which a Plan's positional source image is built.
Ordinary audio plans select `tick`/`rate`; tile materializers additionally
select `tilePhase`/`tileTick` in their declared order. -/
structure PlanSourceImage (α : Type) where
  tick : Value α
  rate : Value α
  tilePhase : Value α
  tileTick : Value α

def PlanSourceImage.value (image : PlanSourceImage α) : SourceKind → Value α
  | .tick => image.tick
  | .rate => image.rate
  | .tilePhase => image.tilePhase
  | .tileTick => image.tileTick

def PlanInputs.withDeclaredSources (inputs : PlanInputs α)
    (kinds : Array SourceKind) (image : PlanSourceImage α) : PlanInputs α :=
  { inputs with sources := kinds.map image.value }

/-- One open structured-region binder. Binder identifiers are nominal and are
resolved innermost-first, matching both production emitters. -/
structure LoopFrame (α : Type) where
  binderId : Nat
  index : Value α
deriving Repr

/-- Mutable reference state of one Plan execution. Temps are SSA-shaped in
well-formed plans but remain explicit because regions reuse scratch indices. -/
structure PlanState (α : Type) where
  temps : Array (Value α) := #[]
  slots : Array (Value α) := #[]
  arrays : Array (Array (Value α)) := #[]
  openLoops : List (LoopFrame α) := []
deriving Repr

private def planError (operation detail : String) : Outcome β :=
  .error { operation, detail }

private def lookupPlanValue (operation space : String)
    (xs : Array (Value α)) (idx : Nat) : Result α :=
  match xs[idx]? with
  | some value => .ok value
  | none => planError operation
      s!"{space} index {idx} is out of bounds (size {xs.size})"

private def lookupPlanArray (operation space : String)
    (xs : Array (Array (Value α))) (idx : Nat) : Result α :=
  match xs[idx]? with
  | some values => .ok (.array values)
  | none => planError operation
      s!"{space} index {idx} is out of bounds (size {xs.size})"

private def findLoopValue (frames : List (LoopFrame α)) (id : Nat) :
    Option (Value α) :=
  match frames with
  | [] => none
  | frame :: rest =>
      if frame.binderId == id then some frame.index else findLoopValue rest id

/-- Evaluate one operand against the current namespace image. Pre-remap session
arrays are deliberately refused at the wire-ready Plan boundary. -/
def evalOperand (alg : Algebra α) (inputs : PlanInputs α)
    (state : PlanState α) : NOperand → Result α
  | .const value _ => alg.literal value
  | .input slot _ => lookupPlanValue "operand.input" "input" inputs.inputs slot
  | .reg slot _ => lookupPlanValue "operand.reg" "temp" state.temps slot
  | .arrayReg slot => lookupPlanArray "operand.arrayReg" "array" state.arrays slot
  | .sessionArrayReg slot =>
      planError "operand.sessionArrayReg"
        s!"session array {slot} leaked to a wire-ready Plan"
  | .param ptr _ =>
      match inputs.params ptr with
      | some value => .ok value
      | none => planError "operand.param" s!"parameter '{ptr}' was not supplied"
  | .source index _ =>
      lookupPlanValue "operand.source" "source" inputs.sources index
  | .slot index _ => lookupPlanValue "operand.slot" "slot" state.slots index
  | .loopIdx id =>
      match findLoopValue state.openLoops id with
      | some value => .ok value
      | none => planError "operand.loopIdx" s!"binder {id} is not open"

theorem evalOperand_source_of_getElem (alg : Algebra α) (inputs : PlanInputs α)
    (state : PlanState α) (index : Nat) (ty : ScalarType) (value : Value α)
    (h : inputs.sources[index]? = some value) :
    evalOperand alg inputs state (.source index ty) = .ok value := by
  simp [evalOperand, lookupPlanValue, h]

private def planOpBinaryTag? : PlanOp → Option BinaryOpTag
  | .add => some .add | .sub => some .sub | .mul => some .mul
  | .div => some .div | .mod => some .mod | .floorDiv => some .floorDiv
  | .less => some .lt | .lessEq => some .lte | .greater => some .gt
  | .greaterEq => some .gte | .equal => some .eq | .notEqual => some .neq
  | .and => some .and | .or => some .or
  | .bitAnd => some .bitAnd | .bitOr => some .bitOr | .bitXor => some .bitXor
  | .lshift => some .lshift | .rshift => some .rshift | .ldexp => some .ldexp
  | _ => none

private def planOpUnaryTag? : PlanOp → Option UnaryOpTag
  | .neg => some .neg | .abs => some .abs | .sqrt => some .sqrt
  | .floor => some .floor | .ceil => some .ceil | .round => some .round
  | .not => some .not | .bitNot => some .bitNot
  | .floatExponent => some .floatExponent
  | .toInt => some .toInt | .toBool => some .toBool | .toFloat => some .toFloat
  | _ => none

private def planOpArity : PlanOp → Nat
  | .select | .clamp => 3
  | .neg | .abs | .sqrt | .floor | .ceil | .round
  | .not | .bitNot | .floatExponent | .toInt | .toBool | .toFloat => 1
  | _ => 2

/-- Interpret one scalar symbol from the total Plan signature. Arity errors are
structural refusals; carrier operations retain their own explicit refusals. -/
def evalPlanOp (alg : Algebra α) (op : PlanOp)
    (args : Array (Value α)) : Result α := do
  if args.size != planOpArity op then
    planError "PlanOp" s!"{op.name} expects {planOpArity op} args, got {args.size}"
  else
    match planOpBinaryTag? op, planOpUnaryTag? op with
    | some tag, _ => alg.binary tag args[0]! args[1]!
    | _, some tag => alg.unary tag args[0]!
    | none, none =>
        match op with
        | .clamp => alg.clamp args[0]! args[1]! args[2]!
        | .select => alg.select args[0]! args[1]! args[2]!
        | _ => planError "PlanOp" s!"unhandled scalar symbol {op.name}"

private def writeScalarDst (state : PlanState α) (dst : DstSlot)
    (value : Value α) : Outcome (PlanState α) :=
  match dst with
  | .temp slot =>
      if h : slot < state.temps.size then
        .ok { state with temps := state.temps.set slot value }
      else planError "instruction.dst" s!"temp {slot} is out of bounds"
  | .moduleSlot slot =>
      if h : slot < state.slots.size then
        .ok { state with slots := state.slots.set slot value }
      else planError "instruction.dst" s!"module slot {slot} is out of bounds"
  | .array slot =>
      planError "instruction.dst" s!"array {slot} cannot receive a scalar value"
  | .sessionArray slot =>
      planError "instruction.dst" s!"session array {slot} leaked to a wire-ready Plan"

private def writeArrayDst (state : PlanState α) (dst : DstSlot)
    (values : Array (Value α)) : Outcome (PlanState α) :=
  match dst with
  | .array slot =>
      if h : slot < state.arrays.size then
        .ok { state with arrays := state.arrays.set slot values }
      else planError "instruction.dst" s!"array {slot} is out of bounds"
  | .sessionArray slot =>
      planError "instruction.dst" s!"session array {slot} leaked to a wire-ready Plan"
  | .temp slot => planError "instruction.dst" s!"temp {slot} cannot receive an array value"
  | .moduleSlot slot =>
      planError "instruction.dst" s!"module slot {slot} cannot receive an array value"

private def evalOperands (alg : Algebra α) (inputs : PlanInputs α)
    (state : PlanState α) (operands : Array NOperand) :
    Outcome (Array (Value α)) :=
  sequence (operands.map (evalOperand alg inputs state))

private def evalElementwiseOperand (alg : Algebra α) (inputs : PlanInputs α)
    (state : PlanState α) (instr : NInstr) (position index : Nat) : Result α := do
  let some operand := instr.args[position]?
    | planError "instruction.elementwise"
        s!"argument position {position} is out of bounds"
  if (instr.strides[position]?.getD 0) == 1 then
    match operand with
    | .arrayReg slot =>
        let some values := state.arrays[slot]?
          | planError "instruction.elementwise"
              s!"array argument {slot} is out of bounds"
        lookupPlanValue "instruction.elementwise" "array element" values index
    | .sessionArrayReg slot =>
        planError "instruction.elementwise"
          s!"session array {slot} leaked to a wire-ready Plan"
    | _ =>
        planError "instruction.elementwise"
          s!"strided argument {position} is not an array operand"
  else
    evalOperand alg inputs state operand

/-- Execute the scalar loop represented by an array-destination Plan op. Array
arguments are selected exactly when the parallel stride entry is one; all
other operands are loop-invariant scalar broadcasts. -/
def execElementwiseInstr (alg : Algebra α) (inputs : PlanInputs α)
    (state : PlanState α) (instr : NInstr) (op : PlanOp) :
    Outcome (PlanState α) := do
  let values ← List.foldlM (fun (values : Array (Value α)) index => do
      let args ← List.foldlM (fun (args : Array (Value α)) position => do
          pure (args.push (← evalElementwiseOperand alg inputs state instr position index)))
        #[] (List.range instr.args.size)
      pure (values.push (← evalPlanOp alg op args)))
    #[] (List.range instr.loopCount)
  writeArrayDst state instr.dst values

/-- Execute an instruction that does not delimit a structured region.
Elementwise array scalar-ops and region delimiters are handled by `execBlocks`;
encountering them here is an explicit layering refusal. -/
def execSmallInstr (alg : Algebra α) (inputs : PlanInputs α)
    (state : PlanState α) (instr : NInstr) : Outcome (PlanState α) := do
  if let some op := PlanOp.ofString? instr.tag then
    match instr.dst with
    | .temp _ | .moduleSlot _ =>
        let args ← evalOperands alg inputs state instr.args
        writeScalarDst state instr.dst (← evalPlanOp alg op args)
    | .array _ | .sessionArray _ => execElementwiseInstr alg inputs state instr op
  else
    match instr.tag with
    | "Pack" =>
        writeArrayDst state instr.dst (← evalOperands alg inputs state instr.args)
    | "Index" =>
        if instr.args.size != 2 then
          planError "instruction.Index" s!"expected 2 args, got {instr.args.size}"
        else
          let args ← evalOperands alg inputs state instr.args
          writeScalarDst state instr.dst (← alg.index args[0]! args[1]!)
    | "WriteSlot" =>
        if instr.args.size != 1 then
          planError "instruction.WriteSlot" s!"expected 1 arg, got {instr.args.size}"
        else
          writeScalarDst state instr.dst (← evalOperand alg inputs state instr.args[0]!)
    | "SetElement" =>
        if instr.args.size != 3 then
          planError "instruction.SetElement" s!"expected 3 args, got {instr.args.size}"
        else
          let args ← evalOperands alg inputs state instr.args
          let .array values := args[0]!
            | planError "instruction.SetElement" "first argument is not an array"
          let rawIndex ← alg.dynamicCount args[1]!
          if rawIndex < 0 then
            .ok state
          else
            let index := rawIndex.toNat
            if h : index < values.size then
              writeArrayDst state instr.dst (values.set index args[2]!)
            else
              .ok state
    | "ReduceBegin" | "ReduceEnd" | "RoutedSumBegin" | "RoutedSumYield"
    | "RoutedSumEnd" =>
        planError "instruction.region"
          s!"delimiter {instr.tag} must execute through execBlocks"
    | _ => planError "instruction.tag" s!"unknown Plan instruction '{instr.tag}'"

private def splitRegionAux (beginTag endTag : String) :
    Nat → List NInstr → List NInstr → Outcome (List NInstr × List NInstr)
  | _, [], _ =>
      planError "instruction.region" s!"missing {endTag} for {beginTag}"
  | depth, instr :: rest, reversed =>
      if instr.tag == endTag then
        if depth == 0 then .ok (reversed.reverse, rest)
        else splitRegionAux beginTag endTag (depth - 1) rest (instr :: reversed)
      else if instr.tag == beginTag then
        splitRegionAux beginTag endTag (depth + 1) rest (instr :: reversed)
      else
        splitRegionAux beginTag endTag depth rest (instr :: reversed)

/-- Split the body and suffix at the matching delimiter without unrolling the
region. Nested regions of the same kind remain in the body and are interpreted
recursively. -/
private def splitRegion (beginTag endTag : String) (rest : List NInstr) :
    Outcome (List NInstr × List NInstr) :=
  splitRegionAux beginTag endTag 0 rest []

private theorem splitRegionAux_length {beginTag endTag : String}
    {depth : Nat} {rest reversed body suffix : List NInstr}
    (hsplit : splitRegionAux beginTag endTag depth rest reversed =
      .ok (body, suffix)) :
    body.length + suffix.length + 1 = reversed.length + rest.length := by
  induction rest generalizing depth reversed with
  | nil => simp [splitRegionAux, planError] at hsplit
  | cons instr rest ih =>
      simp only [splitRegionAux] at hsplit
      split at hsplit
      next hend =>
        split at hsplit
        next hzero =>
          rcases hsplit with ⟨rfl, rfl⟩
          simp
          omega
        next hdepth =>
          have hrec := ih hsplit
          simp at hrec ⊢
          omega
      next hnotEnd =>
        split at hsplit
        next hbegin =>
          have hrec := ih hsplit
          simp at hrec ⊢
          omega
        next hnotBegin =>
          have hrec := ih hsplit
          simp at hrec ⊢
          omega

private theorem splitRegion_lengths {beginTag endTag : String}
    {rest body suffix : List NInstr}
    (hsplit : splitRegion beginTag endTag rest = .ok (body, suffix)) :
    body.length < rest.length + 1 ∧ suffix.length < rest.length + 1 := by
  change splitRegionAux beginTag endTag 0 rest [] = .ok (body, suffix) at hsplit
  have hlength := splitRegionAux_length hsplit
  simp at hlength
  omega

private theorem splitRegionAux_append_of_success {beginTag endTag : String}
    {depth : Nat} {rest reversed body suffix : List NInstr}
    (tail : List NInstr)
    (hsplit : splitRegionAux beginTag endTag depth rest reversed =
      .ok (body, suffix)) :
    splitRegionAux beginTag endTag depth (rest ++ tail) reversed =
      .ok (body, suffix ++ tail) := by
  induction rest generalizing depth reversed with
  | nil => simp [splitRegionAux, planError] at hsplit
  | cons instr rest ih =>
      simp only [List.cons_append, splitRegionAux] at hsplit ⊢
      by_cases hend : instr.tag == endTag
      · simp only [if_pos hend] at hsplit ⊢
        by_cases hzero : depth == 0
        · simp only [if_pos hzero] at hsplit ⊢
          rcases hsplit with ⟨rfl, rfl⟩
          rfl
        · simp only [if_neg hzero] at hsplit ⊢
          exact ih hsplit
      · simp only [if_neg hend] at hsplit ⊢
        by_cases hbegin : instr.tag == beginTag
        · simp only [if_pos hbegin] at hsplit ⊢
          exact ih hsplit
        · simp only [if_neg hbegin] at hsplit ⊢
          exact ih hsplit

private theorem splitRegion_append_of_success {beginTag endTag : String}
    {rest body suffix : List NInstr} (tail : List NInstr)
    (hsplit : splitRegion beginTag endTag rest = .ok (body, suffix)) :
    splitRegion beginTag endTag (rest ++ tail) = .ok (body, suffix ++ tail) := by
  exact splitRegionAux_append_of_success tail hsplit

private def splitRoutedYieldAux : Nat → List NInstr → List NInstr →
    Outcome (List NInstr × NInstr × List NInstr)
  | _, [], _ =>
      planError "instruction.RoutedSum" "region is missing RoutedSumYield"
  | reduceDepth, instr :: rest, reversed =>
      if instr.tag == "ReduceBegin" then
        splitRoutedYieldAux (reduceDepth + 1) rest (instr :: reversed)
      else if instr.tag == "ReduceEnd" then
        if reduceDepth == 0 then
          planError "instruction.RoutedSum" "unmatched ReduceEnd in routed body"
        else
          splitRoutedYieldAux (reduceDepth - 1) rest (instr :: reversed)
      else if instr.tag == "RoutedSumBegin" then
        planError "instruction.RoutedSum" "nested routed regions are not supported"
      else if instr.tag == "RoutedSumYield" && reduceDepth == 0 then
        .ok (reversed.reverse, instr, rest)
      else
        splitRoutedYieldAux reduceDepth rest (instr :: reversed)

private def splitRoutedYield (body : List NInstr) :
    Outcome (List NInstr × NInstr × List NInstr) :=
  splitRoutedYieldAux 0 body []

private theorem splitRoutedYieldAux_length {reduceDepth : Nat}
    {body reversed mapped afterYield : List NInstr} {yieldInstr : NInstr}
    (hsplit : splitRoutedYieldAux reduceDepth body reversed =
      .ok (mapped, yieldInstr, afterYield)) :
    mapped.length + afterYield.length + 1 = reversed.length + body.length := by
  induction body generalizing reduceDepth reversed with
  | nil => simp [splitRoutedYieldAux, planError] at hsplit
  | cons instr body ih =>
      simp only [splitRoutedYieldAux] at hsplit
      split at hsplit
      next hreduce =>
        have hrec := ih hsplit
        simp at hrec ⊢
        omega
      next hnotReduce =>
        split at hsplit
        next hend =>
          split at hsplit
          next hzero => simp [planError] at hsplit
          next hdepth =>
            have hrec := ih hsplit
            simp at hrec ⊢
            omega
        next hnotEnd =>
          split at hsplit
          next hnested => simp [planError] at hsplit
          next hnotNested =>
            split at hsplit
            next hyield =>
              rcases hsplit with ⟨rfl, rfl, rfl⟩
              simp
              omega
            next hnotYield =>
              have hrec := ih hsplit
              simp at hrec ⊢
              omega

private theorem splitRoutedYield_lengths {body mapped afterYield : List NInstr}
    {yieldInstr : NInstr}
    (hsplit : splitRoutedYield body = .ok (mapped, yieldInstr, afterYield)) :
    mapped.length < body.length + 1 ∧ afterYield.length < body.length + 1 := by
  change splitRoutedYieldAux 0 body [] =
    .ok (mapped, yieldInstr, afterYield) at hsplit
  have hlength := splitRoutedYieldAux_length hsplit
  simp at hlength
  omega

/-- Syntactic closure at the structured-block boundary used by `execBlocks`.
It records exactly the condition needed for appending a continuation: every
top-level region opened in the prefix is closed before that continuation. -/
def blocksStructurallyClosedList : List NInstr → Bool
  | [] => true
  | instr :: rest =>
      if instr.tag == "ReduceBegin" then
        match _hreduceSplit : splitRegion "ReduceBegin" "ReduceEnd" rest with
        | .ok (_, suffix) => blocksStructurallyClosedList suffix
        | .error _ => false
      else if instr.tag == "RoutedSumBegin" then
        match _hroutedSplit : splitRegion "RoutedSumBegin" "RoutedSumEnd" rest with
        | .ok (_, suffix) => blocksStructurallyClosedList suffix
        | .error _ => false
      else if instr.tag == "ReduceEnd" || instr.tag == "RoutedSumYield"
          || instr.tag == "RoutedSumEnd" then
        false
      else
        blocksStructurallyClosedList rest
termination_by blocks => blocks.length
decreasing_by
  · have hsizes := splitRegion_lengths _hreduceSplit
    simpa using hsizes.2
  · have hsizes := splitRegion_lengths _hroutedSplit
    simpa using hsizes.2
  · simp

def BlocksStructurallyClosed (instrs : Array NInstr) : Prop :=
  blocksStructurallyClosedList instrs.toList = true

instance (instrs : Array NInstr) : Decidable (BlocksStructurallyClosed instrs) := by
  unfold BlocksStructurallyClosed
  infer_instance

private def loopIdIsOpen (state : PlanState α) (id : Nat) : Bool :=
  state.openLoops.any (fun frame => frame.binderId == id)

private def dynamicTrips (alg : Algebra α) (inputs : PlanInputs α)
    (state : PlanState α) (capacity : Nat) (count? : Option NOperand) :
    Outcome Nat :=
  bankTrips alg capacity (count?.map (evalOperand alg inputs state))

private def applyRoutedValues (alg : Algebra α) (state : PlanState α)
    (dst item fanout outputCount : Nat) (routes : Array (Option Nat))
    (values : Array (Value α)) : Outcome (PlanState α) := do
  let some outputs := state.arrays[dst]?
    | planError "instruction.RoutedSum" s!"array {dst} is out of bounds"
  if outputs.size != outputCount then
    planError "instruction.RoutedSum"
      s!"destination width {outputs.size} does not match output count {outputCount}"
  else
    let next ← List.foldlM (fun current emit => do
        match routes[item * fanout + emit]? with
        | none =>
            planError "instruction.RoutedSum" "route metadata is out of bounds"
        | some none => pure current
        | some (some output) =>
            if output >= outputCount then
              planError "instruction.RoutedSum"
                s!"route target {output} is outside output count {outputCount}"
            else
              let value ← alg.binary .add current[output]! values[emit]!
              pure (current.set! output value))
      outputs (List.range fanout)
    writeArrayDst state (.array dst) next

private def execReductionRegion
    (runBlocks : PlanState α → List NInstr → Outcome (PlanState α))
    (alg : Algebra α) (inputs : PlanInputs α) (state : PlanState α)
    (instr : NInstr) (accTemp : Nat) (body : List NInstr) :
    Outcome (PlanState α) := do
  let init ← evalOperand alg inputs state instr.args[0]!
  let trips ← dynamicTrips alg inputs state instr.loopCount instr.args[1]?
  let seeded ← writeScalarDst state instr.dst init
  let entryTemps := seeded.temps
  let outerLoops := seeded.openLoops
  List.foldlM (fun current item => do
      let acc ← lookupPlanValue "instruction.Reduce" "accumulator"
        current.temps accTemp
      let loopValue ← alg.loopIndex item
      let iteration : PlanState α := {
        current with
        temps := entryTemps.set! accTemp acc
        openLoops := { binderId := instr.loopId, index := loopValue } :: outerLoops
      }
      let bodyState ← runBlocks iteration body
      let nextAcc ← lookupPlanValue "instruction.Reduce" "accumulator"
        bodyState.temps accTemp
      pure {
        bodyState with
        temps := entryTemps.set! accTemp nextAcc
        openLoops := outerLoops
      }) seeded (List.range trips)

private def execRoutedRegion
    (runBlocks : PlanState α → List NInstr → Outcome (PlanState α))
    (alg : Algebra α) (inputs : PlanInputs α) (state : PlanState α)
    (instr : NInstr) (dst : Nat) (mapped : List NInstr)
    (yieldInstr : NInstr) (afterYield : List NInstr) :
    Outcome (PlanState α) := do
  let (.array yieldDst) := yieldInstr.dst
    | planError "instruction.RoutedSumYield" "destination is not an array"
  let fanout := yieldInstr.args.size
  if yieldDst != dst then
    planError "instruction.RoutedSumYield" "destination does not match its region"
  else if instr.loopCount == 0 || instr.routedOutputCount == 0 || fanout == 0 then
    planError "instruction.RoutedSumBegin" "capacity, output count, and fanout must be nonzero"
  else if instr.routedRoutes.size != instr.loopCount * fanout then
    planError "instruction.RoutedSumBegin"
      s!"route count {instr.routedRoutes.size} does not equal capacity×fanout {instr.loopCount * fanout}"
  else
    let trips ← dynamicTrips alg inputs state instr.loopCount instr.args[0]?
    let zero ← alg.zero
    let seeded ← writeArrayDst state instr.dst
      (Array.replicate instr.routedOutputCount zero)
    let entryTemps := seeded.temps
    let outerLoops := seeded.openLoops
    List.foldlM (fun current item => do
        let loopValue ← alg.loopIndex item
        let iteration : PlanState α := {
          current with
          temps := entryTemps
          openLoops := { binderId := instr.loopId, index := loopValue } :: outerLoops
        }
        let mappedState ← runBlocks iteration mapped
        let values ← evalOperands alg inputs mappedState yieldInstr.args
        let accumulated ← applyRoutedValues alg mappedState dst item fanout
          instr.routedOutputCount instr.routedRoutes values
        let afterState ← runBlocks accumulated afterYield
        pure { afterState with temps := entryTemps, openLoops := outerLoops })
      seeded (List.range trips)

private theorem execReductionRegion_congr
    {runLeft runRight : PlanState α → List NInstr → Outcome (PlanState α)}
    (body : List NInstr)
    (hrun : ∀ state, runLeft state body = runRight state body)
    (alg : Algebra α) (inputs : PlanInputs α) (state : PlanState α)
    (instr : NInstr) (accTemp : Nat) :
    execReductionRegion runLeft alg inputs state instr accTemp body =
      execReductionRegion runRight alg inputs state instr accTemp body := by
  simp [execReductionRegion, hrun]

private theorem execRoutedRegion_congr
    {runLeft runRight : PlanState α → List NInstr → Outcome (PlanState α)}
    (mapped afterYield : List NInstr)
    (hmapped : ∀ state, runLeft state mapped = runRight state mapped)
    (hafter : ∀ state, runLeft state afterYield = runRight state afterYield)
    (alg : Algebra α) (inputs : PlanInputs α) (state : PlanState α)
    (instr : NInstr) (dst : Nat) (yieldInstr : NInstr) :
    execRoutedRegion runLeft alg inputs state instr dst mapped yieldInstr afterYield =
      execRoutedRegion runRight alg inputs state instr dst mapped yieldInstr afterYield := by
  simp [execRoutedRegion, hmapped, hafter]

/-- Execute one structurally closed instruction block. Regions are parsed at
their delimiters and evaluated directly: no unrolled instruction stream is
constructed. Reduction and routed bodies run in increasing item order; their
body-local temps and binder frames do not escape the region. -/
def execBlocksListFuel (fuel : Nat) (alg : Algebra α)
    (inputs : PlanInputs α) (state : PlanState α) (blocks : List NInstr) :
    Outcome (PlanState α) :=
  match fuel with
  | 0 => planError "instruction.region" "structured execution fuel exhausted"
  | fuel + 1 =>
    match blocks with
    | [] => .ok state
    | instr :: rest => do
      if instr.tag == "ReduceBegin" then
        let (.temp accTemp) := instr.dst
          | planError "instruction.ReduceBegin" "destination is not a temp"
        if loopIdIsOpen state instr.loopId then
          planError "instruction.ReduceBegin"
            s!"binder {instr.loopId} collides with an open region"
        else if instr.args.size != 1 && instr.args.size != 2 then
          planError "instruction.ReduceBegin"
            s!"expected init and optional count, got {instr.args.size} args"
        else
          let (body, suffix) ← splitRegion "ReduceBegin" "ReduceEnd" rest
          let reduced ← execReductionRegion
            (execBlocksListFuel fuel alg inputs) alg inputs state instr accTemp body
          execBlocksListFuel fuel alg inputs reduced suffix
      else if instr.tag == "RoutedSumBegin" then
        let (.array dst) := instr.dst
          | planError "instruction.RoutedSumBegin" "destination is not an array"
        if loopIdIsOpen state instr.loopId then
          planError "instruction.RoutedSumBegin"
            s!"binder {instr.loopId} collides with an open region"
        else if instr.args.size > 1 then
          planError "instruction.RoutedSumBegin"
            s!"expected at most one dynamic count, got {instr.args.size} args"
        else
          let (body, suffix) ← splitRegion "RoutedSumBegin" "RoutedSumEnd" rest
          let (mapped, yieldInstr, afterYield) ← splitRoutedYield body
          let routed ← execRoutedRegion
            (execBlocksListFuel fuel alg inputs) alg inputs state instr dst mapped
              yieldInstr afterYield
          execBlocksListFuel fuel alg inputs routed suffix
      else if instr.tag == "ReduceEnd" || instr.tag == "RoutedSumYield"
          || instr.tag == "RoutedSumEnd" then
        planError "instruction.region" s!"unmatched delimiter {instr.tag}"
      else
        execBlocksListFuel fuel alg inputs
          (← execSmallInstr alg inputs state instr) rest
termination_by fuel

theorem execBlocksListFuel_irrel (alg : Algebra α) (inputs : PlanInputs α)
    (state : PlanState α) (blocks : List NInstr) (left right : Nat)
    (hleft : blocks.length < left) (hright : blocks.length < right) :
    execBlocksListFuel left alg inputs state blocks =
      execBlocksListFuel right alg inputs state blocks := by
  cases left with
  | zero => omega
  | succ left =>
    cases right with
    | zero => omega
    | succ right =>
      cases hblocks : blocks with
      | nil => simp [execBlocksListFuel]
      | cons instr rest =>
        simp only [execBlocksListFuel]
        by_cases hreduce : instr.tag == "ReduceBegin"
        · simp only [if_pos hreduce]
          cases hdst : instr.dst with
          | temp accTemp =>
            by_cases hcollision : loopIdIsOpen state instr.loopId
            · simp [hcollision, planError]
            · simp only [if_neg hcollision]
              by_cases harity : instr.args.size != 1 && instr.args.size != 2
              · simp [harity, planError]
              · simp only [if_neg harity]
                cases hsplit : splitRegion "ReduceBegin" "ReduceEnd" rest with
                | error error => rfl
                | ok split =>
                  rcases split with ⟨body, suffix⟩
                  have hsizes := splitRegion_lengths hsplit
                  have hblocksLength : blocks.length = rest.length + 1 := by
                    rw [hblocks]
                    simp
                  have hbodyLeft : body.length < left := by omega
                  have hbodyRight : body.length < right := by omega
                  have hsuffixLeft : suffix.length < left := by omega
                  have hsuffixRight : suffix.length < right := by omega
                  have hbody : ∀ current,
                      execBlocksListFuel left alg inputs current body =
                        execBlocksListFuel right alg inputs current body := fun current =>
                    execBlocksListFuel_irrel alg inputs current body left right
                      hbodyLeft hbodyRight
                  have hregion := execReductionRegion_congr body hbody alg inputs
                    state instr accTemp
                  change (execReductionRegion (execBlocksListFuel left alg inputs)
                      alg inputs state instr accTemp body >>= fun reduced =>
                        execBlocksListFuel left alg inputs reduced suffix) =
                    (execReductionRegion (execBlocksListFuel right alg inputs)
                      alg inputs state instr accTemp body >>= fun reduced =>
                        execBlocksListFuel right alg inputs reduced suffix)
                  rw [hregion]
                  cases hresult : execReductionRegion
                      (execBlocksListFuel right alg inputs) alg inputs state instr
                        accTemp body with
                  | error error => rfl
                  | ok reduced =>
                    simp
                    exact execBlocksListFuel_irrel alg inputs reduced suffix left right
                      hsuffixLeft hsuffixRight
          | array _ | sessionArray _ | moduleSlot _ => simp
        · simp only [if_neg hreduce]
          by_cases hrouted : instr.tag == "RoutedSumBegin"
          · simp only [if_pos hrouted]
            cases hdst : instr.dst with
            | array dst =>
              by_cases hcollision : loopIdIsOpen state instr.loopId
              · simp [hcollision]
              · simp only [if_neg hcollision]
                by_cases harity : instr.args.size > 1
                · simp [harity]
                · simp only [if_neg harity]
                  cases hsplit : splitRegion "RoutedSumBegin" "RoutedSumEnd" rest with
                  | error error => rfl
                  | ok split =>
                    rcases split with ⟨body, suffix⟩
                    cases hyield : splitRoutedYield body with
                    | error error => simp [bind, Except.bind, hyield]
                    | ok yieldSplit =>
                      rcases yieldSplit with ⟨mapped, yieldInstr, afterYield⟩
                      have hsizes := splitRegion_lengths hsplit
                      have hyieldSizes := splitRoutedYield_lengths hyield
                      have hblocksLength : blocks.length = rest.length + 1 := by
                        rw [hblocks]
                        simp
                      have hmappedLeft : mapped.length < left := by omega
                      have hmappedRight : mapped.length < right := by omega
                      have hafterLeft : afterYield.length < left := by omega
                      have hafterRight : afterYield.length < right := by omega
                      have hsuffixLeft : suffix.length < left := by omega
                      have hsuffixRight : suffix.length < right := by omega
                      have hmapped : ∀ current,
                          execBlocksListFuel left alg inputs current mapped =
                            execBlocksListFuel right alg inputs current mapped := fun current =>
                        execBlocksListFuel_irrel alg inputs current mapped left right
                          hmappedLeft hmappedRight
                      have hafter : ∀ current,
                          execBlocksListFuel left alg inputs current afterYield =
                            execBlocksListFuel right alg inputs current afterYield := fun current =>
                        execBlocksListFuel_irrel alg inputs current afterYield left right
                          hafterLeft hafterRight
                      have hregion := execRoutedRegion_congr mapped afterYield hmapped hafter
                        alg inputs state instr dst yieldInstr
                      simp only [bind, Except.bind, hyield]
                      change (execRoutedRegion (execBlocksListFuel left alg inputs)
                          alg inputs state instr dst mapped yieldInstr afterYield >>= fun routed =>
                            execBlocksListFuel left alg inputs routed suffix) =
                        (execRoutedRegion (execBlocksListFuel right alg inputs)
                          alg inputs state instr dst mapped yieldInstr afterYield >>= fun routed =>
                            execBlocksListFuel right alg inputs routed suffix)
                      rw [hregion]
                      cases hresult : execRoutedRegion
                          (execBlocksListFuel right alg inputs) alg inputs state instr dst
                            mapped yieldInstr afterYield with
                      | error error => rfl
                      | ok routed =>
                        simp
                        exact execBlocksListFuel_irrel alg inputs routed suffix left right
                          hsuffixLeft hsuffixRight
            | temp _ | sessionArray _ | moduleSlot _ => simp
          · simp only [if_neg hrouted]
            by_cases hdelimiter : instr.tag == "ReduceEnd" ||
                instr.tag == "RoutedSumYield" || instr.tag == "RoutedSumEnd"
            · simp [hdelimiter]
            · simp only [if_neg hdelimiter]
              cases hsmall : execSmallInstr alg inputs state instr with
              | error error => rfl
              | ok next =>
                simp
                apply execBlocksListFuel_irrel alg inputs next rest left right
                · have hblocksLength : blocks.length = rest.length + 1 := by
                    rw [hblocks]
                    simp
                  omega
                · have hblocksLength : blocks.length = rest.length + 1 := by
                    rw [hblocks]
                    simp
                  omega
termination_by blocks.length
decreasing_by
  · rw [hblocks]
    simpa using hsizes.1
  · rw [hblocks]
    simpa using hsizes.2
  · rw [hblocks]
    have : mapped.length < rest.length + 1 := by omega
    simpa using this
  · rw [hblocks]
    have : afterYield.length < rest.length + 1 := by omega
    simpa using this
  · rw [hblocks]
    simpa using hsizes.2
  · rw [hblocks]
    simp

theorem execBlocksListFuel_append_of_structurallyClosed
    (alg : Algebra α) (inputs : PlanInputs α) (state : PlanState α)
    (head suffix : List NInstr) (fuel suffixFuel : Nat)
    (hclosed : blocksStructurallyClosedList head = true)
    (hfuel : (head ++ suffix).length < fuel)
    (hsuffixFuel : suffix.length < suffixFuel) :
    execBlocksListFuel fuel alg inputs state (head ++ suffix) =
      (execBlocksListFuel fuel alg inputs state head >>= fun next =>
        execBlocksListFuel suffixFuel alg inputs next suffix) := by
  cases fuel with
  | zero => omega
  | succ fuel =>
    cases suffixFuel with
    | zero => omega
    | succ suffixFuel =>
      cases hprefix : head with
      | nil =>
        simpa [execBlocksListFuel] using
          execBlocksListFuel_irrel alg inputs state suffix (fuel + 1)
            (suffixFuel + 1) (by simpa [hprefix] using hfuel) hsuffixFuel
      | cons instr rest =>
        have hheadLength : head.length = rest.length + 1 := by
          rw [hprefix]
          simp
        simp only [hprefix, blocksStructurallyClosedList] at hclosed
        simp only [List.cons_append, execBlocksListFuel, bind, Except.bind]
        by_cases hreduce : instr.tag == "ReduceBegin"
        · simp only [if_pos hreduce] at hclosed ⊢
          cases hdst : instr.dst with
          | temp accTemp =>
            by_cases hcollision : loopIdIsOpen state instr.loopId
            · simp only [if_pos hcollision]
              rfl
            · simp only [if_neg hcollision]
              by_cases harity : instr.args.size != 1 && instr.args.size != 2
              · simp only [if_pos harity]
                rfl
              · simp only [if_neg harity]
                cases hsplit : splitRegion "ReduceBegin" "ReduceEnd" rest with
                | error error =>
                  rw [hsplit] at hclosed
                  contradiction
                | ok split =>
                  rcases split with ⟨body, closedSuffix⟩
                  rw [hsplit] at hclosed
                  simp at hclosed
                  have hsizes := splitRegion_lengths hsplit
                  have happend := splitRegion_append_of_success suffix hsplit
                  simp only [happend]
                  cases hregion : execReductionRegion
                      (execBlocksListFuel fuel alg inputs) alg inputs state instr
                        accTemp body with
                  | error error => rfl
                  | ok reduced =>
                    exact execBlocksListFuel_append_of_structurallyClosed
                      alg inputs reduced closedSuffix suffix fuel (suffixFuel + 1)
                      hclosed (by simp at hfuel ⊢; omega) hsuffixFuel
          | array _ | sessionArray _ | moduleSlot _ => rfl
        · simp only [if_neg hreduce] at hclosed ⊢
          by_cases hrouted : instr.tag == "RoutedSumBegin"
          · simp only [if_pos hrouted] at hclosed ⊢
            cases hdst : instr.dst with
            | array dst =>
              by_cases hcollision : loopIdIsOpen state instr.loopId
              · simp only [if_pos hcollision]
                rfl
              · simp only [if_neg hcollision]
                by_cases harity : instr.args.size > 1
                · simp only [if_pos harity]
                  rfl
                · simp only [if_neg harity]
                  cases hsplit : splitRegion "RoutedSumBegin" "RoutedSumEnd" rest with
                  | error error =>
                    rw [hsplit] at hclosed
                    contradiction
                  | ok split =>
                    rcases split with ⟨body, closedSuffix⟩
                    rw [hsplit] at hclosed
                    simp at hclosed
                    have hsizes := splitRegion_lengths hsplit
                    have happend := splitRegion_append_of_success suffix hsplit
                    simp only [happend]
                    cases hyield : splitRoutedYield body with
                    | error error => rfl
                    | ok yieldSplit =>
                      rcases yieldSplit with ⟨mapped, yieldInstr, afterYield⟩
                      simp only [hyield, bind, Except.bind]
                      cases hregion : execRoutedRegion
                          (execBlocksListFuel fuel alg inputs) alg inputs state instr dst
                            mapped yieldInstr afterYield with
                      | error error => rfl
                      | ok routed =>
                        exact execBlocksListFuel_append_of_structurallyClosed
                          alg inputs routed closedSuffix suffix fuel (suffixFuel + 1)
                          hclosed (by simp at hfuel ⊢; omega) hsuffixFuel
            | temp _ | sessionArray _ | moduleSlot _ => rfl
          · simp only [if_neg hrouted] at hclosed ⊢
            by_cases hdelimiter : instr.tag == "ReduceEnd" ||
                instr.tag == "RoutedSumYield" || instr.tag == "RoutedSumEnd"
            · simp [hdelimiter] at hclosed
            · simp only [if_neg hdelimiter] at hclosed ⊢
              cases hsmall : execSmallInstr alg inputs state instr with
              | error error => rfl
              | ok next =>
                exact execBlocksListFuel_append_of_structurallyClosed
                  alg inputs next rest suffix fuel (suffixFuel + 1)
                  hclosed (by simp at hfuel ⊢; omega) hsuffixFuel
termination_by head.length
decreasing_by
  · rw [hprefix]
    simpa using hsizes.2
  · rw [hprefix]
    simpa using hsizes.2
  · omega

/-- Public structured-block semantic interface. The fuel is a termination
witness only: a recursive region body always has strictly fewer delimiters than
its enclosing call. -/
def execBlocks (alg : Algebra α) (inputs : PlanInputs α)
    (state : PlanState α) (instrs : Array NInstr) : Outcome (PlanState α) :=
  execBlocksListFuel (instrs.size + 1) alg inputs state instrs.toList

theorem execBlocks_append_of_structurallyClosed
    (alg : Algebra α) (inputs : PlanInputs α) (state : PlanState α)
    (head suffix : Array NInstr) (hclosed : BlocksStructurallyClosed head) :
    execBlocks alg inputs state (head ++ suffix) =
      (execBlocks alg inputs state head >>= fun next =>
        execBlocks alg inputs next suffix) := by
  unfold execBlocks
  unfold BlocksStructurallyClosed at hclosed
  rw [Array.toList_append]
  have hcompose := execBlocksListFuel_append_of_structurallyClosed
    alg inputs state head.toList suffix.toList ((head ++ suffix).size + 1)
      (suffix.size + 1) hclosed (by simp) (by simp)
  rw [hcompose]
  rw [execBlocksListFuel_irrel alg inputs state head.toList
    ((head ++ suffix).size + 1) (head.size + 1) (by simp; omega) (by simp)]

theorem execBlocks_deterministic (alg : Algebra α) (inputs : PlanInputs α)
    (state : PlanState α) (instrs : Array NInstr)
    {first second : Outcome (PlanState α)}
    (hfirst : execBlocks alg inputs state instrs = first)
    (hsecond : execBlocks alg inputs state instrs = second) : first = second :=
  hfirst.symm.trans hsecond

/-- Execute one recursive instance in the production order: preamble, then
each child's pre-input block and recursive body, then the parent body. -/
def execInstanceFunction (alg : Algebra α) (inputs : PlanInputs α)
    (state : PlanState α) (inst : InstanceFunction) : Outcome (PlanState α) := do
  let mut current ← execBlocks alg inputs state inst.preambleInstructions
  for _h : child in inst.children do
    current ← execBlocks alg inputs current child.preInputInstructions
    current ← execInstanceFunction alg inputs current child
  execBlocks alg inputs current inst.instructions
termination_by sizeOf inst
decreasing_by
  exact Tropical.Plan.InstanceFunction.sizeOf_lt_of_mem_children _h

theorem execInstanceFunction_deterministic
    (alg : Algebra α) (inputs : PlanInputs α) (state : PlanState α)
    (inst : InstanceFunction) {first second : Outcome (PlanState α)}
    (hfirst : execInstanceFunction alg inputs state inst = first)
    (hsecond : execInstanceFunction alg inputs state inst = second) :
    first = second :=
  hfirst.symm.trans hsecond

/-- Execute all top-level instance functions in authored array order. -/
def execPlanFunctions (alg : Algebra α) (inputs : PlanInputs α)
    (state : PlanState α) (plan : FlatPlan) : Outcome (PlanState α) :=
  List.foldlM (execInstanceFunction alg inputs) state plan.instanceFunctions.toList

/-- Construct the explicit initial state. Hosts must supply the decoded slot
and array images; Plan semantics does not silently reinterpret JSON defaults. -/
def initialPlanState (alg : Algebra α) (inputs : PlanInputs α)
    (plan : FlatPlan) : Outcome (PlanState α) := do
  if inputs.initialSlots.size != plan.slotCount then
    planError "plan.initialSlots"
      s!"expected {plan.slotCount} slots, got {inputs.initialSlots.size}"
  else if inputs.initialArrays.size != plan.arraySlotCount then
    planError "plan.initialArrays"
      s!"expected {plan.arraySlotCount} arrays, got {inputs.initialArrays.size}"
  else if inputs.sources.size != plan.sources.size then
    planError "plan.sources"
      s!"expected {plan.sources.size} sources, got {inputs.sources.size}"
  else if plan.arraySlotSizes.size != plan.arraySlotCount then
    planError "plan.arraySlotSizes" "array size metadata is not aligned"
  else
    for index in [0:plan.arraySlotCount] do
      if inputs.initialArrays[index]!.size != plan.arraySlotSizes[index]! then
        planError "plan.initialArrays"
          s!"array {index} has size {inputs.initialArrays[index]!.size}, expected {plan.arraySlotSizes[index]!}"
    let zero ← alg.zero
    pure {
      temps := Array.replicate plan.registerCount zero
      slots := inputs.initialSlots
      arrays := inputs.initialArrays
    }

/-- Target-indexed output values produced from a completed Plan state. The
array index is the logical output channel. -/
abbrev SinkImage (α : Type) := Array (Value α)

private def denoteSink (alg : Algebra α) (state : PlanState α)
    (sink : SinkSpec) : Outcome (Value α) := do
  let zero : Value α ← alg.zero
  let mixed : Value α ← List.foldlM (fun acc slot => do
      alg.binary .add acc (← lookupPlanValue "sink.input" "slot" state.slots slot))
    zero sink.inputs.toList
  let gain : Value α ← alg.literal sink.gain
  alg.binary .mul mixed gain

/-- Observe all pushed outputs after instruction execution. Missing channels
are zero; malformed duplicate or out-of-range targets are refused rather than
assigned an order-dependent meaning. -/
def denoteSinks (alg : Algebra α) (plan : FlatPlan)
    (state : PlanState α) : Outcome (SinkImage α) := do
  if !plan.outputLayoutWellFormed then
    planError "sinks" "output channel count must be positive and targets unique and in range"
  else
    let zero : Value α ← alg.zero
    List.foldlM (fun image sink => do
        pure (image.set! sink.target (← denoteSink alg state sink)))
      (Array.replicate plan.outputChannelCount zero) plan.sinks.toList

/-- Execute a complete Plan from its explicit initial image and observe the
target-indexed sink image. -/
def denoteFlatPlan (alg : Algebra α) (inputs : PlanInputs α)
    (plan : FlatPlan) : Outcome (SinkImage α) := do
  let initial ← initialPlanState alg inputs plan
  denoteSinks alg plan (← execPlanFunctions alg inputs initial plan)

theorem denoteFlatPlan_deterministic
    (alg : Algebra α) (inputs : PlanInputs α) (plan : FlatPlan)
    {first second : Outcome (SinkImage α)}
    (hfirst : denoteFlatPlan alg inputs plan = first)
    (hsecond : denoteFlatPlan alg inputs plan = second) : first = second :=
  hfirst.symm.trans hsecond

end Tropical.Semantics
