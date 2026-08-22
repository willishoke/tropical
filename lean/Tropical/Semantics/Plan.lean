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

end Tropical.Semantics
