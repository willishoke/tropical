import Tropical.Compile
import Tropical.PlanDecode

/-!
# Plan output-layout contract fixtures

Pure compile-time checks for Plan-6 mono compatibility and canonical
multichannel sink grouping. Native output-buffer behavior belongs to the
emitter/runtime integration lane.
-/

namespace Tropical.Testing.PlanOutputs

open Lean (Json)
open Tropical.Plan

private def emptyPlan (channels : Nat := 1)
    (sinks : Array SinkSpec := #[]) : FlatPlan := {
  arraySlotNames := #[]
  registerCount := 0
  arraySlotCount := 0
  arraySlotSizes := #[]
  instanceFunctions := #[]
  sinks
  outputChannelCount := channels
  slotCount := 0
  slotNames := #[]
  slotDefaults := #[] }

private def refused : Except String α → Bool
  | .error _ => true
  | .ok _ => false

private def monoFieldOmitted : Bool :=
  match (emptyPlan).toWire with
  | .error _ => false
  | .ok wire => refused (wire.getObjVal? "output_channel_count")

private def stereoRoundTrip : Bool :=
  let plan := emptyPlan 2 #[
    { inputs := #[], gain := defaultSinkGain, target := 0 },
    { inputs := #[], gain := defaultSinkGain, target := 1 }]
  match plan.toWire with
  | .error _ => false
  | .ok wire =>
    match wire.getObjVal? "output_channel_count", FlatPlan.ofWire wire with
    | .ok (.num count), .ok decoded =>
      count.toFloat == 2 && decoded.outputChannelCount == 2 &&
        decoded.sinks.size == 2
    | _, _ => false

private def malformedLayoutsRefused : Bool :=
  let zero := (emptyPlan 0).toWire
  let outOfRange := (emptyPlan 1 #[
    { inputs := #[], gain := defaultSinkGain, target := 1 }]).toWire
  let duplicate := (emptyPlan 2 #[
    { inputs := #[], gain := defaultSinkGain, target := 1 },
    { inputs := #[], gain := defaultSinkGain, target := 1 }]).toWire
  refused zero && refused outOfRange && refused duplicate

private def grouped : Nat × Array SinkSpec :=
  Tropical.Compile.groupSinks #[(2, 20), (0, 1), (2, 21), (1, 10)]

example : monoFieldOmitted = true := by native_decide
example : stereoRoundTrip = true := by native_decide
example : malformedLayoutsRefused = true := by native_decide
example : grouped.1 = 3 := by native_decide
example : grouped.2.size = 3 := by native_decide
example : grouped.2[0]!.target = 0 := by native_decide
example : grouped.2[1]!.target = 1 := by native_decide
example : grouped.2[2]!.target = 2 := by native_decide
example : grouped.2[2]!.inputs = #[20, 21] := by native_decide

end Tropical.Testing.PlanOutputs
