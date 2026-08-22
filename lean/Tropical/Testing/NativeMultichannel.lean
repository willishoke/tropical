import Tropical.Ir.EmitLlvm

/-!
# Native multichannel LLVM contract

Focused compile-time checks for the compact frame-major sink layout. The
runtime-side execution and zero-fill contract is exercised by
`engine/tests/test_module_process.cpp`.
-/

namespace Tropical.Testing.NativeMultichannel

open Tropical.Plan

private def jn (mantissa : Int) : Lean.JsonNumber :=
  { mantissa, exponent := 0 }

private def stereoPlan : FlatPlan :=
  { arraySlotNames := #[]
    outputChannelCount := 2
    registerCount := 0
    arraySlotCount := 0
    arraySlotSizes := #[]
    instanceFunctions := #[]
    sinks := #[
      { inputs := #[0], gain := jn 1, target := 0 },
      { inputs := #[0], gain := jn 1, target := 1 }]
    slotCount := 1
    slotNames := #["out"]
    slotDefaults := #[Lean.Json.num (jn 0)] }

private def hasSubstring (text needle : String) : Bool :=
  decide ((text.splitOn needle).length > 1)

private def stereoEmissionLooksInterleaved : Bool :=
  match Tropical.Ir.EmitLlvm.emitKernel stereoPlan with
  | .error _ => false
  | .ok ir =>
      hasSubstring ir "mul i64 %s, 2" &&
      hasSubstring ir "add i64" &&
      hasSubstring ir ", 0\n" &&
      hasSubstring ir ", 1\n"

example : stereoEmissionLooksInterleaved = true := by native_decide

private def duplicateTargetsRejected : Bool :=
  let duplicate :=
    { stereoPlan with
      sinks := #[
        { inputs := #[0], gain := jn 1, target := 0 },
        { inputs := #[0], gain := jn 1, target := 0 }] }
  match Tropical.Ir.EmitLlvm.emitKernel duplicate with
  | .error _ => true
  | .ok _ => false

example : duplicateTargetsRejected = true := by native_decide

end Tropical.Testing.NativeMultichannel
