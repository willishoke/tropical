import Tropical.Session.Compile
import Tropical.StagedLoad
import Tropical.Tropicaltest.Reversibility
import Tropical.Tropicaltest.Gates

/-!
# Param order at the port

`design/typed-kernel.md`: the kernel takes `tau`, then one `f64` per `param:`
slot in port order, then `out`. The manifest, the emitted signature, the argv
thunk and the host's argv table each spell that order; nothing in the corpus
declares more than one param, so a permutation between any two of them would
pass every other gate.

This gate drives three params through the whole path — session compile,
typed load, the host's `set_param`, the audio kernel (`process`) and the
observation kernel (`renderTau`, its own artifact and argv) — on a patch
whose output weighs each param by a distinct power of ten (the sum feeds a
unity `VCA`, since `dac.out` must be an instance output):

    dac.out = zeta + 10·alpha + 100·mid

declared in that (non-alphabetical) order, so a name-sorted order anywhere
also shows. The defaults (4, 5, 6) must render 654 before any write, the
writes (1, 2, 3) must render 321; any permutation renders another number.
-/

namespace Tropical.Tropicaltest.ParamOrder

private def frames : Nat := 64

private def param (name : String) : String :=
  "{\"op\":\"param\",\"name\":\"" ++ name ++ "\"}"

private def weighted (w : Nat) (name : String) : String :=
  "{\"op\":\"mul\",\"args\":[" ++ toString w ++ "," ++ param name ++ "]}"

private def patch : String :=
  "{\"schema\":\"tropical_program_2\",\"name\":\"param_order\",\"body\":{\"op\":\"block\",\"decls\":[" ++
  "{\"op\":\"paramDecl\",\"name\":\"zeta\",\"value\":4}," ++
  "{\"op\":\"paramDecl\",\"name\":\"alpha\",\"value\":5}," ++
  "{\"op\":\"paramDecl\",\"name\":\"mid\",\"value\":6}," ++
  "{\"op\":\"instanceDecl\",\"name\":\"sum\",\"program\":\"VCA\",\"inputs\":{\"cv\":1,\"audio\":" ++
  "{\"op\":\"add\",\"args\":[" ++ param "zeta" ++ ",{\"op\":\"add\",\"args\":[" ++
  weighted 10 "alpha" ++ "," ++ weighted 100 "mid" ++ "]}]}}}]," ++
  "\"assigns\":[{\"op\":\"outputAssign\",\"name\":\"dac.out\",\"expr\":" ++
  "{\"op\":\"ref\",\"instance\":\"sum\",\"output\":\"out\"}}]}}"

private def loadPatch : IO (Except String Tropical.Ffi.Runtime) := do
  let registry ← match Tropical.Session.bootRegistry with
    | .error f => return .error s!"stdlib boot failed: {f.toJson.compress}"
    | .ok st => pure st
  match (Tropical.Session.compilePatch patch).runOn registry with
  | (.error f, _) => return .error s!"compile: {f.toJson.compress}"
  | (.ok (plan, blocks), _) =>
    let rt ← Tropical.Ffi.Runtime.new frames.toUInt32
    Tropical.StagedLoad.loadTyped rt plan blocks
    return .ok rt

/-- Every frame of the audio kernel and of the observation kernel. -/
private def renderBoth (rt : Tropical.Ffi.Runtime) : IO (Array Float × Array Float) := do
  rt.process
  let audio := decodeF64LE (← rt.outputBytes)
  let tau : Array UInt64 := (Array.range frames).map fun k => k.toUInt64 <<< Tropical.Ir.clockFracBits.toUInt64
  let observed := decodeF64LE (← rt.renderTau tau #[0])
  return (audio, observed)

private def allEqual (xs : Array Float) (v : Float) : Bool :=
  xs.size == frames && xs.all (· == v)

private def firstOr (xs : Array Float) : String :=
  match xs[0]? with | some x => toString x | none => "nothing"

def runParamOrder : IO Bool := do
  let label := "param-order"
  let rt ← match ← loadPatch with
    | .error e => return ← failGate label e
    | .ok rt => pure rt
  let (audio0, observed0) ← renderBoth rt
  for (name, value) in [("zeta", 1.0), ("alpha", 2.0), ("mid", 3.0)] do
    if (← rt.setParam name value).isNone then
      return ← failGate label s!"set_param: the loaded manifest has no param '{name}'"
  let (audio1, observed1) ← renderBoth rt
  IO.println s!"        defaults: audio {firstOr audio0} · observed {firstOr observed0} (want 654) · after writes: audio {firstOr audio1} · observed {firstOr observed1} (want 321)"
  if allEqual audio0 654 && allEqual observed0 654
      && allEqual audio1 321 && allEqual observed1 321 then
    passGate label "three params reach their own slots through compile, load, set_param and both kernels' argv"
  else
    failGate label s!"audio {firstOr audio0}→{firstOr audio1}, observed {firstOr observed0}→{firstOr observed1}; want 654→321"

end Tropical.Tropicaltest.ParamOrder
