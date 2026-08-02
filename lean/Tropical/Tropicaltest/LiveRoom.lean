import Tropical.Tropicaltest.Patcher

/-!
# Tropical.Tropicaltest.LiveRoom

Served-modal truth for the production graph path.  This deliberately uses the
public `string` and `resonator` kinds; it neither exposes the withheld
`bloomgong` kind nor treats the lower-level gong seam witness as a served graph.
-/

open Tropical

namespace Tropical.Tropicaltest.LiveRoom

private def num (mantissa : Int) (exponent : Nat := 0) : Lean.Json :=
  .num { mantissa, exponent }

private def inlet (source : String) : Lean.Json :=
  Lean.Json.mkObj [("in", .arr #[.str source])]

private def modeRow (freq sigma amp phase : Lean.Json) : Lean.Json :=
  .arr #[freq, sigma, amp, phase]

/-- A small authored string table keeps this production-path gate quick while
    retaining the exact serialized `string` semantics: these are structural
    score rows, not live controls. -/
private def stringModes (edited : Bool) : Lean.Json :=
  .arr #[
    modeRow (if edited then num 247 else num 220) (num 2) (num 1) (num 0),
    modeRow (num 330) (num 3) (num 55 2) (num 4 1),
    modeRow (num 495) (num 4) (num 3 1) (num (-7) 1)]

private def roomParams : Lean.Json :=
  Lean.Json.mkObj [
    ("rt60", num 2), ("dir", num 15 2), ("sway", num 35 2),
    ("rate", num 3 1), ("modes", num 4)]

private def roomNode (source? : Option String) : Lean.Json :=
  Lean.Json.mkObj <| [
    ("id", .str "room"), ("kind", .str "reverb"),
    ("params", roomParams)] ++
    match source? with
    | some source => [("in", inlet source)]
    | none => []

private def outNode : Lean.Json :=
  Lean.Json.mkObj [
    ("id", .str "sink"), ("kind", .str "out"),
    ("in", inlet "room")]

private def stringRoomGraph (edited : Bool := false) : Lean.Json :=
  Lean.Json.mkObj [
    ("nodes", .arr #[
      Lean.Json.mkObj [
        ("id", .str "string"), ("kind", .str "string"),
        ("params", Lean.Json.mkObj [("modes", stringModes edited)])],
      roomNode (some "string"), outNode]),
    ("out", .str "sink")]

private def resonatorRoomGraph : Lean.Json :=
  Lean.Json.mkObj [
    ("nodes", .arr #[
      Lean.Json.mkObj [
        ("id", .str "ring"), ("kind", .str "resonator"),
        ("params", Lean.Json.mkObj [
          ("freq", num 220), ("decay", num 4), ("partials", num 3)])],
      roomNode (some "ring"), outNode]),
    ("out", .str "sink")]

/-- The legal incomplete-patch reading of source removal: an unwired modal
    inlet is the empty modal bank and therefore exact silence. -/
private def sourceRemovedGraph : Lean.Json :=
  Lean.Json.mkObj [
    ("nodes", .arr #[roomNode none, outNode]),
    ("out", .str "sink")]

private def natField? (j : Lean.Json) (name : String) : Option Nat :=
  match j.getObjVal? name with
  | .ok (.num n) => some n.toFloat.toUInt64.toNat
  | _ => none

/-- Enter through the production graph handler: graph decode, arrow lowering,
    session compilation, staged IR load, and runtime publication. -/
private def loadServedGraph (env : Tropical.Engine.Env) (graph : Lean.Json) :
    IO (Except String Nat) := do
  match ← (Tropical.Engine.handleLoadPatchGraph env graph).run with
  | .error failure => pure (.error failure.toJson.compress)
  | .ok report =>
    match natField? report "program_version" with
    | some version => pure (.ok version)
    | none => pure (.error s!"load report omitted program_version: {report.compress}")

/-- Route every live write through the one public `set_param` dispatcher. -/
private def setParam (env : Tropical.Engine.Env) (name : String)
    (value : Lean.Json) : IO (Except String Unit) := do
  let reply ← Tropical.Engine.handleTool env "set_param" <|
    Lean.Json.mkObj [("name", .str name), ("value", value)]
  match reply.getObjVal? "isError" with
  | .ok (.bool true) => pure (.error reply.compress)
  | _ => pure (.ok ())

private def renderAt (env : Tropical.Engine.Env) (start : Nat) :
    IO (Array Float) := do
  env.runtime.setSampleIndex start.toUInt64
  env.runtime.process
  pure (decodeF64LE (← env.runtime.outputBytes))

private def loadAndRender (env : Tropical.Engine.Env) (graph : Lean.Json)
    (start : Nat) : IO (Except String (Nat × Array Float)) := do
  match ← loadServedGraph env graph with
  | .error e => pure (.error e)
  | .ok version => pure (.ok (version, ← renderAt env start))

private def deltaEnergy (a b : Array Float) : Float := Id.run do
  let mut e := 0.0
  for i in [0:min a.size b.size] do
    let d := a[i]! - b[i]!
    e := e + d * d
  return e

private def materiallyDifferent (a b : Array Float) : Bool :=
  let d := deltaEnergy a b
  let scale := energyOf a + energyOf b
  d > 1.0e-12 * (if scale > 1.0e-300 then scale else 1.0)

private def reverseSamples (xs : Array Float) : Array Float :=
  (Array.range xs.size).map fun i => xs[xs.size - 1 - i]!

private def exactSamples (a b : Array Float) : Bool :=
  a.size == b.size && bitDiffCount a b == 0

private def exceptError (result : Except String α) : String :=
  match result with
  | .error e => e
  | .ok _ => "unreachable successful result"

/-- F1's noncontroversial served-modal baseline.

    Structural source/topology edits intentionally republish programs.  The
    live room controls and transport section performs one graph load only; its
    operation trace after that token consists solely of `set_param`, seek, and
    process calls, so no program-version-producing compile/load can occur. -/
def runLiveRoom : IO Bool := do
  let structuralEnv ← Tropical.Engine.boot
  let probeStart := 22050

  let baseResult ← loadAndRender structuralEnv stringRoomGraph probeStart
  let .ok (baseVersion, baseWet) := baseResult
    | return ← failGate "served-live-room" s!"base string→reverb load/render: {firstLine (exceptError baseResult)}"
  let removedResult ← loadAndRender structuralEnv sourceRemovedGraph probeStart
  let .ok (removedVersion, removedWet) := removedResult
    | return ← failGate "served-live-room" s!"source-removed load/render: {firstLine (exceptError removedResult)}"
  let subResult ← loadAndRender structuralEnv resonatorRoomGraph probeStart
  let .ok (subVersion, substituteWet) := subResult
    | return ← failGate "served-live-room" s!"resonator substitution load/render: {firstLine (exceptError subResult)}"
  let editResult ← loadAndRender structuralEnv (stringRoomGraph true) probeStart
  let .ok (editVersion, editedWet) := editResult
    | return ← failGate "served-live-room" s!"structural string-mode edit load/render: {firstLine (exceptError editResult)}"

  let baseEnergy := energyOf baseWet
  let removedEnergy := energyOf removedWet
  let structuralVersions :=
    baseVersion < removedVersion && removedVersion < subVersion && subVersion < editVersion
  let sourceTruth := baseEnergy > 1.0e-12 && removedEnergy == 0.0
    && materiallyDifferent baseWet substituteWet
    && materiallyDifferent baseWet editedWet
  IO.println s!"        served source truth: wet energy={baseEnergy}; removed={removedEnergy}; resonator ΔE={deltaEnergy baseWet substituteWet}; mode-frequency ΔE={deltaEnergy baseWet editedWet}; structural program_versions={baseVersion},{removedVersion},{subVersion},{editVersion}"

  -- One publication for all live room controls and every transport operation.
  let liveEnv ← Tropical.Engine.boot
  let liveResult ← loadServedGraph liveEnv stringRoomGraph
  let .ok liveVersion := liveResult
    | return ← failGate "served-live-room" s!"live string→reverb load: {firstLine (exceptError liveResult)}"
  let controlStart := 22050
  let liveBaseline ← renderAt liveEnv controlStart
  let controls : Array (String × Lean.Json × Lean.Json) := #[
    ("room.rt60", num 4, num 2),
    ("room.dir", num 8 1, num 15 2),
    ("room.sway", num 75 2, num 35 2),
    ("room.rate", num 12 1, num 3 1)]
  let mut controlsOk := true
  let mut controlFacts : Array (String × Float) := #[]
  for (name, changed, restored) in controls do
    match ← setParam liveEnv name changed with
    | .error e =>
      controlsOk := false
      IO.println s!"        set_param {name}: ERROR {firstLine e}"
    | .ok () =>
      let wet ← renderAt liveEnv controlStart
      let d := deltaEnergy liveBaseline wet
      controlFacts := controlFacts.push (name, d)
      if !materiallyDifferent liveBaseline wet then controlsOk := false
      match ← setParam liveEnv name restored with
      | .ok () => pure ()
      | .error e =>
        controlsOk := false
        IO.println s!"        set_param restore {name}: ERROR {firstLine e}"
  for (name, d) in controlFacts do
    IO.println s!"        unified set_param {name}: rendered wet ΔE={d}"

  -- Pick the boundary after the first 512-sample forward block at exactly one
  -- second.  That makes both velocity re-anchors integers in sample space:
  -- hold gets tau_base=1, reverse at physical sample 88200 gets tau_base=3.
  let referenceEnv ← Tropical.Engine.boot
  let referenceResult ← loadServedGraph referenceEnv stringRoomGraph
  let .ok _ := referenceResult
    | return ← failGate "served-live-room" s!"random-access reference load: {firstLine (exceptError referenceResult)}"
  let forwardStart := 44100 - 512
  let forwardLive ← renderAt liveEnv forwardStart
  let forwardReference ← renderAt referenceEnv forwardStart
  let forwardOk := exactSamples forwardLive forwardReference

  let holdSet ← setParam liveEnv Tropical.Playground.masterVelocityParam (num 0)
  let held ← if holdSet.isOk then
      liveEnv.runtime.process
      pure (decodeF64LE (← liveEnv.runtime.outputBytes))
    else pure #[]
  let pointReference := (← renderAt referenceEnv 44100)[0]?
  let holdOk := holdSet.isOk && pointReference.isSome
    && held.size == 512 && held.all (· == pointReference.getD 0.0)

  -- Move physical time while frozen, then reverse at an exact whole-second
  -- coordinate.  No room/source state is reset because there is no state.
  liveEnv.runtime.setSampleIndex 87688
  liveEnv.runtime.process
  let reverseSet ← setParam liveEnv Tropical.Playground.masterVelocityParam (num (-1))
  let reversed ← if reverseSet.isOk then
      liveEnv.runtime.process
      pure (decodeF64LE (← liveEnv.runtime.outputBytes))
    else pure #[]
  let reverseReference := reverseSamples (← renderAt referenceEnv 43589)
  let reverseOk := reverseSet.isOk && exactSamples reversed reverseReference

  -- Seek the SAME reverse-running runtime to a distant physical coordinate;
  -- tau_base=3 maps it to logical 30000, so the comparison window is known.
  let seekPhysical := 102300
  let seeked ← renderAt liveEnv seekPhysical
  let seekReference := reverseSamples (← renderAt referenceEnv 29489)
  let seekOk := exactSamples seeked seekReference
  IO.println s!"        random access under one program_version={liveVersion}: forward bit-diff={bitDiffCount forwardLive forwardReference}; hold samples at coordinate 44100={held.size}; reverse bit-diff={bitDiffCount reversed reverseReference}; reverse-seek bit-diff={bitDiffCount seeked seekReference}"

  let transportOk := forwardOk && holdOk && reverseOk && seekOk
  if structuralVersions && sourceTruth && controlsOk && transportOk then
    passGate "served-live-room"
      s!"serialized string→reverb is source-dependent; substitution/structural frequency edit move wet; rt60/dir/sway/rate move wet through unified set_param under the single published program_version {liveVersion}; forward/hold/reverse/seek equal random-access coordinates without a tail reset"
  else
    failGate "served-live-room"
      s!"structuralVersions={structuralVersions} sourceTruth={sourceTruth} controls={controlsOk} transport={transportOk} (forward={forwardOk}, hold={holdOk}, reverse={reverseOk}, seek={seekOk})"

end Tropical.Tropicaltest.LiveRoom
