import Tropical.Tropicaltest.AnchorContinuity
import Tropical.NativeBoot
import Tropical.Testing.EngineMirror

/-!
# Patch-bay anchors: the declarations, the refusals, the engine path

`design/host-param-dispatch.md`, "Patch-bay params". The rendered continuity
of an anchored patch-bay param is `anchor-continuity/patch-bay-*`
(`AnchorContinuity`); this module checks what is around it:

* **refusals** — every declaration the compile cannot honour is refused
  through the real front door (`Session.compilePatch`) with its own reason;
  fan-out to two phasors on τ is accepted with one companion.
* **engine** — through the session engine: `set_param` on the anchored param
  moves its companion by exactly the predicted amount (the host's dispatch
  reached from MCP), the companion itself is refused by name, `save` writes
  the declaration and not the companion, and the saved document compiles to
  the same discipline table; `export_program` keeps a phasor's anchor when
  its phase and clock inputs are exposed too, and drops it otherwise.
-/

namespace Tropical.Tropicaltest.PatchAnchor

open Tropical

private def q (s : String) : String := "\"" ++ s ++ "\""

private def paramDecl (name : String) (extra : String := "") (value : String := "220") : String :=
  "{\"op\":\"paramDecl\",\"name\":" ++ q name ++ ",\"value\":" ++ value ++ extra ++ "}"

private def inst (name program inputs : String) : String :=
  "{\"op\":\"instanceDecl\",\"name\":" ++ q name ++ ",\"program\":" ++ q program ++
    ",\"inputs\":{" ++ inputs ++ "}}"

private def pref (name : String) : String := "{\"op\":\"param\",\"name\":" ++ q name ++ "}"
private def clockE : String := "{\"op\":\"clock\"}"
private def refE (i o : String) : String :=
  "{\"op\":\"ref\",\"instance\":" ++ q i ++ ",\"output\":" ++ q o ++ "}"

private def patch (decls : List String) (out : String) : String :=
  "{\"schema\":\"tropical_program_2\",\"name\":\"p\",\"body\":{\"op\":\"block\",\"decls\":[" ++
    String.intercalate "," decls ++ "],\"assigns\":[{\"op\":\"outputAssign\",\"name\":\"dac.out\"," ++
    "\"expr\":" ++ out ++ "}]}}"

private def anchored : String := ",\"discipline\":\"anchor\""

private def phasor (name : String) (freq : String) (clk : String := clockE)
    (extra : String := "") : String :=
  inst name "ClockPhasor" ("\"clk\":" ++ clk ++ ",\"freq\":" ++ freq ++ extra)

private def contains (s sub : String) : Bool := (s.splitOn sub).length > 1

def runRefusals : IO Bool := do
  let label := "patch-anchor/refusals"
  let cases : List (String × String × String) := [
    ("an integer param",
      patch [paramDecl "pitch" (",\"type\":\"int32\",\"frac_bits\":16" ++ anchored),
        phasor "ph" (pref "pitch")] (refE "ph" "phase"),
      "an integer param is written raw"),
    ("the param through an expression",
      patch [paramDecl "pitch" anchored,
        phasor "ph" ("{\"op\":\"mul\",\"args\":[" ++ pref "pitch" ++ ",2]}")] (refE "ph" "phase"),
      "reads it through an expression"),
    ("the param bare into an input with no anchor",
      patch [paramDecl "pitch" anchored, phasor "ph" (pref "pitch"),
        inst "vca" "VCA" ("\"audio\":" ++ refE "ph" "phase" ++ ",\"cv\":" ++ pref "pitch")]
        (refE "vca" "out"),
      "declares no phase anchor"),
    ("a phasor on a scrubbed clock",
      patch [paramDecl "pitch" anchored,
        inst "sc" "ScrubClock" "\"tau_base\":0,\"velocity\":0.5",
        phasor "ph" (pref "pitch") (refE "sc" "clk")] (refE "ph" "phase"),
      "holds on the sample clock only"),
    ("a phasor whose clock is left on its default",
      patch [paramDecl "pitch" anchored,
        inst "ph" "ClockPhasor" ("\"freq\":" ++ pref "pitch")] (refE "ph" "phase"),
      "holds on the sample clock only"),
    ("a phasor whose phase input is wired",
      patch [paramDecl "pitch" anchored, phasor "ph" (pref "pitch") clockE ",\"offset\":0.25"]
        (refE "ph" "phase"),
      "owns its phase input"),
    ("a param named like the companion",
      patch [paramDecl "pitch" anchored, paramDecl "pitch#phase" "" "0",
        phasor "ph" (pref "pitch"), phasor "ph2" (pref "pitch#phase")] (refE "ph" "phase"),
      "is also declared as a param"),
    ("a discipline a patch cannot declare",
      patch [paramDecl "pitch" ",\"discipline\":\"glide\"", phasor "ph" (pref "pitch")]
        (refE "ph" "phase"),
      "is not declarable on a patch-bay param"),
    ("an anchor_clock",
      patch [paramDecl "pitch" ",\"discipline\":\"anchor_clock\"", phasor "ph" (pref "pitch")]
        (refE "ph" "phase"),
      "is not declarable on a patch-bay param")]
  let mut problems : Array String := #[]
  for (what, text, reason) in cases do
    match AnchorContinuity.compilePatchBay text with
    | .ok _ => problems := problems.push s!"{what}: compiles"
    | .error e =>
      unless contains e reason do
        problems := problems.push s!"{what}: refused for another reason: {firstLine e}"
  -- Accepted: one param fanned out bare to two phasors on τ shares one
  -- companion, wired into both phase inputs.
  let fan := patch [paramDecl "pitch" anchored, phasor "a" (pref "pitch"), phasor "b" (pref "pitch"),
    inst "mix" "VCA" ("\"audio\":" ++ refE "a" "phase" ++ ",\"cv\":" ++ refE "b" "phase")]
    (refE "mix" "out")
  match AnchorContinuity.compilePatchBay fan with
  | .error e => problems := problems.push s!"fan-out: refused: {firstLine e}"
  | .ok (plan, _) =>
    let want : Array Plan.ParamDiscipline :=
      #[{ name := "pitch", discipline := "anchor", companions := #["pitch#phase"] }]
    unless (plan.paramDisciplines.map (·.toWire.compress)) == want.map (·.toWire.compress) do
      problems := problems.push s!"fan-out: table {(plan.paramDisciplines.map (·.toWire.compress))}"
    unless plan.slotNames.contains "param:pitch#phase" do
      problems := problems.push "fan-out: no companion slot"
  -- Undeclared: the same patch is raw, with no companion.
  match AnchorContinuity.compilePatchBay (AnchorContinuity.splineVoicePatch none) with
  | .error e => problems := problems.push s!"undeclared: refused: {firstLine e}"
  | .ok (plan, _) =>
    unless plan.paramDisciplines.isEmpty && !plan.slotNames.contains "param:pitch#phase" do
      problems := problems.push "undeclared: the plan carries a discipline or a companion"
  if problems.isEmpty then
    passGate label s!"{cases.length} declarations refused, each for its own reason; fan-out shares one companion; an undeclared param stays raw"
  else failGate label (String.intercalate "; " problems.toList)

private def runE {α : Type} (act : Tropical.EngineM α) : IO (Except String α) := do
  match ← act.run with
  | .ok a => pure (.ok a)
  | .error f => pure (.error f.toJson.compress)

private def loadArgs (text : String) : Lean.Json :=
  Lean.Json.mkObj [("program", (Lean.Json.parse text).toOption.getD Lean.Json.null)]

private def setArgs (name : String) (v : Float) : Lean.Json :=
  Lean.Json.mkObj [("name", Lean.Json.str name),
    ("value", match Lean.JsonNumber.fromFloat? v with
      | .inr n => Lean.Json.num n
      | .inl _ => Lean.Json.null)]

private def tableOf (plan : Plan.FlatPlan) : Array String :=
  plan.paramDisciplines.map (·.toWire.compress)

def runEngine : IO Bool := do
  let label := "patch-anchor/engine"
  let mut problems : Array String := #[]
  let (rt, env) ← NativeBoot.bootWithRuntime
  let text := AnchorContinuity.splineVoicePatch
  match ← runE (Engine.handleLoad env (loadArgs text)) with
  | .error e => return ← failGate label s!"load: {firstLine e}"
  | .ok _ => pure ()
  let original ← match ← runE (Engine.compileMirrorStaged env .fused) with
    | .error e => return ← failGate label s!"compile: {firstLine e}"
    | .ok m => pure m.plan
  -- set_param through the engine: the host moves the companion exactly.
  for _ in [0:3] do rt.process
  let rate ← rt.sampleRate
  let now := (← rt.currentSampleIndexU64).toNat
  let slotOf := fun (n : String) => rt.slotIndex? s!"param:{n}"
  match ← slotOf "pitch", ← slotOf "pitch#phase" with
  | some ip, some iph =>
    let f0 ← rt.getSlot ip
    let phi ← rt.getSlot iph
    match ← runE (Engine.handleSetParam env (setArgs "pitch" 330)) with
    | .error e => problems := problems.push s!"set_param pitch: {firstLine e}"
    | .ok _ =>
      let want := AnchorContinuity.anchoredPhase f0 330 phi 1 0 rate now
      let got ← rt.getSlot iph
      unless got.toBits == want.toBits && (← rt.getSlot ip) == 330 do
        problems := problems.push s!"set_param pitch: companion {got}, predicted {want}"
  | _, _ => problems := problems.push "the loaded plan has no pitch / pitch#phase slots"
  -- The companion is not a session param.
  match ← runE (Engine.handleSetParam env (setArgs "pitch#phase" 0.5)) with
  | .ok _ => problems := problems.push "set_param on the companion is accepted"
  | .error e =>
    unless contains e "phase companion of the anchored param" do
      problems := problems.push s!"set_param on the companion: another reason: {firstLine e}"
  -- save writes the declaration, not the companion; the saved document
  -- compiles to the same table.
  match ← runE (Engine.handleSave env) with
  | .error e => problems := problems.push s!"save: {firstLine e}"
  | .ok doc =>
    let s := doc.compress
    unless contains s "\"discipline\":\"anchor\"" do
      problems := problems.push "save dropped the paramDecl's discipline"
    if contains s "pitch#phase" then
      problems := problems.push "save wrote the companion"
    match ← runE (Engine.handleLoad env (loadArgs (doc.getObjValD "program").compress)) with
    | .error e => problems := problems.push s!"reload the saved document: {firstLine e}"
    | .ok _ =>
      match ← runE (Engine.compileMirrorStaged env .fused) with
      | .error e => problems := problems.push s!"compile the saved document: {firstLine e}"
      | .ok m =>
        unless tableOf m.plan == tableOf original && !(tableOf original).isEmpty do
          problems := problems.push s!"save → load: table {tableOf m.plan}, was {tableOf original}"
  -- export_program: the anchor survives when the phase and clock are exposed.
  -- (Exposed inputs take their wires as defaults, so the phasor is exported
  -- from a session where its freq is a literal.)
  match ← runE (Engine.handleLoad env (loadArgs (patch [phasor "ph" "220"] (refE "ph" "phase")))) with
  | .error e => problems := problems.push s!"load the export source: {firstLine e}"
  | .ok _ => pure ()
  let exportArgs := fun (name : String) (inputs : List (String × Lean.Json)) =>
    Lean.Json.mkObj [("name", Lean.Json.str name),
      ("outputs", Lean.Json.mkObj [("phase", Lean.Json.mkObj
        [("instance", "ph"), ("output", "phase")])]),
      ("inputs", Lean.Json.mkObj inputs)]
  let full ← runE (Engine.handleExportProgram env (exportArgs "AnchoredPhasor"
    [("f", "ph:freq"), ("o", "ph:offset"), ("c", "ph:clk")]))
  let partial' ← runE (Engine.handleExportProgram env (exportArgs "FreqOnlyPhasor"
    [("f", "ph:freq")]))
  match full, partial' with
  | .ok _, .ok _ =>
    let use := fun (program : String) (inputs : String) =>
      patch [paramDecl "hz" anchored, inst "x" program inputs] (refE "x" "phase")
    match ← runE (Engine.handleLoad env (loadArgs (use "AnchoredPhasor"
        ("\"c\":" ++ clockE ++ ",\"f\":" ++ pref "hz")))) with
    | .error e => problems := problems.push s!"exported AnchoredPhasor: load: {firstLine e}"
    | .ok _ =>
      match ← runE (Engine.compileMirrorStaged env .fused) with
      | .error e => problems := problems.push s!"exported AnchoredPhasor: refused: {firstLine e}"
      | .ok m =>
        let want : Plan.ParamDiscipline :=
          { name := "hz", discipline := "anchor", companions := #["hz#phase"] }
        unless tableOf m.plan == #[want.toWire.compress] do
          problems := problems.push s!"exported AnchoredPhasor: table {tableOf m.plan}"
    -- `load` compiles, so the refusal may come from it or from the compile.
    let refusal ← match ← runE (Engine.handleLoad env (loadArgs (use "FreqOnlyPhasor" ("\"f\":" ++ pref "hz")))) with
      | .error e => pure (some e)
      | .ok _ =>
        match ← runE (Engine.compileMirrorStaged env .fused) with
        | .ok _ => pure none
        | .error e => pure (some e)
    match refusal with
    | none => problems := problems.push "exported FreqOnlyPhasor: an anchor with no exposed phase compiles"
    | some e =>
      unless contains e "declares no phase anchor" do
        problems := problems.push s!"exported FreqOnlyPhasor: another reason: {firstLine e}"
  | .error e, _ | _, .error e => problems := problems.push s!"export: {firstLine e}"
  if problems.isEmpty then
    passGate label "set_param moves the companion bit-exactly; the companion is refused by name; save writes the declaration and reloads to the same table; export keeps the anchor only with phase and clock exposed"
  else failGate label (String.intercalate "; " problems.toList)

def runPatchAnchor : IO Bool := do
  let a ← runRefusals
  let b ← runEngine
  return a && b

end Tropical.Tropicaltest.PatchAnchor
