import Tropical.Tropicaltest.Gates
import Tropical.NativeBoot
import Tropical.Testing.EngineMirror

/-!
# Failed re-export gate

`export_program` onto a name that is already registered must change nothing
when it fails. `SessionM` keeps its state across a failure, so a registration
that wrote to the store before its last fallible step would leave the store
split: `resolvedByName` on the new body, `programs`/`templateByName` on the
old one, and the next `add_instance` would pair old port metadata with the
new body.

The failure is the catalog entry: an exposed input's default must be a
literal-class expression. Wiring the second exposed port to a param that
shares its name with the first exposed input makes that default an
`inputRef`, which strata accepts and the entry refuses.
-/

namespace Tropical.Tropicaltest.ReexportFailure

open Lean (Json)

private def sinePatch : Json :=
  Json.mkObj [
    ("schema", .str "tropical_program_2"), ("name", .str "reexport_a"),
    ("body", Json.mkObj [("op", .str "block"),
      ("decls", .arr #[Json.mkObj [("op", .str "instanceDecl"),
        ("name", .str "osc1"), ("program", .str "FixedSinOsc"),
        ("inputs", Json.mkObj [("freq", .num 440)])]]),
      ("assigns", .arr #[Json.mkObj [("op", .str "outputAssign"),
        ("name", .str "dac.out"),
        ("expr", Json.mkObj [("op", .str "ref"), ("instance", .str "osc1"),
          ("output", .str "sine")])]])])]

/-- Two oscillators; `osc2.freq` reads the param `a`, the name the failing
    export gives to `osc1.freq`. -/
private def paramPatch : Json :=
  let osc (n : String) (freq : Json) := Json.mkObj [("op", .str "instanceDecl"),
    ("name", .str n), ("program", .str "FixedSinOsc"),
    ("inputs", Json.mkObj [("freq", freq)])]
  Json.mkObj [
    ("schema", .str "tropical_program_2"), ("name", .str "reexport_b"),
    ("body", Json.mkObj [("op", .str "block"),
      ("decls", .arr #[
        Json.mkObj [("op", .str "paramDecl"), ("name", .str "a"), ("value", .num 220)],
        osc "osc1" (.num 440),
        osc "osc2" (Json.mkObj [("op", .str "param"), ("name", .str "a")])]),
      ("assigns", .arr #[Json.mkObj [("op", .str "outputAssign"),
        ("name", .str "dac.out"),
        ("expr", Json.mkObj [("op", .str "ref"), ("instance", .str "osc2"),
          ("output", .str "sine")])]])])]

private def outRef (inst port : String) : Json :=
  Json.mkObj [("instance", .str inst), ("output", .str port)]

def runReexportFailure : IO Bool := do
  let label := "reexport-failure"
  let env ← Tropical.NativeBoot.boot
  let name := "ReexportProbe"
  let action : Tropical.EngineM (Option String) := do
    -- First export: one input `pitch`, one output `out`.
    let _ ← Tropical.Engine.handleLoad env (Json.mkObj [("program", sinePatch)])
    let _ ← Tropical.Engine.handleExportProgram env (Json.mkObj [
      ("name", .str name),
      ("inputs", Json.mkObj [("pitch", .str "osc1:freq")]),
      ("outputs", Json.mkObj [("out", outRef "osc1" "sine")])])
    let _ ← Tropical.Engine.handleAddInstance env (Json.mkObj [
      ("program", .str name), ("instance_name", .str "before")])
    let st0 ← env.state.get
    let some old := st0.templateByName.get? name
      | return some "first export did not register"
    -- Second export of the same name: its entry render must fail.
    let _ ← Tropical.Engine.handleLoad env (Json.mkObj [("program", paramPatch)])
    let st1 ← env.state.get
    -- `none`: accepted; `some msg`: refused with `msg`.
    let refusal ← try
        let _ ← Tropical.Engine.handleExportProgram env (Json.mkObj [
          ("name", .str name),
          ("inputs", Json.mkObj [("a", .str "osc1:freq"), ("b", .str "osc2:freq")]),
          ("outputs", Json.mkObj [("out", outRef "osc1" "sine"),
            ("out2", outRef "osc2" "sine")])])
        pure none
      catch f => pure (some f.toJson.compress)
    let some msg := refusal
      | return some "the second export was accepted; the probe no longer fails"
    -- The refusal must come from the entry render, the step after strata:
    -- an earlier refusal would leave the store untouched under any order.
    unless (msg.splitOn "not a literal-class form").length > 1 do
      return some s!"the second export failed before its entry render: {firstLine msg}"
    let st2 ← env.state.get
    if st2.templateByName.get? name != some old then
      return some "templateByName moved"
    if st2.resolvedByName.get? name != some old then
      return some s!"resolvedByName moved to {(st2.resolvedByName.get? name).map (·.idx)} (templateByName stays at {old.idx})"
    if st2.arena.programs.size != st1.arena.programs.size then
      return some "the failed export grew the store"
    let _ ← Tropical.Engine.handleAddInstance env (Json.mkObj [
      ("program", .str name), ("instance_name", .str "after")])
    let st3 ← env.state.get
    let some info := st3.findInstance? "after"
      | return some "add_instance after the failure did not land"
    if info.resolvedIdx != some old then
      return some "the new instance does not snapshot the old body"
    if info.progMeta.inputNames != #["pitch"] then
      return some s!"the new instance has inputs {info.progMeta.inputNames}, expected [pitch]"
    let _ ← Tropical.Engine.compileMirrorFlatPlan env .fused
    pure none
  match ← action.run with
  | .error f => failGate label s!"engine: {firstLine f.toJson.compress}"
  | .ok (some e) => failGate label e
  | .ok none =>
    passGate label "a failed re-export of a live name leaves the old type in place; add_instance still gets the old ports and body"

end Tropical.Tropicaltest.ReexportFailure
