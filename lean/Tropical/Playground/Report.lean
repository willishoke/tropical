import Tropical.Playground.Decode

/-!
# Playground.Report

Host parameter disciplines, realized-state reporting, and scope-tap metadata.
-/

namespace Tropical.Playground

open Lean (Json JsonNumber)
open Tropical.Ir
open Tropical.EmitArrow
open Tropical.Exact (DyadicI)

-- ── Host-contract dispatch table (param_disciplines in the manifest) ────────
/-- The per-param write-discipline table this graph's plan carries — the same
    walk and skip rules as `collectParams`, projected to base names. A host
    reads this from the manifest and dispatches param writes itself
    (design/host-param-dispatch.md is the normative math); no client ever
    chooses a write verb. -/
def paramDisciplinesOf (raws : Array Raw) :
    Array Tropical.Plan.ParamDiscipline := Id.run do
  let mut out : Array Tropical.Plan.ParamDiscipline := #[
    { name := masterVelocityParam, discipline := "velocity",
      companions := #[masterTauBaseParam] },
    { name := masterTauBaseParam, discipline := "raw" },
    { name := masterGainParam, discipline := "raw" }]
  for r in raws do
    if r.kind == "out" then continue
    for spec in portSpecs r.kind do
      if spec.knob.isNone then continue
      let selfWired := !(portSources r.inObj spec.name).isEmpty
      let ownerWired := match spec.ownerPort with
        | some o => !(portSources r.inObj o).isEmpty
        | none => false
      if !selfWired && !ownerWired then
        let base := s!"{r.id}.{spec.name}"
        let d : Tropical.Plan.ParamDiscipline := match spec.discipline with
          | .glide =>
            -- 0.02 s: the engine's glide window
            { name := base, discipline := "glide", glideDurSec := some ⟨2, 2⟩,
              companions := #[s!"{base}#v0", s!"{base}#v1", s!"{base}#t0"] }
          | .anchor =>
            { name := base, discipline := "anchor", companions := #[s!"{base}#phase"] }
          | .raw =>
            { name := base, discipline := "raw" }
        out := out.push d
    -- Trip-count-as-data: the live `partials` slot (present only with the
    -- STATIC `partials_max` capacity) is a plain raw write — same walk and
    -- skip rules as `collectParams`.
    if r.kind == "resonator" && (jNum? r.params "partials_max").isSome then
      out := out.push { name := s!"{r.id}.partials", discipline := "raw" }
  return out

-- ── The realized-state report (the load_patch_graph reply) ──────────────────
/-- FACTS about what compiled — never warnings (legal-but-incomplete states
    compile to silence by contract; the report is how a surface renders truth
    instead of guessing policy). Per user node: `active` (reachable from the
    `out` node, walking inlet edges backwards) or `excluded`. Per inlet:
    `wired` (with sources) or `normalled` (running on its default). Per live
    param: the collected value and write discipline, base names only — the
    glide/anchor companion slots are the discipline's implementation detail,
    not surface. Plus the vocabulary identity, exact published generation, and
    complete tap bindings needed to adopt this response atomically. The
    silence-with-`{ok:true}` class dies here: a patch that gracefully compiled
    to nothing now SAYS so, as facts. -/
def realizedReportForGeneration (args : Json) (taps : Array Tropical.ScopeTap)
    (programVersion controlVersion : UInt64) : Json := Id.run do
  let raws := rawsOf args
  let outId := match (args.getObjVal? "out").toOption with
    | some (.str s) => s
    | _ => ""
  -- reachability fixed point (≤ |nodes| rounds; GUI graphs are small)
  let mut reach : Array String := #[outId]
  for _ in [0:raws.size] do
    for r in raws do
      if reach.contains r.id then
        for spec in portSpecs r.kind do
          for src in portSources r.inObj spec.name do
            if !reach.contains src then reach := reach.push src
  let nodesJ := raws.map fun r =>
    Json.mkObj [("id", Json.str r.id), ("kind", Json.str r.kind),
      ("status", Json.str (if reach.contains r.id then "active" else "excluded"))]
  let inputsJ := raws.foldl (init := #[]) fun acc r =>
    (portSpecs r.kind).foldl (init := acc) fun acc spec =>
      if spec.accepts.isEmpty then acc else
      let srcs := portSources r.inObj spec.name
      acc.push (Json.mkObj (
        [("node", Json.str r.id), ("port", Json.str spec.name)] ++
        (if srcs.isEmpty then [("state", Json.str "normalled")]
         else [("state", Json.str "wired"),
               ("sources", Json.arr (srcs.map Json.str))])))
  let kindOf : String → Option String := fun id => (raws.find? (·.id == id)).map (·.kind)
  let paramsJ := (collectParams raws).filterMap fun (nm, v) =>
    if nm.endsWith "#v1" || nm.endsWith "#t0" || nm.endsWith "#phase" then none
    else if nm.endsWith "#v0" then
      some (Json.mkObj [("name", Json.str (nm.dropEnd 3).toString),
        ("value", Json.num v), ("discipline", Json.str "glide")])
    else
      let disc :=
        if nm == masterVelocityParam then "velocity"
        else match nm.splitOn "." with
          | [id, kname] => (kindOf id).map (fun k => discStr (disciplineOf k kname)) |>.getD "raw"
          | _ => "raw"
      some (Json.mkObj [("name", Json.str nm),
        ("value", Json.num v), ("discipline", Json.str disc)])
  let tapsJ := taps.map fun tap =>
    Json.mkObj [("name", Json.str tap.name),
      ("instance", Json.str tap.sourceInstance),
      ("output", Json.str tap.sourceOutput),
      ("slot", Json.str tap.slot)]
  return Json.mkObj [("ok", Json.bool true),
    ("vocabulary_fingerprint", Json.str vocabularyFingerprint),
    -- `ToJson UInt64` uses a quoted compatibility spelling; the socket data
    -- plane's render versions are JSON integers, so keep the handshake shape
    -- identical by widening losslessly through Nat.
    ("program_version", Lean.toJson programVersion.toNat),
    ("control_version", Lean.toJson controlVersion.toNat),
    ("nodes", Json.arr nodesJ),
    ("inputs", Json.arr inputsJ), ("params", Json.arr paramsJ),
    ("taps", Json.arr tapsJ)]

/-- Pure-report compatibility seam for tests and non-publishing consumers.
    Production `load_patch_graph` always uses `realizedReportForGeneration`
    with the runtime token; zero explicitly means no publication was supplied. -/
def realizedReport (args : Json) (taps : Array Tropical.ScopeTap) : Json :=
  realizedReportForGeneration args taps 0 0

-- ── Scope taps ──────────────────────────────────────────────────────────────
/-- For the arrow path the source instance is always the synthetic root, and
    the output is a dedicated `tap:<id>` port. -/
abbrev Tap := Tropical.ScopeTap

structure CompiledPatch where
  plan : Tropical.Plan.FlatPlan
  taps : Array Tap
  stageBlocks : Array (Array (Option Tropical.Ir.Stage))

/-- The user-facing nodes worth tapping: the raw GUI nodes, minus the reserved
    `out`/`__silence__` and any synthesized helper (`__lfo_*`) — i.e. every node
    that has a visible output jack. `lowerNode` gives each one's closed-form
    output signal (the signal at its jack), which is exactly what a scope shows. -/
def tapNodeIds (raws : Array Raw) : Array String :=
  raws.filterMap fun r =>
    if r.kind == "out" || "__".isPrefixOf r.id then none else some r.id


end Tropical.Playground
