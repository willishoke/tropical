import Tropical.Engine.Audio

/-!
# Engine.Front — the tool dispatcher and boot

`handleTool` routes a named MCP tool to its handler; `boot` constructs the
`Env`, wiring the native runtime and DAC. `handleLoadPatchGraph` compiles a
downstream-only patch graph through the EmitArrow lowering to a session root.
-/

namespace Tropical.Engine

open Lean (Json toJson)
open Tropical.Expr (getField? getStrField? opOf? validateExpr exprDependencies prettyExpr)
open Tropical.Wiring (parsePortType? checkArrayConnection PortType)

-- ── Dispatcher ───────────────────────────────────────────────────────────────

def handleListScopeTaps (env : Env) : EngineM Json := do
  let st ← env.state.get
  let taps := st.scopeTaps.map fun (name, inst, out) =>
    Json.mkObj [("name", Json.str name), ("instance", Json.str inst),
                ("output", Json.str out), ("slot", Json.str s!"{inst}.{out}")]
  pure <| Json.mkObj [("taps", Json.arr taps)]

/-- EXPERIMENT (`load_patch_graph`): compile a downstream-only patch graph (the
    playground GUI) through the EmitArrow arrow lowering — `lowerGraph → normalize
    (the slide) → emitTerm` — to a session root, then the production
    `compileSession → buildKernelIr → loadIrStaged` tail. A compile failure
    errors BEFORE the load, so the previous kernel keeps playing. -/
def handleLoadPatchGraph (env : Env) (args : Json) : EngineM Json := do
  let (plan, taps, stageBlocks) ← match ← Tropical.Playground.compilePlan args with
    | .error e => internalError e
    | .ok p => pure p
  loadKernel env plan (some stageBlocks)
  -- Seed the session param mirror with the graph's knobs so `set_param` — which
  -- guards on the mirror, then drives the live `param:<name>` slot — reaches them
  -- without a relower. Replaces (not appends): the mirror tracks the current graph.
  -- Also publish the arrow taps as `scopeTaps` (each already routed to a
  -- `render_window`-readable root output slot), so an attached scope discovers
  -- this graph's inspection points via `list_scope_taps` with no session wiring.
  env.state.modify (fun st => { st with
    params := Tropical.Playground.knobParams args
    scopeTaps := taps
    paramDisciplines := plan.paramDisciplines })
  -- The realized-state report: facts about what compiled (active/excluded
  -- nodes, wired/normalled inputs, live params with disciplines, taps) —
  -- never warnings. `ok` stays for callers that only ever looked at it.
  pure <| Tropical.Playground.realizedReport args taps

def handleTool (env : Env) (name : String) (args : Json) : IO Json :=
  wrap <| match name with
  | "load_patch_graph" => handleLoadPatchGraph env args
  -- The vocabulary (port-spec table as data: ports, accepts, defaults, write
  -- disciplines, display metadata) — clients render it, never re-encode it.
  -- Session-independent, so it just echoes.
  | "get_vocabulary" => pure Tropical.Playground.vocabularyJson
  | "define_program"  => handleDefineProgram env args
  | "add_instance"    => handleAddInstance env args
  | "remove_instance" => handleRemoveInstance env args
  | "replicate"       => handleReplicate env args
  | "wire_chain"      => handleWireChain env args
  | "wire_zip"        => handleWireZip env args
  | "fan_out"         => handleFanOut env args
  | "fan_in"          => handleFanIn env args
  | "export_program"  => handleExportProgram env args
  | "list_programs"   => handleListPrograms env
  | "list_instances"  => handleListInstances env
  | "get_info"        => handleGetInfo env args
  | "wire"            => handleWire env args
  | "list_wiring"     => handleListWiring env args
  | "list_scope_taps" => handleListScopeTaps env
  | "load"            => handleLoad env args
  | "save"            => handleSave env
  | "merge"           => handleMerge env args
  | "start_audio"     => handleStartAudio env args
  | "stop_audio"      => handleStopAudio env
  | "audio_status"    => handleAudioStatus env
  -- ONE set_param: discipline-dispatched from the loaded plan's table
  -- (raw for table-less names). The three verbs below are migration
  -- aliases into the same internals.
  | "set_param"       => handleSetParamDispatch env args
  | "set_param_glide" => handleSetParamGlide env args
  | "set_param_freq"  => handleSetParamFreq env args
  | "set_param_velocity" => handleSetParamVelocity env args
  | "list_params"     => handleListParams env
  | "debug_render"    => handleDebugRender env args
  | _ => internalError s!"Unknown tool: '{name}'"

-- ── Boot ─────────────────────────────────────────────────────────────────────

/-- Spawn the compiler service, boot the stdlib from the pre-parsed
    bridge, build the Env.

    The service no longer loads the stdlib (Phase 4 stage 4b): the
    engine reads `stdlib/parsed/manifest.json` (the registration order
    `loadStdlib` produced) and each `stdlib/parsed/<Name>.json` from
    the repo root, and registers each through the SAME
    elaborate→register→adopt flow `define_program` uses. Mirroring
    `loadStdlibFromResolved`: stdlib elaboration resolves siblings
    through the RAW elaborated map (TS `localResolved`), and the
    service relinks concrete registrations to the post-strata canon
    before strata (its `processedByName` step) — so the registered
    catalog is byte-faithful to the old TS `loadStdlib`. Any failure
    here is fatal: the engine cannot compile without its store. -/
def boot : IO Env := do
  let state ← IO.mkRef ({} : SessionSt)
  let runtime ← Ffi.Runtime.new 512
  let dac ← IO.mkRef (none : Option Ffi.Dac)
  let metalBackend := (← IO.getEnv "TROPICAL_BACKEND") == some "metal"
  let arrowRoot := (← IO.getEnv "TROPICAL_ARROW").isSome
  let env : Env := { state, runtime, dac, metalBackend, arrowRoot }
  let manifestText ← IO.FS.readFile "stdlib/parsed/manifest.json"
  let names ← match Json.parse manifestText with
    | .error e => throw <| IO.userError s!"stdlib/parsed/manifest.json: {e}"
    | .ok j =>
      match j.getObjVal? "programs" with
      | .ok (.arr ns) => pure <| ns.filterMap fun n =>
          match n with | .str s => some s | _ => none
      | _ => throw <| IO.userError "stdlib/parsed/manifest.json: missing programs[]"
  let registerAll : EngineM Unit := do
    let mut raw : Std.HashMap String Tropical.Ir.ProgramIdx := {}
    for name in names do
      let path := s!"stdlib/parsed/{name}.json"
      let text ← IO.FS.readFile path
      let prog ← match Tropical.Parse.JsonV.parse text with
        | .error e => internalError s!"{path}: JSON parse failed: {e}"
        | .ok jv =>
          match Tropical.Parse.decodeProgram jv with
          | .error e => internalError s!"{path}: {e}"
          | .ok p => pure p
      let rawMap := raw
      let (_, rawIdx) ← registerOne env name prog (some fun n => rawMap.get? n)
      raw := raw.insert name rawIdx
  match ← registerAll.run with
  | .ok () => pure ()
  | .error f => throw <| IO.userError s!"stdlib boot failed: {f.toJson.compress}"
  pure env

end Tropical.Engine
