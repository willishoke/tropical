import Tropical.Engine.Audio
import Tropical.Stdlib

/-!
# Engine.Front — the tool dispatcher and boot

`handleTool` routes a named MCP tool to its handler; `boot` constructs the
`Env`, wiring the native runtime and DAC. `handleLoadPatchGraph` compiles a
downstream-only patch graph through the EmitArrow lowering to a session root.
-/

namespace Tropical.Engine

open Lean (Json toJson)
open Tropical.Expr (getField? getStrField? opOf?)
open Tropical.Wiring (parsePortType? checkArrayConnection PortType)

-- ── Dispatcher ───────────────────────────────────────────────────────────────

def handleListScopeTaps (env : Env) : EngineM Json := do
  let st ← env.state.get
  let taps := st.scopeTaps.map fun tap =>
    Json.mkObj [("name", Json.str tap.name),
                ("instance", Json.str tap.sourceInstance),
                ("output", Json.str tap.sourceOutput),
                ("slot", Json.str tap.slot)]
  pure <| Json.mkObj [("taps", Json.arr taps)]

/-- EXPERIMENT (`load_patch_graph`): compile a downstream-only patch graph (the
    playground GUI) through the EmitArrow arrow lowering — `lowerGraph → normalize
    (the slide) → emitTerm` — to a session root, then the production
    `compileSession → buildKernelIr → loadIrStaged` tail. A compile failure
    errors BEFORE the load, so the previous kernel keeps playing. -/
def handleLoadPatchGraph (env : Env) (args : Json) : EngineM Json := do
  let compiled ← match ← Tropical.Playground.compilePlan args with
    | .error e => internalError e
    | .ok p => pure p
  loadKernel env compiled.plan (some compiled.stageBlocks)
  -- Seed the session param mirror with the graph's knobs so `set_param` — which
  -- guards on the mirror, then drives the live `param:<name>` slot — reaches them
  -- without a relower. Replaces (not appends): the mirror tracks the current graph.
  -- Also publish the arrow taps as `scopeTaps` (each already routed to a
  -- `render_window`-readable root output slot), so an attached scope discovers
  -- this graph's inspection points via `list_scope_taps` with no session wiring.
  env.state.modify (fun st => { st with
    params := Tropical.Playground.knobParams args
    scopeTaps := compiled.taps
    paramDisciplines := compiled.plan.paramDisciplines })
  -- The realized-state report: facts about what compiled (active/excluded
  -- nodes, wired/normalled inputs, live params with disciplines, taps) —
  -- never warnings. `ok` stays for callers that only ever looked at it.
  pure <| Tropical.Playground.realizedReport args compiled.taps

def handleTool (env : Env) (name : String) (args : Json) : IO Json :=
  wrap <| match name with
  | "load_patch_graph" => handleLoadPatchGraph env args
  -- The vocabulary (port-spec table as data: ports, accepts, defaults, write
  -- disciplines, display metadata) — clients render it, never re-encode it.
  -- Session-independent, so it just echoes.
  | "get_vocabulary" => pure Tropical.Playground.vocabularyJson
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

/-- Qualification-only runtime block-length seam. The live default stays 512;
    the opt-in environment value is read before Runtime/DAC construction and
    is already visible through runtime telemetry's `buffer_length`. -/
private def bootBufferLength : IO UInt32 := do
  match ← IO.getEnv "TROPICAL_BUFFER_LENGTH" with
  | none => pure 512
  | some raw =>
    match raw.toNat? with
    | some n =>
      if n >= 16 && n <= 16384 then
        pure n.toUInt32
      else
        throw <| IO.userError
          "TROPICAL_BUFFER_LENGTH must be an integer in [16,16384]"
    | none =>
      throw <| IO.userError
        "TROPICAL_BUFFER_LENGTH must be an integer in [16,16384]"

/-- Boot the stdlib and build the Env.

    The stdlib is the arrow-builder chain (`Tropical.EmitArrow.stdlibBuilders`),
    the 15 live programs authored directly as `Sig`/`assemble` builders — no
    `.md` source, no parsed bridge, no elaboration. Each builder appends its raw
    program to the arena, linking sub-programs by name against the chain built so
    far (deps precede dependents in manifest order); `registerResolved` relinks
    that raw registry to the post-strata canon and adopts the catalog entry. Any
    failure here is fatal: the engine cannot compile without its store. -/
def boot : IO Env := do
  let state ← IO.mkRef ({} : SessionSt)
  let runtime ← Ffi.Runtime.new (← bootBufferLength)
  let dac ← IO.mkRef (none : Option Ffi.Dac)
  let metalBackend := (← IO.getEnv "TROPICAL_BACKEND") == some "metal"
  let env : Env := { state, runtime, dac, metalBackend }
  -- The stdlib is the arrow-builder chain (`Tropical.EmitArrow.stdlibBuilders`),
  -- no longer the parsed-.md bridge. Each builder appends its RAW program to the
  -- arena, linking sub-programs by name against the chain built so far;
  -- `registerResolved` relinks that raw registry to the post-strata canon and
  -- adopts the catalog entry. Manifest order == the chain order, so the catalog
  -- (and `list_programs`) ordering is preserved.
  let registerAll : EngineM Unit := do
    let mut chain : Array (String × Tropical.Ir.ProgramIdx) := #[]
    for (name, build) in Tropical.EmitArrow.stdlibBuilders do
      let st ← env.state.get
      let (arena', rawIdx) ← match build st.arena chain with
        | .error e => internalError s!"stdlib builder '{name}': {e}"
        | .ok r => pure r
      let _ ← registerResolved env name arena' rawIdx
      chain := chain.push (name, rawIdx)
  match ← registerAll.run with
  | .ok () => pure ()
  | .error f => throw <| IO.userError s!"stdlib boot failed: {f.toJson.compress}"
  pure env

end Tropical.Engine
