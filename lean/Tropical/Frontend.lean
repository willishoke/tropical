import Turnstile
import Tropical.Tools
import Tropical.Engine
import Tropical.Rpc
import Tropical.Serve
import Tropical.Resources

/-!
# Tropical Lean front-door

The native Lean MCP server (built on Turnstile). As of Phase 6 the
whole stack is in-process (`Tropical.Engine`): session, compiler,
runtime FFI, resources. No subprocess is spawned.

Two modes:
- default — the MCP server (stdio), launched by `make mcp-lean`
- `--rpc` — the ir_service-compatible newline JSON-RPC surface used by
  the bun test suites and the differential harness

Run from the tropical repo root (so `stdlib/parsed/` resolves):
  lean/.lake/build/bin/frontend [--rpc]
-/

open Lean Turnstile

namespace Tropical

/-- A tool whose arguments are validated against `α`'s schema but whose
    handler receives the *original* params Json — number lexical forms
    and absent-vs-null distinctions survive intact. -/
private def inProcTool (α : Type) [FromJson α] [ToJsonSchema α]
    (env : Engine.Env) (name desc : String) : Tool :=
  { name, description := desc,
    inputSchema := @ToJsonSchema.jsonSchema α _,
    handler := fun args =>
      match (fromJson? args : Except String α) with
      | .ok _    => Engine.handleTool env name args
      | .error e => pure (errorResult s!"invalid arguments: {e}") }

private def inProcNoArg (env : Engine.Env) (name desc : String) : Tool :=
  { name, description := desc,
    inputSchema := Json.mkObj [("type", Json.str "object"), ("properties", Json.mkObj [])],
    handler := fun _ => Engine.handleTool env name (Json.mkObj []) }

def tropicalEngineTools (env : Engine.Env) : List Tool := [
  -- program management
  inProcTool DefineProgram  env "define_program"  "Define and register a reusable DSP program type from a tropical_program_2 object.",
  inProcTool AddInstance    env "add_instance"    "Create a named instance of a registered program type.",
  inProcTool RemoveInstance env "remove_instance" "Remove a program instance from the session.",
  inProcTool Replicate      env "replicate"       "Create N instances of a program type in one call (does not recompile; follow with wire).",
  inProcNoArg               env "list_programs"   "List all registered program types with their input/output ports and defaults.",
  inProcNoArg               env "list_instances"  "List all live program instances with their ports.",
  inProcTool GetInfo        env "get_info"        "Detailed info about one instance: ports, wiring, registers.",
  -- wiring
  inProcTool _root_.Wire    env "wire"            "Set and/or remove input wiring in one recompile. Audio output is instance \"dac\", input \"out\"; the expr there must be a ref node.",
  inProcTool WireChain      env "wire_chain"      "Wire N instances in series: output[i] → input[i+1]. One recompile.",
  inProcTool WireZip        env "wire_zip"        "Wire two equal-length lists of ports pairwise. One recompile.",
  inProcTool FanOut         env "fan_out"         "Wire one source (instance output or ExprNode) to many target inputs.",
  inProcTool FanIn          env "fan_in"          "Sum N instance outputs (optional per-source gain) into one input.",
  inProcTool Feedback       env "feedback"        "Wire an output back to an input through a 1-sample delay (no extra instance).",
  inProcTool ListWiring     env "list_wiring"     "List all wired inputs and the expression assigned to each.",
  -- program I/O
  inProcTool ExportProgram  env "export_program"  "Crystallize selected session instances into a reusable program type.",
  inProcTool Load           env "load"            "Load a tropical_program_2 program (path or inline); stops audio and recreates the session.",
  inProcNoArg               env "save"            "Serialize the current session to a tropical_program_2 object.",
  inProcTool Merge          env "merge"           "Merge a program/patch into the current session without clearing it.",
  -- control + audio
  inProcTool SetParam       env "set_param"       "Set the value of a named Param (thread-safe, smoothed).",
  inProcNoArg               env "list_params"     "List all registered Params with their current values.",
  inProcTool StartAudio     env "start_audio"     "Start audio output.",
  inProcNoArg               env "stop_audio"      "Stop audio output.",
  inProcNoArg               env "audio_status"    "Return current audio status including callback statistics."
]

-- ── Resources & prompts (engine-side; Tropical.Resources statics) ───────────

private def engineResources (env : Engine.Env) : List Resource := [
  { uri := "tropical://programs",
    name := "Program catalog",
    description := "Markdown catalog of all registered program types with inputs, outputs, and default values.",
    mimeType := "text/markdown",
    read := do
      let st ← env.state.get
      pure (Resources.renderProgramCatalog st) },
  { uri := "tropical://program-format",
    name := "Program format",
    description := "Reference doc for the tropical_program_2 schema.",
    mimeType := "text/markdown",
    read := pure Resources.programFormatDoc }]

private def enginePrompts : List Prompt := [
  { name := "build-patch",
    description := "Three-tiered workflow guidance for building and editing tropical patches efficiently.",
    get := match Resources.getPromptMessages "build-patch" with
      | .ok j => pure j
      | .error e => throw (IO.userError e) }]

-- ── Entry points ─────────────────────────────────────────────────────────────

def runFrontend : IO Unit := do
  let env ← Engine.boot
  let srv : Server := {
    name := "tropical", version := "0.1.0",
    tools := tropicalEngineTools env,
    resources := engineResources env,
    prompts := enginePrompts }
  srv.run                       -- runs until the MCP client closes stdin

def runRpc : IO Unit := do
  let env ← Engine.boot
  Rpc.run env

/-- `--serve <addr>` — the multi-client socket endpoint (C++-owned socket,
    control/data plane split). See `Tropical.Serve`. -/
def runServe (addr : String) : IO Unit :=
  Tropical.Serve.run addr

end Tropical
