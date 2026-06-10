import Turnstile
import Tropical.Relay
import Tropical.Tools
import Tropical.Engine
import Tropical.Rpc

/-!
# Tropical Lean front-door

The native Lean MCP server (built on Turnstile). As of Phase 1 of the
Lean port, tool semantics run **in-process** (`Tropical.Engine`): the
session lives in Lean, and the spawned subprocess is the *compiler
service* (`mcp/compiler_service.ts`) — program registration,
compilation, and the runtime FFI — not a full TS engine.

Two modes:
- default — the MCP server (stdio), launched by `make mcp-lean`
- `--rpc` — the ir_service-compatible newline JSON-RPC surface used by
  the bun test suites and the differential harness

Run from the tropical repo root (so `mcp/compiler_service.ts` resolves):
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

-- ── Resources & prompts ─────────────────────────────────────────────────────
-- Discovered from the compiler service at startup; read/get relay back.

private def buildResources (r : Relay) (listResult : Json) : List Resource :=
  match listResult.getObjVal? "resources" with
  | .ok (.arr arr) => arr.toList.filterMap fun e =>
      match e.getObjValAs? String "uri" with
      | .ok uri => some {
          uri, name := (e.getObjValAs? String "name").toOption.getD uri,
          description := (e.getObjValAs? String "description").toOption.getD "",
          mimeType := (e.getObjValAs? String "mimeType").toOption.getD "text/plain",
          read := do
            let res ← r.call "resources/read" (Json.mkObj [("uri", Json.str uri)])
            pure ((res.getObjValAs? String "text").toOption.getD "") }
      | .error _ => none
  | _ => []

private def buildPrompts (r : Relay) (listResult : Json) : List Prompt :=
  match listResult.getObjVal? "prompts" with
  | .ok (.arr arr) => arr.toList.filterMap fun e =>
      match e.getObjValAs? String "name" with
      | .ok name => some {
          name, description := (e.getObjValAs? String "description").toOption.getD "",
          get := r.call "prompts/get" (Json.mkObj [("name", Json.str name)]) }
      | .error _ => none
  | _ => []

-- ── Entry points ─────────────────────────────────────────────────────────────

def runFrontend : IO Unit := do
  let (env, child) ← Engine.boot
  let relay := env.service.relay
  let resources := buildResources relay (← relay.call "resources/list" (Json.mkObj []))
  let prompts   := buildPrompts   relay (← relay.call "prompts/list"   (Json.mkObj []))
  let srv : Server := {
    name := "tropical", version := "0.1.0",
    tools := tropicalEngineTools env, resources, prompts }
  srv.run                       -- runs until the MCP client closes stdin
  child.kill
  let _ ← child.wait

def runRpc : IO Unit := do
  let (env, child) ← Engine.boot
  Rpc.run env
  child.kill
  let _ ← child.wait

end Tropical
