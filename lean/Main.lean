import Turnstile

/-!
# Tropical Lean front-door

A native Lean MCP server (built on Turnstile) that an MCP client talks to. It
spawns the tropical IR service (`mcp/ir_service.ts`), validates each tool call
against a typed schema, and relays the validated call over JSON-RPC/stdio.

Run from the tropical repo root (so `mcp/ir_service.ts` resolves):
  lean/.lake/build/bin/frontend
-/

open Lean Turnstile

-- ── Relay to the IR service ─────────────────────────────────────────────────

/-- A relay to `mcp/ir_service.ts` over newline JSON-RPC. The MCP server loop is
    serial (one call in flight), so a write-then-read-one-line suffices. -/
structure Relay where
  stdin  : IO.FS.Handle
  stdout : IO.FS.Handle

/-- Drop top-level `null`-valued keys so omitted optionals forward as absent
    (tropical's handlers treat absent and present-but-null differently). -/
private def stripNulls (j : Json) : Json :=
  match j with
  | .obj m => Json.mkObj (m.toList.filter (fun p => p.2.compress != "null"))
  | _      => j

/-- Forward one tool call to the IR service and return its `result` payload. -/
def Relay.call (r : Relay) (method : String) (params : Json) : IO Json := do
  let req := Json.mkObj [
    ("jsonrpc", Json.str "2.0"), ("id", toJson 0),
    ("method", Json.str method), ("params", stripNulls params)]
  r.stdin.putStr (req.compress ++ "\n")
  r.stdin.flush
  let line ← r.stdout.getLine
  match Json.parse line with
  | .error e => pure (errorResult s!"relay parse error: {e}")
  | .ok j =>
    match j.getObjVal? "result" with
    | .ok result => pure result
    | .error _   => pure (errorResult s!"relay: no result in {j.compress}")

-- ── Tool argument types (typed front; validated before forwarding) ──────────

tool_args AddInstance where
  /-- Registered program/type name (builtin or user-defined) -/
  program : String
  /-- Unique name for this instance -/
  instance_name : String
  /-- Compile-time type args for generic programs, e.g. {"N": 44100}. Omit for non-generic. -/
  type_args : Option Json
deriving instance ToJson for AddInstance

tool_args Wire where
  /-- Inputs to set: each {instance, input, expr, [combine]} -/
  set : Option (Array Json)
  /-- Inputs to disconnect: each {instance, input} -/
  remove : Option (Array Json)
deriving instance ToJson for Wire

tool_args StartAudio where
  /-- Optional partial device name match -/
  device_name : Option String
  /-- DAC sample rate in Hz (default 44100). Used only on first DAC creation. -/
  sample_rate : Option Nat where 1 ≤ sample_rate
  /-- DAC output channel count (default 2). Used only on first DAC creation. -/
  channels : Option Nat where 1 ≤ channels
deriving instance ToJson for StartAudio

-- ── Tool surface ────────────────────────────────────────────────────────────

/-- A no-argument tool that forwards an empty params object. -/
private def noArgTool (r : Relay) (name desc : String) : Tool :=
  { name, description := desc,
    inputSchema := Json.mkObj [("type", Json.str "object"), ("properties", Json.mkObj [])],
    handler := fun _ => r.call name (Json.mkObj []) }

def tropicalTools (r : Relay) : List Tool := [
  noArgTool r "list_programs"  "List all registered program types with their input/output ports.",
  noArgTool r "list_instances" "List all live program instances in the current session.",
  Tool.typed "add_instance" "Create a named instance of a registered program type."
    (fun (a : AddInstance) => r.call "add_instance" (toJson a)),
  Tool.typed "wire" "Set and/or remove input wiring in a single recompile. Audio output is instance \"dac\", input \"out\"."
    (fun (a : Wire) => r.call "wire" (toJson a)),
  Tool.typed "start_audio" "Start audio output."
    (fun (a : StartAudio) => r.call "start_audio" (toJson a)),
  noArgTool r "stop_audio" "Stop audio output."
]

-- ── Entry point ─────────────────────────────────────────────────────────────

def main : IO Unit := do
  let child ← IO.Process.spawn {
    cmd := "bun", args := #["run", "mcp/ir_service.ts"],
    stdin := .piped, stdout := .piped, stderr := .inherit
  }
  let relay : Relay := { stdin := child.stdin, stdout := child.stdout }
  let srv : Server := { name := "tropical", version := "0.1.0", tools := tropicalTools relay }
  srv.run                       -- runs until the MCP client closes stdin
  child.kill
  let _ ← child.wait
