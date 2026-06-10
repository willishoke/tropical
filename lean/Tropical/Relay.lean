import Turnstile

/-!
# Relay to the TypeScript IR service

The front door spawns `mcp/ir_service.ts` and forwards validated tool
calls to it over newline JSON-RPC. The MCP server loop is serial (one
call in flight), so a write-then-read-one-line suffices.

As the Lean port proceeds top-down, tools migrate from `fwd` (relayed)
to in-process handlers; the relay shrinks until Phase 6 deletes it.
-/

open Lean Turnstile

/-- A relay to a TS service over newline JSON-RPC. -/
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

/-- A no-argument tool that forwards an empty params object. -/
def noArgTool (r : Relay) (name desc : String) : Tool :=
  { name, description := desc,
    inputSchema := Json.mkObj [("type", Json.str "object"), ("properties", Json.mkObj [])],
    handler := fun _ => r.call name (Json.mkObj []) }

/-- A typed tool that validates against `α`'s schema, then re-encodes and
    forwards to the IR service under `name`. -/
def fwd (α : Type) [FromJson α] [ToJsonSchema α] [ToJson α]
    (r : Relay) (name desc : String) : Tool :=
  Tool.typed name desc (fun (a : α) => r.call name (toJson a))
