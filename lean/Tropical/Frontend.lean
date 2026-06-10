import Turnstile
import Tropical.Relay
import Tropical.Tools

/-!
# Tropical Lean front-door

A native Lean MCP server (built on Turnstile) that an MCP client talks
to. It spawns the tropical IR service (`mcp/ir_service.ts`), validates
each tool call against a typed schema, and relays the validated call
over JSON-RPC/stdio.

Run from the tropical repo root (so `mcp/ir_service.ts` resolves):
  lean/.lake/build/bin/frontend
-/

open Lean Turnstile

namespace Tropical

-- ── Resources & prompts ─────────────────────────────────────────────────────
-- Discovered from the IR service at startup; read/get forward back to it.

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

-- ── Entry point ─────────────────────────────────────────────────────────────

def runFrontend : IO Unit := do
  let child ← IO.Process.spawn {
    cmd := "bun", args := #["run", "mcp/ir_service.ts"],
    stdin := .piped, stdout := .piped, stderr := .inherit
  }
  let relay : Relay := { stdin := child.stdin, stdout := child.stdout }
  let resources := buildResources relay (← relay.call "resources/list" (Json.mkObj []))
  let prompts   := buildPrompts   relay (← relay.call "prompts/list"   (Json.mkObj []))
  let srv : Server := {
    name := "tropical", version := "0.1.0",
    tools := tropicalTools relay, resources, prompts }
  srv.run                       -- runs until the MCP client closes stdin
  child.kill
  let _ ← child.wait

end Tropical
