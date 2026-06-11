import Tropical.Engine
import Tropical.Resources

/-!
# `--rpc` mode — the ir_service-compatible stdio surface

Speaks the exact newline JSON-RPC protocol `mcp/ir_service.ts` spoke
(method = tool name, plus `resources/*` and `prompts/*`), so the
existing `mcp/*.test.ts` suites and `scripts/diff/diff_engine.ts` drive
the Lean engine without modification. This is the Phase 1 gate surface.
-/

namespace Tropical.Rpc

open Lean (Json)

private def send (j : Json) : IO Unit := do
  let out ← IO.getStdout
  out.putStr (j.compress ++ "\n")
  out.flush

private def respond (id : Json) (result : Json) : IO Unit :=
  send <| Json.mkObj [("jsonrpc", Json.str "2.0"), ("id", id), ("result", result)]

private def respondError (id : Json) (message : String) : IO Unit :=
  send <| Json.mkObj [("jsonrpc", Json.str "2.0"), ("id", id),
    ("error", Json.mkObj [("code", Lean.toJson (-32603 : Int)), ("message", Json.str message)])]

def run (env : Engine.Env) : IO Unit := do
  let stdin ← IO.getStdin
  repeat
    let line ← stdin.getLine
    if line.isEmpty then break  -- EOF
    let trimmed := line.trim
    if trimmed.isEmpty then continue
    match Json.parse trimmed with
    | .error e => respondError Json.null s!"parse error: {e}"
    | .ok req =>
      let id := (req.getObjVal? "id").toOption.getD Json.null
      let method := match req.getObjVal? "method" with
        | .ok (.str m) => m
        | _ => ""
      let params := (req.getObjVal? "params").toOption.getD (Json.mkObj [])
      try
        let internalEnv := fun (msg : String) =>
          Tropical.failResult (.env { code := .internalError, message := msg, retryable := false })
        let result ←
          if method == "resources/list" then
            pure (Json.mkObj [("resources", Resources.resourcesList)])
          else if method == "resources/read" then
            let uri := match req.getObjVal? "params" with
              | .ok p => ((p.getObjValAs? String "uri").toOption).getD ""
              | .error _ => ""
            let st ← env.state.get
            match Resources.readResourceText st uri with
            | .ok text => pure (Json.mkObj [("text", Json.str text)])
            | .error msg => pure (internalEnv msg)
          else if method == "prompts/list" then
            pure (Json.mkObj [("prompts", Resources.promptsList)])
          else if method == "prompts/get" then
            let name := ((params.getObjValAs? String "name").toOption).getD ""
            match Resources.getPromptMessages name with
            | .ok j => pure j
            | .error msg => pure (internalEnv msg)
          else
            Engine.handleTool env method params
        respond id result
      catch e =>
        respondError id (toString e)

end Tropical.Rpc
