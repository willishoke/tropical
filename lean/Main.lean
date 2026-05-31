import Turnstile

/-!
Relay spike. Spawns `mcp/ir_service.ts`, sends one `list_programs` request over
JSON-RPC/stdio, and prints the response. Proves two things at once:
  1. the cross-repo Lake dependency on Turnstile resolves and builds, and
  2. the Lean → ir_service subprocess/stdio relay works.

Run from the tropical repo root (so `mcp/ir_service.ts` resolves):
  lean/.lake/build/bin/frontend
-/

open Lean

def main : IO Unit := do
  IO.eprintln "[spike] spawning: bun run mcp/ir_service.ts"
  let child ← IO.Process.spawn {
    cmd    := "bun"
    args   := #["run", "mcp/ir_service.ts"]
    stdin  := .piped
    stdout := .piped
    stderr := .inherit
  }

  -- one request, serial: write a line, read a line back
  let req := "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"list_programs\",\"params\":{}}"
  child.stdin.putStr (req ++ "\n")
  child.stdin.flush
  let line ← child.stdout.getLine

  child.kill
  let _ ← child.wait

  if line.isEmpty then
    IO.eprintln "[spike] ✗ no response (child died — check the bun stderr above)"
    throw (IO.userError "ir_service produced no output")

  match Json.parse line with
  | .error e => IO.eprintln s!"[spike] ✗ parse error: {e}"
  | .ok j =>
    IO.println s!"[spike] ✓ {line.length} bytes back, valid JSON"
    match j.getObjVal? "result" with
    | .ok _    => IO.println "[spike] ✓ response has a `result` field"
    | .error _ => IO.println "[spike] ✗ missing `result` field"
    IO.println s!"[spike] head: {line.take 200}"
