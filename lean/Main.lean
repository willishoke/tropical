import Tropical.Frontend

/-!
The `frontend` executable: the tropical MCP server (default) or the
ir_service-compatible JSON-RPC surface (`--rpc`, used by the bun test
suites and the differential harness). All logic lives in the `Tropical`
library.
-/

def main (args : List String) : IO Unit :=
  if args.contains "--rpc" then Tropical.runRpc else Tropical.runFrontend
