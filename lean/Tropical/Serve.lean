import Tropical.Engine
import Tropical.Rpc
import Tropical.Resources
import Tropical.Ffi

/-!
# `--serve` mode — the C++-owned socket endpoint

A single Unix-domain socket (owned by the C++ engine) fans many clients onto
one shared session. C++ handles the data plane locally (param writes, telemetry
GETs) and queues control-plane requests; this single Lean driver loop PULLS each
control request, dispatches it through the same `Rpc.handleRequest` the `--rpc`
stdio surface uses, and PUSHES the response back. One Lean thread ⇒ control
dispatch stays serial against the one `SessionSt`, exactly like the stdio loop.

The data plane never reaches here — high-rate param writes hit the runtime
slot directly in C++ and never queue.
-/

namespace Tropical.Serve

open Lean (Json)

def run (addr : String) : IO Unit := do
  let env ← Engine.boot
  let sock ← Ffi.Socket.listen env.runtime addr
  IO.eprintln s!"tropical: serving on {addr}"
  repeat
    match ← Ffi.Socket.nextControl sock with
    | none => break  -- socket shut down
    | some (clientId, line) =>
      let resp ←
        match Json.parse line.trim with
        | .error e => pure (Rpc.errorEnvelope Json.null s!"parse error: {e}")
        | .ok req  => Rpc.handleRequest env req
      Ffi.Socket.sendResponse sock clientId resp.compress

end Tropical.Serve
