import Tropical.Engine
import Tropical.NativeHost

/-!
# NativeBoot — the composition root for a native-hosted engine

Creates the libtropical runtime and boots the session engine against its
`HostPort`. Every executable that drives a live engine (the MCP front door,
`--rpc`, `--serve`, diffcli, tropicaltest) boots through here; the engine
itself never constructs or names the runtime.
-/

namespace Tropical.NativeBoot

/-- Qualification-only runtime block-length seam. The live default stays 512;
    the opt-in environment value is read before Runtime/DAC construction and
    is already visible through runtime telemetry's `buffer_length`. -/
private def bufferLength : IO UInt32 := do
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

/-- `TROPICAL_BACKEND=metal` selects the Metal realization for live audio:
    every session compile dual-loads (LLVM IR → JIT + MSL → GPU pipeline)
    and audio dispatches on Metal; the JIT keeps serving render_window and
    the scope either way. Read once here, at the composition root. -/
private def backend : IO Host.Backend := do
  pure (if (← IO.getEnv "TROPICAL_BACKEND") == some "metal" then .metal else .jit)

/-- Boot an engine on a fresh native runtime, returning both. The runtime is
    for host-side services that need the raw handle (the `--serve` socket). -/
def bootWithRuntime : IO (Ffi.Runtime × Engine.Env) := do
  let frames ← bufferLength
  let runtime ← Ffi.Runtime.new frames
  let env ← Engine.boot (runtime.hostPort frames.toNat (← backend))
  pure (runtime, env)

/-- Boot an engine on a fresh native runtime. -/
def boot : IO Engine.Env := do
  let (_, env) ← bootWithRuntime
  pure env

end Tropical.NativeBoot
