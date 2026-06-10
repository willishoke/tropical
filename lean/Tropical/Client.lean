import Tropical.Relay
import Tropical.Errors

/-!
# Compiler-service client

Typed access to `mcp/compiler_service.ts` over the relay. Service
payloads are `{...}` on success or `{error: <ErrorEnvelope>}` on a
tool-level failure; the envelope is propagated verbatim as a
`Failure.raw` so nothing is lost in translation.
-/

namespace Tropical

open Lean (Json)

structure Service where
  relay : Relay

namespace Service

/-- Call a service method; raise relayed error envelopes as failures. -/
def call (s : Service) (method : String) (params : Json) : EngineM Json := do
  let payload ← s.relay.call method params
  match payload.getObjVal? "error" with
  | .ok envelope =>
    -- `isError` marks a transport-level errorResult from the relay itself;
    -- a service envelope is a plain `{error: {...}}` payload.
    throw (.raw envelope)
  | .error _ =>
    match payload.getObjVal? "isError" with
    | .ok _ =>
      -- Relay-level fault (parse error, missing result): internal.
      throw (.env { code := .internalError, message := payload.compress, retryable := false })
    | .error _ => pure payload

/-- Field accessor on a service payload, failing internal on absence. -/
def field (payload : Json) (k : String) : EngineM Json :=
  match payload.getObjVal? k with
  | .ok v => pure v
  | .error _ =>
    let context := payload.compress.take 200
    let env : ErrorEnvelope :=
      { code := .internalError
        message := s!"compiler service: missing '{k}' in {context}"
        retryable := false }
    throw (.env env)

end Service

end Tropical
