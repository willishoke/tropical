import Tropical.Engine.ProgramIO.Export

/-!
# Engine.ProgramIO.Ingest — v2 load and merge

The tool shells over `Tropical.Session.{normalizePatch, loadPatch,
mergePatch}`: read the document, run the ingest against the session, then
compile and publish. The walk itself lives in `Session.Ingest`.
-/

namespace Tropical.Engine

open Lean (Json toJson)
open Tropical.Session (normalizePatch loadPatch mergePatch)

def handleLoad (env : Env) (args : Json) : EngineM Json := do
  let path := arg? args "path"
  let program := arg? args "program"
  if path.isNone && program.isNone then
    throwBare .missingArgument "Provide either path (file) or program (inline JSON)."
  -- Stop audio before replacing the session (TS handleLoad semantics).
  if let some dac := ← env.dac.get then
    if ← dac.isRunning then dac.stop
  let rawText ← match path with
    | some (.str p) =>
      match ← (IO.FS.readFile p).toBaseIO with
      | .ok text => pure text
      | .error e => internalError (toString e)
    | _ => pure (program.getD jsonNull).compress
  let t0 ← IO.monoMsNow
  env.session do
    let node ← normalizePatch rawText
    loadPatch node
  syncCompile env
  let t1 ← IO.monoMsNow
  let st ← env.state.get
  pure <| Json.mkObj [
    ("instances", toJson st.instanceNames),
    ("wiring", toJson st.wires.size),
    ("outputs", toJson st.graphOutputs.size),
    ("params", toJson (st.params.map (·.1))),
    ("timing", Json.mkObj [("wall_ms", toJson (t1 - t0))])]

def handleMerge (env : Env) (args : Json) : EngineM Json := do
  let raw := match arg? args "program" with
    | some p => some p
    | none => arg? args "patch"
  let some program := raw
    | throwBare .missingArgument "Provide a program or patch object."
  env.session do
    let node ← normalizePatch program.compress
    mergePatch node
  syncCompile env
  let st ← env.state.get
  pure <| Json.mkObj [
    ("instances", toJson st.instanceNames),
    ("wiring", toJson st.wires.size),
    ("outputs", toJson st.graphOutputs.size),
    ("params", toJson (st.params.map (·.1)))]

end Tropical.Engine
