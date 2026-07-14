import Tropical.Parse.Raise
import Tropical.Ir.Elaborator
import Tropical.Stdlib

/-!
# StdlibChain — the parsed-stdlib elaboration chain

Reads the committed parse bridge (`stdlib/parsed/*.json` + `manifest.json`)
and elaborates it in order, threading an external resolver over the earlier
results — producing the arena and name→index table every consumer links
stdlib programs against. One driver serves three callers with different
needs: the playground boot (`Playground.elabStdlib`, cached behind an
`IO.Ref`) takes the whole manifest and a flat error message; the `diffcli`
elab/strata/emit verbs take arbitrary name prefixes and need the STRUCTURED
`ElabError` (their output prints it as comparable JSON); the tropicaltest
arrow gates take the whole manifest. `elabChain` is the core (explicit
names, structured errors); `elabStdlib` is the whole-manifest, flattened
convenience over it.
-/

namespace Tropical.StdlibChain

/-- The committed parse bridge directory. -/
def parsedDir : String := "stdlib/parsed"

/-- Read + strictly decode a serialized ParsedProgram. Decode failures
    are harness errors (there is no second decoder to disagree with). -/
def readParsed (path : String) : IO (Except String Tropical.Parse.Program) := do
  let text ← IO.FS.readFile path
  pure <| do
    let jv ← Tropical.Parse.JsonV.parse text |>.mapError (s!"JSON parse error: {·}")
    Tropical.Parse.decodeProgram jv

/-- The bridge manifest's program names, in registration order. -/
def manifestNames : IO (Except String (Array String)) := do
  let text ← IO.FS.readFile s!"{parsedDir}/manifest.json"
  pure <| do
    let jv ← Tropical.Parse.JsonV.parse text |>.mapError (s!"manifest parse error: {·}")
    let some (Tropical.Parse.JsonV.arr items) := jv.getField? "programs"
      | .error "manifest missing 'programs' array"
    items.mapM fun
      | .str s => .ok s
      | _ => .error "manifest 'programs' entries must be strings"

/-- The raw-elaborated chain over `names` (elaborate-only — no strata), in
    the given order, threading an external resolver over earlier results.
    The outer `Except` is a harness error (unreadable/undecodable bridge
    file); the inner one preserves the structured `ElabError` together with
    the failing program's name. -/
def elabChain (names : Array String) :
    IO (Except String (Except (String × Tropical.Ir.ElabError)
      (Tropical.Ir.Arena × Array (String × Tropical.Ir.ProgramIdx)))) := do
  let mut arena : Tropical.Ir.Arena := {}
  let mut resolved : Array (String × Tropical.Ir.ProgramIdx) := #[]
  for name in names do
    match ← readParsed s!"{parsedDir}/{name}.json" with
    | .error e => return .error s!"{name}.json: {e}"
    | .ok prog =>
      let r := resolved
      match Tropical.Ir.elaborateInto arena prog
          (some fun n => (r.find? (·.1 == n)).map (·.2)) with
      | .error e => return .ok (.error (name, e))
      | .ok (arena', idx) =>
        arena := arena'
        resolved := resolved.push (name, idx)
  return .ok (.ok (arena, resolved))

/-- The whole stdlib chain the arena consumers (playground, arrow gates, diffcli)
    link against — now the arrow-builder chain (`buildStdlibChain`), not the
    parsed bridge. Pure, but kept in `IO` for its callers' signatures. The
    bridge-reading `elabChain`/`manifestNames`/`readParsed` above survive only
    for the diffcli `*-stdlib` verbs until those are retired. -/
def elabStdlib :
    IO (Except String (Tropical.Ir.Arena × Array (String × Tropical.Ir.ProgramIdx))) :=
  pure Tropical.EmitArrow.buildStdlibChain

end Tropical.StdlibChain
