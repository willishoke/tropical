import Tropical.Session.Compile
import Tropical.Engine.ProgramIO.Export
import Tropical.Tropicaltest.Spline

/-!
# Type-family gates: a registered type at a size

`lean/Tropical/Family.lean` and `design/type-family.md` are what these back.
The periodic splines are the families under test; the reference is the
independent f64 spline oracle of `Tropicaltest/Spline.lean` (`refEval`).

* **family-json** — one `tropical_program_2` patch through the real front door
  (`compilePatch`: normalize, ingest, compile) instantiating
  `PeriodicCatmullRom<K>` and `PeriodicBSpline<K>` at sizes 1 to 100,
  including the boot-registered size 8, each read by a hashed and a grid
  phase, one DAC channel per instance. Every channel matches the reference.
  Each member is registered once with `K + 1` inputs `phase, c0 … c(K−1)`.
* **family-session** — the `add_instance` path (`resolveInstanceMeta` with
  tool envelopes, then `addInstance` and wires), at K = 12 and 5: renders
  match the reference, and naming a member again reuses its registration.
* **family-refusals** — unknown family, missing size, size 0, size 2³¹, a
  leading zero, and a size on a type that is not a family, through ingest
  (`internal_error` messages) and through the tool shape (`unknown_program`,
  `invalid_value`, the range as a predicate); plus the boot-time family
  contract (`familyConflicts`) on a bad family set.
* **family-roundtrip** — save a session holding K = 12 and K = 5 members
  (`saveDocument`, the `save` result), load the document into a fresh session:
  the instance types, their ports and the rendered audio are identical.
-/

namespace Tropical.Tropicaltest.TypeFamily

open Tropical
open Tropical.EmitArrow
open Tropical.Tropicaltest.Spline
open Tropical.Tropicaltest.Spectral (range maxAbsDiff peakOf sci)

private def frames : Nat := 1024

private def boot : Except String SessionSt :=
  Session.bootRegistry.mapError (·.toJson.compress)

/-- A member under test: the family, its basis (for the reference) and K. -/
structure Member where
  family : TypeFamily
  basis : SplineBasis
  k : Nat
deriving Inhabited

def Member.name (m : Member) : String := m.family.memberName m.k

/-- Non-zero control values (`m/64`), so a K = 1 spline is not silent. -/
def Member.controls (m : Member) : Array Int :=
  (controlsOf (m.k * 7919 + 5) m.k).map fun c => if c == 0 then 1 else c

def cr (k : Nat) : Member := { family := periodicCatmullRomFamily, basis := catmullRomBasis, k }
def bs (k : Nat) : Member := { family := periodicBSplineFamily, basis := uniformBSplineBasis, k }

def members : Array Member :=
  #[cr 1, cr 2, cr 3, cr 5, cr 8, cr 12, cr 33, bs 1, bs 2, bs 4, bs 8, bs 12, bs 100]

def laws : Array PhaseLaw := #[.hashed, .grid]

private def phaseJson : PhaseLaw → String
  | .hashed => "{\"op\":\"mul\",\"args\":[{\"op\":\"sampleIndex\"}," ++ toString hashMul ++ "]}"
  | .grid => "{\"op\":\"mul\",\"args\":[{\"op\":\"sampleIndex\"},16777216]}"

private def instanceJson (inst : String) (m : Member) (law : PhaseLaw) : String :=
  let cs := m.controls.toList.zipIdx.map fun (c, j) =>
    "\"c" ++ toString j ++ "\":" ++ toString (controlFloat c)
  "{\"op\":\"instanceDecl\",\"name\":\"" ++ inst ++ "\",\"program\":\"" ++ m.name ++
  "\",\"inputs\":{\"phase\":" ++ phaseJson law ++ "," ++ ",".intercalate cs ++ "}}"

private def outJson (inst : String) (channel : Nat) : String :=
  "{\"op\":\"outputAssign\",\"name\":\"dac.out\",\"channel\":" ++ toString channel ++
  ",\"expr\":{\"op\":\"ref\",\"instance\":\"" ++ inst ++ "\",\"output\":\"out\"}}"

/-- One patch: every member under every law, instance `i` on channel `i`. -/
def patchOf (ms : Array Member) : String := Id.run do
  let mut decls := #[]
  let mut outs := #[]
  for m in ms do
    for law in laws do
      let inst := s!"s{decls.size}"
      outs := outs.push (outJson inst decls.size)
      decls := decls.push (instanceJson inst m law)
  "{\"schema\":\"tropical_program_2\",\"name\":\"type_family\",\"body\":{\"op\":\"block\",\"decls\":[" ++
    ",".intercalate decls.toList ++ "],\"assigns\":[" ++ ",".intercalate outs.toList ++ "]}}"

def reference (m : Member) (law : PhaseLaw) : Array Float :=
  let cs := m.controls.map controlFloat
  (range 0 frames).map fun n => refEval m.basis .wrap cs (phaseInt law .wrap n)

/-- Render `channels` channels of the session's compiled plan over
    `n ∈ [0, frames)`. -/
def renderSession (st : SessionSt) (channels : Nat) :
    IO (Except String (Array (Array Float))) := do
  match (Session.compileStaged "type-family").runOn st with
  | (.error f, _) => return .error s!"compile: {f.toJson.compress}"
  | (.ok (plan, blocks), _) =>
    let rt ← Tropical.Ffi.Runtime.new frames.toUInt32
    Tropical.StagedLoad.loadTyped rt plan blocks
    let taus := (range 0 frames).map fun n => (n * Tropical.Ir.clockOne).toUInt64
    let chans := (Array.range channels).map UInt32.ofNat
    let values := decodeF64LE (← rt.renderTau taus chans)
    return .ok ((Array.range channels).map fun c =>
      values.extract (c * frames) ((c + 1) * frames))

/-- The worst error of rendered channels against `(member, law)` references,
    in channel order. -/
def worstAgainst (got : Array (Array Float)) (refs : Array (String × Array Float)) :
    Float × String × Option String := Id.run do
  let mut worst := 0.0
  let mut at_ := ""
  for i in [0:refs.size] do
    let (label, want) := refs[i]!
    let g := got[i]!
    if g.size != want.size || peakOf want == 0.0 then
      return (worst, at_, some s!"{label}: rendered {g.size} samples, reference peak {peakOf want}")
    let e := maxAbsDiff g want
    if !worst.isNaN && (e.isNaN || e > worst || at_ == "") then
      worst := e
      at_ := label
  return (worst, at_, none)

private def lawName : PhaseLaw → String
  | .hashed => "hashed"
  | .grid => "grid"

private def loadText (st : SessionSt) (text : String) : Except String SessionSt :=
  match (do Session.loadPatch (← Session.normalizePatch text)).runOn st with
  | (.error f, _) => .error f.toJson.compress
  | (.ok (), st') => .ok st'

-- ── family-json ──────────────────────────────────────────────────────────

def runJson : IO Bool := do
  let label := "family-json"
  let st0 ← match boot with
    | .error e => return ← failGate label s!"boot: {e}"
    | .ok st => pure st
  let st ← match loadText st0 (patchOf members) with
    | .error e => return ← failGate label s!"load: {e}"
    | .ok st => pure st
  -- Each member registered once, under its own name, with K + 1 inputs.
  for m in members do
    let some pm := st.programs.get? m.name
      | return ← failGate label s!"{m.name} is not registered after ingest"
    let want := #["phase"] ++ (Array.range m.k).map (s!"c{·}")
    if pm.inputNames != want || pm.outputNames != #["out"] then
      return ← failGate label s!"{m.name}: ports {pm.inputNames} → {pm.outputNames}"
    if (st.catalogOrder.filter (· == m.name)).size != 1 then
      return ← failGate label s!"{m.name} is in the catalog {(st.catalogOrder.filter (· == m.name)).size} times"
  let refs := members.flatMap fun m =>
    laws.map fun law => (s!"{m.name}/{lawName law}", reference m law)
  let got ← match ← renderSession st refs.size with
    | .error e => return ← failGate label e
    | .ok g => pure g
  let (worst, at_, bad?) := worstAgainst got refs
  if let some e := bad? then return ← failGate label e
  if worst ≤ 1e-12 then
    passGate label s!"{refs.size} channels ({members.size} members, K = 1 … 100, both bases, hashed + grid phases) through load ≡ f64 reference, worst |Δ| {sci worst} ({at_})"
  else
    failGate label s!"worst |Δ| {sci worst} at {at_} (tolerance 1e-12)"

-- ── family-session ───────────────────────────────────────────────────────

private def wireOf (j : String) : Except String WireExpr := do
  WireExpr.ofJson (← Lean.Json.parse j)

/-- `add_instance` + `wire` + a DAC route, as the tools do it. -/
private def addMember (inst : String) (m : Member) (law : PhaseLaw) (channel : Nat) :
    Session.SessionM Unit := do
  let resolved ← Session.resolveInstanceMeta m.name none "program"
  modify (·.addInstance inst (resolved.toInstanceInfo m.name))
  let wire : String → String → Session.SessionM Unit := fun port json => do
    match wireOf json with
    | .error e => internalError e
    | .ok w => modify (·.setWireRaw inst port w)
  wire "phase" (phaseJson law)
  for (c, j) in m.controls.toList.zipIdx do
    wire s!"c{j}" (toString (controlFloat c))
  let route : GraphOutput := { channel, sourceInstance := inst, sourceOutput := "out" }
  modify fun st => { st with graphOutputs := st.graphOutputs.push route }

def runSession : IO Bool := do
  let label := "family-session"
  let st0 ← match boot with
    | .error e => return ← failGate label s!"boot: {e}"
    | .ok st => pure st
  let ms := #[cr 12, bs 5]
  let build : Session.SessionM Unit := do
    addMember "a" ms[0]! .hashed 0
    addMember "b" ms[1]! .grid 1
    addMember "c" ms[0]! .grid 2
  let st ← match build.runOn st0 with
    | (.error f, _) => return ← failGate label s!"add_instance: {f.toJson.compress}"
    | (.ok (), st) => pure st
  let added := st.catalogOrder.size - st0.catalogOrder.size
  if added != 2 then
    return ← failGate label s!"{added} types registered for two members named three times"
  let snap := fun (n : String) => (st.findInstance? n).bind (·.resolvedIdx) |>.map (·.idx)
  if snap "a" != snap "c" || (snap "a").isNone then
    return ← failGate label s!"the second {ms[0]!.name} instance took snapshot {snap "c"}, the first {snap "a"}"
  let refs := #[(s!"{ms[0]!.name}/hashed", reference ms[0]! .hashed),
    (s!"{ms[1]!.name}/grid", reference ms[1]! .grid),
    (s!"{ms[0]!.name}/grid", reference ms[0]! .grid)]
  let got ← match ← renderSession st refs.size with
    | .error e => return ← failGate label e
    | .ok g => pure g
  let (worst, at_, bad?) := worstAgainst got refs
  if let some e := bad? then return ← failGate label e
  if worst ≤ 1e-12 then
    passGate label s!"add_instance of {ms[0]!.name} (twice, one registration) and {ms[1]!.name} ≡ f64 reference, worst |Δ| {sci worst} ({at_})"
  else
    failGate label s!"worst |Δ| {sci worst} at {at_} (tolerance 1e-12)"

-- ── family-refusals ──────────────────────────────────────────────────────

/-- A name, the ingest message fragment, and the tool code (and whether the
    tool envelope carries a range predicate). -/
def refusals : Array (String × String × String × Bool) := #[
  ("PeriodicHermite12", "Unknown program type 'PeriodicHermite12'", "unknown_program", false),
  ("PeriodicCatmullRom", "is a type family and needs a size", "invalid_value", false),
  ("PeriodicBSpline0", "size 0 is out of range for the type family 'PeriodicBSpline' (sizes 1 to 2147483647", "invalid_value", true),
  ("PeriodicCatmullRom2147483648", "size 2147483648 is out of range", "invalid_value", true),
  ("PeriodicCatmullRom012", "write the size without leading zeros, 'PeriodicCatmullRom12'", "invalid_value", false),
  ("VCA3", "'VCA' is not a type family and takes no size", "invalid_value", false)]

private def onePatch (program : String) : String :=
  "{\"schema\":\"tropical_program_2\",\"name\":\"refuse\",\"body\":{\"op\":\"block\",\"decls\":[" ++
  "{\"op\":\"instanceDecl\",\"name\":\"x\",\"program\":\"" ++ program ++ "\"}]}}"

private def has (s fragment : String) : Bool := (s.splitOn fragment).length > 1

def runRefusals : IO Bool := do
  let label := "family-refusals"
  let st0 ← match boot with
    | .error e => return ← failGate label s!"boot: {e}"
    | .ok st => pure st
  for (name, fragment, code, predicate) in refusals do
    -- Ingest: `internal_error` with the message.
    match loadText st0 (onePatch name) with
    | .ok _ => return ← failGate label s!"load of '{name}' was accepted"
    | .error e =>
      if !(has e fragment && has e "internal_error") then
        return ← failGate label s!"load of '{name}': {e} (want internal_error with “{fragment}”)"
    -- The tool shape (`add_instance`).
    match (Session.resolveInstanceMeta name none "program").runOn st0 with
    | (.ok _, _) => return ← failGate label s!"add_instance of '{name}' was accepted"
    | (.error f, st') =>
      let j := f.toJson
      let got := (j.getObjValAs? String "code").toOption.getD ""
      let kind := ((j.getObjVal? "valid").toOption.bind
        (·.getObjValAs? String "kind" |>.toOption)).getD ""
      if got != code || (predicate && kind != "predicate") then
        return ← failGate label s!"add_instance of '{name}': {j.compress} (want {code}{if predicate then ", a predicate" else ""})"
      if st'.catalogOrder.size != st0.catalogOrder.size then
        return ← failGate label s!"add_instance of '{name}' registered a type"
  -- The boot contract on a bad family set.
  let bad : Array TypeFamily := #[{ periodicCatmullRomFamily with name := "Saw8" },
    periodicBSplineFamily, periodicBSplineFamily]
  let conflicts := familyConflicts bad #["PeriodicBSpline", "PeriodicBSpline07"]
  let stdConflicts := familyConflicts stdlibFamilies st0.catalogOrder
  -- Saw8 ends in a digit; the second PeriodicBSpline is a duplicate; both
  -- PeriodicBSpline entries are a registered type; PeriodicBSpline07 is a
  -- registered non-canonical member.
  if conflicts.size != 5 || !stdConflicts.isEmpty then
    return ← failGate label s!"familyConflicts: {conflicts.size} conflicts on the bad set (want 5: {conflicts}), {stdConflicts.size} on the stdlib"
  passGate label s!"{refusals.size} family spellings refused through load and add_instance with their reasons; the boot contract flags a digit-final name, a duplicate, a family that is a type, and a registered non-canonical member"

-- ── family-roundtrip ─────────────────────────────────────────────────────

private def roundTripPatch : String :=
  let m := cr 12
  let params := m.controls.toList.zipIdx.map fun (c, j) =>
    "{\"op\":\"paramDecl\",\"name\":\"k" ++ toString j ++ "\",\"value\":" ++ toString (controlFloat c) ++ "}"
  let wires := (List.range m.k).map fun j =>
    "\"c" ++ toString j ++ "\":{\"op\":\"param\",\"name\":\"k" ++ toString j ++ "\"}"
  "{\"schema\":\"tropical_program_2\",\"name\":\"round_trip\",\"body\":{\"op\":\"block\",\"decls\":[" ++
  ",".intercalate params ++ "," ++
  "{\"op\":\"instanceDecl\",\"name\":\"wave\",\"program\":\"" ++ m.name ++ "\",\"inputs\":{\"phase\":" ++
    phaseJson .hashed ++ "," ++ ",".intercalate wires ++ "}}," ++
  instanceJson "env" (bs 5) .grid ++ "]," ++
  "\"assigns\":[" ++ outJson "wave" 0 ++ "," ++ outJson "env" 1 ++ "]}}"

def runRoundTrip : IO Bool := do
  let label := "family-roundtrip"
  let st0 ← match boot with
    | .error e => return ← failGate label s!"boot: {e}"
    | .ok st => pure st
  let st1 ← match loadText st0 roundTripPatch with
    | .error e => return ← failGate label s!"load: {e}"
    | .ok st => pure st
  let saved := Engine.saveDocument st1
  let some doc := (saved.getObjVal? "program").toOption
    | return ← failGate label s!"save: no program in {saved.compress}"
  let st2 ← match loadText st0 doc.compress with
    | .error e => return ← failGate label s!"reload of the saved document: {e}"
    | .ok st => pure st
  let shape := fun (st : SessionSt) => st.instances.map fun (n, i) =>
    (n, i.baseTypeName, i.progMeta.inputNames)
  if shape st1 != shape st2 then
    return ← failGate label s!"instances before {shape st1}, after {shape st2}"
  let types := (shape st2).map (·.2.1)
  if types != #["PeriodicCatmullRom12", "PeriodicBSpline5"] then
    return ← failGate label s!"reloaded instance types {types}"
  let a ← match ← renderSession st1 2 with
    | .error e => return ← failGate label s!"render before: {e}"
    | .ok g => pure g
  let b ← match ← renderSession st2 2 with
    | .error e => return ← failGate label s!"render after: {e}"
    | .ok g => pure g
  let diff := bitDiffCount a[0]! b[0]! + bitDiffCount a[1]! b[1]!
  let err := maxAbsDiff a[0]! (reference (cr 12) .hashed)
  if diff == 0 && a[0]!.size == frames && err ≤ 1e-12 then
    passGate label s!"save → load keeps {types} (sizes and ports) and renders bit for bit the same ({2 * frames} samples)"
  else
    failGate label s!"{diff} samples differ after the round trip; before vs reference {sci err}"

def runTypeFamily : IO (Nat × Nat) := do
  let results := #[← runJson, ← runSession, ← runRefusals, ← runRoundTrip]
  pure (results.size, results.filter (!·) |>.size)

end Tropical.Tropicaltest.TypeFamily
