import Tropical.Playground.Report

/-!
# Playground.Compile

Graph-to-plan compilation and the cached stdlib entry point.
-/

namespace Tropical.Playground

open Lean (Json JsonNumber)
open Tropical.Ir
open Tropical.EmitArrow
open Tropical.Exact (DyadicI)

-- ── Graph → loadable FlatPlan (mirrors Tropicaltest.compileArrowCarrier) ─────
/-- Compile the GUI graph to a loadable `FlatPlan`, plus the scope taps it
    materializes. The final mix is always tap `out` (`__root__.out`); each user
    node gets a `tap:<id>` output port carrying `emit(normalize(lowerNode id))`
    — the signal at that node's jack — so the collapse keeps the slot alive and
    `render_window` can read it. Taps cost extra kernel compute (each re-emits its
    upstream cone), so they're for the inspection build, not a lean audio path. -/
def compilePlanPure (arena : Arena) (resolved : Array (String × ProgramIdx)) (j : Json) :
    Except String CompiledPatch := do
  let raws := rawsOf j
  let usesGroupedRoom := raws.any (·.kind == "groupedroom")
  let usesGroupedRoomCache := raws.any (·.kind == "groupedroomcache")
  if usesGroupedRoom && usesGroupedRoomCache then
    throw "groupedroom: direct and fixed-scene cache profiles cannot be mixed in one graph"
  checkServedKinds raws                              -- reject withheld/unknown kinds FIRST (honest msg over the misleading type error)
  checkEdgeTypes raws                                -- reject ill-typed / dangling / wire-into-knob edges pre-lowering
  checkOutTarget j raws                              -- reject an "out" id that names no node
  -- (cycles are refused by `lowerGraph` itself now — the topo sort that
  -- funds the lowering's termination IS the cycle check, loop named)
  let (g, paramTable) ← decodeGraph j
  let term ← lowerGraph g
  let (out, b0) := emitTerm (normalize term) {}
  -- Per-node taps are opt-in: `"taps": true` retains the inspection/debug
  -- surface, while `"taps": ["id", ...]` emits only named projections. The
  -- latter is the production scope path: it avoids re-emitting every visible
  -- node's upstream cone when the UI needs a small, explicit set of signals.
  -- The final mix (`out`) remains tappable regardless.
  let visibleTapIds := tapNodeIds (rawsOf j)
  let tapIds := match j.getObjVal? "taps" with
    | .ok (.bool true) => visibleTapIds
    | .ok (.arr requested) => requested.filterMap fun
        | Json.str id => if visibleTapIds.contains id then some id else none
        | _ => none
    | _ => #[]
  -- Emit each user node's sub-term into the SAME builder (so instance names stay
  -- unique by `decls.size`), collecting `(id, signal)`. A node that fails to
  -- lower is simply not tapped.
  let (tapSigs, b) : Array (String × Sig) × Builder :=
    tapIds.foldl (fun (acc : Array (String × Sig) × Builder) id =>
      match lowerAt g id with
      | .ok t => let (s, b') := emitTerm (normalize t) acc.2; (acc.1.push (id, s), b')
      | .error _ => acc) (#[], b0)
  let registry ← buildRegistry arena resolved #["FixedSinOsc", "MorphOsc", "PluckedMorphOsc"]
  -- One `.param` decl per live knob, appended AFTER the instance decls: declaration
  -- order = `ParamIdx` = the index each `paramRef` carries, and keeping params after
  -- instances leaves the emitted voices' `InstanceIdx` untouched. `compileSession`
  -- allocates each a `param:<id>.<knob>` module slot, driven live by `set_param`.
  let paramDecls : Array BodyDecl := paramTable.map (fun (nm, v) => .param nm (some v))
  -- Output port 0 is the audible mix; ports 1.. are the taps (`tap:<id>`).
  let tapOutputs := tapSigs.map fun (id, _) =>
    ({ name := s!"tap:{id}", type? := some (.scalar .float) } : OutputDecl)
  let tapAssigns : Array (Tropical.Ir.OutputTarget × Sig) :=
    tapSigs.mapIdx fun i (_, s) => (.port ⟨i + 1⟩, s)
  -- Assemble through EmitArrow's lowering boundary (interns every `Sig` into the
  -- arena's DAG); the live param decls append after the instance decls.
  let rootInputs := if usesGroupedRoom then groupedRoomInputDecls
    else if usesGroupedRoomCache then groupedRoomCacheInputDecls else #[]
  let (arena1, idx) := Tropical.EmitArrow.assemble arena "__patch__" rootInputs
    (#[{ name := "out", type? := some (.scalar .float) }] ++ tapOutputs)
    b.decls (#[(.port ⟨0⟩, out)] ++ tapAssigns) registry
    (extraDecls := paramDecls)
  let (coreArena, core) ← (Tropical.Ir.Strata.runResolved {} arena1 idx).mapError (·.message)
  let input := Tropical.Compile.SessionInput.forRoot core coreArena
    (params := paramTable.map (fun (nm, v) => (nm, Json.num v)))
    (alloc := Tropical.Lowering.allocate (paramTable.map (·.1)) #[])
  let (plan, stageBlocks) ← Tropical.Compile.compileSessionStaged input
  let plan ← if usesGroupedRoom then bindGroupedRoomAsset plan else pure plan
  let plan ← if usesGroupedRoomCache then bindGroupedRoomCacheAsset plan else pure plan
  -- The host-contract dispatch table rides the manifest: any runtime host
  -- (C++ today, Swift/Metal or wasm tomorrow) reads per-slot disciplines from
  -- the plan itself and dispatches param writes locally.
  let plan := { plan with paramDisciplines := paramDisciplinesOf (rawsOf j) }
  -- The final mix (`out`) plus one tap per user node, all routed to the synthetic
  -- root's output slots (`__root__.<port>`), ready for `render_window`.
  let root := Tropical.Compile.rootInstancePath
  let taps : Array Tap := #[{
      name := "out"
      sourceInstance := root
      sourceOutput := "out" }]
    ++ tapSigs.map (fun (id, _) => {
      name := id
      sourceInstance := root
      sourceOutput := s!"tap:{id}" })
  pure { plan, taps, stageBlocks }

-- ── Stdlib-into-arena (the shared chain; cached below via `getStdlib`) ───────
def elabStdlib : IO (Except String (Arena × Array (String × ProgramIdx))) :=
  Tropical.StdlibChain.elabStdlib

initialize stdlibCache : IO.Ref (Option (Arena × Array (String × ProgramIdx))) ← IO.mkRef none

def getStdlib : IO (Except String (Arena × Array (String × ProgramIdx))) := do
  match ← stdlibCache.get with
  | some v => pure (.ok v)
  | none =>
    match ← elabStdlib with
    | .error e => pure (.error e)
    | .ok v => stdlibCache.set (some v); pure (.ok v)

/-- Every live param as `(name, valueJson)` — the session param mirror the engine
    seeds after a successful load, so `set_param` (which guards on the mirror and
    drives the `param:<name>` slot) reaches EVERY live knob of a `load_patch_graph`
    plan, with no relower. Same `collectParams` the compile uses, so the mirror and
    the allocated slots agree name-for-name; value kept as raw JSON. -/
def knobParams (j : Json) : Array (String × Json) :=
  (collectParams (rawsOf j)).map (fun (nm, v) => (nm, Json.num v))

/-- Decode + lower + compile the GUI graph to a loadable `FlatPlan` + its
    taps + typed stage blocks (the split classification). -/
def compilePlan (j : Json) : IO (Except String CompiledPatch) := do
  match ← getStdlib with
  | .error e => pure (.error s!"stdlib elaboration: {e}")
  | .ok (arena, resolved) => pure (compilePlanPure arena resolved j)
