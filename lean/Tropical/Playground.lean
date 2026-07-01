import Tropical.EmitArrow
import Tropical.Ir.Strata
import Tropical.Ir.Core
import Tropical.Compile
import Tropical.Parse.Raise
import Tropical.Ir.Elaborator
import Lean.Data.Json

/-!
# `load_patch_graph` — the playground's live arrow entry point (EXPERIMENT)

A downstream-only patch graph (the PureData-style playground GUI) decoded into the
`EmitArrow` `PatchGraph`, lowered through the SAME `lowerGraph → normalize → emitTerm`
the corpus gates exercise, wrapped as a session root, and compiled to a `FlatPlan`
via `compileSession` — the production loadable tail. The slide (`normalize`) pushes
each effect's warps up onto the generators' clocks, so a `flange`/`fm` dropped
downstream of an oscillator genuinely re-clocks that oscillator.

Knobs are baked literals (EmitArrow emits no `paramRef`), so a knob change re-sends
the whole graph and hot-swaps — clickless, since there is no per-sample state.

Uncommitted: this file makes `EmitArrow` reachable from the live `frontend`.
-/

namespace Tropical.Playground

open Lean (Json JsonNumber)
open Tropical.Ir
open Tropical.EmitArrow

-- ── JSON field helpers (over `Lean.Json` objects) ───────────────────────────
private def jNum? (obj : Json) (key : String) : Option JsonNumber :=
  match obj.getObjVal? key with
  | .ok j => j.getNum?.toOption
  | .error _ => none

/-- A numeric param as a scalar `Expr` (`Expr.num`), carrying the JSON decimal
    straight through (mantissa · 10^-exponent). -/
private def jExpr (obj : Json) (key : String) (dflt : Expr) : Expr :=
  match jNum? obj key with
  | some n => lit n.mantissa n.exponent
  | none => dflt

/-- A numeric param truncated to `Int` (the `fm` depth, in samples). -/
private def jInt (obj : Json) (key : String) (dflt : Int) : Int :=
  match jNum? obj key with
  | some n => if n.exponent == 0 then n.mantissa else n.mantissa / ((10 : Int) ^ n.exponent)
  | none => dflt

private def jStr (obj : Json) (key : String) (dflt : String) : String :=
  match obj.getObjVal? key with
  | .ok (.str s) => s
  | _ => dflt

-- ── Voices (literal pitch, so the knob bakes into the emitted clock) ─────────
/-- `FixedSinOsc`: freq (port 0), clk (port 1), phase (port 2). `phaseE` drives the
    phasor's continuity-correction offset — supplied (as a live slot) for a source
    whose `freq` is phase-anchored, omitted (defaults to 0) otherwise. -/
private def sineVoiceE (pitchE : Expr) (phaseE : Option Expr := none) : Voice :=
  { programName := "FixedSinOsc",
    wire := fun clkE => match phaseE with
      | some ph => #[ ⟨⟨0⟩, pitchE⟩, ⟨⟨1⟩, clkE⟩, ⟨⟨2⟩, ph⟩ ]
      | none    => #[ ⟨⟨0⟩, pitchE⟩, ⟨⟨1⟩, clkE⟩ ] }

/-- `MorphOsc`: freq (port 0), morph (port 1), clk (port 2), phase (port 3).
    `morph = 0` is saw, `morph = 1` is sine; `phaseE` drives the phase-anchor
    offset (a live slot when the source's freq is anchored). -/
private def morphVoiceE (pitchE morphE : Expr) (phaseE : Option Expr := none) : Voice :=
  { programName := "MorphOsc",
    wire := fun clkE => match phaseE with
      | some ph => #[ ⟨⟨0⟩, pitchE⟩, ⟨⟨1⟩, morphE⟩, ⟨⟨2⟩, clkE⟩, ⟨⟨3⟩, ph⟩ ]
      | none    => #[ ⟨⟨0⟩, pitchE⟩, ⟨⟨1⟩, morphE⟩, ⟨⟨2⟩, clkE⟩ ] }

/-- Build a voice at an explicit pitch/morph expression. `pitchE` is either the
    baked `freq` knob or a live slot; `phaseE` is the sine voice's phase-anchor
    offset (a live slot when the source's freq is anchored, else `lit 0`). -/
private def voiceOf (sel : Json) (pitchE morphE phaseE : Expr) : Voice :=
  match jStr sel "voice" "sine" with
  | "saw"   => morphVoiceE pitchE (lit 0) (some phaseE)
  | "morph" => morphVoiceE pitchE morphE (some phaseE)
  | _       => sineVoiceE pitchE (some phaseE)

/-- `δ = toInt(seconds · sampleRate · 2³²)` — a Q32.32 sample offset (the stdlib
    flanger's own delay form), but with the `seconds` taken from the knob. -/
private def deltaOf (secondsE : Expr) : Expr :=
  toIntE (mul (mul secondsE .sampleRate) (lit 4294967296))

-- ── Node decode (named inlets: `in` is an object {port: [srcId,…]}) ──────────
private def portSources (inObj : Json) (port : String) : Array String :=
  match (inObj.getObjVal? port).toOption.bind (·.getArr?.toOption) with
  | some arr => arr.filterMap (·.getStr?.toOption)
  | none => #[]

/-- Resolve a live param name to a `paramRef` slot read, falling back to `dflt`
    (used only if the param table somehow lacks the entry — the collector always
    allocates one). Every continuous knob is a live slot, so its value is READ from
    the slot at runtime, never baked — turning it drives `set_param`, no relower. -/
private def pref (pidx : String → Option Nat) (name : String) (dflt : Expr) : Expr :=
  match pidx name with
  | some i => .paramRef ⟨i⟩
  | none => dflt

/-- Which (kind, knob) pairs are driven by a CLOSED-FORM GLIDE (3 slots + a ramp
    expr) instead of a raw slot. PROTOTYPE: just the flanger's `depth`, so a turn
    of it eases as a click-free smoothstep; every other knob stays a raw slot (the
    A/B). -/
private def isGlided (kind kname : String) : Bool := kind == "flange" && kname == "depth"

/-- A closed-form smoothstep GLIDE of τ from three slots: `v0 + (v1−v0)·s²(3−2s)`,
    `s = clamp((τ − t0)/dur, 0, 1)`, `dur = 0.02·sampleRate` samples (20 ms at any
    rate, matching the engine's `set_param_glide`). The value eases from `v0` to
    `v1` starting at tick `t0`; the control plane re-anchors the slots on each turn.
    Stateless — the ramp is a pure function of the ambient clock, not an accumulator. -/
private def glideExpr (pidx : String → Option Nat) (base : String) (dflt : Expr) : Expr :=
  let v0 := pref pidx s!"{base}#v0" dflt
  let v1 := pref pidx s!"{base}#v1" dflt
  let t0 := pref pidx s!"{base}#t0" (lit 0)
  let dur := mul (lit 2 2) .sampleRate   -- 0.02·SR = 20 ms
  let s  := clampE (div (sub (toFloatE .sampleIndex) t0) dur) (lit 0) (lit 1)
  let ss := mul (mul s s) (sub (lit 3) (mul (lit 2) s))
  add v0 (mul (sub v1 v0) ss)

/-- Build a node, plus any synthesized helper nodes (a default LFO source for a
    swept flange with no modulator patched). `pidx` maps a live param name
    `<nodeId>.<knob>` to its `ParamIdx`. Every continuous knob reads its slot via
    `paramRef`; only structural selectors (`voice`, warp `mode`) and topology are
    baked, so only they trigger a relower. -/
private def buildNode (pidx : String → Option Nat) (id kind : String)
    (sel params inObj : Json) : Node × Array PatchNode :=
  let sig := (portSources inObj "in")[0]?.getD "__silence__"
  -- this node's own knob `kname` as a live value: a closed-form glide if glided,
  -- else a raw slot read; falling back to the baked literal if unallocated.
  let p := fun (kname : String) (dflt : Expr) =>
    if isGlided kind kname then glideExpr pidx s!"{id}.{kname}" dflt
    else pref pidx s!"{id}.{kname}" dflt
  match kind with
  | "knob" =>
    match pidx s!"{id}.value" with
    | some i => (.knob i, #[])
    | none => (.mix #[], #[])   -- a knob missing from the table (unreachable): silence
  | "source" =>
    -- a Knob wired into `freq` shadows the baked freq slot: read the WIRED knob's
    -- `<id>.value` slot instead. (Only a knob may wire here — audio-rate into the
    -- pitch PORT is not integration; audio-rate modulation is the FM/sflange path.)
    let pitchE := match (portSources inObj "freq")[0]? with
      | some w => pref pidx s!"{w}.value" (jExpr params "freq" (lit 220))
      | none => p "freq" (jExpr params "freq" (lit 220))
    -- the phase-anchor offset slot (present only for a sine source's own freq);
    -- falls back to lit 0 (no offset) for saw/morph or a knob-driven freq.
    let phaseE := pref pidx s!"{id}.freq#phase" (lit 0)
    (.source (voiceOf sel pitchE (p "morph" (jExpr params "morph" (lit 0))) phaseE) clockLit, #[])
  | "flange" =>
    let d := deltaOf (p "depth" (jExpr params "depth" (lit 7 4)))
    (.flange sig (fun c => sub c d) (fun c => add c d), #[])
  | "warp" =>
    if jStr sel "mode" "delay" == "reverse" then
      (.warpFx sig (fun c => neg c), #[])
    else
      let d := deltaOf (p "amount" (jExpr params "amount" (lit 4 3)))
      (.warpFx sig (fun c => sub c d), #[])
  | "fm" =>
    (.fm sig (sineVoiceE (p "carrier" (jExpr params "carrier" (lit 330))))
      clockLit (p "depth" (jExpr params "depth" (lit 8))), #[])
  | "sflange" =>
    let depthSec := p "depth" (jExpr params "depth" (lit 2 3))
    match (portSources inObj "mod")[0]? with
    | some modId => (.sflange sig modId depthSec, #[])
    | none =>
      -- no modulator patched: synthesize a built-in LFO sine at the `rate` knob.
      let lfoId := s!"__lfo_{id}"
      (.sflange sig lfoId depthSec,
       #[ { id := lfoId, node := .source (sineVoiceE (p "rate" (jExpr params "rate" (lit 3 1)))) clockLit } ])
  | "mix" => (.mix (portSources inObj "in"), #[])
  | _ => (.mix (portSources inObj "in"), #[])

private structure Raw where
  id : String
  kind : String
  sel : Json
  params : Json
  inObj : Json

private def decodeRaw (nj : Json) : Option Raw :=
  match nj.getObjVal? "id", nj.getObjVal? "kind" with
  | .ok (.str id), .ok (.str kind) =>
    let sel := (nj.getObjVal? "sel").toOption.getD (Json.mkObj [])
    let params := (nj.getObjVal? "params").toOption.getD (Json.mkObj [])
    let inObj := (nj.getObjVal? "in").toOption.getD (Json.mkObj [])
    some { id, kind, sel, params, inObj }
  | _, _ => none

private def rawsOf (j : Json) : Array Raw :=
  match (j.getObjVal? "nodes").toOption.bind (·.getArr?.toOption) with
  | some arr => arr.filterMap decodeRaw
  | none => #[]

/-- The continuous knobs each node kind carries — the ones that become live param
    slots (`<nodeId>.<knob>`). Structural selectors (`voice`, warp `mode`) are NOT
    here: changing one alters the graph topology, so it relowers. -/
private def knobNamesOf : String → Array String
  | "source"  => #["freq", "morph"]
  | "flange"  => #["depth"]
  | "sflange" => #["depth", "rate"]
  | "fm"      => #["carrier", "depth"]
  | "warp"    => #["amount"]
  | "knob"    => #["value"]
  | _         => #[]

/-- The live param table: every node's continuous knobs as `(<id>.<knob>, default)`
    in scan order (node order, then knob order). The position IS the `ParamIdx` the
    node's `paramRef`s carry; `compileSession` allocates each `param:<id>.<knob>`. A
    knob shadowed by a wired control inlet of the same name (a Knob patched into
    `freq`) is skipped — the cord's own slot drives it. -/
private def collectParams (raws : Array Raw) : Array (String × JsonNumber) := Id.run do
  let mut out : Array (String × JsonNumber) := #[]
  for r in raws do
    if r.kind == "out" then continue
    for kname in knobNamesOf r.kind do
      if (portSources r.inObj kname).isEmpty then
        let base := s!"{r.id}.{kname}"
        let dflt := (jNum? r.params kname).getD ⟨0, 0⟩
        if isGlided r.kind kname then
          -- three anchor slots for the closed-form ramp; v0=v1=current value and
          -- t0=0 means it starts flat at the knob's value (no ramp until re-anchored).
          out := out.push (s!"{base}#v0", dflt)
          out := out.push (s!"{base}#v1", dflt)
          out := out.push (s!"{base}#t0", ⟨0, 0⟩)
        else
          out := out.push (base, dflt)
          -- every source's own freq carries a phase-anchor offset slot (sine, saw,
          -- morph — all phasor-based), so a live freq change bumps `#phase` and
          -- keeps the phase continuous (click-free).
          if r.kind == "source" && kname == "freq" then
            out := out.push (s!"{base}#phase", ⟨0, 0⟩)
  return out

/-- Decode the GUI graph, returning the patch plus the knob param table. The
    reserved `out` node is not an arrow node; it becomes a synthetic `mix` of its
    inputs (the dac mix-bus). Effects with no input point at `__silence__` (an
    empty sum), so a half-built patch compiles to silence with no error. -/
def decodeGraph (j : Json) : Except String (PatchGraph × Array (String × JsonNumber)) := do
  let outId := match (j.getObjVal? "out").toOption with
    | some (.str s) => s
    | _ => ""
  let raws := rawsOf j
  let params := collectParams raws
  let pidx : String → Option Nat := fun nm => params.findIdx? (·.1 == nm)
  let mut pnodes : Array PatchNode := #[]
  for r in raws do
    if r.kind == "out" then continue
    let (node, extras) := buildNode pidx r.id r.kind r.sel r.params r.inObj
    pnodes := pnodes.push { id := r.id, node }
    for e in extras do pnodes := pnodes.push e
  let outIns := match raws.find? (·.id == outId) with
    | some r => if r.kind == "out" then portSources r.inObj "in" else #[r.id]
    | none => #[]
  pnodes := pnodes.push { id := "__out__", node := .mix outIns }
  pnodes := pnodes.push { id := "__silence__", node := .mix #[] }
  pure ({ nodes := pnodes, output := "__out__" }, params)

-- ── Graph → loadable FlatPlan (mirrors Tropicaltest.compileArrowCarrier) ─────
def compilePlanPure (arena : Arena) (resolved : Array (String × ProgramIdx)) (j : Json) :
    Except String Tropical.Plan.FlatPlan := do
  let (g, paramTable) ← decodeGraph j
  let term ← lowerGraph g
  let (out, b) := emitTerm (normalize term) {}
  let registry ← buildRegistry arena resolved #["FixedSinOsc", "MorphOsc"]
  -- One `.param` decl per live knob, appended AFTER the instance decls: declaration
  -- order = `ParamIdx` = the index each `paramRef` carries, and keeping params after
  -- instances leaves the emitted voices' `InstanceIdx` untouched. `compileSession`
  -- allocates each a `param:<id>.<knob>` module slot, driven live by `set_param`.
  let paramDecls : Array BodyDecl := paramTable.map (fun (nm, v) => .param nm (some v))
  let prog : Program := {
    name := "__patch__"
    inputs := #[]
    outputs := #[{ name := "out", type? := some (.scalar .float) }]
    decls := b.decls ++ paramDecls
    assigns := #[{ target := .port ⟨0⟩, expr := out }]
    binderCount := 0
    registry }
  let idx : ProgramIdx := ⟨arena.programs.size⟩
  let arena1 : Arena := { arena with programs := arena.programs.push prog }
  let (arena2, root') ← (Tropical.Ir.Strata.run { upto := 5 } arena1 idx).mapError (·.message)
  let core ← Tropical.Ir.Core.check arena2 root'
  let input : Tropical.Compile.SessionInput := {
    instances := #[(Tropical.Compile.rootInstancePath, core)]
    wiresPost := #[]
    graphOutputs := #[(Tropical.Compile.rootInstancePath, "out")]
    params := paramTable.map (fun (nm, v) => (nm, Json.num v))
    alloc := Tropical.Lowering.allocate (paramTable.map (·.1)) #[]
    root := core
    mode := .fused }
  Tropical.Compile.compileSession input

-- ── Stdlib-into-arena (cached; mirrors Tropicaltest.arrowElabStdlib) ─────────
def elabStdlib : IO (Except String (Arena × Array (String × ProgramIdx))) := do
  let manifestText ← IO.FS.readFile "stdlib/parsed/manifest.json"
  let names : Except String (Array String) := do
    let jv ← Tropical.Parse.JsonV.parse manifestText |>.mapError (s!"manifest parse error: {·}")
    let some (Tropical.Parse.JsonV.arr items) := jv.getField? "programs"
      | .error "manifest missing 'programs' array"
    items.mapM fun | .str s => .ok s | _ => .error "manifest 'programs' entries must be strings"
  match names with
  | .error e => pure (.error e)
  | .ok names => do
    let mut arena : Arena := {}
    let mut resolved : Array (String × ProgramIdx) := #[]
    for name in names do
      let text ← IO.FS.readFile s!"stdlib/parsed/{name}.json"
      match Tropical.Parse.JsonV.parse text with
      | .error e => return .error s!"{name}.json: JSON parse error: {e}"
      | .ok jv =>
        match Tropical.Parse.decodeProgram jv with
        | .error e => return .error s!"{name}.json: {e}"
        | .ok prog =>
          let r := resolved
          match Tropical.Ir.elaborateInto arena prog (some fun n => (r.find? (·.1 == n)).map (·.2)) with
          | .error e => return .error s!"{name}: {e.message}"
          | .ok (arena', idx) => arena := arena'; resolved := resolved.push (name, idx)
    pure (.ok (arena, resolved))

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

/-- Decode + lower + compile the GUI graph to a loadable `FlatPlan`. -/
def compilePlan (j : Json) : IO (Except String Tropical.Plan.FlatPlan) := do
  match ← getStdlib with
  | .error e => pure (.error s!"stdlib elaboration: {e}")
  | .ok (arena, resolved) => pure (compilePlanPure arena resolved j)

end Tropical.Playground
