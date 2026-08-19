import Tropical.EmitArrow.Modal
import Tropical.Ir.TileTime
import Tropical.EmitArrow.Modal.OrientedRealize
import Tropical.Ir.Cycles

/-!
# EmitArrow.Patch — the patcher lowering: a downstream-only graph → arrow term

A patch is a downstream-only DAG of modules (the "you may only patch
forward" UX rule — the acyclicity invariant made visible). Lowering:
a wire is `⋙`, fan-out is the shared upstream term (Δ), fan-in is the sum,
a generator is `gen`, an effect contributes `warp` nodes — and `normalize`
(the slide) then pushes those warps up to the generators. The modal island
composes by pole algebra (`lowerModal`) and realizes to a `Sig` only at its
boundary with the signal mainland (`lowerInput`).
-/

namespace Tropical.EmitArrow

open Tropical.Ir

/-- An ABSOLUTE address warp: the modulator signal `s` (read as SECONDS) BECOMES
    the clock, the base discarded — `c ↦ toInt(s·SR·2³²)`, so the modal bank's
    `dSec = clk/2³²/SR − anchor` reduces to `s − anchor`. Unlike the RELATIVE
    flange/fm warps (`c ± δ·m`, a bounded perturbation of the ambient clock), this
    RELOCATES the causal gate to `s`'s crossing of the anchor: a patched CF signal
    triggers and scrubs the resonance by its own zero-crossing, and its read rate
    `ds/dτ` pitch-bends the ring. The master scrub still reaches it — `s` is itself
    read through the enclosing clock transform (`emitTermC`'s `swarp` case). -/
def modalAddrWarp : Clock → Sig → BuildM Clock :=
  fun _ s => do
    let sr ← sampleRate
    let twoPow32 ← lit 4294967296
    let scale ← mul sr twoPow32
    let fixed ← mul s scale
    toIntE fixed

-- ─────────────────────────────────────────────────────────────
-- M9 — the PATCHER LOWERING: a downstream-only patch graph → arrow term
-- ─────────────────────────────────────────────────────────────

/-! The MVP front end. A patch is a downstream-only DAG of modules (the "you may
    only patch forward" UX rule — which is just the acyclicity invariant the
    whole language rests on, made visible). Lowering reads the doc's slogan
    literally:

      * a WIRE `A.out → B.in`  ⟶  B's effect morphism APPLIED to A's term  (⋙)
      * a FAN-OUT (one output, many inputs)  ⟶  the shared upstream term  (Δ / &&&)
      * a FAN-IN (a mixer)  ⟶  the sum  (the product collapse)
      * a GENERATOR  ⟶  `gen`;  an EFFECT  ⟶  a `ArrowTerm → ArrowTerm`

    An EFFECT contributes `warp` nodes (a flanger fans its input into warped
    taps; a delay/reverse is a bare warp), so it is authored and wired DOWNSTREAM
    — and then `normalize` (the slide) pushes those warps UP to the generators.
    That is the whole trick: the user patches forward, the compiler reads
    backward in time. Because tropical has no state primitive, the slide is
    always total — every module is stateless, so the warp always reaches the
    generators. `lowerGraph` produces the UNREDUCED (downstream) term; the slide
    is the separate pass that follows. -/

/-- A patch-graph node. Generators carry their clock; effects carry the id of the
    upstream node they consume (`⋙`); the mixer carries the ids it sums (fan-in).
    Generators read the master clock; the flanger/`warp`/shaper are the effect
    morphisms. -/
inductive Node where
  | source (v : Voice) (clk : Clock)
  | flange (input : String) (back fwd : Clock → BuildM Clock)
  | shaper (input : String) (f : Sig → BuildM Sig)
  | warpFx (input : String) (φ : Clock → BuildM Clock)
  | mix (inputs : Array String)
  /-- A MODULATED carrier: `input`'s signal modulates this carrier's clock
      (FM/PM). The signal-into-clock edge — patch `mod.out → carrier.fm`. -/
  | fm (input : String) (carrier : Voice) (baseClk : Clock) (depthE : Sig)
  /-- A PHASE-modulated carrier (through-zero PM): `input`'s signal, scaled by
      `depthE` (the modulation index, in cycles), adds to this carrier's phase
      port — NOT its clock. The direct `osc → osc` patch. Sidebands without pitch
      droop (the carrier clock is untouched). -/
  | pm (input : String) (carrier : Voice) (baseClk : Clock) (depthE : Sig)
  /-- A SWEPT flanger: `input` is the signal flanged, `modInput` the modulator that
      sweeps the ±δ taps (an LFO, or any patched signal). `depthSec` is the sweep
      depth in seconds. The signal-warp distributes onto `input`'s generators. -/
  | sflange (input modInput : String) (depthSec : Sig)
  /-- A KNOB: a program that is nothing but a param with one output. `idx` is the
      `ParamIdx` of the root's param slot; it lowers to a τ-constant `paramRef`
      leaf, so wiring it into a modulator/pitch position binds that parameter to a
      live module slot (`param:<name>`), driven by `set_param` without a relower. -/
  | knob (idx : Nat)
  /-- A one-sided resonant COMB: `input` read at a bank of clock-warped offsets,
      each weighted — `Σ wₖ·(warpₖ ⋙ input)`. `taps` is `(weight, clockShift)` per
      tap: tap 0 is usually the dry `(1, id)`, the rest a decaying series `(gᵏ,
      c ↦ c + k·D)`. One-sided (all shifts the same direction) makes it
      time-ASYMMETRIC — its impulse tail rings on one side, so under a global clock
      reverse the tail flips (echo ↔ pre-echo). A future shift (`c + kD`, `D > 0`)
      reads AHEAD — an audible pre-echo, impossible on a stream. The slide
      distributes each tap's warp onto the generators, exactly like the flanger. -/
  | comb (input : String) (taps : Array (Sig × (Clock → BuildM Clock)))
  /-- The multiplicative fan-in — `⊗` over its inputs, the ring-product twin of
      `mix`'s `sum`. Two inputs is ring modulation; an input × an envelope-generator
      is a VCA; and because the slide distributes over `prod`, a downstream warp
      reclocks every factor, so each factor's own generators stay in step. Empty ⇒
      silence (a graceful half-built patch), like `mix`. -/
  | ring (inputs : Array String)
  /-- MODAL-island nodes. They carry pole banks (`Array ModalMode`), not pointwise
      signals, and compose by the residue calculus at BUILD time. They consume only
      modal (a Sig can't be un-realized into poles) and REALIZE to a `Sig` at their
      boundary with the Sig mainland (a Sig consumer or the tap). `modalSource`
      carries the master clock it will realize against; `modalReverb` composes its
      modal input with a room (`voice ⋙ reverb`); `modalMix` is the pole union.
      `modalSource.addr` is an OPTIONAL Sig node id: when present, that signal
      becomes the bank's absolute time-address (`modalAddrWarp`) — the temporal
      inlet (a clock driven by a signal is a warp), distinct from the spectral
      (modal) inlets `modalReverb`/`modalMix` consume. It propagates through
      composition to the realization boundary. -/
  -- `modalSource.count` is the optional LIVE effective mode count
  -- (trip-count-as-data): a `Sig` (a param-slot read) that becomes the bank's
  -- dynamic trip count, clamped to `modes.size` (the capacity). Applies only
  -- to a DIRECTLY realized bank — residue composition and pole union break
  -- the mode-prefix alignment, so the count is dropped at those seams.
  -- `modalSource.bloom` is the optional PITCH-BLOOM `(B, g)` (baked Floats,
  -- `B = β·scale/g`): a bloomed struck resonator (a gong register). It keeps
  -- the source MODAL until the realization boundary — a downstream reverb folds
  -- into the room chain and the bloom is crossed ONCE at the tap (bloomCompose),
  -- rather than realizing at the warp up front. `none` = an ordinary bank.
  | modalSource (modes : Array ModalMode) (anchor : Sig) (clk : Clock)
      (addr : Option String) (count : Option Sig := none)
      (bloom : Option (Float × Float) := none)
  /-- A fixed modal room used by internal fixtures.  Its optional
      direction is still room-local; this constructor remains convenient for
      structural banks that do not own a wireable RT60 control. -/
  | modalReverb (input : String) (room : Array ModalMode) (dir : Option ModalDir)
  /-- The public ordinary room. Its topology is a function of the frozen RT60
      value and its local direction remains a live terminal control. Decay
      sway remains a low-level modal operation. -/
  | modalRoom (input : String) (build : Sig → BuildM (Array ModalMode))
      (rt60 direction : ModalControlRef)
  /-- One generic linear modal-kernel action.  The stage builds a retained
      identity/proper/parallel/cascade expression only after its controls are
      frozen at the terminal response coordinate. -/
  | modalLinear (input : String) (stage : ModalLinearStage)
  /-- A dry/wet value junction.  When `wet` is structurally a feed-forward
      linear chain from `dry`, lowering factors the shared prefix and retains
      one kernel blend.  Other legal inputs remain an ordered two-branch sum. -/
  | modalBlend (dry wet : String) (mix : ModalControlRef)
  | modalMix (inputs : Array String)
  -- `modalGauge` is the excitation-gauge adapter (§5): it re-levels its modal input's
  -- residues by the self-measured `‖H‖^{−g}` (`normalizePeak`), a pure Modal ⇝ Modal
  -- effect. `g` is a live slot. The norm is measured on the complete current modal
  -- universe at this exact authored stage; it is not independently settled or
  -- distributed over branches/support arms.
  | modalGauge (input : String) (g : Sig)
  | modalGaugeControl (input : String) (g : ModalControlRef)

structure PatchNode where
  id : String
  node : Node

/-- A downstream-only patch DAG: named nodes plus the id wired to the output. -/
structure PatchGraph where
  nodes : Array PatchNode
  output : String

/-- The standard FM modulation law: a modulator signal `m` shifts the carrier's
    Q32.32 clock by `depth · m` samples — `clk − toInt(depth · m · 2³²)`. `depth`
    is an EXPRESSION (a literal, or a live `paramRef` slot), so the depth knob can
    be a live param. Closed form in τ (`m` is a closed-form signal) either way. -/
def fmWarp (depthE : Sig) : Clock → Sig → BuildM Clock :=
  fun base m => do
    let modulation ← mul depthE m
    let twoPow32 ← lit 4294967296
    let fixed ← mul modulation twoPow32
    let shift ← toIntE fixed
    sub base shift

/-- The proper tail half of one swept first-order all-pass section.  This is a
    generic linear modal stage: the all-pass direct path is supplied by an
    ordinary topology junction, not encoded in this builder. -/
def modalAllpassTailStage (center sweep rate : ModalControlRef)
    (ratio : Float) : ModalLinearStage where
  controls := #[center, sweep, rate]
  build := fun responseClock values => do
    let centerValue ← match values[0]? with | some value => pure value | none => lit 700
    let centerLo ← lit 40
    let centerHi ← lit 4000
    let center ← clampE centerValue centerLo centerHi
    let sweepValue ← match values[1]? with | some value => pure value | none => lit 15 1
    let zero ← lit 0
    let three ← lit 3
    let sweep ← clampE sweepValue zero three
    let rateValue ← match values[2]? with | some value => pure value | none => lit 2 1
    let rateLo ← lit 2 2
    let rateHi ← lit 8
    let rate ← clampE rateValue rateLo rateHi
    let phase ← phasorPhaseSig rate zero responseClock
    let twoPi ← twoPiE
    let radians ← mul twoPi phase
    let sine ← sinSig radians
    let octaveOffset ← mul sweep sine
    let ln2 ← lit 6931471805599453 16
    let exponent ← mul octaveOffset ln2
    let frequencyScale ← expSig exponent
    let ratioSig ← litF ratio
    let centerFrequency ← mul center ratioSig
    let frequency ← mul centerFrequency frequencyScale
    let a ← mul twoPi frequency
    let sigmaLo : Float := 6.283185307179586 * 40.0 * ratio * 0.125
    let sigmaHi : Float := 6.283185307179586 * 4000.0 * ratio * 8.0
    let tail ← Oriented.allpassTail a (some (sigmaLo, sigmaHi))
    pure (.proper (.causalTail tail))

/-- Expand a convenience phaser module into ordinary modal patch topology:
    six `identity + tail` junctions in cascade, followed by a dry/wet blend.
    The returned node is the named public module; the array contains its
    compiler-visible internal nodes. -/
def modalPhaserTopology (id input : String) (center sweep rate mix : ModalControlRef)
    (ratios : Array Float) : Node × Array PatchNode := Id.run do
  let mut nodes := #[]
  let mut current := input
  for (ratio, index) in ratios.zipIdx do
    let tailId := s!"__modal_phaser_{id}_tail_{index}"
    let sectionId := s!"__modal_phaser_{id}_section_{index}"
    nodes := nodes.push {
      id := tailId
      node := .modalLinear current (modalAllpassTailStage center sweep rate ratio) }
    nodes := nodes.push {
      id := sectionId
      node := .modalMix #[current, tailId] }
    current := sectionId
  return (.modalBlend input current mix, nodes)

/-- Every input id a node's lowering recurses into — the wires of the
    graph, `modalSource`'s optional address inlet included. These are the
    edges `lowerAt` ranks (and refuses cycles over). -/
def Node.inputIds : Node → Array String
  | .source .. | .knob _ => #[]
  | .flange i _ _ | .shaper i _ | .warpFx i _ | .comb i _ => #[i]
  | .mix is | .ring is | .modalMix is => is
  | .fm i .. | .pm i .. => #[i]
  | .sflange i m _ => #[i, m]
  | .modalSource _ _ _ addr _ _ => (addr.map (#[·])).getD #[]
  | .modalReverb i _ _ | .modalGauge i _ => #[i]
  | .modalLinear i stage =>
      #[i] ++ stage.controls.filterMap (fun control => control.signalNode?)
  | .modalBlend dry wet mixControl =>
      #[dry, wet] ++ (mixControl.signalNode?.map (#[·])).getD #[]
  | .modalGaugeControl i g => #[i] ++ (g.signalNode?.map (#[·])).getD #[]
  | .modalRoom i _ rt60 direction =>
      #[i] ++ (#[rt60, direction].filterMap (fun control => control.signalNode?))

/-- Is `id` a modal-island node? Its output wire carries poles, not a `Sig`. A
    missing node reads as Sig (graceful — a half-built patch stays lowerable). -/
def nodeIsModal (g : PatchGraph) (id : String) : Bool :=
  match g.nodes.find? (·.id == id) with
  | none => false
  | some pn => match pn.node with
    | .modalSource .. => true
    | .modalReverb .. => true
    | .modalRoom .. => true
    | .modalLinear .. => true
    | .modalBlend .. => true
    | .modalMix .. => true
    | .modalGauge .. => true
    | .modalGaugeControl .. => true
    | _ => false

/-- Recognize the literal topology `base + linear(base)` in authored order.
    This is an exact factoring rule over shared node identity, not an effect-name
    heuristic. -/
private def directLinearFactor? (g : PatchGraph) (inputs : Array String) :
    Option (String × ModalLinearStage) := do
  let [base, transformed] := inputs.toList | none
  let node ← g.nodes.find? (·.id == transformed)
  let .modalLinear input stage := node.node | none
  if input == base then some (base, stage.withDirect) else none

/-- Recover a feed-forward linear stage chain from `base` to `current`.  A
    factored direct-plus-tail junction counts as one stage.  Fuel is bounded by
    the patch size; the total graph entry independently rejects cycles. -/
private def linearChainFrom? (g : PatchGraph) (base current : String) :
    Option (Array ModalLinearStage) :=
  let rec go : Nat → String → Option (Array ModalLinearStage)
    | 0, _ => none
    | fuel + 1, id =>
      if id == base then some #[] else do
      let node ← g.nodes.find? (·.id == id)
      match node.node with
      | .modalLinear input stage =>
        return (← go fuel input).push stage
      | .modalMix inputs =>
        let (input, stage) ← directLinearFactor? g inputs
        return (← go fuel input).push stage
      | _ => none
  go (g.nodes.size + 1) current

/-- The empty modal value (`__silence__`, the graceful-silence contract). -/
private def silentModal : BuildM ModalBranch := do
  let strikeAnchor ← lit 0
  let realizationClock ← clockLit
  pure {
    source := .plain #[]
    strikeAnchor
    realizationClock
    addressNode? := none
    modeCount? := none }

/-- Silence remains a singleton so terminals that historically accepted an
    unwired modal inlet retain their exact graceful-silence path. -/
private def silentForest : BuildM ModalForest := do
  pure #[← silentModal]

/-- The rank-certificate breach message. Dead by invariant — `lowerAt`
    refuses cyclic graphs (with the full loop named) before any lowering
    runs — but a direct caller handing in an unranked recursion still
    gets the honest refusal, never an overflow. -/
private def cycleGuardMsg (src dst : String) : String :=
  s!"connection cycle: {src} → {dst} — patch graphs must be acyclic (you may only patch forward; there is no delay to break a loop through)"

/-- Lower a modal-island subgraph to its `ModalBank` + strike anchor + the master
    clock it realizes against. Pure pole algebra at BUILD time: a reverb is the
    residue calculus (`residueComposeEC`) on a plain source, or a room-chain FOLD
    on a bloomed one; a mix is the pole union. Never touches `Sig` — a modal
    node's inputs are modal by construction, so it only recurses into
    `lowerModal`. Total: recursion descends the caller-supplied topological
    rank (`rankOf`, built by `lowerAt`); `r` is this node's own rank. -/
def lowerModal (g : PatchGraph) (rankOf : String → Option Nat) (id : String) (r : Nat) :
    BuildM ModalForest := do
  -- An unconnected modal inlet points at `__silence__`; a modal node with no modal
  -- source is a legal, incomplete patch — it compiles to an EMPTY bank (silence),
  -- not an error. (The graceful-silence contract, same as `mix []`.)
  if id == "__silence__" then
    return ← silentForest
  let some pn := g.nodes.find? (·.id == id)
    | throw s!"lowerModal: node '{id}' not found"
  -- The gated modal recursion (inlined at each site for the WF measure):
  -- `__silence__` short-circuits, a dangling id keeps today's message, and
  -- a rank breach is the dead-by-invariant refusal.
  match pn.node with
  | .modalSource ms a clk addr count bloom? =>
    -- unit modes enter the island CLAMPED to their declared σ intervals
    -- (`clampSigmas` — the uniform-bank clamp; every downstream fold,
    -- partition, gauge, and mix inherits it, so the whole bank saturates
    -- together under knob overdrive). In-span drives are value-identical.
    let ms ← clampSigmas ms
    match bloom? with
    | none => pure #[{
        source := .plain ms
        strikeAnchor := a
        realizationClock := clk
        addressNode? := addr
        modeCount? := count }]
    -- a bloomed source starts modal with an EMPTY room; reverbs fold in downstream
    | some (B, gr) => pure #[{
        source := .bloomed ms B gr
        strikeAnchor := a
        realizationClock := clk
        addressNode? := addr
        modeCount? := count }]
  | .modalReverb inId room dir => do
    let lowered ←
      if inId == "__silence__" then silentForest
      else match rankOf inId with
        | none => throw s!"lowerModal: node '{inId}' not found"
        | some ri =>
          if _h : ri < r then lowerModal g rankOf inId ri
          else throw (cycleGuardMsg id inId)
    -- The live count does NOT survive composition (the composed modes are no
    -- longer a prefix of the source's) — realize at full capacity (graceful-
    -- silent through a reverb; v1, no warning, legal state).
    -- Room unit modes enter clamped (the uniform-bank σ clamp — see
    -- `.modalSource`); in-span drives are value-identical.
    let room ← clampSigmas room
    let directionSig ← match dir with
      | some d => pure d.dir
      | none => lit 0
    let direction := ModalControlRef.constant directionSig
    let sway? := dir.bind (fun d => d.damp) |>.map fun (sway, rate) =>
      (ModalControlRef.constant sway, ModalControlRef.constant rate)
    let stage : ModalStage := .ordinaryRoom {
      kernel := .fixed room
      direction
      sway? }
    return lowered.map fun branch =>
      { branch with stages := branch.stages.push stage, modeCount? := none }
  | .modalRoom inId build rt60 direction => do
    let lowered ←
      if inId == "__silence__" then silentForest
      else match rankOf inId with
        | none => throw s!"lowerModal: node '{inId}' not found"
        | some ri =>
          if _h : ri < r then lowerModal g rankOf inId ri
          else throw (cycleGuardMsg id inId)
    let stage : ModalStage := .ordinaryRoom {
      kernel := .controlled rt60 (fun value => do
        let modes ← build value
        clampSigmas modes)
      direction
      sway? := none
      normalizeDirectionLevel := true }
    return lowered.map fun branch =>
      { branch with stages := branch.stages.push stage, modeCount? := none }
  | .modalLinear inId linear => do
    let lowered ←
      if inId == "__silence__" then silentForest
      else match rankOf inId with
        | none => throw s!"lowerModal: node '{inId}' not found"
        | some ri =>
          if _h : ri < r then lowerModal g rankOf inId ri
          else throw (cycleGuardMsg id inId)
    return lowered.map fun branch =>
      { branch with stages := branch.stages.push (.linear linear), modeCount? := none }
  | .modalBlend dryId wetId mix => do
    match linearChainFrom? g dryId wetId with
    | some stages =>
      let dry ←
        if dryId == "__silence__" then silentForest
        else match rankOf dryId with
          | none => throw s!"lowerModal: node '{dryId}' not found"
          | some ri =>
            if _h : ri < r then lowerModal g rankOf dryId ri
            else throw (cycleGuardMsg id dryId)
      let stage : ModalStage := .linear (ModalLinearStage.dryWet stages mix)
      return dry.map fun branch =>
        { branch with stages := branch.stages.push stage, modeCount? := none }
    | none =>
      let dry ←
        if dryId == "__silence__" then silentForest
        else match rankOf dryId with
          | none => throw s!"lowerModal: node '{dryId}' not found"
          | some ri =>
            if _h : ri < r then lowerModal g rankOf dryId ri
            else throw (cycleGuardMsg id dryId)
      let wet ←
        if wetId == "__silence__" then silentForest
        else match rankOf wetId with
          | none => throw s!"lowerModal: node '{wetId}' not found"
          | some ri =>
            if _h : ri < r then lowerModal g rankOf wetId ri
            else throw (cycleGuardMsg id wetId)
      let dryStage : ModalStage := .linear (ModalLinearStage.scale mix true)
      let wetStage : ModalStage := .linear (ModalLinearStage.scale mix false)
      return (dry.map fun branch =>
          { branch with stages := branch.stages.push dryStage, modeCount? := none }) ++
        (wet.map fun branch =>
          { branch with stages := branch.stages.push wetStage, modeCount? := none })
  | .modalGauge inId gExpr => do
    let lowered ←
      if inId == "__silence__" then silentForest
      else match rankOf inId with
        | none => throw s!"lowerModal: node '{inId}' not found"
        | some ri =>
          if _h : ri < r then lowerModal g rankOf inId ri
          else throw (cycleGuardMsg id inId)
    let stage : ModalStage := .gauge (ModalControlRef.constant gExpr)
    return lowered.map fun branch =>
      { branch with stages := branch.stages.push stage }
  | .modalGaugeControl inId control => do
    let lowered ←
      if inId == "__silence__" then silentForest
      else match rankOf inId with
        | none => throw s!"lowerModal: node '{inId}' not found"
        | some ri =>
          if _h : ri < r then lowerModal g rankOf inId ri
          else throw (cycleGuardMsg id inId)
    let stage : ModalStage := .gauge control
    return lowered.map fun branch =>
      { branch with stages := branch.stages.push stage }
  | .modalMix inputs => do
    match directLinearFactor? g inputs with
    | some (baseId, linear) =>
      let lowered ←
        if baseId == "__silence__" then silentForest
        else match rankOf baseId with
          | none => throw s!"lowerModal: node '{baseId}' not found"
          | some ri =>
            if _h : ri < r then lowerModal g rankOf baseId ri
            else throw (cycleGuardMsg id baseId)
      return lowered.map fun branch =>
        { branch with stages := branch.stages.push (.linear linear), modeCount? := none }
    | none =>
      let forests ← inputs.mapM fun i =>
        if i == "__silence__" then silentForest
        else match rankOf i with
          | none => throw s!"lowerModal: node '{i}' not found"
          | some ri =>
            if _h : ri < r then lowerModal g rankOf i ri
            else throw (cycleGuardMsg id i)
      let parts := forests.foldl (fun acc forest => acc ++ forest) #[]
      if parts.isEmpty then
        throw s!"modalMix '{id}': no inputs"
      else
        -- A general modal mix is the authored-order forest constructor.  The
        -- exact shared `x + linear(x)` topology above retains one factor; all
        -- other sums preserve independent branch metadata and order.
        pure parts
  | _ => throw s!"a modal inlet (reverb/modal-mix) needs a modal SOURCE — resonator or modal-mix — but '{id}' is a signal node; a Sig has no poles to compose"
termination_by r
decreasing_by all_goals assumption

/-- Resolve a missing control to an arena-owned zero. -/
private def valueOrZero (values : Array Sig) (cursor : Nat) : BuildM Sig :=
  match values[cursor]? with
  | some value => pure value
  | none => lit 0

/-- Resolve one room's controls in `ModalStage.controls` order. -/
private def resolveRoomStage (room : OrdinaryRoomStage) (responseClock : Sig)
    (values : Array Sig) (cursor : Nat) :
    BuildM (Array ModalMode × Sig × Option Sig × Nat) := do
  let (kernel, cursor) ← match room.kernel with
    | .fixed modes => pure (modes, cursor)
    | .controlled _ build => do
        let value ← valueOrZero values cursor
        pure (← build value, cursor + 1)
  let directionValue ← valueOrZero values cursor
  let zero ← lit 0
  let one ← lit 1
  let direction ← clampE directionValue zero one
  let cursor := cursor + 1
  let (kernel, cursor) ← match room.sway? with
    | none => pure (kernel, cursor)
    | some _ => do
        let swayValue ← valueOrZero values cursor
        let rateValue ← valueOrZero values (cursor + 1)
        let zero ← lit 0
        let swayMax ← lit 9 1
        let rateMax ← lit 8
        let sway ← clampE swayValue zero swayMax
        let rate ← clampE rateValue zero rateMax
        pure (← Oriented.swayKernel kernel sway rate responseClock, cursor + 2)
  -- Fixed-coordinate qualification of the shipped phaser room gives the
  -- reverse/forward level curve closely enough with `1 + 3.5·d²` (endpoints
  -- within 0.1 dB, interior within 1.2 dB). Apply it once after realization,
  -- never through each modal row, so it costs one multiply and does not deepen
  -- or duplicate the specialized phaser→room graph.
  let levelGain? ← if room.normalizeDirectionLevel then do
      let directionSquared ← mul direction direction
      let scale ← lit 35 1
      let correction ← mul scale directionSquared
      pure (some (← add one correction))
    else
      pure none
  pure (kernel, direction, levelGain?, cursor)

private def resolveLinearStage (stage : ModalLinearStage) (responseClock : Sig)
    (values : Array Sig) (cursor : Nat) : BuildM (ModalKernelExpr × Nat) := do
  let next := cursor + stage.controls.size
  pure (← stage.build responseClock (values.extract cursor next), next)

private structure PlainStageState where
  cursor : Nat
  bank : Oriented.Bank
  levelGain? : Option Sig := none

private def combineLevelGain (left right : Option Sig) : BuildM (Option Sig) :=
  match left, right with
  | none, value | value, none => pure value
  | some a, some b => return some (← mul a b)

private def resolvePlainStageState (initial : PlainStageState)
    (stages : Array ModalStage) (responseClock : Sig) (values : Array Sig) :
    BuildM PlainStageState :=
  stages.foldlM (fun (state : PlainStageState) stage => do
    let cursor := state.cursor
    let bank := state.bank
    match stage with
    | .ordinaryRoom room =>
      let (kernel, direction, gain?, cursor) ←
        resolveRoomStage room responseClock values cursor
      let bank ← bank.convolveKernel kernel direction Oriented.syntacticSameSideClassifier
      let levelGain? ← combineLevelGain state.levelGain? gain?
      pure { cursor, bank, levelGain? }
    | .linear linear =>
      let (kernel, cursor) ← resolveLinearStage linear responseClock values cursor
      pure { state with cursor, bank := ← kernel.applyGeneric bank }
    | .gauge _ =>
      let value ← valueOrZero values cursor
      let zero ← lit 0
      let one ← lit 1
      let g ← clampE value zero one
      pure { state with cursor := cursor + 1, bank := ← bank.gauge g }) initial

private structure HigherOrderPhaserTerminal where
  source : Array ModalMode
  sourceAtTile : Array ModalMode
  linear : ModalLinearStage
  responseClock : Sig
  values : Array Sig
  exactRows : Array ModalMode

private inductive PlainTerminal where
  | generic (terminal : Oriented.TerminalBank)
  | factored (terminal : Oriented.FactoredTwoRoomTerminal)
  | factoredPhaser (terminal : Oriented.FactoredTwoRoomPhaserTerminal)
  | higherOrderPhaser (terminal : HigherOrderPhaserTerminal)
  | scaled (gain : Sig) (terminal : PlainTerminal)

private def PlainTerminal.withLevelGain (terminal : PlainTerminal)
    (gain? : Option Sig) : PlainTerminal :=
  match gain? with
  | none => terminal
  | some gain => .scaled gain terminal

/-- Reconstruct the same mode identities at `sampleIndex + frames`.  Every
    signal field participates, so a warped address or a glide straddling the
    endpoint is evaluated at its true absolute coordinate. -/
private def shiftModalInputs (voice : Array ModalMode) (responseClock : Sig)
    (values : Array Sig) (frames : Nat) :
    BuildM (Array ModalMode × Sig × Array Sig) := do
  let mut roots : Array Sig := #[]
  for mode in voice do
    roots := roots ++ #[mode.sigma, mode.omega, mode.cre, mode.cim]
  roots := roots.push responseClock ++ values
  let shifted ← shiftSampleIndex roots frames
  let mut modes : Array ModalMode := #[]
  let mut cursor := 0
  for mode in voice do
    modes := modes.push { mode with
      sigma := shifted[cursor]!
      omega := shifted[cursor + 1]!
      cre := shifted[cursor + 2]!
      cim := shifted[cursor + 3]! }
    cursor := cursor + 4
  let shiftedResponse := shifted[cursor]!
  pure (modes, shiftedResponse,
    shifted.extract (cursor + 1) shifted.size)

private def stablePhaserTails (tails : Array ModalMode) : Bool :=
  tails.all fun tail =>
    tail.deg == 0 && match tail.sigmaRange with
      | some (lo, hi) => 0.0 < lo && lo <= hi
      | none => false

private def admittedPhaserSectionCount (count : Nat) : Bool :=
  count == 6 || count == 12 || count == 18

private def admittedPhaserSourceCount (count : Nat) : Bool :=
  count == 6 || count == 32

private def lerpSig (phase a b : Sig) : BuildM Sig := do
  add a (← mul phase (← sub b a))

private def sameModeFields (left right : ModalMode) : Bool :=
  left.sigma == right.sigma && left.omega == right.omega
    && left.cre == right.cre && left.cim == right.cim

private def higherOrderSourceOffsets : Nat → Option (Array Nat)
  | 32 => some #[0, 2, 6, 12, 20, 26, 30, 32]
  | 64 => some #[0, 3, 12, 25, 39, 52, 61, 64]
  | 128 => some #[0, 4, 15, 32, 53, 75, 96, 113, 124, 128]
  | _ => none

private def higherOrderWeightSupport (interval : Nat) : Nat :=
  if interval == 32 then 6 else 8

private def chebyshevEvalSig (coefficients : Array Sig)
    (coordinate : Sig) : BuildM Sig := do
  if coefficients.isEmpty then return ← lit 0
  let mut following ← lit 0
  let mut afterFollowing ← lit 0
  for coefficient in (coefficients.extract 1 coefficients.size).reverse do
    let twice ← mul (← lit 2) coordinate
    let current ← add (← sub (← mul twice following) afterFollowing) coefficient
    afterFollowing := following
    following := current
  add (← sub (← mul coordinate following) afterFollowing) coefficients[0]!

/-- Evaluate the published source-gain field inside one indexed modal loop.
    Keeping the Chebyshev reads in the loop body avoids constructing `P`
    separate per-sample amplitude expressions before entering `bankSum`; plan
    size and register pressure are therefore independent of the source row
    count. -/
private def higherOrderSourceSig (modes : Array ModalMode)
    (sourceSupport : Nat) (image coordinate : Sig)
    (clkInt anchorSamples : Sig) : BuildM Sig := do
  let relative ← relClockQ clkInt anchorSamples
  let relativeFloat ← toFloatE relative
  let secondsTimesRate ← div relativeFloat (← lit 4294967296)
  let dSec ← div secondsTimesRate (← sampleRate)
  let twoPi ← twoPiE
  let twoPow32 ← lit 4294967296
  let sr ← sampleRate
  let increments ← modes.mapM fun mode => do
    let frequency ← div mode.omega twoPi
    let scaled ← mul frequency twoPow32
    div scaled sr
  let incrementTable ← arr increments
  let sigmaTable ← arr (modes.map (·.sigma))
  let binderId := 6200
  let modeIndex ← loopIdx binderId
  let coefficientStride := 2 * sourceSupport
  let coefficientBase ← add (← litI 8)
    (← mul (← litI (Int.ofNat coefficientStride)) modeIndex)
  let realCoefficients ← (Array.range sourceSupport).mapM fun degree => do
    let offset ← add coefficientBase (← litI (Int.ofNat (2 * degree)))
    index image offset
  let imagCoefficients ← (Array.range sourceSupport).mapM fun degree => do
    let offset ← add coefficientBase (← litI (Int.ofNat (2 * degree + 1)))
    index image offset
  let cre ← chebyshevEvalSig realCoefficients coordinate
  let cim ← chebyshevEvalSig imagCoefficients coordinate
  let increment ← toIntE (← index incrementTable modeIndex)
  let phase ← modePhaseQFromIncr increment relative
  let scale ← lit 1073741824
  let cosine ← div (← toFloatE (← fixedCosCycSig phase)) scale
  let sine ← div (← toFloatE (← fixedSinCycSig phase)) scale
  let carrier ← sub (← mul cre cosine) (← mul cim sine)
  let sigma ← index sigmaTable modeIndex
  let envelope ← expSig (← neg (← mul sigma dSec))
  let contribution ← mul envelope carrier
  let value ← bankSum modes.size #[incrementTable, sigmaTable, image]
    contribution none binderId
  selectE (← gt relative (← litI 0)) value (← lit 0)

private def higherOrderPhaserSig (terminal : HigherOrderPhaserTerminal)
    (clkInt anchorSamples : Sig) (count? : Option Sig) : BuildM Sig := do
  -- A dynamic source trip count would make the worker's whole-tail sum and the
  -- audio source bank disagree.  Keep that graph on the exact path.
  if count?.isSome then
    let bank ← Oriented.Bank.ofFuture terminal.exactRows
    return ← (Oriented.TerminalBank.ofBank bank).realizeSig
      clkInt anchorSamples count?
  let interval := Tropical.Ir.phaserTimeStagingInterval
  let some supportOffsets := higherOrderSourceOffsets interval
    | let bank ← Oriented.Bank.ofFuture terminal.exactRows
      return ← (Oriented.TerminalBank.ofBank bank).realizeSig
        clkInt anchorSamples count?
  let sourceCount := terminal.source.size
  let sectionCount := terminal.exactRows.size - sourceCount
  let sourceSupport := supportOffsets.size
  let weightSupport := higherOrderWeightSupport interval
  let headerCount := 8
  let sourceCoefficientCount := 2 * sourceSupport * sourceCount
  let tailBase := headerCount + sourceCoefficientCount
  let primitiveBase := tailBase + interval

  -- The unsplit LLVM form is the exact fallback.  At every JIT sample
  -- tilePhase=0 and tileTick=current_idx, so c[0] holds the current exact
  -- decorated source amplitude and tail[0] holds the current exact real-pole
  -- sum.  Keeping the small tail fold scalar (rather than nesting a BankSum in
  -- the Pack) lets TileStage move the complete dependency slice.  The Metal
  -- worker overwrites this prefix with the stable higher-order image.
  let shiftedClockAnchor ← shiftSampleIndex #[clkInt, anchorSamples] 0
  let zero ← lit 0
  let relative ← relClockQ shiftedClockAnchor[0]! shiftedClockAnchor[1]!
  let relativeFloat ← toFloatE relative
  let secondsTimesRate ← div relativeFloat (← lit 4294967296)
  let dSec ← div secondsTimesRate (← sampleRate)
  let tailRows := terminal.exactRows.extract sourceCount terminal.exactRows.size
  let fallbackTail ← tailRows.foldlM (fun total mode => do
    let envelope ← expSig (← neg (← mul mode.sigma dSec))
    add total (← mul envelope mode.cre)) zero
  let fallbackTail ← selectE (← gt relative (← litI 0)) fallbackTail zero
  let mut items : Array Sig := #[]
  for value in #[8430261, 1, sourceCount, sectionCount, interval,
      sourceSupport, weightSupport, primitiveBase] do
    items := items.push (← lit (Int.ofNat value))
  for sourceIndex in [0:sourceCount] do
    items := items ++ #[terminal.exactRows[sourceIndex]!.cre,
      terminal.exactRows[sourceIndex]!.cim]
      ++ Array.replicate (2 * sourceSupport - 2) zero
  items := items.push fallbackTail
  items := items ++ Array.replicate (interval - 1) zero

  -- Static source pole/amplitude fields are private materializer inputs.
  for mode in terminal.sourceAtTile do
    items := items ++ #[mode.sigma, mode.omega, mode.cre, mode.cim]

  -- Integer-coordinate control images feed both the source-gain and internal
  -- rate interpolation.  The 128-frame form uses ten source supports and
  -- eight private weight supports; shorter images retain the measured 8/6
  -- form.  The worker derives weight supports from this common field, avoiding
  -- a second JIT-side trajectory image.
  for offset in supportOffsets do
    let (source, response, values) ← shiftModalInputs terminal.source
      terminal.responseClock terminal.values offset
    let (kernel, _) ← resolveLinearStage terminal.linear response values 0
    let some (tails, mix) := kernel.dryWetAllpassCascadeShape?
      | throw "higherOrderPhaserSig: shifted all-pass topology changed"
    if source.size != sourceCount || tails.size != sectionCount
        || !stablePhaserTails tails then
      throw "higherOrderPhaserSig: shifted source/tail shape changed"
    items := items.push mix ++ tails.map (·.sigma)

  -- The phase-type fold needs the exact bounded response time at every output
  -- sample.  This retains warped clocks, anchors, glides, and random seeks.
  for offset in [0:interval] do
    let shifted ← shiftSampleIndex #[terminal.responseClock, anchorSamples] offset
    let relative ← relClockQ shifted[0]! shifted[1]!
    let relativeFloat ← toFloatE relative
    let secondsTimesRate ← div relativeFloat (← lit 4294967296)
    items := items.push (← div secondsTimesRate (← sampleRate))

  let image ← tileArray items
  let phase ← tilePhase
  let coordinate ← sub (← mul (← lit 2) phase) (← lit 1)
  let sourceValue ← higherOrderSourceSig terminal.source sourceSupport image coordinate
    clkInt anchorSamples
  let localOffset ← toIntE (← mul phase (← lit (Int.ofNat interval)))
  let tailIndex ← add (← lit (Int.ofNat tailBase)) localOffset
  add sourceValue (← index image tailIndex)

private def PlainTerminal.realizeSig (terminal : PlainTerminal)
    (clkInt anchorSamples : Sig) (count? : Option Sig) : BuildM Sig :=
  match terminal with
  | .generic value => value.realizeSig clkInt anchorSamples count?
  | .factored value => value.realizeSig clkInt anchorSamples
  | .factoredPhaser value => value.realizeSig clkInt anchorSamples
  | .higherOrderPhaser value => higherOrderPhaserSig value clkInt anchorSamples count?
  | .scaled gain value => do
      mul gain (← value.realizeSig clkInt anchorSamples count?)

private structure MixedPhaserRows where
  simple : Array ModalMode
  paired : Array PairedMode

/-- Rewrite two ordinary exponential rows into one first-order divided-difference
    atom plus their summed residue on the right pole:

      A e^(λt) + B e^(νt)
        = A(λ-ν) e^(νt) t cexpm1((λ-ν)t) + (A+B)e^(νt).

    This is an algebraic identity for every frozen endpoint.  Pair selection is
    structural and authored-order; it never depends on floating residue size. -/
private def fusePhaserRows (left right : ModalMode) : BuildM (ModalMode × PairedMode) := do
  let lam ← left.poleE
  let nu ← right.poleE
  let delta ← csubE lam nu
  let c ← cmulE left.ampE delta
  let remainder ← caddE left.ampE right.ampE
  pure ({ right with cre := remainder.1, cim := remainder.2 }, { lam, nu, c })

/-- The all-pass tail poles form one same-frequency cluster.  This falsification
    experiment pairs adjacent structural rows, leaving source rows and
    at most one odd tail as simple exponentials.  Candidate density is reported
    by the benchmark from the section count; this representation deliberately
    does not pretend disjoint first-order pairs solve a dense higher-order
    cluster. -/
private def mixedPhaserRows (sourceCount : Nat) (rows : Array ModalMode) :
    BuildM MixedPhaserRows := do
  if rows.size < sourceCount then
    throw "mixedPhaserRows: decorated row count is smaller than source count"
  let source := rows.extract 0 sourceCount
  let tails := rows.extract sourceCount rows.size
  let mut simple := source
  let mut paired : Array PairedMode := #[]
  let mut i := 0
  while i + 1 < tails.size do
    let (remainder, pair) ← fusePhaserRows tails[i]! tails[i + 1]!
    simple := simple.push remainder
    paired := paired.push pair
    i := i + 2
  if i < tails.size then simple := simple.push tails[i]!
  pure { simple, paired }

private def interpolateModes (phase : Sig) (startRows endRows : Array ModalMode) :
    BuildM (Array ModalMode) := do
  let startSigma ← tileArray (startRows.map (·.sigma))
  let startOmega ← tileArray (startRows.map (·.omega))
  let startCre ← tileArray (startRows.map (·.cre))
  let startCim ← tileArray (startRows.map (·.cim))
  let endSigma ← tileArray (endRows.map (·.sigma))
  let endOmega ← tileArray (endRows.map (·.omega))
  let endCre ← tileArray (endRows.map (·.cre))
  let endCim ← tileArray (endRows.map (·.cim))
  (Array.range startRows.size).mapM fun i => do
    let indexSig ← litI (Int.ofNat i)
    pure ({
      sigma := ← lerpSig phase (← index startSigma indexSig)
        (← index endSigma indexSig)
      omega := ← lerpSig phase (← index startOmega indexSig)
        (← index endOmega indexSig)
      cre := ← lerpSig phase (← index startCre indexSig)
        (← index endCre indexSig)
      cim := ← lerpSig phase (← index startCim indexSig)
        (← index endCim indexSig)
      deg := 0 } : ModalMode)

private def interpolatePairedModes (phase : Sig)
    (startRows endRows : Array PairedMode) : BuildM (Array PairedMode) := do
  let startLamRe ← tileArray (startRows.map (·.lam.1))
  let startLamIm ← tileArray (startRows.map (·.lam.2))
  let startNuRe ← tileArray (startRows.map (·.nu.1))
  let startNuIm ← tileArray (startRows.map (·.nu.2))
  let startCre ← tileArray (startRows.map (·.c.1))
  let startCim ← tileArray (startRows.map (·.c.2))
  let endLamRe ← tileArray (endRows.map (·.lam.1))
  let endLamIm ← tileArray (endRows.map (·.lam.2))
  let endNuRe ← tileArray (endRows.map (·.nu.1))
  let endNuIm ← tileArray (endRows.map (·.nu.2))
  let endCre ← tileArray (endRows.map (·.c.1))
  let endCim ← tileArray (endRows.map (·.c.2))
  (Array.range startRows.size).mapM fun i => do
    let indexSig ← litI (Int.ofNat i)
    pure ({
      lam := (
        ← lerpSig phase (← index startLamRe indexSig)
          (← index endLamRe indexSig),
        ← lerpSig phase (← index startLamIm indexSig)
          (← index endLamIm indexSig))
      nu := (
        ← lerpSig phase (← index startNuRe indexSig)
          (← index endNuRe indexSig),
        ← lerpSig phase (← index startNuIm indexSig)
          (← index endNuIm indexSig))
      c := (
        ← lerpSig phase (← index startCre indexSig)
          (← index endCre indexSig),
        ← lerpSig phase (← index startCim indexSig)
          (← index endCim indexSig)) } : PairedMode)

/-- Build the compact audio side of the staged terminal.  The higher-order
    experiment publishes a single primitive image for the native worker to
    replace with stable source polynomials and a bounded whole-tail table.  The
    retained endpoint paths remain available only to reproduce the earlier
    ordinary and mixed-DD falsification evidence. -/
private def stagedPhaserTerminal? (voice : Array ModalMode)
    (linear : ModalLinearStage) (responseClock : Sig) (values : Array Sig) :
    BuildM (Option PlainTerminal) := do
  if !Tropical.Ir.phaserTimeStagingEnabled then return none
  if !(voice.all (·.deg == 0)) then return none
  -- Patched signal-rate controls are not part of the first proof surface.
  if !(linear.controls.all (·.signalNode?.isNone)) then return none
  let (startVoice, startResponse, startValues) ←
    shiftModalInputs voice responseClock values 0
  let (startKernel, _) ← resolveLinearStage linear startResponse startValues 0
  let some (startTails, startMix) := startKernel.dryWetAllpassCascadeShape?
    | return none
  if !admittedPhaserSectionCount startTails.size
      || !stablePhaserTails startTails then return none
  let interval := Tropical.Ir.phaserTimeStagingInterval
  let (endVoice, endResponse, endValues) ←
    shiftModalInputs voice responseClock values interval
  let (endKernel, _) ← resolveLinearStage linear endResponse endValues 0
  let some (endTails, endMix) := endKernel.dryWetAllpassCascadeShape?
    | return none
  if endTails.size != startTails.size || !stablePhaserTails endTails then
    return none
  let startRows ← Oriented.decorateDegreeZeroCausalPhaser
    startVoice startTails startMix
  let endRows ← Oriented.decorateDegreeZeroCausalPhaser
    endVoice endTails endMix
  if startRows.size != endRows.size || startRows.isEmpty then return none
  if Tropical.Ir.phaserTimeStagingHigherOrderEnabled then
    if (higherOrderSourceOffsets interval).isNone
        || !admittedPhaserSourceCount startVoice.size
        || startVoice.size != endVoice.size
        || !(startVoice.zip endVoice).all fun (left, right) =>
          sameModeFields left right then
      return none
    return some (.higherOrderPhaser {
      source := voice
      sourceAtTile := startVoice
      linear
      responseClock
      values
      exactRows := startRows })
  let phase ← tilePhase
  if Tropical.Ir.phaserTimeStagingMixedDDEnabled then
    let startMixed ← mixedPhaserRows voice.size startRows
    let endMixed ← mixedPhaserRows voice.size endRows
    if startMixed.simple.size != endMixed.simple.size
        || startMixed.paired.size != endMixed.paired.size then return none
    let rows ← interpolateModes phase startMixed.simple endMixed.simple
    let paired ← interpolatePairedModes phase startMixed.paired endMixed.paired
    let bank ← Oriented.Bank.ofFuture rows
    pure (some (.generic ({ bank, futurePaired := paired } : Oriented.TerminalBank)))
  else
    let rows ← interpolateModes phase startRows endRows
    let bank ← Oriented.Bank.ofFuture rows
    pure (some (.generic (Oriented.TerminalBank.ofBank bank)))

/-- Fold an authored stage spine after binding the current static universe.  A
    final room uses the stable EC/DD carrier for hot same-side couplings. -/
private def resolvePlainStages (voice : Array ModalMode) (stages : Array ModalStage)
    (responseClock : Sig) (values : Array Sig) : BuildM PlainTerminal := do
  let initialBank ← Oriented.Bank.ofFuture voice
  let initial : PlainStageState := { cursor := 0, bank := initialBank }
  if stages.size == 1 then
    match stages[0]? with
    | some (ModalStage.linear linear) =>
      match ← stagedPhaserTerminal? voice linear responseClock values with
      | some terminal => pure terminal
      | none =>
        let (kernel, _) ← resolveLinearStage linear responseClock values 0
        match kernel.dryWetAllpassCascadeShape? with
        | some (tails, mix) =>
          let decorated ← Oriented.decorateDegreeZeroCausalPhaser voice tails mix
          let bank ← Oriented.Bank.ofFuture decorated
          pure (.generic (Oriented.TerminalBank.ofBank bank))
        | none => match kernel.orientedShape? with
          | some (modes, direction) =>
            pure (.generic (← initialBank.convolveKernelTerminal modes direction))
          | none =>
            let bank ← kernel.applyGeneric initialBank
            pure (.generic (Oriented.TerminalBank.ofBank bank))
    | some (ModalStage.ordinaryRoom room) =>
      let (kernel, direction, gain?, _) ← resolveRoomStage room responseClock values 0
      pure <| (PlainTerminal.generic (← initialBank.convolveKernelTerminal kernel direction))
        |>.withLevelGain gain?
    | _ =>
      let state ← resolvePlainStageState initial stages responseClock values
      pure <| (PlainTerminal.generic (Oriented.TerminalBank.ofBank state.bank))
        |>.withLevelGain state.levelGain?
  else if stages.size == 2 then
    match stages[0]?, stages[1]? with
    | some (ModalStage.ordinaryRoom first), some (ModalStage.ordinaryRoom second) =>
      let (room1, direction1, gain1?, cursor) ←
        resolveRoomStage first responseClock values 0
      let (room2, direction2, gain2?, _) ←
        resolveRoomStage second responseClock values cursor
      let gain? ← combineLevelGain gain1? gain2?
      match ← Oriented.factoredTwoRoomTerminal? voice room1 room2 direction1 direction2 with
      | some terminal => pure <| (PlainTerminal.factored terminal).withLevelGain gain?
      | none =>
        let bank ← initialBank.convolveKernel room1 direction1
          Oriented.syntacticSameSideClassifier
        pure <| (PlainTerminal.generic (← bank.convolveKernelTerminal room2 direction2))
          |>.withLevelGain gain?
    | some (ModalStage.linear linear), some (ModalStage.ordinaryRoom room) =>
      let (linearKernel, cursor) ← resolveLinearStage linear responseClock values 0
      match linearKernel.dryWetAllpassCascadeShape? with
      | some (tails, mix) =>
        let decorated ← Oriented.decorateDegreeZeroCausalPhaser voice tails mix
        let decoratedBank ← Oriented.Bank.ofFuture decorated
        let (kernel, direction, gain?, _) ←
          resolveRoomStage room responseClock values cursor
        pure <| (PlainTerminal.generic (← decoratedBank.convolveKernelTerminal kernel direction))
          |>.withLevelGain gain?
      | none =>
        let state ← resolvePlainStageState initial stages responseClock values
        pure <| (PlainTerminal.generic (Oriented.TerminalBank.ofBank state.bank))
          |>.withLevelGain state.levelGain?
    | _, _ =>
      let state ← resolvePlainStageState initial stages responseClock values
      pure <| (PlainTerminal.generic (Oriented.TerminalBank.ofBank state.bank))
        |>.withLevelGain state.levelGain?
  else if stages.size == 3 then
    match stages[0]?, stages[1]?, stages[2]? with
    | some (ModalStage.ordinaryRoom first), some (ModalStage.linear linear),
        some (ModalStage.ordinaryRoom second) =>
      let (room1, direction1, gain1?, cursor) ←
        resolveRoomStage first responseClock values 0
      let (linearKernel, cursor) ← resolveLinearStage linear responseClock values cursor
      match linearKernel.dryWetAllpassCascadeShape? with
      | some (tails, mix) =>
        let (room2, direction2, gain2?, _) ←
          resolveRoomStage second responseClock values cursor
        let gain? ← combineLevelGain gain1? gain2?
        match ← Oriented.factoredTwoRoomPhaserTerminal? voice tails room1 room2 mix
            direction1 direction2 with
        | some terminal =>
          pure <| (PlainTerminal.factoredPhaser terminal).withLevelGain gain?
        | none =>
          let bank ← initialBank.convolveKernel room1 direction1
            Oriented.syntacticSameSideClassifier
          let bank ← linearKernel.applyGeneric bank
          pure <| (PlainTerminal.generic (← bank.convolveKernelTerminal room2 direction2))
            |>.withLevelGain gain?
      | none =>
        let state ← resolvePlainStageState initial stages responseClock values
        pure <| (PlainTerminal.generic (Oriented.TerminalBank.ofBank state.bank))
          |>.withLevelGain state.levelGain?
    | _, _, _ =>
      let state ← resolvePlainStageState initial stages responseClock values
      pure <| (PlainTerminal.generic (Oriented.TerminalBank.ofBank state.bank))
        |>.withLevelGain state.levelGain?
  else
    match stages.back? with
    | some (.ordinaryRoom room) =>
      let state ← resolvePlainStageState initial stages.pop responseClock values
      let (kernel, direction, gain?, _) ←
        resolveRoomStage room responseClock values state.cursor
      let gain? ← combineLevelGain state.levelGain? gain?
      pure <| (PlainTerminal.generic (← state.bank.convolveKernelTerminal kernel direction))
        |>.withLevelGain gain?
    | _ =>
      let state ← resolvePlainStageState initial stages responseClock values
      pure <| (PlainTerminal.generic (Oriented.TerminalBank.ofBank state.bank))
        |>.withLevelGain state.levelGain?

/-- The present composable carrier can safely cross one nonterminal room.  A
    second room must remain terminal: making it cross a later room, phaser, or gauge
    would require generalized composable divided differences when independently
    authored damping expressions happen to have the same runtime value. -/
private def plainStageSpineAdmitted (stages : Array ModalStage) : Bool := Id.run do
  let mut roomsSeen := 0
  for (stage, index) in stages.zipIdx do
    if stage matches .ordinaryRoom _ then
      if roomsSeen > 0 && index + 1 < stages.size then
        return false
      roomsSeen := roomsSeen + 1
  return true

/-- The already-proven causal bloom bridge remains available for a spine of
    fixed, unswayed, exactly-forward rooms.  Any live/local direction, sway,
    controlled kernel, or gauge requires the oriented Gamma extension and is
    refused as one named unsupported crossing. -/
private def fixedForwardBloomRooms? (stages : Array ModalStage) :
    BuildM (Option (Array (Array ModalMode))) := do
  let mut rooms := #[]
  for stage in stages do
    let .ordinaryRoom room := stage | return none
    let .fixed modes := room.kernel | return none
    if room.direction.signalNode?.isSome || room.sway?.isSome then return none
    let .konst direction := room.direction.fallback | return none
    let some value ← sigConstF? direction | return none
    if value != 0.0 then return none
    rooms := rooms.push modes
  pure rooms

mutual
/-- The gated `Sig`-side recursion step: rank lookup, dangling refusal
    (today's message), and the strict-decrease check that carries the
    whole mutual block's termination. -/
def lowerInputGated (g : PatchGraph) (rankOf : String → Option Nat)
    (srcId i : String) (r : Nat) : BuildM ArrowTerm := do
  match rankOf i with
  | none => throw s!"lower: node '{i}' not found"
  | some ri =>
    if _h : ri < r then lowerInput g rankOf i ri
    else throw (cycleGuardMsg srcId i)
termination_by (r, 0)
/-- Lower one Sig node to its arrow term, recursing UP its input wires via
    `lowerInput` (which realizes a modal input at its edge). A wire is `⋙`; fan-out
    is the shared upstream term; a mixer is the sum; `fm` routes its input's signal
    into the carrier's clock. Result is the UNREDUCED downstream term. -/
def lowerNode (g : PatchGraph) (rankOf : String → Option Nat)
    (id : String) (r : Nat) : BuildM ArrowTerm := do
  let some pn := g.nodes.find? (·.id == id)
    | throw s!"lower: node '{id}' not found"
  match pn.node with
  | .source v clk => pure (.gen v id clk)
  | .flange inId back fwd =>
    flangeEffectWith back fwd (← lowerInputGated g rankOf id inId r)
  | .shaper inId f => return .arrUn f (← lowerInputGated g rankOf id inId r)
  | .warpFx inId φ => return .warp φ (← lowerInputGated g rankOf id inId r)
  | .mix inputs => return .sum (← inputs.mapM (lowerInputGated g rankOf id · r))
  | .fm inId carrier base depth =>
    return .swarp (fmWarp depth) (← lowerInputGated g rankOf id inId r) (.gen carrier id base)
  | .pm inId carrier base depth =>
    return .pmGen carrier id base depth (← lowerInputGated g rankOf id inId r)
  | .sflange inId modId depthSec =>
    sweptFlangeEffect (sflangeBack depthSec) (sflangeFwd depthSec)
      (← lowerInputGated g rankOf id modId r) (← lowerInputGated g rankOf id inId r)
  | .knob idx => pure (.konst (← paramRef ⟨idx⟩))
  | .comb inId taps => do
    let s ← lowerInputGated g rankOf id inId r
    return .sum (taps.map fun (w, φ) => .scale w (.warp φ s))
  | .ring inputs => do
    let terms ← inputs.mapM (lowerInputGated g rankOf id · r)
    match terms.toList with
    | [] => pure (.konst (← lit 0))
    | t :: ts => return ts.foldl (fun acc u => .prod acc u) t
  | _ =>
    throw s!"lower: modal node '{id}' reached Sig lowering — realize via lowerInput"
termination_by (r, 1)

/-- Lower an input wire: a modal node realizes its island (pole bank →
    `modalBankTerm` at the master clock) at this edge; a Sig node lowers normally.
    This is the one-directional seam — modal grafts into Sig here, never the
    reverse (a Sig node has no modal input, so lowerModal is never asked for one). -/
def lowerInput (g : PatchGraph) (rankOf : String → Option Nat)
    (id : String) (r : Nat) : BuildM ArrowTerm := do
  if nodeIsModal g id then
    let forest ← lowerModal g rankOf id r
    let terms ← forest.mapM fun modal => do
      -- Realize each branch independently at THIS edge. The forest order is the
      -- authored order and therefore the signal-sum order.
      let controlTerms ← modal.controls.mapM fun control =>
        match control.signalNode? with
        | none => pure control.fallback
        | some controlId => lowerInputGated g rankOf id controlId r
      match modal.source with
      | .plain modes =>
        if !plainStageSpineAdmitted modal.stages then
          throw s!"lower: nonterminal repeated-room crossing at '{id}' refused (a later room, phaser, or gauge requires the composable divided-difference carrier)"
        else if modal.stages.isEmpty then
          let bare := modalBankTerm modes modal.strikeAnchor modal.realizationClock
            modal.modeCount?
          match modal.addressNode? with
          | none => pure bare
          | some addrId =>
            pure (.swarp modalAddrWarp
              (← lowerInputGated g rankOf id addrId r) bare)
        else
          -- The address warp is confined to the response-clock child. Control
          -- terms remain siblings, so branch address does not implicitly retime
          -- an independent control source; an ordinary downstream warp still
          -- reaches every child through `arrN`'s shared clock context.
          let response : ArrowTerm ← match modal.addressNode? with
            | none => pure (.clk modal.realizationClock)
            | some addrId => pure (.swarp modalAddrWarp
                (← lowerInputGated g rankOf id addrId r)
                (.clk modal.realizationClock))
          pure (.arrN (fun inputs => do
            let responseClock := (inputs[0]?).getD modal.realizationClock
            let values := inputs.extract 1 inputs.size
            let bank ← resolvePlainStages modes modal.stages responseClock values
            bank.realizeSig responseClock modal.strikeAnchor modal.modeCount?)
            (#[response] ++ controlTerms))
      | .bloomed voice B gr =>
        let hasLinear := modal.stages.any fun stage => stage matches .linear _
        if hasLinear then
          throw s!"lower: bloomed linear-kernel crossing at '{id}' refused (the live linear/Gamma crossing is not implemented)"
        else
        let term ← match ← fixedForwardBloomRooms? modal.stages with
          | none =>
            throw s!"lower: bloomed stage crossing at '{id}' refused (live room-kernel direction/sway or gauge requires the oriented Gamma bridge)"
          | some rooms =>
            if rooms.isEmpty then
              pure (ArrowTerm.warp (bloomWarpClock modal.strikeAnchor B gr)
                (modalBankTerm voice modal.strikeAnchor modal.realizationClock modal.modeCount?))
            else
              let folded ← foldRoomsEC rooms[0]! (rooms.extract 1 rooms.size)
              let composed ← bloomComposeChecked voice folded B gr
              if composed.isComplete then
                pure (bloomComposedTerm composed.pairs modal.strikeAnchor modal.realizationClock)
              else
                throw s!"lower: bloomed room crossing at '{id}' refused ({composed.refusalSummary})"
        match modal.addressNode? with
        | none => pure term
        | some addrId =>
          pure (.swarp modalAddrWarp
            (← lowerInputGated g rankOf id addrId r) term)
    -- Keep every pre-forest singleton plan structurally identical. Forests with
    -- multiple branches cross once as a stable authored-order signal sum.
    match terms.toList with
    | [] => pure (.sum #[])
    | [term] => pure term
    | _ => pure (.sum terms)
  else lowerNode g rankOf id r
termination_by (r, 2)
end

/-- Lower from an arbitrary node id — THE total entry. Interns the graph,
    refuses cycles with the loop named (the topo sort IS the cycle check,
    and it covers every caller — hand-built `PatchGraph`s included, which
    used to bypass the playground's separate pre-pass), then lowers with
    the Kahn rank as the termination measure. -/
def lowerAt (g : PatchGraph) (rootId : String) : BuildM ArrowTerm := do
  let ids := g.nodes.map (·.id)
  let deps : Array (Array Nat) := g.nodes.map fun pn =>
    pn.node.inputIds.filterMap ids.idxOf?
  let some ranks := Tropical.Ir.topoRanks? deps
    | let names := match Tropical.Ir.findCycle deps with
        | some cyc => cyc.map fun i => ((ids[i]?).getD "?")
        | none => #[]
      throw s!"connection cycle: {Tropical.Ir.renderLoop names} — patch graphs must be acyclic (you may only patch forward; there is no delay to break a loop through)"
  let rankOf := fun (s : String) => (ids.idxOf? s).bind (ranks[·]?)
  lowerInput g rankOf rootId ((rankOf rootId).getD g.nodes.size)

/-- Lower a whole patch to its (downstream, unreduced) arrow term. The output may
    be a modal island (realized at the tap) or a Sig graph. Compose with
    `normalize` (the slide), then `emitTerm`. -/
def lowerGraph (g : PatchGraph) : BuildM ArrowTerm := lowerAt g g.output

end Tropical.EmitArrow
