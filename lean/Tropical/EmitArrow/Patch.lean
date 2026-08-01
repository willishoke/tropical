import Tropical.EmitArrow.Modal
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
def modalAddrWarp : Clock → Sig → Clock :=
  fun _ s => toIntE (mul s (mul .sampleRate (lit 4294967296)))

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
  | flange (input : String) (back fwd : Clock → Clock)
  | shaper (input : String) (f : Sig → Sig)
  | warpFx (input : String) (φ : Clock → Clock)
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
  /-- A CONTROL: a program that is nothing but a scalar expression with one
      output. A raw Knob supplies a `paramRef`; a GlideKnob supplies the same
      closed-form smoothstep used by glided module parameters. Keeping the
      expression here lets a shared control preserve its write discipline when
      it is wired into several consumers. -/
  | knob (value : Sig)
  /-- A one-sided resonant COMB: `input` read at a bank of clock-warped offsets,
      each weighted — `Σ wₖ·(warpₖ ⋙ input)`. `taps` is `(weight, clockShift)` per
      tap: tap 0 is usually the dry `(1, id)`, the rest a decaying series `(gᵏ,
      c ↦ c + k·D)`. One-sided (all shifts the same direction) makes it
      time-ASYMMETRIC — its impulse tail rings on one side, so under a global clock
      reverse the tail flips (echo ↔ pre-echo). A future shift (`c + kD`, `D > 0`)
      reads AHEAD — an audible pre-echo, impossible on a stream. The slide
      distributes each tap's warp onto the generators, exactly like the flanger. -/
  | comb (input : String) (taps : Array (Sig × (Clock → Clock)))
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
  | modalReverb (input : String) (room : Array ModalMode) (dir : Option ModalDir)
  | modalMix (inputs : Array String)
  -- `modalGauge` is the excitation-gauge adapter (§5): it re-levels its modal input's
  -- residues by the self-measured `‖H‖^{−g}` (`normalizePeak`), a pure Modal ⇝ Modal
  -- effect. `g` is a live slot. The norm is measured on the SETTLED poles, so an
  -- un-settleable (per-sample-modulated) input DECLINES to identity, never an s1 norm.
  | modalGauge (input : String) (g : Sig)
  /-- The frozen grouped-room TERMINAL: consumes an untouched modal source and
      produces a signal from the approved immutable prefix profile. -/
  | groupedRoom (input profile : String) (position : Sig)
  /-- The bounded release-scene cache selected only after direct evaluator
      reserve failure. It is a signal source addressed by the master clock. -/
  | groupedRoomCache (profile : String) (position : Sig) (clk : Clock)

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
def fmWarp (depthE : Sig) : Clock → Sig → Clock :=
  fun base m => sub base (toIntE (mul (mul depthE m) (lit 4294967296)))

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
  | .modalReverb i _ _ | .modalGauge i _ | .groupedRoom i _ _ => #[i]
  | .groupedRoomCache .. => #[]

/-- Is `id` a modal-island node? Its output wire carries poles, not a `Sig`. A
    missing node reads as Sig (graceful — a half-built patch stays lowerable). -/
def nodeIsModal (g : PatchGraph) (id : String) : Bool :=
  match g.nodes.find? (·.id == id) with
  | none => false
  | some pn => match pn.node with
    | .modalSource .. => true
    | .modalReverb .. => true
    | .modalMix .. => true
    | .modalGauge .. => true
    | _ => false

/-- The lowered modal value: a PLAIN source with its room chain DEFERRED
    (accumulated, folded at realization — the EC/DD partition site) or a
    BLOOMED source kept modal — its voice modes and the FOLDED room chain held
    SEPARATE, so the pitch bloom is crossed ONCE (`bloomCompose`) at
    realization, AFTER the rooms fold. Rooms fold by the residue calculus
    reinterpreted as filter∘filter (the two-hats reading: a bank is both a
    struck source `Σaᵢe^{λᵢd}` and a transfer function `Σaᵢ/(s−λᵢ)`, same
    data), so gong⋙reverb⋙reverb crosses each nonlinearity exactly once.

    Why the plain arm defers (fork 3′): rooms COMMUTE (filter∘filter, the
    bit-exact registered EC-commute law), so the fold order is free — and the
    partition wants hot rooms (couplings near coincidence, `couplingHot`) to
    fold LAST, where their couplings become `PairedMode` atoms that never need
    to re-compose. A cold chain folds in arrival order through
    `residueComposeEC` — byte-identical to the eager fold this replaced. -/
inductive ModalBank where
  | plain (voice : Array ModalMode) (rooms : Array (Array ModalMode))
  | bloomed (voice room : Array ModalMode) (bloomB gRate : Float)

/-- The result of lowering a modal island before it crosses into the signal
    graph. These values travel together, but are not interchangeable tuple
    positions: the bank is the spectral payload, while the remaining fields
    describe how and where that payload is realized. -/
structure LoweredModal where
  bank : ModalBank
  strikeAnchor : Sig
  realizationClock : Clock
  addressNode? : Option String
  direction? : Option ModalDir
  modeCount? : Option Sig

/-- The COLLECTED chain fold, arrival order — today's exact expressions (the
    byte-identity comparator, and the fallback for the surfaces that need a
    plain `Array ModalMode`: gauge, mix, direction). -/
def foldRoomsEC (voice : Array ModalMode) (rooms : Array (Array ModalMode)) :
    Array ModalMode :=
  rooms.foldl residueComposeEC voice

/-- The PARTITIONED chain fold: cold rooms fold first in arrival order
    (collected, verbatim), rooms carrying a hot coupling fold LAST, and only
    the final fold partitions — so `PairedMode`s form once and never
    re-compose. When no room is hot this is `foldRoomsEC` VERBATIM.

    THE DETECTION INVARIANT (stated precisely, not hedged): heat is decided
    twice — the sort HEURISTIC here reads source-UNIT poles/amps; the
    partition re-decides per coupling on the COMPOSED amps, but only at the
    final fold. The two predicates cohere exactly this far:
    · θ_acc (accuracy-lens) heat is a pole-distance test — amp-independent —
      and composition never moves poles, so the sort CANNOT miss it; every
      near-coincident coupling reaches the final fold and partitions.
    · Rail-lens heat is amp-dependent, and composed ringing amps carry `1/Δ`
      factors from earlier folds — so a coupling can be rail-cold on unit
      amps yet rail-hot on composed amps. If it forms BEFORE the final fold
      it folds collected, silently, at the status-quo floor. That is: **amp-
      dependent heat is only detected at the final fold.** The exposure is
      narrow today — the rail lens's routed service region is itself cap-
      bounded to well-damped couplings (`ecddRailCeil` doc) — but it WIDENS
      if the cap or lens is ever retuned; re-examine this invariant then.
    A multi-hot residual (a hot coupling forming before the final fold — a
    deliberate triple/multi-unison) likewise stays collected at the
    status-quo floor, served gracefully, not silently wrong. -/
def foldRoomsPartitioned (voice : Array ModalMode)
    (rooms : Array (Array ModalMode)) :
    Array ModalMode × Array PairedMode :=
  if rooms.isEmpty then (voice, #[]) else
  let units := #[voice] ++ rooms
  let roomHot := fun (j : Nat) (room : Array ModalMode) =>
    room.any fun q => units.zipIdx.any fun (u, i) =>
      i != j + 1 && u.any fun p => couplingHot p q || couplingHot q p
  if (rooms.zipIdx.all fun (r, j) => !roomHot j r) then
    (foldRoomsEC voice rooms, #[])
  else
    let cold := rooms.zipIdx.filterMap fun (r, j) =>
      if roomHot j r then none else some r
    let hot := rooms.zipIdx.filterMap fun (r, j) =>
      if roomHot j r then some r else none
    let ordered := cold ++ hot
    let front := ordered.pop
    let last := ordered.back!
    residueComposePartitioned (foldRoomsEC voice front) last

/-- The empty modal value (`__silence__`, the graceful-silence contract). -/
private def silentModal : LoweredModal where
  bank := .plain #[] #[]
  strikeAnchor := lit 0
  realizationClock := clockLit
  addressNode? := none
  direction? := none
  modeCount? := none

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
    Except String LoweredModal := do
  -- An unconnected modal inlet points at `__silence__`; a modal node with no modal
  -- source is a legal, incomplete patch — it compiles to an EMPTY bank (silence),
  -- not an error. (The graceful-silence contract, same as `mix []`.)
  if id == "__silence__" then
    return silentModal
  let some pn := g.nodes.find? (·.id == id)
    | .error s!"lowerModal: node '{id}' not found"
  -- The gated modal recursion (inlined at each site for the WF measure):
  -- `__silence__` short-circuits, a dangling id keeps today's message, and
  -- a rank breach is the dead-by-invariant refusal.
  match pn.node with
  | .modalSource ms a clk addr count bloom? =>
    -- unit modes enter the island CLAMPED to their declared σ intervals
    -- (`clampSigmas` — the uniform-bank clamp; every downstream fold,
    -- partition, gauge, and mix inherits it, so the whole bank saturates
    -- together under knob overdrive). In-span drives are value-identical.
    let ms := clampSigmas ms
    match bloom? with
    | none => .ok {
        bank := .plain ms #[]
        strikeAnchor := a
        realizationClock := clk
        addressNode? := addr
        direction? := none
        modeCount? := count }
    -- a bloomed source starts modal with an EMPTY room; reverbs fold in downstream
    | some (B, gr) => .ok {
        bank := .bloomed ms #[] B gr
        strikeAnchor := a
        realizationClock := clk
        addressNode? := addr
        direction? := none
        modeCount? := count }
  | .modalReverb inId room dir => do
    let lowered ←
      if inId == "__silence__" then pure silentModal
      else match rankOf inId with
        | none => throw s!"lowerModal: node '{inId}' not found"
        | some ri =>
          if h : ri < r then lowerModal g rankOf inId ri
          else throw (cycleGuardMsg id inId)
    -- The live count does NOT survive composition (the composed modes are no
    -- longer a prefix of the source's) — realize at full capacity (graceful-
    -- silent through a reverb; v1, no warning, legal state).
    -- Room unit modes enter clamped (the uniform-bank σ clamp — see
    -- `.modalSource`); in-span drives are value-identical.
    let room := clampSigmas room
    match lowered.bank with
    | .plain v rooms =>
      -- source ⋙ filter: ACCUMULATE the room; the fold (and the EC/DD
      -- partition) happens once, at realization — where the whole chain is
      -- visible and hot rooms can sort last. Amps stay symbolic, so live
      -- source knobs survive the room.
      .ok { lowered with
        bank := .plain v (rooms.push room)
        direction? := dir.orElse (fun _ => lowered.direction?)
        modeCount? := none }
    | .bloomed voice acc B gr =>
      -- filter ∘ filter: FOLD this room into the accumulated room chain (voice
      -- stays separate, bloom uncrossed). An empty accumulator is the fold
      -- identity (the first reverb in a chain).
      let folded := if acc.isEmpty then room else residueComposeEC acc room
      .ok { lowered with
        bank := .bloomed voice folded B gr
        direction? := dir.orElse (fun _ => lowered.direction?)
        modeCount? := none }
  | .modalGauge inId gExpr => do
    let lowered ←
      if inId == "__silence__" then pure silentModal
      else match rankOf inId with
        | none => throw s!"lowerModal: node '{inId}' not found"
        | some ri =>
          if h : ri < r then lowerModal g rankOf inId ri
          else throw (cycleGuardMsg id inId)
    -- re-level in place: `normalizePeak` scales residues by the SETTLED norm (s0),
    -- preserving the pole set — so `count` (the live mode count) survives, unlike a
    -- reverb's composition. An un-settleable input declines to identity inside
    -- `normalizePeak`. A bloomed source gauges its VOICE modes; the folded room and
    -- the uncrossed bloom are untouched.
    match lowered.bank with
    -- gauge is a COLLECTED surface in v1: it needs the composed bank to
    -- self-measure, so it forces the fold here (today's exact expressions) —
    -- and its output amps carry the norm's expSig, so downstream couplings
    -- are unmeasurable and stay collected (status quo, documented). The fold
    -- is SAFE against hot chains even though the collected amps carry the
    -- huge cancelling ±c/Δ pair: the norm reads the transfer function H,
    -- which has NO 1/Δ pole (the partial fractions recombine into the
    -- bounded product form), so the f64 cancellation is benign — measured,
    -- not assumed (the ecdd-gauge gate mirrors the norm independently and
    -- pins gauged ≡ scale × collected). The residual under a gauge is the
    -- hot pair rendering at the collected floor — which at sub-grid detune
    -- means the pair is silent AND its ±c/Δ amps size the bank's landing
    -- exponent, quantizing the cold modes coarsely (the LANDING POISON, the
    -- gate's own finding) — the stated status quo of every collected
    -- surface, one more reason the routed path is the served one.
    | .plain v rooms =>
      .ok { lowered with
        bank := .plain (normalizePeak gExpr (foldRoomsEC v rooms)) #[] }
    | .bloomed voice acc B gr =>
      .ok { lowered with
        bank := .bloomed (normalizePeak gExpr voice) acc B gr }
  | .modalMix inputs => do
    let parts ← inputs.mapM fun i =>
      if i == "__silence__" then pure silentModal
      else match rankOf i with
        | none => throw s!"lowerModal: node '{i}' not found"
        | some ri =>
          if h : ri < r then lowerModal g rankOf i ri
          else throw (cycleGuardMsg id i)
    match parts.toList with
    | [] => .error s!"modalMix '{id}': no inputs"
    -- head carries the shared anchor/clock/address/direction for the union.
    | first :: rest =>
      -- pole union is defined on PLAIN banks; a bloomed source carries a warp
      -- the union can't merge (v1) — reverb it before mixing.
      -- mix is a COLLECTED surface in v1: pole union needs plain mode arrays,
      -- so a deferred chain folds here (today's exact expressions). A hot
      -- coupling BELOW a mix stays collected (status quo, documented); one
      -- formed by a reverb ABOVE the mix partitions normally.
      let modesOf : ModalBank → Except String (Array ModalMode) := fun b => match b with
        | .plain ms rooms => .ok (foldRoomsEC ms rooms)
        | .bloomed .. => .error s!"modalMix '{id}': a bloomed source has a pitch-bloom warp pole-union can't merge (v1) — cross it through a reverb before mixing"
      let ms0 ← modesOf first.bank
      let union ← rest.foldlM (fun acc p => do
        let modes ← modesOf p.bank
        pure (acc ++ modes)) ms0
      .ok { first with bank := .plain union #[], modeCount? := none }
  | _ => .error s!"a modal inlet (reverb/modal-mix) needs a modal SOURCE — resonator or modal-mix — but '{id}' is a signal node; a Sig has no poles to compose"
termination_by r
decreasing_by all_goals exact h

mutual
/-- The gated `Sig`-side recursion step: rank lookup, dangling refusal
    (today's message), and the strict-decrease check that carries the
    whole mutual block's termination. -/
def lowerInputGated (g : PatchGraph) (rankOf : String → Option Nat)
    (srcId i : String) (r : Nat) : Except String ArrowTerm := do
  match rankOf i with
  | none => throw s!"lower: node '{i}' not found"
  | some ri =>
    if h : ri < r then lowerInput g rankOf i ri
    else throw (cycleGuardMsg srcId i)
termination_by (r, 0)
/-- Lower one Sig node to its arrow term, recursing UP its input wires via
    `lowerInput` (which realizes a modal input at its edge). A wire is `⋙`; fan-out
    is the shared upstream term; a mixer is the sum; `fm` routes its input's signal
    into the carrier's clock. Result is the UNREDUCED downstream term. -/
def lowerNode (g : PatchGraph) (rankOf : String → Option Nat)
    (id : String) (r : Nat) : Except String ArrowTerm := do
  let some pn := g.nodes.find? (·.id == id)
    | .error s!"lower: node '{id}' not found"
  match pn.node with
  | .source v clk => .ok (.gen v id clk)
  | .flange inId back fwd =>
    return flangeEffectWith back fwd (← lowerInputGated g rankOf id inId r)
  | .shaper inId f => return .arrUn f (← lowerInputGated g rankOf id inId r)
  | .warpFx inId φ => return .warp φ (← lowerInputGated g rankOf id inId r)
  | .mix inputs => return .sum (← inputs.mapM (lowerInputGated g rankOf id · r))
  | .fm inId carrier base depth =>
    return .swarp (fmWarp depth) (← lowerInputGated g rankOf id inId r) (.gen carrier id base)
  | .pm inId carrier base depth =>
    return .pmGen carrier id base depth (← lowerInputGated g rankOf id inId r)
  | .sflange inId modId depthSec =>
    return sweptFlangeEffect (sflangeBack depthSec) (sflangeFwd depthSec)
      (← lowerInputGated g rankOf id modId r) (← lowerInputGated g rankOf id inId r)
  | .knob value => .ok (.konst value)
  | .comb inId taps => do
    let s ← lowerInputGated g rankOf id inId r
    return .sum (taps.map fun (w, φ) => .scale w (.warp φ s))
  | .ring inputs => do
    let terms ← inputs.mapM (lowerInputGated g rankOf id · r)
    match terms.toList with
    | [] => return .konst (lit 0)
    | t :: ts => return ts.foldl (fun acc u => .prod acc u) t
  | .groupedRoom inId profile position => do
    if profile != groupedRoomProfile then
      throw s!"groupedroom '{id}': unsupported profile '{profile}'; expected '{groupedRoomProfile}'"
    let lowered ←
      if inId == "__silence__" then pure silentModal
      else match rankOf inId with
        | none => throw s!"lowerModal: node '{inId}' not found"
        | some ri =>
          if h : ri < r then lowerModal g rankOf inId ri
          else throw (cycleGuardMsg id inId)
    if lowered.addressNode?.isSome then
      throw s!"groupedroom '{id}': the frozen profile requires the source's direct relative clock (an address inlet is not supported)"
    if lowered.direction?.isSome then
      throw s!"groupedroom '{id}': incumbent modal direction is not part of the frozen grouped-room evaluator"
    if lowered.modeCount?.isSome then
      throw s!"groupedroom '{id}': the frozen profile requires all twelve source poles (a live mode count is not supported)"
    match lowered.bank with
    | .bloomed .. =>
      throw s!"groupedroom '{id}': a pitch-bloomed source does not match the frozen source profile"
    | .plain modes rooms =>
      if !rooms.isEmpty then
        throw s!"groupedroom '{id}': connect the modal source directly; ordinary modal rooms are not composed before this terminal"
      validateGroupedRoomModes modes
      return groupedRoomTerm modes lowered.strikeAnchor position lowered.realizationClock
  | .groupedRoomCache profile position clk => do
    if profile != groupedRoomCacheProfile then
      throw s!"groupedroomcache '{id}': unsupported profile '{profile}'; expected '{groupedRoomCacheProfile}'"
    return groupedRoomCacheTerm position clk
  | _ =>
    .error s!"lower: modal node '{id}' reached Sig lowering — realize via lowerInput"
termination_by (r, 1)

/-- Lower an input wire: a modal node realizes its island (pole bank →
    `modalBankTerm` at the master clock) at this edge; a Sig node lowers normally.
    This is the one-directional seam — modal grafts into Sig here, never the
    reverse (a Sig node has no modal input, so lowerModal is never asked for one). -/
def lowerInput (g : PatchGraph) (rankOf : String → Option Nat)
    (id : String) (r : Nat) : Except String ArrowTerm := do
  if nodeIsModal g id then
    let modal ← lowerModal g rankOf id r
    -- realize the lowered bank at THIS edge (the one-directional Modal→Sig seam).
    let term ← match modal.bank with
      | .plain ms rooms =>
        -- a DIRECTION rotates the poles and reads them through the per-mode
        -- `sign(σ·d)` gate (`modalBankTermDir`), else the plain forward causal
        -- bank. `count?` (trip-count-as-data) is the bank's live mode count.
        -- The deferred room chain folds HERE, partitioned (`couplingHot` routes
        -- near-coincident couplings to the paired DD body; a cold chain emits
        -- today's collected term verbatim). Direction is a collected surface
        -- in v1 (its pole-rotation machinery has no paired form) — a hot
        -- coupling under a direction stays collected, status quo.
        .ok (match modal.direction? with
          | none =>
            let (plain, paired) := foldRoomsPartitioned ms rooms
            modalBankTermPartitioned plain paired modal.strikeAnchor
              modal.realizationClock modal.modeCount?
          | some d => modalBankTermDir (foldRoomsEC ms rooms) modal.strikeAnchor
              modal.realizationClock d.dir d.damp modal.modeCount?)
      | .bloomed voice room B gr =>
        -- the bare bloomed source: the bloom-warped bank (identical to a gong
        -- register's `warpFx`-around-`modalSource`). Also the graceful fallback
        -- when the crossing produces nothing (a live pole outside WS-LP's lift —
        -- live voice/ω, live σ without a declared range, a glided σ — or every
        -- pair fell outside admission) — legal, no warning. A live-rt60 room
        -- with `sigmaRange` DOES cross (WS-LP): its serOnly pairs lift to s0
        -- expressions of the live pole.
        let bare := ArrowTerm.warp (bloomWarpClock modal.strikeAnchor B gr)
          (modalBankTerm voice modal.strikeAnchor modal.realizationClock modal.modeCount?)
        if room.isEmpty then .ok bare
        else match bloomCompose voice room B gr with
          | some pairs =>
            if pairs.isEmpty then .ok bare
            else .ok (bloomComposedTerm pairs modal.strikeAnchor modal.realizationClock)
          | none => .ok bare
    match modal.addressNode? with
    | none => .ok term
    -- a patched address signal BECOMES the bank's clock (absolute warp): the
    -- signal is lowered as an ordinary Sig here at the seam, then `modalAddrWarp`
    -- routes it into the bank's clock leaf. The master scrub still reaches the
    -- bank through this modulator (it too rides the enclosing clock transform).
    | some addrId => .ok (.swarp modalAddrWarp (← lowerInputGated g rankOf id addrId r) term)
  else lowerNode g rankOf id r
termination_by (r, 2)
end

/-- Lower from an arbitrary node id — THE total entry. Interns the graph,
    refuses cycles with the loop named (the topo sort IS the cycle check,
    and it covers every caller — hand-built `PatchGraph`s included, which
    used to bypass the playground's separate pre-pass), then lowers with
    the Kahn rank as the termination measure. -/
def lowerAt (g : PatchGraph) (rootId : String) : Except String ArrowTerm := do
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
def lowerGraph (g : PatchGraph) : Except String ArrowTerm := lowerAt g g.output

end Tropical.EmitArrow
