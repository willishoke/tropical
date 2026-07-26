import Tropical.EmitArrow.Sig

/-!
# EmitArrow.Term — voices, the cartesian surface, and the reified arrow

`Voice` (the primitive morphism `Clock ⇝ Sig`), `Builder.osc`, the cartesian
combinator surface (`Mor`: `seq`/`fan`/`par`/`first`/`instMor` — products,
diagonal, composition; no exponentials, no runtime closures), and `ArrowTerm`,
the inspectable arrow AST whose `normalize`/`emitTermC` realize THE SLIDE:
a warp presented downstream of a signal is pushed up through the stateless
cone until it lands as a generator's clock argument. Downstream effect
morphisms (`flangeEffectWith`, `sweptFlangeEffect`) live here too.
-/

namespace Tropical.EmitArrow

open Tropical.Ir

-- ─────────────────────────────────────────────────────────────
-- A "warp" is just a clock EXPRESSION (no separate algebra)
-- ─────────────────────────────────────────────────────────────

/-! There is no `Warp` type. The clock is a first-class expression and a warp is
    any operation applied to it, drawn from the SAME operation set used on values
    — one algebra, not a clock algebra distinct from the value algebra:

    * `reverse  = neg clk`          (the same `neg` that negates a signal)
    * `delay δ  = sub clk δ`
    * `advance δ = add clk δ`
    * `timescale k = mul clk k`
    * `modulated δ(t) = sub clk (m clk)`  for any signal `m`  (i.e. PM)

    all just clock expressions. No constructor enumerates them and none can be
    left out; invertibility never enters (you only ever apply operations, never
    invert — `reverse` is `neg`, not an inverse). A combinator generic over
    "which warp" takes a `Clock → Clock` (an operation on the clock), e.g.
    `fun c => sub c δ`. -/

-- ─────────────────────────────────────────────────────────────
-- The `Voice` primitive — `Clock ⇝ Sig`, generic over the stdlib closed-form
-- ─────────────────────────────────────────────────────────────

/-- A **voice** is the primitive morphism `Clock ⇝ Sig`, made generic over the
    stdlib program that realizes it. `osc` does NOT hand-roll a phasor +
    polynomial — it references the elaborated voice body and lets strata inline
    it, which is what buys byte-identity with the hand-written programs.

    * `programName` — the stdlib program (`FixedSinOsc`, `ModalVoice`, …).
    * `wire` — how to fill the voice instance's inputs from the (warped) clock.
      This captures the per-voice port ORDER and which port is the clock vs the
      pitch (`FixedSinOsc`: `freq→0, clk→1`; `ModalVoice`: `clk→0, f0→1`).
    * `output` — which voice output carries the signal (both voices: index 0). -/
structure Voice where
  programName : String
  wire : Clock → Array AInput
  output : OutputIdx := ⟨0⟩
  /-- The phase-anchor hook (the slide in the phase domain). When set to
      `(phasePort, corr)`, the SLIDE (`emitTermC`) adds `corr shift` to this voice's
      `phasePort` input for each warped copy, where `shift = clk − warpedClk` is the
      clock shift the slide pushed onto that copy. Since phase = inc·clk, a clock
      shift maps to a phase shift via `inc`, so threading the anchor across the
      cartesian copies keeps every warped read (delay/flange tap) phase-continuous
      across a live freq change — not just the un-warped read. `none` = no
      correction (byte-gated voices are untouched). -/
  phaseAnchor : Option (InputIdx × (Clock → Sig)) := none

/-- Accumulates the instance declarations the `osc` morphisms emit. The
    `InstanceIdx` of a declared voice is its position here, which is what the
    summed output reads back. -/
structure Builder where
  decls : Array AInst := #[]
deriving Inhabited

/-- `osc` — emit one `Voice` instance named `name`, wired from the warped clock
    `clkE`, and return its signal output (`nestedOut`). Generic over the voice:
    the program name, the input wiring, and the output index all come from `v`. -/
def Builder.osc (b : Builder) (v : Voice) (name : String) (clkE : Clock) :
    Sig × Builder :=
  let inst : AInst := { name, programName := v.programName, inputs := v.wire clkE }
  let i := b.decls.size
  (.nestedOut ⟨i⟩ v.output, { b with decls := b.decls.push inst })

-- ─────────────────────────────────────────────────────────────
-- PRODUCTS / MIMO — the cartesian combinator surface (the DATA axis)
-- ─────────────────────────────────────────────────────────────

/-! `warp` is the CLOCK axis; this is the orthogonal DATA axis. Post-strata, a
    "value" is not one scalar but a *bundle* of scalar wires — a categorical
    PRODUCT. A morphism `A ⇝ B` is then a function from a wire-bundle to a
    wire-bundle that accretes the instance decls it sources along the way:

      `Mor := tuple of input wires → Builder → (tuple of output wires, Builder)`

    The structure is **cartesian, not closed**: it has products (concatenation
    of bundles), the diagonal (duplication), and composition — but no
    exponentials, no closures, no runtime higher-order programs. Every
    combinator is a SMART CONSTRUCTOR that emits the post-strata scalar DAG at
    build time; none survives to run time. Crucially `⋙` (composition) **is**
    `inlineInstances`: `g ⋙ f` feeds f's output wires straight into g, building
    one flattened DAG with no instance boundary between them — the combinator
    layer's image of the strata pass it absorbs. (Hughes' `first` is the whole
    of MIMO; the named-port ⟷ tuple bridge — `instMor` — is its primitive.) -/

/-- A cartesian morphism in the wire category: consumes a product of input
    wires, accretes instance decls, produces a product of output wires. Objects
    are wire-bundle ARITIES (the splitter combinators carry them explicitly,
    since the category is untyped-but-arity-indexed). -/
abbrev Mor := Array Sig → Builder → (Array Sig × Builder)

/-- Identity (`arr id`) — passes its whole bundle through untouched. Absorbed
    by `identityElim` at construction: `idMor ⋙ f = f` holds definitionally. -/
def idMor : Mor := fun xs b => (xs, b)

/-- Sequential composition `g ⋙ f` (read left-to-right): the LEFT runs first and
    its outputs feed the right. THIS is inlining — there is no instance boundary
    between the two, just one threaded DAG (the absorbed `inlineInstances`). -/
def seq (f g : Mor) : Mor := fun xs b => let (ys, b) := f xs b; g ys b

/-- Fan-out / the cartesian diagonal `f &&& g`: run BOTH on the same input,
    concatenating their outputs. The Builder threads f-then-g, so the sourced
    instances get deterministic indices. The shared input bundle is the diagonal
    `Δ` — at the Lean level it is plain value reuse, which is exactly the
    post-strata DAG's shared sub-node. -/
def fan (f g : Mor) : Mor := fun xs b =>
  let (ys, b) := f xs b
  let (zs, b) := g xs b
  (ys ++ zs, b)

/-- Parallel product `f *** g`, split at arity `m`: `f` consumes the first `m`
    wires, `g` the rest; outputs concatenate. The bifunctor on products. -/
def par (m : Nat) (f g : Mor) : Mor := fun xs b =>
  let (ys, b) := f (xs.extract 0 m) b
  let (zs, b) := g (xs.extract m xs.size) b
  (ys ++ zs, b)

/-- `first f` (Hughes): apply `f` to the first `m` wires, pass the remaining
    bundle through unchanged. `first m f = par m f idMor`. The single primitive
    from which all of MIMO is generated. -/
def first (m : Nat) (f : Mor) : Mor := par m f idMor

/-- `second g`: dual of `first` — pass the first `m` wires through, apply `g` to
    the rest. -/
def second (m : Nat) (g : Mor) : Mor := par m idMor g

/-- The diagonal `Δ : A ⇝ A × A` — duplicate the whole bundle. -/
def dup : Mor := fun xs b => (xs ++ xs, b)

/-- Left projection `π₁` — keep the first `m` wires, drop the rest. -/
def exl (m : Nat) : Mor := fun xs b => (xs.extract 0 m, b)

/-- Right projection `π₂` — drop the first `m` wires, keep the rest. -/
def exr (m : Nat) : Mor := fun xs b => (xs.extract m xs.size, b)

/-- Lift a pure wire-bundle function into a morphism (`arr`, RESTRICTED): no
    instances, just structural/arithmetic rewiring of scalar `Sig`s. This is
    the `arr` of a cartesian (not closed) arrow — a fixed structural map, never
    an arbitrary host closure. -/
def arrMor (f : Array Sig → Array Sig) : Mor := fun xs b => (f xs, b)

/-- THE NAMED-PORT ⟷ TUPLE BRIDGE — the MIMO primitive. Instantiate program
    `programName` (a named multi-port morphism `A ⇝ B`) by assigning the input
    wire bundle to ports `portOrder` positionally, reading its `numOut` outputs
    back as a bundle. The categorical content: a named multi-port instance IS a
    morphism between products; this bridge is the iso between its named (record)
    presentation and the positional (tuple) one. Emits a COARSE instance — `⋙`
    and the lowering's `inlineInstances` flatten it away; only the combinator surface
    survives to the emitted bytes. -/
def instMor (name programName : String) (portOrder : Array InputIdx)
    (numOut : Nat) : Mor := fun args b =>
  let inputs : Array AInput :=
    (portOrder.zip args).map (fun (p, v) => ⟨p, v⟩)
  let i := b.decls.size
  let inst : AInst := { name, programName, inputs }
  let outs : Array Sig := (Array.range numOut).map (fun o => .nestedOut ⟨i⟩ ⟨o⟩)
  (outs, { b with decls := b.decls.push inst })

/-- The root clock `sampleIndex << 32` as a closed-form value — the inlined
    `clkInputDecl` default, so the carrier needs no `clk` input port. -/
def clockLit : Clock := .binary .lshift .sampleIndex (lit 32)

-- ─────────────────────────────────────────────────────────────
-- The SLIDE (WARP-PUSH): a REIFIED arrow term + the τ-push rewrite
-- ─────────────────────────────────────────────────────────────

/-! Everything above is the SMART-CONSTRUCTOR (deep-by-emission) layer: the
    combinators build the IR directly, with nothing to inspect or rewrite. The
    slide needs the opposite — a REIFIED arrow term it can pattern-match — because
    WARP-PUSH is a *rewrite*: it takes an effect presented DOWNSTREAM (`warp`
    applied to a signal) and pushes the warp UP through the stateless cone until
    it lands as a generator's clock argument. This is where the COMPILER performs
    downstream→upstream: the user patches an effect after a signal, the emitted
    kernel reads the generators at warped clocks.

    `ArrowTerm` is a tiny inspectable arrow AST. The only non-trivial node is
    `warp φ t` — `warp(φ) ⋙ t`, kept UNREDUCED. `normalize` (the slide) is three
    law-justified rules plus the fork over products:

      * `warp φ ⋙ arr f      ⟶ arr f ⋙ warp φ`     (slide past a pointwise node, R1)
      * `warp φ ⋙ warp ψ     ⟶ warp (φ∘ψ)`          (fuse, R2)
      * `warp φ ⋙ gen[clk:=c] ⟶ gen[clk:=φ c]`       (absorb into the clock, R4)
      * `warp φ ⋙ (a + b)    ⟶ (warp φ ⋙ a) + (warp φ ⋙ b)`, and through `scale`
                                                     (fork over the product, R3)

    Crucially the warp/arr COMMUTATION (R1) is the one law here that the arrow
    structure makes true and that plain `let`/composition does NOT give you for
    free — this is where the EDSL stops being re-spelled `let` and starts
    earning its keep. The denotation is `⟦warp φ t⟧ = ⟦t⟧ ∘ φ`, so the slide is
    just ∘-associativity. -/

/-- A reified, inspectable arrow term. `gen` is a voice instance whose clock arg
    is the warp target; `warp φ t` is `warp(φ) ⋙ t` kept unreduced; `scale`/`arrUn`
    are pointwise (clock-agnostic) `arr`s; `sum` is the left-assoc weighted sum
    (the cartesian product collapse). Functions are carried opaquely — the slide
    composes/applies them, never inspects them. -/
inductive ArrowTerm where
  | gen (v : Voice) (name : String) (clk : Clock)
  | warp (φ : Clock → Clock) (t : ArrowTerm)
  | scale (w : Sig) (t : ArrowTerm)
  | arrUn (f : Sig → Sig) (t : ArrowTerm)
  | sum (ts : Array ArrowTerm)
  /-- A SIGNAL-dependent warp — the data-into-clock edge, made first-class. Bends
      the clock of `t` by `mw baseClk modSig`, where `modSig` is the SIGNAL of
      `modulator` (another sub-term, a function of τ, so the warp stays a closed
      form — the PM-of-PM lawfulness). A WRAPPER (like `warp`), so signal-warps
      compose with plain warps and with each other — `swarp ∘ swarp`,
      `warp ∘ swarp`, all nest; an FM voice is just
      `swarp mw mod (gen carrier base)`. The slide (in `emitTermC`) distributes
      it onto the generators `t` feeds. -/
  | swarp (mw : Clock → Sig → Clock) (modulator : ArrowTerm) (t : ArrowTerm)
  /-- A τ-CONSTANT leaf — a bare signal `s` that does not read the clock (a param
      slot read `paramRef`, or any literal). It has no generator, so the slide's
      clock transform threads STRAIGHT PAST it (warping a constant is the
      constant). This is what a "knob" node lowers to: its value is a module slot,
      wireable into any modulator (`swarp`/`fm`) or generator-pitch position, and
      driven live by `set_param` — no per-sample state, so no relower needed. -/
  | konst (s : Sig)
  /-- The pointwise ring PRODUCT of two sub-terms — `x ⊗ y`, the multiplicative
      dual of `sum`. `scale` multiplies a term by a fixed value `w`; `prod`
      multiplies two TERMS, each with its own generators. The slide law holds
      because pre-composition distributes over pointwise ×: `warp φ (x ⊗ y) =
      (x ∘ φ)·(y ∘ φ) = (warp φ x) ⊗ (warp φ y)`, so `emitTermC` threads the SAME
      clock transform into BOTH factors — a downstream warp reclocks both. This is
      the VCA / amplitude-multiply that `scale` cannot express: a warp on
      `scale w t` leaves `w` un-reclocked, but a warp on `prod x y` reclocks `x`
      AND `y`, so an envelope factored as its own term rides every delay tap. -/
  | prod (x y : ArrowTerm)
  /-- The CLOCK LEAF — the one atom warp actually acts on. Its signal IS the
      (warped) clock: `emitTermC` applies the ambient clock transform to `c` and
      hands the result back as a `Sig`. Everything periodic is built on this — a
      phasor is pointwise arithmetic over `clk`, and because the leaf is warped,
      so is every oscillator built from it (reverse, scrub, future-tap for free).
      This is what lets the stdlib's generators be TERMS over `{clk, +, ×, frac}`
      instead of opaque `.trop` instances — the bootstrap's ground floor. -/
  | clk (c : Clock)
  /-- THROUGH-ZERO PHASE MODULATION — the data-into-PHASE edge (the dual of
      `swarp`'s data-into-clock FM). A single carrier `v` whose phase port
      receives `depth · modSig` added on top of its anchor correction, where
      `modSig` is the SIGNAL of `mod` (another sub-term, a closed form in τ).
      Unlike FM it does NOT bend the carrier's clock, so there is no pitch droop
      — the sidebands sit symmetrically about a fixed carrier. `mod` is emitted
      through the SAME enclosing clock transform, so a downstream warp reclocks
      the modulator too (the PM-of-PM / delay-continuity rule, as for `swarp`). -/
  | pmGen (v : Voice) (name : String) (baseClk : Clock) (depth : Sig)
      (mod : ArrowTerm)

instance : Inhabited ArrowTerm := ⟨.sum #[]⟩

/-- Normalize a term's sub-structure (the modulators, the branches). The SLIDE
    itself — pushing every `warp`/`swarp` onto the generators it feeds — is
    realized in `emitTermC`, which threads the composed clock transform down to
    each generator. Keeping warps as WRAPPERS here (rather than eagerly fusing
    them into generator clocks) is exactly what makes the algebra TOTAL: the slide
    is function composition of clock transforms, so `warp ∘ warp`, `warp ∘ swarp`,
    and `swarp ∘ swarp` all compose — there is no case it cannot reduce. -/
def normalize : ArrowTerm → ArrowTerm
  | .gen v name clk => .gen v name clk
  | .scale w t => .scale w (normalize t)
  | .arrUn f t => .arrUn f (normalize t)
  | .sum ts => .sum (ts.attach.map fun ⟨x, _⟩ => normalize x)
  | .warp φ t => .warp φ (normalize t)
  | .swarp mw mod t => .swarp mw (normalize mod) (normalize t)
  | .konst s => .konst s
  | .prod x y => .prod (normalize x) (normalize y)
  | .clk c => .clk c
  | .pmGen v name baseClk depth mod => .pmGen v name baseClk depth (normalize mod)

/-- Emit a (normalized) arrow term to its output signal. Each `gen` sources a
    voice instance in left-to-right order (matching `warpBank`'s instance order);
    `scale`/`arrUn` are the pointwise ops; `sum` is the left-assoc fold
    `((t₀ + t₁) + t₂)`. Instance names are uniquified by position (names are
    inlined away post-strata, so they never reach the emitted bytes). -/
def emitTermC (cmod : Clock → Builder → Clock × Builder) :
    ArrowTerm → Builder → Sig × Builder
  | .gen v name clk, b =>
    let (clk', b) := cmod clk b
    -- Thread the phase anchor across this warped copy: the slide shifted the clock
    -- by `clk − clk'`, so add `corr (clk − clk')` to the voice's phase port. Keeps
    -- delay/flange taps phase-continuous under a live freq change (not just the
    -- un-warped read). A `phaseAnchor := none` voice passes through untouched.
    let v := match v.phaseAnchor with
      | some (port, corr) =>
        let c := corr (sub clk clk')
        { v with wire := fun cc => (v.wire cc).map fun ii =>
            if ii.port.idx == port.idx then { ii with value := add ii.value c } else ii }
      | none => v
    b.osc v s!"{name}{b.decls.size}" clk'
  | .scale w t, b => let (s, b) := emitTermC cmod t b; (mul w s, b)
  | .arrUn f t, b => let (s, b) := emitTermC cmod t b; (f s, b)
  -- a plain warp composes into the threaded transform (R1/R2/R4 in one line).
  | .warp φ t, b => emitTermC (fun c b => cmod (φ c) b) t b
  -- a signal warp: source the modulator (pinned through the SAME enclosing `cmod`,
  -- so a downstream warp reclocks it too — the lawful PM-of-PM rule), read its
  -- signal, then bend the clock by `mw` before the enclosing transform.
  | .swarp mw mod t, b =>
    emitTermC (fun c b =>
      let (mSig, b) := emitTermC cmod mod b
      cmod (mw c mSig) b) t b
  -- a τ-constant leaf: no generator to reclock, so the clock transform `cmod` is
  -- discarded and the bare signal is emitted as-is.
  | .konst s, b => (s, b)
  -- the ring product: thread the SAME clock transform into both factors (the slide
  -- distributes over ×), emit both signals, multiply. A downstream warp thus
  -- reclocks x and y together — the amplitude/VCA multiply `scale` can't give.
  | .prod x y, b =>
    let (sx, b) := emitTermC cmod x b
    let (sy, b) := emitTermC cmod y b
    (mul sx sy, b)
  -- the clock leaf: apply the ambient clock transform and hand the warped clock
  -- back AS the signal (Clock and Sig are both `Sig`). This is the one leaf warp
  -- acts on; a phasor over it inherits reverse/scrub/future-tap.
  | .clk c, b => cmod c b
  -- through-zero PM: emit the modulator's signal (reclocked through the same
  -- ambient transform), then source THIS carrier with `depth·modSig` added to its
  -- phase port — on top of the anchor's clock-shift correction. The carrier's
  -- clock is NOT bent, so the carrier pitch stays put (no FM droop).
  | .pmGen v name baseClk depth mod, b =>
    let (mSig, b) := emitTermC cmod mod b
    let (clk', b) := cmod baseClk b
    let pmOff := mul depth mSig
    let v := match v.phaseAnchor with
      | some (port, corr) =>
        let c := add (corr (sub baseClk clk')) pmOff
        { v with wire := fun cc => (v.wire cc).map fun ii =>
            if ii.port.idx == port.idx then { ii with value := add ii.value c } else ii }
      | none => v
    b.osc v s!"{name}{b.decls.size}" clk'
  | .sum ts, b =>
    -- emit every summand in order, then fold the left-assoc sum — the
    -- same builder sequence and Sig tree as the old head/tail fold.
    let (sigs, b) := ts.attach.foldl
      (fun (acc : Array Sig × Builder) ti =>
        let (s, b') := emitTermC cmod ti.1 acc.2
        (acc.1.push s, b'))
      ((#[] : Array Sig), b)
    match sigs[0]? with
    | none => (lit 0, b)
    | some s0 => ((sigs.extract 1 sigs.size).foldl add s0, b)
termination_by t _ => sizeOf t
decreasing_by
  all_goals first
    | decreasing_tactic
    | (have := Array.sizeOf_lt_of_mem ti.2; simp_all; omega)

/-- Emit a (normalized) term at the identity clock context — the public entry. -/
def emitTerm (t : ArrowTerm) (b : Builder) : Sig × Builder :=
  emitTermC (fun c b => (c, b)) t b

/-- A 3-tap flanger as a DOWNSTREAM-PRESENTED effect (a morphism on terms):
    `s ↦ 0.5·s + 0.25·(warp(−δ) ⋙ s) + 0.25·(warp(+δ) ⋙ s)`. The warps are
    UNREDUCED — they sit on the signal `s`, exactly as "I dropped a flanger after
    this signal" reads. The slide is what turns this into the upstream form. -/
def flangeEffectWith (back fwd : Clock → Clock) (s : ArrowTerm) : ArrowTerm :=
  .sum #[ .scale (lit 5 1) s,
          .scale (lit 25 2) (.warp back s),
          .scale (lit 25 2) (.warp fwd s) ]

/-- A SWEPT delay offset: `δ(τ) = toInt(seconds · m(τ) · sampleRate · 2³²)` — the
    static flanger's δ, but the depth is scaled by a modulator signal `m ∈ [−1,1]`,
    so the comb delay sweeps (through zero, as `m` crosses 0). -/
def sweptDelta (secondsE : Sig) (m : Sig) : Sig :=
  toIntE (mul (mul (mul secondsE m) .sampleRate) (lit 4294967296))

def sflangeBack (secondsE : Sig) : Clock → Sig → Clock := fun c m => sub c (sweptDelta secondsE m)
def sflangeFwd (secondsE : Sig) : Clock → Sig → Clock := fun c m => add c (sweptDelta secondsE m)

/-- A SWEPT (through-zero) flanger as a DOWNSTREAM effect: like `flangeEffect`, but
    the ±δ taps are SIGNAL-modulated (`swarp`) by `mod` — a modulator term (an LFO,
    or any patched signal). The slide distributes the signal-warp onto the input's
    generators (each becomes a modulated carrier); the dry tap is unwarped. Because
    `swarp` is a wrapper, this stacks freely: a swept flange after a plain flange,
    after another swept flange, all compose — totality, not a special case. -/
def sweptFlangeEffect (back fwd : Clock → Sig → Clock) (mod s : ArrowTerm) : ArrowTerm :=
  .sum #[ .scale (lit 5 1) s,
          .scale (lit 25 2) (.swarp back mod s),
          .scale (lit 25 2) (.swarp fwd mod s) ]

end Tropical.EmitArrow
