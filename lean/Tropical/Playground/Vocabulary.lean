import Tropical.EmitArrow
import Tropical.Ir.Strata
import Tropical.Ir.Core
import Tropical.Compile
import Tropical.StdlibChain
import Lean.Data.Json

/-!
# `load_patch_graph` — the playground's live arrow entry point (EXPERIMENT)

A downstream-only patch graph (the PureData-style playground GUI) decoded into the
`EmitArrow` `PatchGraph`, lowered through the SAME `lowerGraph → normalize → emitTerm`
the corpus gates exercise, wrapped as a session root, and compiled to a `FlatPlan`
via `compileSession` — the production loadable tail. The slide (`normalize`) pushes
each effect's warps up onto the generators' clocks, so a `flange`/`fm` dropped
downstream of an oscillator genuinely re-clocks that oscillator.

Continuous knobs are LIVE `param:<id>.<knob>` module slots — a knob change drives
`set_param` on the running kernel with no recompile (the modal path emits `paramRef`
via `pref`; the earlier "all knobs baked" design is retired). Only STRUCTURAL edits —
topology, mode-bank size, and baked strike data (gong/string mode tables, anchors) —
re-send the whole graph and hot-swap. Clickless either way, since there is no
per-sample state to carry.

Uncommitted: this file makes `EmitArrow` reachable from the live `frontend`.
-/

namespace Tropical.Playground

open Lean (Json JsonNumber)
open Tropical.Ir
open Tropical.EmitArrow
open Tropical.Exact (DyadicI)

-- ── JSON field helpers (over `Lean.Json` objects) ───────────────────────────
def jNum? (obj : Json) (key : String) : Option JsonNumber :=
  match obj.getObjVal? key with
  | .ok j => j.getNum?.toOption
  | .error _ => none

/-- A numeric param as a scalar `Sig` (`Sig.num`), carrying the JSON decimal
    straight through (mantissa · 10^-exponent). -/
def jExpr (obj : Json) (key : String) (dflt : Sig) : Sig :=
  match jNum? obj key with
  | some n => lit n.mantissa n.exponent
  | none => dflt

/-- A numeric param truncated to `Int` (the `fm` depth, in samples). -/
def jInt (obj : Json) (key : String) (dflt : Int) : Int :=
  match jNum? obj key with
  | some n => if n.exponent == 0 then n.mantissa else n.mantissa / ((10 : Int) ^ n.exponent)
  | none => dflt

def jStr (obj : Json) (key : String) (dflt : String) : String :=
  match obj.getObjVal? key with
  | .ok (.str s) => s
  | _ => dflt

/-- A numeric param as a build-time `Float`. RETAINED only for the readers that
    hand a `Float` to a function whose signature this pass does not reach
    (`defaultGongModes`, and `bloomgong`'s withheld `(Float × Float)` bloom
    pair). Every other bake-time reader takes `jDec`/`jExactD` below: a `Float`
    here throws the exact decimal away before a structural decision can read it,
    and `JsonNumber.toFloat` inherits core's double rounding. -/
def jFloat (obj : Json) (key : String) (dflt : Float) : Float :=
  ((jNum? obj key).map (·.toFloat)).getD dflt

/-- A numeric param as its exact DECIMAL `(mantissa, exponent)` — `m·10^{−e}`,
    the shape `JsonNumber` already carries, with no `Float` in between. Read from
    a PARSED number and defaulted with a plain `(Int × Nat)` tuple, never with a
    `⟨m, e⟩ : JsonNumber` source literal: the linux-x86 miscompile that made every
    carrier sink unity-gain on CI bites SOURCE literals only. -/
def jDec (obj : Json) (key : String) (dflt : Int × Nat) : Int × Nat :=
  match jNum? obj key with
  | some n => (n.mantissa, n.exponent)
  | none => dflt

/-- A decimal `(m, e)` as its certified enclosure `m·10^{−e}`. Decimals are not
    dyadic, so this is where the authoring layer's exactness genuinely ends —
    and the enclosure says so, to the working precision, instead of pretending. -/
def decD (d : Int × Nat) : DyadicI :=
  DyadicI.ofJsonNumber ⟨d.1, d.2⟩

/-- A numeric param as a certified enclosure. -/
def jExactD (obj : Json) (key : String) (dflt : Int × Nat) : DyadicI :=
  decD (jDec obj key dflt)

/-- The emit funnel on a CERTIFIED value: `litF` of the enclosure's midpoint —
    the nearest `Float` to the exact value. `litF`'s 12-decimal quantization and
    its own f64 multiply stay exactly where they are (the `litF` FORMAT is a
    separate later decision with its own one-time golden migration); all that
    changes is the PROVENANCE of the double it rounds — a correctly-rounded value
    rather than a platform `libm`'s. Poison is `none`, never a fabricated `0`:
    that is the `sigConstF?` pathology, one floor down. -/
def litOfD? (x : DyadicI) : Option Sig :=
  if x.ok then some (litF x.toFloat) else none

/-- `litOfD?` at a site where poison is unreachable BY CONSTRUCTION (every
    argument is a `sin`/`cos` of a finite enclosure, or a `pow`/`log` of a
    certifiably positive one) AND the bank's LENGTH is contractual — a
    resonator's `partials_max` capacity, a reverb's `nmode` — so dropping a mode
    is not an available answer. The `lit 0` arm is therefore dead code, and the
    `exact-playground` gate asserts it STAYS dead over the whole served
    vocabulary rather than trusting this sentence. -/
def litOfD (x : DyadicI) : Sig := (litOfD? x).getD (lit 0)

/-- The AUTHORED `2π`, `π`, golden ratio and golden angle as exact DECIMALS —
    the same numbers the incumbent `Float` literals spell, entering the carrier as
    decimals rather than as the doubles nearest them. Deliberately NOT
    `Tropical.Exact.{twoPiI, piI}`: swapping an authored rounding for the true
    constant is a VALUE change, and this campaign moves the arithmetic. `twoPiD`
    in particular must agree with the symbolic `twoPiE` (`lit 6283185307179586
    e−15`, Numerics.lean) or one emitted plan would carry two spellings of 2π. -/
def twoPiD : DyadicI := decD (6283185307179586, 15)
def piD : DyadicI := decD (3141592653589793, 15)
def goldenRatioD : DyadicI := decD (6180339887, 10)
def goldenAngleD : DyadicI := decD (2399963, 6)

/-- `ln 80`, certified once at module init (the `eulerI` precedent) — the
    constant `filterPair`'s `Q = 0.55·80^res` mapping is written in terms of.
    MEASURED: the exact value's nearest double IS the authored literal
    `4.382026634673881` and its 12-place quantization is identical, so this is a
    provenance change with no value change. -/
def ln80D : DyadicI := DyadicI.log (DyadicI.ofNat 80)

/-- `ln 80` as the emitted literal `filterPair` consumes. Named so the
    `exact-playground` probe can hand the gate THIS `Sig` rather than a second
    spelling of it — `filterPair`'s own `q` is wrapped in an `expSig`, which does
    not fold, so the constant has to be reachable on its own to be observable at
    all. -/
def lnEightyLit : Sig := litOfD ln80D

/-- A gong register's mode table: an array of `[freqHz, sigma, amp, phase]`
    rows → `ModalMode`s (rectangular: `cre = a·cos φ`, `cim = a·sin φ`, so
    the bank's `cre·cos ωd − cim·sin ωd` is `a·cos(ωd + φ)`). Amplitude-bloom
    pairs arrive pre-expanded (two rows, `±a`, two σ). Malformed rows drop. -/
def jModes (obj : Json) (key : String) : Array ModalMode :=
  match (obj.getObjVal? key).toOption.bind (·.getArr?.toOption) with
  | none => #[]
  | some arr => arr.filterMap fun mj =>
    match mj.getArr?.toOption with
    | some fs =>
      if fs.size < 4 then none else
      -- each cell enters as the DECIMAL the score wrote, not as the double
      -- nearest it: `JsonNumber.toFloat` double-rounds (the same conversion the
      -- CI miscompile incident was about), and these rows are a gong's or a
      -- string's whole timbre.
      let numD := fun (i : Nat) =>
        match fs[i]!.getNum?.toOption with
        | some n => DyadicI.ofJsonNumber n
        | none => DyadicI.zero
      let a := numD 2
      let ph := numD 3
      some { sigma := litOfD (numD 1),
             omega := litOfD (DyadicI.mul twoPiD (numD 0)),
             cre := litOfD (DyadicI.mul a (DyadicI.cos ph)),
             cim := litOfD (DyadicI.mul a (DyadicI.sin ph)) }
    | none => none

-- ── Voices (literal pitch, so the knob bakes into the emitted clock) ─────────
/-- The phase-anchor correction — the slide, in the phase domain. A clock `shift`
    (Q32.32) of `shift/2³²` samples maps to a phase shift of
    `(freq − freqInit)·(shift/2³²)/SR` cycles: the phase the oscillator advances over
    the shift, referenced to the compile-time `freqInit` so the effect's delay is
    preserved as a fixed phase. Added to a warped copy's phase port by `emitTermC`,
    it keeps that copy phase-continuous across a live freq change. -/
def phaseCorr (pitchE freqInit : Sig) : Clock → Sig :=
  fun shift => div (mul (sub pitchE freqInit) (toFloatE shift)) (mul (lit 4294967296) .sampleRate)

/-- The anchor payload: `(phaseSlot, freqInit)`. Present when the voice's freq is a
    live phase-anchored slot; the voice then wires the phase port and installs the
    `phaseAnchor` so every warped copy self-corrects. -/
abbrev Anchor := Sig × Sig

/-- `FixedSinOsc`: freq (port 0), clk (port 1), phase (port 2). -/
def sineVoiceE (pitchE : Sig) (anchor : Option Anchor := none) : Voice :=
  match anchor with
  | some (phaseE, freqInit) =>
    { programName := "FixedSinOsc",
      wire := fun clkE => #[ ⟨⟨0⟩, pitchE⟩, ⟨⟨1⟩, clkE⟩, ⟨⟨2⟩, phaseE⟩ ],
      phaseAnchor := some (⟨2⟩, phaseCorr pitchE freqInit) }
  | none =>
    { programName := "FixedSinOsc",
      wire := fun clkE => #[ ⟨⟨0⟩, pitchE⟩, ⟨⟨1⟩, clkE⟩ ] }

/-- `MorphOsc`: freq (port 0), morph (port 1), clk (port 2), phase (port 3).
    `morph = 0` is saw, `morph = 1` is sine. -/
def morphVoiceE (pitchE morphE : Sig) (anchor : Option Anchor := none) : Voice :=
  match anchor with
  | some (phaseE, freqInit) =>
    { programName := "MorphOsc",
      wire := fun clkE => #[ ⟨⟨0⟩, pitchE⟩, ⟨⟨1⟩, morphE⟩, ⟨⟨2⟩, clkE⟩, ⟨⟨3⟩, phaseE⟩ ],
      phaseAnchor := some (⟨3⟩, phaseCorr pitchE freqInit) }
  | none =>
    { programName := "MorphOsc",
      wire := fun clkE => #[ ⟨⟨0⟩, pitchE⟩, ⟨⟨1⟩, morphE⟩, ⟨⟨2⟩, clkE⟩ ] }

/-- The `source` node is always a `MorphOsc` (morph = 0 is saw, morph = 1 is sine),
    so the morph knob is always meaningful. -/
def voiceOf (pitchE morphE : Sig) (anchor : Option Anchor) : Voice :=
  morphVoiceE pitchE morphE anchor

/-- `PluckedMorphOsc`: freq (0), morph (1), clk (2), event_rate (3), phase (4).
    A `MorphOsc` with the closed-form pluck envelope baked in — dynamic content
    that reverses with the master clock, so any downstream warp (a comb tap) reads
    a delayed PLUCKED copy (an audible echo/pre-echo), not a silent bulk delay. -/
def pluckedVoiceE (pitchE morphE eventRateE : Sig) (anchor : Option Anchor := none) : Voice :=
  match anchor with
  | some (phaseE, freqInit) =>
    { programName := "PluckedMorphOsc",
      wire := fun clkE => #[ ⟨⟨0⟩, pitchE⟩, ⟨⟨1⟩, morphE⟩, ⟨⟨2⟩, clkE⟩, ⟨⟨3⟩, eventRateE⟩, ⟨⟨4⟩, phaseE⟩ ],
      phaseAnchor := some (⟨4⟩, phaseCorr pitchE freqInit) }
  | none =>
    { programName := "PluckedMorphOsc",
      wire := fun clkE => #[ ⟨⟨0⟩, pitchE⟩, ⟨⟨1⟩, morphE⟩, ⟨⟨2⟩, clkE⟩, ⟨⟨3⟩, eventRateE⟩ ] }

/-- `δ = toInt(seconds · sampleRate · 2³²)` — a Q32.32 sample offset (the stdlib
    flanger's own delay form), but with the `seconds` taken from the knob. Signed:
    a negative `seconds` reads the past, a positive one the future. -/
def deltaOf (secondsE : Sig) : Sig :=
  toIntE (mul (mul secondsE .sampleRate) (lit 4294967296))

/-- `gᵏ` as a product of `k` live-`g` reads (`g` may be a `paramRef`), so the comb's
    decay is a live knob with no relower. `k = 0 ⇒ 1`. -/
def gPow (g : Sig) (k : Nat) : Sig :=
  (Array.range k).foldl (fun acc _ => mul acc g) (lit 1)

/-- A struck resonator's modal bank: `npart` harmonics of `f0`, decay
    `σ_k = decay·(1 + 0.4k)`, amplitude `1/k^1.1`. `f0` and `decay` may be live
    `paramRef`s (the pole frequencies/decays sweep with the knobs), because the
    downstream residue calculus is emitted symbolically. -/
def resonatorBank (f0 decay : Sig) (npart : Nat) : Array ModalMode :=
  (Array.range npart).map fun j =>
    let k := j + 1
    let kD := DyadicI.ofNat k
    -- σ factor `1 + 0.4k`: `0.4` is an AUTHORED DECIMAL, so it enters as a tight
    -- enclosure instead of pretending to be the dyadic 0.4000000000000000222…
    let sigFac := DyadicI.add DyadicI.one (DyadicI.mul (decD (4, 1)) kD)
    -- amplitude `k^{−1.1}` — the one transcendental here, now `exp(−1.1·ln k)`
    -- in the carrier. `k ≥ 1 > 0`, so `pow`'s positivity precondition is
    -- certified and poison is unreachable (which is why `litOfD` is total here:
    -- the array LENGTH is the bank's `partials_max` capacity contract, so a
    -- dropped mode is not an available answer).
    let ampD := DyadicI.div DyadicI.one (DyadicI.pow kD (decD (11, 1)))
    ModalMode.hz (mul (lit (Int.ofNat k)) f0)
                 (mul decay (litOfD sigFac))
                 (litOfD ampD)

/-- The default plucked-string bank a bare `string` node strikes: the exact
    Jaffe-Smith pole table of the Karplus-Strong loop `y = x + ρ·½(y[n-N] +
    y[n-N-1])` at fundamental `f0`, closed form — the loop diagonalized, not
    simulated. Mode `k` sits at `f_k = k·SR/(N+½)` (`N = round(SR/f0)`); the
    loop's averaging filter has `|G| = cos(π f_k/SR)`, so one transit multiplies
    partial `k` by `g_k = ρ·|G|` and its decay is `σ_k = −SR·ln(g_k)/(N+½)` —
    highs die faster, which IS the plucked-string sound. Residues use a `1/k`
    displacement rolloff at golden-angle phases (projecting a real burst onto
    the modes is the DATA path's job — this is the audible default). SR is baked
    at 44100 for the default timbre; the `string` kind's real content arrives as
    `modes` data rows. Capped at 48 partials (higher ones decay in ms). -/
def defaultStringModes (f0 rho : Int × Nat) : Array ModalMode := Id.run do
  let srI : Int := 44100
  let (m, e) := f0
  -- `f0 ≤ 0` is not a string: an EMPTY bank — silence, the house's graceful
  -- exclusion. (The incumbent `Float` path reached `44100/0 = ∞` here, whose
  -- `.round + 0.5` is `∞`, whose `Float.floor … |>.toUInt64` SATURATES, so
  -- `{"kind":"string","params":{"freq":0}}` emitted 48 undamped DC modes. Exact
  -- arithmetic has no ∞ to saturate; the change is forced by the carrier, it is
  -- the better answer, and `exact-playground` pins it so it is a decision rather
  -- than a surprise.)
  if m ≤ 0 then return #[]
  -- THE FIRST CLIFF, decided rather than estimated. `N = round(SR/f0)` HALF AWAY
  -- FROM ZERO in exact `Int` arithmetic on the decimal the JSON carried:
  -- `f0 = m·10^{−e}` ⇒ `SR/f0 = SR·10^e/m`, a RATIONAL, and
  -- `round(p/q) = ⌊(2p + q)/(2q)⌋` for `p, q > 0`. No `Float`, no enclosure and
  -- no overlap arm — this cliff does not need a policy, it needs integers.
  -- (MEASURED against the incumbent over 95 000 plausible decimal `f0`: two
  -- disagreements, both `f0 = 2.24`, where `44100/2.24 = 19687.5` exactly but
  -- the f64 quotient is `19687.499999999996`, so `Float.round` answered 19687
  -- and the exact round answers 19688.)
  let p : Int := srI * (10 : Int) ^ e
  let q : Int := m
  let N : Int := (2 * p + q) / (2 * q)   -- p, q > 0 ⇒ every division convention agrees
  if N ≤ 0 then return #[]
  -- span = N + ½, exactly: the dyadic `(2N+1)·2^{−1}`
  let span : DyadicI := (DyadicI.ofInt (2 * N + 1)).shift (-1 : Int)
  -- THE EMITTED PARTIAL COUNT. The incumbent's `Float.floor` was already inert —
  -- `span/2 = N/2 + ¼` is exactly representable, so its floor is `N div 2` on
  -- every platform — and here it is that integer fact and nothing else.
  let kmax : Nat := min 48 (N / 2).toNat
  let rhoD := decD rho
  let srD := DyadicI.ofInt srI
  let halfSR := srD.shift (-1 : Int)
  let mut modes : Array ModalMode := #[]
  for j in [0:kmax] do
    let k := DyadicI.ofNat (j + 1)
    let fk := DyadicI.div (DyadicI.mul k srD) span            -- f_k = k·SR/span
    -- ρ·|G(ω_k)| per loop transit. `π·f_k/SR = π·k/span` exactly, so the
    -- reduction is taken algebraically instead of through two roundings.
    let g := DyadicI.mul rhoD (DyadicI.cos (DyadicI.div (DyadicI.mul piD k) span))
    -- THE SECOND CLIFF — the per-partial EMIT/SKIP fork, certified. Two
    -- conjuncts, one policy: EMIT only on a certified verdict, DROP otherwise.
    --  · `f_k < SR/2` is REDUNDANT — `k ≤ ⌊span/2⌋ = N div 2 < span/2` forces it
    --    for every integer `k`. It is kept, as a certified check rather than a
    --    float one, so a future change to `kmax` cannot silently re-open the
    --    band edge (the `exact-playground` gate proves it never fires today).
    --  · `g > 0` is the live half. `σ = −SR·ln g/span` is UNBOUNDED as `g → 0⁺`,
    --    so an `overlap` verdict (the enclosure straddles zero) admits NO finite
    --    decay: there is no conservative side to take, and an emitted σ would be
    --    a fabricated number of exactly the class this campaign deletes.
    -- The drop reproduces the incumbent on every reachable input: the only
    -- reachable overlap is `ρ = 0`, where `g` is exactly `[0,0]` and the float
    -- test `g > 0.0` is false too; a hypothetical straddling `g` is within
    -- `2^{−128}` of zero — a partial losing `e^{−88}` per transit, inaudible
    -- whichever way it goes. (`g < 0` needs `ρ < 0`.)
    if DyadicI.certLt fk halfSR && DyadicI.certGt g DyadicI.zero then
      let sigma := DyadicI.neg (DyadicI.div (DyadicI.mul srD (DyadicI.log g)) span)
      let amp := DyadicI.inv k                        -- 1/k displacement rolloff
      let ph := DyadicI.mul goldenAngleD k            -- golden-angle phases
      modes := modes.push
        { sigma := litOfD sigma, omega := litOfD (DyadicI.mul twoPiD fk),
          cre := litOfD (DyadicI.mul amp (DyadicI.cos ph)),
          cim := litOfD (DyadicI.mul amp (DyadicI.sin ph)) }
  return modes

/-- A reverb room as a `ModalMode` bank (pole + residue-as-coeff): `nmode`
    log-spaced modes over `[flo,fhi]` with damping `σ = 6.91/rt60` (live), unit
    residues at golden-ratio phases so the tail isn't a pure comb. Freqs and count
    are structural (baked); only the damping is a live knob. `rtRange` (the rt60
    knob's declared span) maps through σ = 6.91/rt60 to each mode's `sigmaRange` —
    what lets a bloomed source CROSS this room with the rt60 still live (WS-LP:
    `bloomCompose` classifies the live pole over that interval). -/
def reverbRoom (rt60 : Sig) (rtRange : Option (Float × Float))
    (nmode : Nat) (flo fhi : Int × Nat) : Array ModalMode :=
  let sigma := div (lit 691 2) rt60           -- 6.91 / rt60
  -- `6.91` as the SAME exact decimal the EMITTED σ carries (`lit 691 e−2`), so
  -- the declared range and the emitted value are two readings of ONE constant
  -- rather than of a decimal and of the double nearest it — the
  -- quantization-vs-classification hazard, at a site that genuinely decides
  -- (`sigmaIntervalD?` feeds this range to the EC/DD router, and `clampSigmas`
  -- emits its endpoints). The FIELD stays `Float` — its type is `ModalMode`'s —
  -- so the exact quotient is projected to its nearest double here. MEASURED: at
  -- the shipped rt60 span (0.2, 12) both endpoints are bit-identical.
  let c691 : DyadicI := decD (691, 2)
  let sigmaRange := rtRange.map fun (lo, hi) =>
    ((DyadicI.div c691 (DyadicI.ofFloat hi)).toFloat,
     (DyadicI.div c691 (DyadicI.ofFloat lo)).toFloat)
  let floD := decD flo
  let fhiD := decD fhi
  let ratio := DyadicI.div fhiD floD          -- certifiably positive: `pow` is safe
  let denom : DyadicI := DyadicI.ofNat (if nmode ≤ 1 then 1 else nmode - 1)
  (Array.range nmode).map fun j =>
    -- log-spacing `flo·(fhi/flo)^{j/(n−1)}` — `pow` in the carrier
    -- (`exp(y·ln x)`), not the platform's. `ph` reaches ~120 rad at `j = 31`,
    -- where a platform `cos` spends reduction bits and this one does not.
    let fq := DyadicI.mul floD (DyadicI.pow ratio (DyadicI.div (DyadicI.ofNat j) denom))
    let ph := DyadicI.mul twoPiD (DyadicI.mul goldenRatioD (DyadicI.ofNat j))
    { sigma, omega := mul twoPiE (litOfD fq),
      cre := litOfD (DyadicI.cos ph), cim := litOfD (DyadicI.sin ph), sigmaRange }

/-- A 2-pole resonant filter as its EXACT complex-conjugate pole pair — the
    modal island's filter (the Serge-VCFQ move). "Filtering" a modal source is
    composing its poles with the filter's by the residue calculus, so the
    filter node is a `modalReverb` whose room is one conjugate pair:

      H(s) = ω₀² / (s² + (ω₀/Q)s + ω₀²)      (lowpass, unity DC gain,
                                               peak ≈ Q at fc — resonance ADDS
                                               presence, it never thins the
                                               passband)
      ν = −α ± i·ω_d,  α = ω₀/(2Q),  ω_d = ω₀·√(1 − 1/(4Q²))
      residues R = ∓ i·ω₀²/(2ω_d)

    BOTH conjugates are stored (amps R, R̄), so the composition's `Σ amp/(λ−ν)`
    IS the true rational H(λ) — exact, not a single-sided approximation — and
    the two rendered modes sum to the real impulse response
    `(ω₀²/ω_d)·e^{−αt}·sin(ω_d t)`. At high Q, α→0: the ringing modes barely
    decay and a strike PINGS at fc — the composed pole pair literally is the
    VCFQ's ring; sweep resonance up and the filter crosses into a struck
    resonator, which is the whole aesthetic reason it lives on the modal
    island. Everything is symbolic (`Sig`), so cutoff and resonance are live
    knobs through the composition — no relower.

    `res ∈ [0,1]` maps log to `Q = 0.55·80^res ∈ [0.55, 44]` (Q > ½ keeps ω_d
    real; the top of the knob rings for seconds). -/
def filterPair (fc res : Sig) : Array ModalMode :=
  let w0 := mul twoPiE fc
  -- Q = 0.55·e^{res·ln 80}. `ln 80` is certified once at module init (`ln80D`)
  -- rather than transcribed as a decimal literal; MEASURED byte-identical — the
  -- exact value's nearest double IS `4.382026634673881` and its 12-place
  -- quantization is unchanged. Provenance, not value.
  let q := mul (lit 55 2) (expSig (mul res lnEightyLit))
  let alpha := div w0 (mul (lit 2) q)
  let wd := mul w0 (.unary .sqrt (sub (lit 1) (div (lit 1) (mul (lit 4) (mul q q)))))
  let rim := div (mul w0 w0) (mul (lit 2) wd)          -- |Im R|
  #[ { sigma := alpha, omega := wd,     cre := lit 0, cim := neg rim },
     { sigma := alpha, omega := neg wd, cre := lit 0, cim := rim } ]

-- ── Node decode (named inlets: `in` is an object {port: [srcId,…]}) ──────────
def portSources (inObj : Json) (port : String) : Array String :=
  match (inObj.getObjVal? port).toOption.bind (·.getArr?.toOption) with
  | some arr => arr.filterMap (·.getStr?.toOption)
  | none => #[]

/-- Resolve a live param name to a `paramRef` slot read, falling back to `dflt`
    (used only if the param table somehow lacks the entry — the collector always
    allocates one). Every continuous knob is a live slot, so its value is READ from
    the slot at runtime, never baked — turning it drives `set_param`, no relower. -/
def pref (pidx : String → Option Nat) (name : String) (dflt : Sig) : Sig :=
  match pidx name with
  | some i => .paramRef ⟨i⟩
  | none => dflt

-- ── The port-spec table: ONE description of the vocabulary ──────────────────
-- Every recent vocabulary bug lived in a gap between two hand-maintained
-- descriptions of one thing (buildNode vs collectParams vs nodeSchema — the
-- sflange.rate dead knob was a NAME mismatch between an inlet and the param it
-- supersedes). The table is the single source; the collector, the lowering,
-- and the schema/vocabulary views are derived READERS of it and cannot drift.

/-- How a continuous port's live writes land — the host contract's dispatch
    key. `raw` = plain slot write; `glide` = closed-form smoothstep re-anchor
    (`#v0/#v1/#t0` companion slots); `anchor` = phase-anchored frequency
    (`#phase` companion — gliding a frequency VALUE would reintroduce the
    τ·f' chirp, so pitch keeps phase continuous instead). -/
inductive Discipline where
  | raw | glide | anchor
deriving BEq, Repr

/-- The connection-typing color a port carries or accepts. -/
inductive PortDomain where
  | signal | modal | control
deriving BEq, Repr

/-- Display metadata for a knob — semantics, not decoration: without it every
    frontend reinvents ranges and scales, badly, and drifts (three copies of
    this table have already existed). Served verbatim by `get_vocabulary`. -/
structure KnobMeta where
  min : Float
  max : Float
  log : Bool := false
  unit : String := ""
deriving Repr

/-- One port of a node kind. A port with `accepts ≠ #[]` is an inlet; a port
    with `knob = some (m, e)` carries a continuous param slot whose
    compile-time fallback is `lit m e` (the value `buildNode` bakes only if the
    collector somehow skipped the slot). A port may be BOTH (source `freq`: a
    control inlet that, unwired, is a knob) — wiring it supersedes the slot. -/
structure PortSpec where
  name : String
  accepts : Array PortDomain := #[]
  multi : Bool := false
  knob : Option (Int × Nat) := none
  discipline : Discipline := .raw
  display : Option KnobMeta := none
  /-- The port whose NORMAL this knob parameterizes (an input of the default
      subgraph, not of the node): sflange's `rate` is an input of `mod`'s
      normalled LFO. Wiring the owner replaces the whole default — circuit,
      knob, and slot vanish together, so a dead knob is unrepresentable (the
      sflange.rate bug was this relationship encoded as a name coincidence
      that didn't hold). -/
  ownerPort : Option String := none
deriving Repr

def sigIn : Array PortDomain := #[.signal, .modal, .control]
def modalIn : Array PortDomain := #[.modal]
def ctrlIn : Array PortDomain := #[.control]

/-- THE table. Order matters twice: knob-bearing entries in declaration order
    define the `ParamIdx` scan order (`collectParams`), and the whole layout is
    what `get_vocabulary` will serve. -/
def portSpecs : String → Array PortSpec
  | "knob" => #[
      { name := "value", knob := some (0, 0),
        display := some { min := 0, max := 1000 } }]
  | "source" => #[
      { name := "freq", accepts := ctrlIn, knob := some (220, 0), discipline := .anchor,
        display := some { min := 0.02, max := 2000, log := true, unit := "Hz" } },
      { name := "morph", knob := some (0, 0), discipline := .glide,
        display := some { min := 0, max := 1 } },
      { name := "pm", accepts := sigIn }]
  | "pluck" => #[
      { name := "freq", accepts := ctrlIn, knob := some (110, 0), discipline := .anchor,
        display := some { min := 20, max := 2000, log := true, unit := "Hz" } },
      { name := "morph", knob := some (0, 0), discipline := .glide,
        display := some { min := 0, max := 1 } },
      { name := "event_rate", knob := some (2, 0), discipline := .glide,
        display := some { min := 0.1, max := 20, log := true, unit := "Hz" } }]
  | "comb" => #[
      { name := "in", accepts := sigIn },
      { name := "delay", knob := some (12, 3), discipline := .glide,
        display := some { min := 0.0005, max := 0.05, log := true, unit := "s" } },
      { name := "decay", knob := some (7, 1), discipline := .glide,
        display := some { min := 0, max := 0.95 } }]
  | "flange" => #[
      { name := "in", accepts := sigIn },
      { name := "depth", knob := some (7, 4), discipline := .glide,
        display := some { min := 0.0001, max := 0.01, log := true, unit := "s" } }]
  | "sflange" => #[
      { name := "in", accepts := sigIn },
      { name := "mod", accepts := sigIn },
      { name := "depth", knob := some (2, 3), discipline := .glide,
        display := some { min := 0.0002, max := 0.02, log := true, unit := "s" } },
      -- `rate` parameterizes `mod`'s normal (the built-in LFO): patch `mod`
      -- and the LFO, this knob, and its slot all vanish together.
      { name := "rate", knob := some (3, 1), ownerPort := some "mod",
        display := some { min := 0.02, max := 12, log := true, unit := "Hz" } }]
  | "fm" => #[
      { name := "in", accepts := sigIn },
      { name := "carrier", knob := some (330, 0), discipline := .anchor,
        display := some { min := 20, max := 2000, log := true, unit := "Hz" } },
      { name := "depth", knob := some (8, 0), discipline := .glide,
        display := some { min := 1, max := 400, log := true } }]
  | "delay" => #[
      { name := "in", accepts := sigIn },
      { name := "amount", knob := some (4, 3), discipline := .glide,
        display := some { min := 0.0001, max := 0.02, log := true, unit := "s" } }]
  | "reverse" => #[{ name := "in", accepts := sigIn }]
  | "mix" => #[{ name := "in", accepts := sigIn, multi := true }]
  | "ring" => #[{ name := "in", accepts := sigIn, multi := true }]
  | "resonator" => #[
      { name := "addr", accepts := sigIn },
      { name := "freq", knob := some (220, 0),
        display := some { min := 20, max := 2000, log := true, unit := "Hz" } },
      { name := "decay", knob := some (4, 0),
        display := some { min := 0.5, max := 50, log := true } }]
  | "reverb" => #[
      { name := "in", accepts := modalIn },
      { name := "rt60", knob := some (2, 0),
        display := some { min := 0.2, max := 12, log := true, unit := "sec" } },
      { name := "dir", knob := some (0, 0),
        display := some { min := 0, max := 1 } },
      { name := "sway", knob := some (0, 0),
        display := some { min := 0, max := 0.9 } },
      { name := "rate", knob := some (3, 1),
        display := some { min := 0.05, max := 8, log := true, unit := "Hz" } }]
  | "filter" => #[
      { name := "in", accepts := modalIn },
      { name := "cutoff", knob := some (800, 0), discipline := .glide,
        display := some { min := 20, max := 8000, log := true, unit := "Hz" } },
      { name := "resonance", knob := some (5, 1), discipline := .glide,
        display := some { min := 0, max := 1 } }]
  | "modalmix" => #[{ name := "in", accepts := modalIn, multi := true }]
  -- gauge: the §5 excitation-gauge adapter — re-levels its modal input's peak by the
  -- self-measured ‖H‖^{−g}. `g` is the gauge: 0 = unity-DC (strike, the identity),
  -- ½ = √Q trim, 1 = unity-peak (tuned-tone level-invariant). Glided (smooth sweep).
  | "gauge" => #[
      { name := "in", accepts := modalIn },
      { name := "g", knob := some (0, 0), discipline := .glide,
        display := some { min := 0, max := 1 } }]
  -- gong: a struck resonator whose strike data (`t`, `g`, `modes_full`,
  -- `modes_half`) is structural (carried in `params`), EXCEPT the pitch-bloom
  -- depth `beta` — promoted to a LIVE slot, so the bloom deepens/relaxes under a
  -- knob with no relower, the score's baked value its initial.
  | "gong" => #[{ name := "beta", knob := some (5, 2), display := some { min := 0, max := 0.5 } }]
  -- string: a plucked/struck string as its diagonalized modal bank. Like gong,
  -- its content (`freq`, `decay`, `t`, `modes`) is structural, carried in
  -- `params`; the optional `addr` inlet drives the pluck's time-address.
  | "string" => #[{ name := "addr", accepts := sigIn }]
  | "out" => #[{ name := "in", accepts := sigIn, multi := true }]
  | _ => #[]

/-- Each kind's outlet color (`none` = no outlet, the dac sink). -/
def outletOf : String → Option PortDomain
  | "knob" => some .control
  | "resonator" | "reverb" | "filter" | "modalmix" | "string" | "gauge" => some .modal
  | "out" => none
  | _ => some .signal

/-- The kinds the table covers, in schema order (`out` last — it has no outlet).
    The SERVED surface vocabulary: `checkServedKinds` admits exactly these; a
    client renders them from `vocabularyJson`. -/
def vocabularyKinds : Array String := #[
  "source", "pluck", "comb", "flange", "delay", "reverse", "fm", "sflange",
  "mix", "ring", "gong", "string", "resonator", "reverb", "filter",
  "modalmix", "gauge", "knob", "out"]

/-- Every kind `buildNode` actually constructs (its match arms — the AUTHORITATIVE
    list, read straight off the arms below). The classification-drift gate and
    `checkServedKinds` both derive from THIS, so `buildNode` cannot grow a kind the
    vocabulary/withholding machinery has not accounted for: the drift gate asserts
    `buildNodeKinds ⊆ vocabularyKinds ∪ withheldKinds` (every built kind is served
    or explicitly withheld) and that no served kind drifts. (`out` is a dac sink,
    not a `buildNode` arm — it is in `vocabularyKinds`, not here.) -/
def buildNodeKinds : Array String := #[
  "knob", "source", "pluck", "comb", "flange", "sflange", "fm", "delay",
  "reverse", "mix", "ring", "resonator", "reverb", "filter", "modalmix",
  "gauge", "gong", "bloomgong", "string"]

/-- Kinds `buildNode` builds but which are WITHHELD from the served surface. Their
    modal factor-site landing (`bloomComposedSig`) still lands `lit 268435456`
    unconditionally — no per-region sup, no admission guard for the conditioning
    hazard when `a` is near a negative integer (the fixed-depth float64 series-M
    Horner catastrophically cancels; see that def's RANGE block). `checkServedKinds`
    rejects a withheld kind with an honest message rather than letting it die
    downstream as a MISLEADING `signal→modal` type error: `outletOf` falls through
    to `signal` for it, which DRIFTS from its modal constructed node — the exact
    drift the `modal-class-agreement` gate now sees because it drives off
    `buildNodeKinds`. Un-withholding one is NOT a one-line `outletOf` edit: it
    re-admits the unguarded factor site, so it waits on the per-region-sup landing
    (`design/seam-hardening-optionE-handoff.local.md`, the factor-site follow-on). -/
def withheldKinds : Array String := #["bloomgong"]

-- Derived views — the ONLY readers of glide/anchor/knob facts from here down.
def portOf (kind kname : String) : Option PortSpec :=
  (portSpecs kind).find? (·.name == kname)
def disciplineOf (kind kname : String) : Discipline :=
  ((portOf kind kname).map (·.discipline)).getD .raw
def isGlided (kind kname : String) : Bool := disciplineOf kind kname == .glide
def isAnchored (kind kname : String) : Bool := disciplineOf kind kname == .anchor
/-- A knob's compile-time fallback from the table (`lit m e`). -/
def fallbackOf (kind kname : String) : Sig :=
  match (portOf kind kname).bind (·.knob) with
  | some (m, e) => lit m e
  | none => lit 0
/-- A knob's display span from the table — the interval a live value is DECLARED
    to range over (WS-LP feeds it to the live-pole region classifier; the lifted
    kernel clamps the live read to it, so the declaration is enforced, not
    advisory). -/
def displayRangeOf (kind kname : String) : Option (Float × Float) :=
  ((portOf kind kname).bind (·.display)).map (fun d => (d.min, d.max))

/-- A closed-form smoothstep GLIDE of τ from three slots: `v0 + (v1−v0)·s²(3−2s)`,
    `s = clamp((τ − t0)/dur, 0, 1)`, `dur = 0.02·sampleRate` samples (20 ms at any
    rate, matching the engine's glide discipline for `set_param`). The value eases from `v0` to
    `v1` starting at tick `t0`; the control plane re-anchors the slots on each turn.
    Stateless — the ramp is a pure function of the ambient clock, not an accumulator. -/
def glideExpr (pidx : String → Option Nat) (base : String) (dflt : Sig) : Sig :=
  let v0 := pref pidx s!"{base}#v0" dflt
  let v1 := pref pidx s!"{base}#v1" dflt
  let t0 := pref pidx s!"{base}#t0" (lit 0)
  let dur := mul (lit 2 2) .sampleRate   -- 0.02·SR = 20 ms
  let s  := clampE (div (sub (toFloatE .sampleIndex) t0) dur) (lit 0) (lit 1)
  let ss := mul (mul s s) (sub (lit 3) (mul (lit 2) s))
  add v0 (mul (sub v1 v0) ss)


end Tropical.Playground
