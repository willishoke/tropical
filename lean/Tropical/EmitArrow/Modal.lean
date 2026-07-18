import Tropical.EmitArrow.Numerics
import Tropical.Ir.BanksFlag
import Tropical.EmitArrow.Term

/-!
# EmitArrow.Modal — the pole island: modal banks + the residue calculus

A modal bank is a gated sum of decaying sinusoids — the real part of
`Σ A·d^deg·e^{μd}` — evaluated as a pure function of the (already-warped)
clock: random-access, no state. `ModalMode` is the rectangular pole/amp
record; `bankFold` is the generic banked reduction over coefficient columns
(`Sig.bankSum` — O(1) plan size in mode count); the `*Table` lowerings are
its bit-identical banked twins. Composition (`voice ⋙ reverb`) is the
SYMBOLIC residue calculus (`residueComposeE`/`residueComposeEC`) over
`CplxE` — pure `+−×÷` on `Sig`, so poles and amps stay live param slots
through a room. The direction operator crossfades causal/anti-causal tails
without touching σ or ω.
-/

namespace Tropical.EmitArrow

open Tropical.Ir

/-- One mode in RECTANGULAR form: `d^deg · e^{−σd} · (c_re·cos(ωd) − c_im·sin(ωd))
    = Re(A · d^deg · e^{μd})` with pole `μ = −σ + iω` and complex amp `A = c_re +
    i·c_im`. Rectangular (not magnitude/phase) so the residue calculus can emit the
    coefficients as pure `+−×÷` expressions — no `sqrt`/`atan2` — which is what lets
    every field be a LIVE param slot, not a baked literal. Authored modes set
    `ω = 2π·f`, `c_re = amp`, `c_im = 0`. -/
structure ModalMode where
  sigma : Sig           -- σ = −Re μ  (decay)
  omega : Sig           -- ω = Im μ   (rad/s)
  cre   : Sig           -- Re A
  cim   : Sig := lit 0   -- Im A
  deg   : Nat := 0

/-- Authored mode from (freq Hz, decay σ, real amp): `ω = 2π·f`, `c_re = amp`,
    `c_im = 0`. Any of `fHz`/`sigma`/`amp` may be a live `paramRef`. -/
def ModalMode.hz (fHz sigma amp : Sig) (deg : Nat := 0) : ModalMode :=
  { sigma, omega := mul twoPiE fHz, cre := amp, deg }

/-- `base^k` by repeated multiply (k is a mode's small polynomial order). -/
def powE (base : Sig) : Nat → Sig
  | 0 => lit 1
  | 1 => base
  | n + 1 => mul (powE base n) base

/-- The RELATIVE clock `clkRel = clk − anchor·2³²` as an EXACT i64 subtract.
    (A float-relative clock — `toFloat(clk)/2³² − anchor` — loses mantissa bits
    as the absolute clock grows, drifting with τ.) Subtracting on
    the integer clock first keeps time-translation exact at any τ; everything
    downstream (phase reduction, the bounded envelope coordinate, the causal gate)
    sees only the bounded relative value. -/
def relClockQ (clkInt anchorSamples : Sig) : Sig :=
  sub clkInt (toIntE (mul anchorSamples (lit 4294967296)))

/-- A mode's oscillator phase over the RELATIVE clock, integer-reduced (the
    FixedPhasor pattern): the frequency lands ONCE as `incr = ⌊(ω/2π)·2³²/SR⌋`
    (float math, so ω may be a live slot), then the exact split-multiply
    `incr·hi + (incr·lo)>>32` reduces `incr·clkRel` on the circle ℤ/2³² — the
    phase argument never leaves `[0, 2π)`, so its precision is τ-INDEPENDENT
    (a raw `ω·dSec` argument is unbounded — its mantissa starves as τ grows).
    ω quantizes to the SR/2³² grid (~1e-5 Hz at 44.1k) — inaudible.
    `rshift` is `ashr`, so `clkRel = (clkRel>>32)·2³² + (clkRel & (2³²−1))` holds
    exactly on NEGATIVE relative clocks (pre-strike, mirrored reads). Returns the
    phase in RADIANS `[0, 2π)` for the `sinSig`/`cosSig` polynomials. -/
def modePhaseQ (omega clkRel : Sig) : Sig :=
  let incr := toIntE (div (mul (div omega twoPiE) (lit 4294967296)) .sampleRate)
  let thi := rshift clkRel (lit 32)
  let tlo := bitAnd clkRel (lit 4294967295)
  bitAnd (add (mul incr thi) (rshift (mul incr tlo) (lit 32))) (lit 4294967295)

/-- `modePhaseQ` scaled to RADIANS `[0, 2π)` — for float-polynomial consumers
    (`sinSig`/`cosSig`). The fixed datapath consumes `modePhaseQ` directly. -/
def modePhaseW (omega clkRel : Sig) : Sig :=
  mul twoPiE (div (toFloatE (modePhaseQ omega clkRel)) (lit 4294967296))

/-- `modePhaseQ` with the quantized increment supplied DIRECTLY, skipping the
    `incr = ⌊(ω/2π)·2³²/SR⌋` step. The banked path stores `incr`'s pre-`toInt`
    float in a coefficient column and reads it per iteration; `toInt` in-loop then
    recovers the SAME i64 (f64 storage round-trips the value exactly, < 2^53), so
    `modePhaseQFromIncr (toIntE incrFloat) clkRel` is bit-identical to
    `modePhaseQ ω clkRel`. -/
def modePhaseQFromIncr (incr clkRel : Sig) : Sig :=
  let thi := rshift clkRel (lit 32)
  let tlo := bitAnd clkRel (lit 4294967295)
  bitAnd (add (mul incr thi) (rshift (mul incr tlo) (lit 32))) (lit 4294967295)

/-- A modal bank as a pure `Sig` over the (already-warped) clock: shift each pole
    to its time-since-strike on the INTEGER clock (`relClockQ`, exact at any τ),
    sum `d^deg·e^{−σd}·(c_re·cos φ − c_im·sin φ)` over the modes with each φ
    integer-reduced (`modePhaseW`), and GATE on `clkRel > 0` (causal: silent
    before the strike, a closed-form tail after — and, read through a reversing
    warp, the tail plays backward). The envelope coordinate `dSec` is float on
    the BOUNDED relative clock (`expSig` clamps; decay makes far-field precision
    irrelevant). Every sample is `f(clk)`: no state, random-access. Rides
    `arrUn … (.clk c)`, so warps reach it through the clock leaf. -/
def modalBankSig (modes : Array ModalMode) (clkInt : Sig) (anchorSamples : Sig) : Sig :=
  let clkRel := relClockQ clkInt anchorSamples
  let dSec := div (div (toFloatE clkRel) (lit 4294967296)) .sampleRate
  -- Q datapath: per mode, the slow scalars (envelope × residue
  -- weight) land ONCE as Q4.28 (headroom |w| < 32, quantum 3.7e-9 ≈ −168 dB —
  -- a tail quieter than that truncates to true silence), the oscillator values
  -- are exact Q2.30 off the integer phase, and the mode SUM is i64 — modular,
  -- hence associative and commutative: reordering modes cannot move a bit,
  -- which float summation never gave us. One float scale at the boundary.
  let bankQ := modes.foldl
    (fun acc m =>
      let phQ := modePhaseQ m.omega clkRel
      let env := expSig (neg (mul m.sigma dSec))
      let env2 := if m.deg == 0 then env else mul (powE dSec m.deg) env
      let wCre := toIntE (mul (mul env2 m.cre) (lit 268435456))
      let wCim := toIntE (mul (mul env2 m.cim) (lit 268435456))
      let oscQ := rshift (sub (mul wCre (fixedCosCycSig phQ))
                              (mul wCim (fixedSinCycSig phQ))) (lit 28)
      add acc oscQ)
    (litI 0)
  selectE (gt clkRel (lit 0)) (fixedOutQ 30 bankQ) (lit 0)

/-- The SYMBOLIC mode: a bank body's per-mode scalars as column reads at the
    loop index. A body written against `ModeSym` is the same lambda the unrolled
    fold applies to a concrete `ModalMode` — only where the scalars COME FROM
    changes (a column read at `loopIdx` instead of a baked subterm). `incr` is
    the pre-`toInt` float increment `(ω/2π)·2³²/SR` (exact < 2^53); `toInt`
    happens in the body (`modePhaseQFromIncr`). -/
structure ModeSym where
  incr  : Sig
  sigma : Sig
  cre   : Sig
  cim   : Sig

/-- A bank's coefficient columns: one `Sig.arr` per `ModalMode` field, one entry
    per mode — the SAME s0 subterms the unrolled path bakes per mode, destined
    for the stage-0 coefficient kernel. Built once per bank and shared by every
    fold over it (a direction bank runs TWO folds over one set of columns; the
    arena's hash-consing would dedupe rebuilt columns anyway, but sharing at the
    source keeps the intent visible). -/
structure BankCols where
  count : Nat
  /-- Optional LIVE effective count (trip-count-as-data): a `Sig` (typically a
      param-slot read) the reduction trips instead of the static `count`, which
      stays the CAPACITY (columns fill to capacity; the emitters clamp the live
      value to `[0, count]` at the loop head). `none` = the static bank. -/
  live? : Option Sig := none
  /-- The bank's binder id (`Sig.bankSum.idxId` / `Sig.loopIdx.id`). Ids need
      only be unique along a NESTING CHAIN, and `bankFold` banks never nest
      (tables materialize before their region, so two banks over one set of
      columns run sequentially) — 0 is correct for every Sig-level bank today.
      If nesting ever arrives here, thread a chain-unique allocator; the
      emitters fail loudly on an ancestor id collision. -/
  idxId : Nat := 0
  incr  : Sig
  sigma : Sig
  cre   : Sig
  cim   : Sig

def bankCols (modes : Array ModalMode) (live? : Option Sig := none) : BankCols where
  count := modes.size
  live? := live?
  incr  := Sig.arr (modes.map fun m =>
    div (mul (div m.omega twoPiE) (lit 4294967296)) .sampleRate)
  sigma := Sig.arr (modes.map (·.sigma))
  cre   := Sig.arr (modes.map (·.cre))
  cim   := Sig.arr (modes.map (·.cim))

/-- THE generic banked fold: `Σₖ body(mode k)` as one indexed reduction
    (`Sig.bankSum` → a `ReduceBegin` region — O(1) plan instructions in mode
    count). The body is an ordinary function of the symbolic mode; this is the
    only place a banked effect touches `loopIdx`/`bankSum`. Banking is a property
    of the fold, not of the effect: every banked lowering is `bankFold` applied
    to its own body, never a hand-built table twin. The loop visits modes in
    array order — the same order the unrolled `foldl` nests its adds — so for
    the i64 mode sum the render is BIT-IDENTICAL to the unroll. -/
def bankFold (cols : BankCols) (body : ModeSym → Sig) : Sig :=
  let k := Sig.loopIdx cols.idxId
  Sig.bankSum cols.count #[cols.incr, cols.sigma, cols.cre, cols.cim]
    (body { incr  := Sig.index cols.incr k
          , sigma := Sig.index cols.sigma k
          , cre   := Sig.index cols.cre k
          , cim   := Sig.index cols.cim k })
    cols.live?
    cols.idxId

/-- The BANKED lowering of a modal bank (banks-as-data): the SAME value
    as `modalBankSig`, but the mode sum is a `bankFold` indexed reduction over
    the coefficient columns instead of an unrolled fold — so the emitted plan is
    O(1) in mode count, not O(modes). Requires every mode `deg == 0` (the uniform
    datapath, true for resonator/reverb/residue banks); ragged banks route to the
    unrolled path. The body is EXACTLY `modalBankSig`'s op sequence over the
    symbolic mode (deg == 0, so `env2 = env`), and the mode sum is i64-modular
    (associative), so the render is BIT-IDENTICAL — the goldens gate the move.
    Drop-in for `modalBankSig`: same `(modes, clkInt, anchor)` signature, so it
    rides `modalBankTerm`'s `arrUn` (warps reach it through the already-warped
    clock leaf) unchanged. -/
def modalBankSigTable (modes : Array ModalMode) (clkInt anchorSamples : Sig)
    (live? : Option Sig := none) : Sig :=
  let clkRel := relClockQ clkInt anchorSamples
  let dSec := div (div (toFloatE clkRel) (lit 4294967296)) .sampleRate
  let bankQ := bankFold (bankCols modes live?) fun m =>
    let phQ  := modePhaseQFromIncr (toIntE m.incr) clkRel
    let env  := expSig (neg (mul m.sigma dSec))
    let wCre := toIntE (mul (mul env m.cre) (lit 268435456))
    let wCim := toIntE (mul (mul env m.cim) (lit 268435456))
    rshift (sub (mul wCre (fixedCosCycSig phQ))
                (mul wCim (fixedSinCycSig phQ))) (lit 28)
  selectE (gt clkRel (lit 0)) (fixedOutQ 30 bankQ) (lit 0)

/-- The ANALYTIC bank: the `(Re, Im)` pair of `Σ A·e^{iφ}·env` over ONE column
    set — the substrate for a phase TWIST (heterodyne) and for the
    divided-difference paired body, both of which need `Im` as well as the `Re`
    every existing lowering emits. Two `bankFold`s over one `BankCols` (the
    direction bank's pattern, `modalBankSigDirTable` — two sequential regions,
    `idxId` reuse is safe because they do not nest). The `re` fold is EXACTLY
    `modalBankSigTable`'s body (`wCre·cos − wCim·sin`); the `im` fold is
    `wCre·sin + wCim·cos`. Both gated `clkRel > 0` (causal). Identity the
    `modal-pair` gate leans on: `Im(A·e^{iφ}) = Re(−iA·e^{iφ})`, so the `im`
    component equals the `re` bank of the amp-rotated modes `(cre,cim)↦(cim,−cre)`
    — a bit-identical oracle, no new numerics. deg-0 (uniform) only. -/
def modalBankSigPairTable (modes : Array ModalMode) (clkInt anchorSamples : Sig)
    (live? : Option Sig := none) : Sig × Sig :=
  let clkRel := relClockQ clkInt anchorSamples
  let dSec := div (div (toFloatE clkRel) (lit 4294967296)) .sampleRate
  let cols := bankCols modes live?
  let body (imag : Bool) : ModeSym → Sig := fun m =>
    let phQ  := modePhaseQFromIncr (toIntE m.incr) clkRel
    let env  := expSig (neg (mul m.sigma dSec))
    let wCre := toIntE (mul (mul env m.cre) (lit 268435456))
    let wCim := toIntE (mul (mul env m.cim) (lit 268435456))
    if imag then
      rshift (add (mul wCre (fixedSinCycSig phQ))
                  (mul wCim (fixedCosCycSig phQ))) (lit 28)
    else
      rshift (sub (mul wCre (fixedCosCycSig phQ))
                  (mul wCim (fixedSinCycSig phQ))) (lit 28)
  let gate := fun q => selectE (gt clkRel (lit 0)) (fixedOutQ 30 q) (lit 0)
  (gate (bankFold cols (body false)), gate (bankFold cols (body true)))

/-- True when a bank is eligible for the table lowering: every mode `deg == 0`
    (the uniform datapath). Ragged banks (mixed degree) must route to the
    unrolled `modalBankSig` (or, later, split into a banked deg-0 part ⊕ an
    unrolled remainder). -/
def bankIsUniform (modes : Array ModalMode) : Bool := modes.all (·.deg == 0)

/-- Complex arithmetic over `Sig` (real, imag) — the residue calculus done
    SYMBOLICALLY, so poles/coeffs can be live param slots. With literal operands
    every op constant-folds to the baked value; with a `paramRef` operand it stays
    a live expression the kernel re-evaluates when the slot moves. Only `+−×÷`. -/
abbrev CplxE := Sig × Sig
def caddE (a b : CplxE) : CplxE := (add a.1 b.1, add a.2 b.2)
def csubE (a b : CplxE) : CplxE := (sub a.1 b.1, sub a.2 b.2)
def cmulE (a b : CplxE) : CplxE := (sub (mul a.1 b.1) (mul a.2 b.2), add (mul a.1 b.2) (mul a.2 b.1))
def cdivE (a b : CplxE) : CplxE :=
  let d := add (mul b.1 b.1) (mul b.2 b.2)
  (div (add (mul a.1 b.1) (mul a.2 b.2)) d, div (sub (mul a.2 b.1) (mul a.1 b.2)) d)
def cnegE (a : CplxE) : CplxE := (neg a.1, neg a.2)

/-- The pole `μ = −σ + iω` and complex amp `A = c_re + i·c_im` of a mode, as `CplxE`. -/
def ModalMode.poleE (m : ModalMode) : CplxE := (neg m.sigma, m.omega)
def ModalMode.ampE (m : ModalMode) : CplxE := (m.cre, m.cim)
def modeOfE (pole amp : CplxE) (deg : Nat := 0) : ModalMode :=
  { sigma := neg pole.1, omega := pole.2, cre := amp.1, cim := amp.2, deg }

/-- The residue calculus, SYMBOLICALLY — `voice ⋙ reverb` with every pole/coeff an
    `Sig`. Same law as the build-time `residueCompose`: forced mode at each voice
    pole λ (amp `a·H(λ)`), ringing mode at each reverb pole ν (amp `−a·r/(λ−ν)`).
    No degeneracy branch — a build-time `|λ−ν|<tol` test is impossible on live
    poles; an exact coincidence is measure-zero for continuous knobs, and a NEAR
    coincidence gives a large finite coupling (the resonance you want). Validated
    against the Float `residueCompose` on literal poles (`symbolic-residue`). -/
def residueComposeE (voice reverb : Array ModalMode) : Array ModalMode :=
  voice.foldl (fun acc v =>
    let lam := v.poleE
    let a := v.ampE
    let Hlam := reverb.foldl (fun s r => caddE s (cdivE r.ampE (csubE lam r.poleE))) ((lit 0, lit 0) : CplxE)
    let acc := acc.push (modeOfE lam (cmulE a Hlam))
    reverb.foldl (fun acc r =>
      acc.push (modeOfE r.poleE (cnegE (cdivE (cmulE a r.ampE) (csubE lam r.poleE))))) acc) #[]

/-- `residueComposeE` COLLECTED to `m + n` modes. The uncollected form pushes one
    ringing mode per (λ, ν) pair — `m·n` modes all sitting at the same `n` reverb
    poles. Partial fractions over distinct simple poles needs only the pole UNION:
    each voice pole λ keeps its forced mode (amp `a·H(λ)`, unchanged), and each
    reverb pole ν keeps ONE ringing mode whose amp is the pair-amps summed over the
    voice, `−r·Σ_k a_k/(λ_k−ν)`. Same signal (the `residue-collected` gate pins it
    pointwise against the uncollected form), a factor `m` fewer transcendentals per
    sample. Amps stay `CplxE` expressions, so live poles (a resonator's `freq`
    slot) stay live through the composition — this is what keeps a voice's pitch
    knob working THROUGH a reverb. Same no-degeneracy-branch stance as
    `residueComposeE` (coincidence is measure-zero on live knobs; near-coincidence
    is a large finite coupling — the sympathetic resonance you want). An empty
    voice composes to the empty bank (silence), preserving the graceful-silence
    contract for a reverb with nothing patched into it. -/
def residueComposeEC (voice reverb : Array ModalMode) : Array ModalMode :=
  if voice.isEmpty then #[] else
  let forced := voice.map (fun v =>
    let Hlam := reverb.foldl (fun s r => caddE s (cdivE r.ampE (csubE v.poleE r.poleE)))
      ((lit 0, lit 0) : CplxE)
    modeOfE v.poleE (cmulE v.ampE Hlam))
  let ringing := reverb.map (fun r =>
    let coupling := voice.foldl (fun s v => caddE s (cdivE v.ampE (csubE v.poleE r.poleE)))
      ((lit 0, lit 0) : CplxE)
    modeOfE r.poleE (cnegE (cmulE r.ampE coupling)))
  forced ++ ringing

/-- `residueComposeEC` with the Cauchy inner sums BANKED (WS-F). Each output mode's
    amp (`Hlam`/`coupling`) becomes a scalar `Sig.bankSum` over the SOURCE columns
    rather than a meta-unrolled fold — same value (per-term identical to `cdivE`,
    left-assoc = the fold's order, so BIT-IDENTICAL to `residueComposeEC`), but the
    coefficient-kernel FILL is O(m+n) reduce regions instead of O(m·n) unrolled ops
    when the poles are live: each inner sum is an all-s0 region that hoists to the
    coeff kernel (`Stage0.tryRegion` — scalar accumulator), and the column writes
    hoist as ordinary array fills. A compile-flatness win, orthogonal to
    `residueComposeDD`'s stability (the `1/Δ` still appears — this is the COLLECTED
    form). Inner binder id 1 (the audio bank is 0; the fills are sequential, emitted
    before the audio region). -/
def residueComposeBanked (voice reverb : Array ModalMode) : Array ModalMode :=
  if voice.isEmpty then #[] else
  let idH : Nat := 1
  -- Σₖ (rRe+i·rIm)/((pRe−nRe[k]) + i(pIm−nIm[k])), real or imag part, as a scalar
  -- bankSum; per term = `cdivE` (r·conj(d)/|d|²), left-associative.
  let cauchy := fun (pRe pIm nRe nIm rRe rIm : Sig) (count : Nat) (imag : Bool) =>
    Sig.bankSum count #[nRe, nIm, rRe, rIm]
      (let k := Sig.loopIdx idH
       let dRe := sub pRe (Sig.index nRe k)
       let dIm := sub pIm (Sig.index nIm k)
       let den := add (mul dRe dRe) (mul dIm dIm)
       let rkRe := Sig.index rRe k
       let rkIm := Sig.index rIm k
       if imag then div (sub (mul rkIm dRe) (mul rkRe dIm)) den
       else div (add (mul rkRe dRe) (mul rkIm dIm)) den)
      none idH
  let revNuRe := Sig.arr (reverb.map (·.poleE.1))
  let revNuIm := Sig.arr (reverb.map (·.poleE.2))
  let revRRe := Sig.arr (reverb.map (·.ampE.1))
  let revRIm := Sig.arr (reverb.map (·.ampE.2))
  let voNuRe := Sig.arr (voice.map (·.poleE.1))
  let voNuIm := Sig.arr (voice.map (·.poleE.2))
  let voARe := Sig.arr (voice.map (·.ampE.1))
  let voAIm := Sig.arr (voice.map (·.ampE.2))
  -- forced (over reverb): Hlam = Σᵣ r/(λ−ν), amp = a·Hlam
  let forced := voice.map (fun v =>
    let hRe := cauchy v.poleE.1 v.poleE.2 revNuRe revNuIm revRRe revRIm reverb.size false
    let hIm := cauchy v.poleE.1 v.poleE.2 revNuRe revNuIm revRRe revRIm reverb.size true
    modeOfE v.poleE (cmulE v.ampE (hRe, hIm)))
  -- ringing (over voice): with pole=ν the bankSum gives Σᵥ a/(ν−λ) = −coupling, so
  -- the EC amp `−r·coupling` becomes `r·(bankSum)` — no `cnegE`.
  let ringing := reverb.map (fun r =>
    let cRe := cauchy r.poleE.1 r.poleE.2 voNuRe voNuIm voARe voAIm voice.size false
    let cIm := cauchy r.poleE.1 r.poleE.2 voNuRe voNuIm voARe voAIm voice.size true
    modeOfE r.poleE (cmulE r.ampE (cRe, cIm)))
  forced ++ ringing

/-- `∫`: the antiderivative of a modal bank, exactly, as a build-time pole move.
    `∫ Σₖ Aₖ e^{μₖ d} dd = Σₖ (Aₖ/μₖ) e^{μₖ d} + C`, and choosing `C = −Σₖ Aₖ/μₖ`
    fixes the integral to 0 at the strike (`d=0`) — a DC atom (`μ=0`, so `e^{0·d}=1`)
    carrying that constant. So each mode's amp divides by its pole (`cdivE`) and one
    `μ=0` mode is appended. Pure `CplxE`, so poles/amps stay live. Stays deg-0
    (bankable). Requires deg-0, NONZERO poles — the modulator case: an LFO is a σ=0
    undamped mode with pole `iω≠0`, and integrating its bank IS the residue transform
    `a ↦ a/μ` behind "FM is PM of the integrated bank" (`demos/modal_vco.py` D3). A DC
    input mode (`μ=0`) would integrate to a `d·e^{0}` deg-1 ramp — out of scope for
    v1, and division by its zero pole is the caller's contract to avoid. -/
def integrateBank (modes : Array ModalMode) : Array ModalMode :=
  let integ := modes.map (fun m => modeOfE m.poleE (cdivE m.ampE m.poleE))
  let sumAmp := integ.foldl (fun s m => caddE s m.ampE) ((lit 0, lit 0) : CplxE)
  integ.push (modeOfE (lit 0, lit 0) (cnegE sumAmp))

/-- `Jₙ(b)` (Bessel, first kind) by trapezoid on the periodic integrand
    `cos(nθ − b·sin θ)` over `[0,π]` — spectral accuracy on the smooth periodic
    integrand. Build-time `Float`; the FM sideband weights of `besselFuse`. The
    index `nf` is the sideband number as a `Float` (so callers avoid `Int→Float`). -/
def besselJ (nf b : Float) (quad : Nat := 256) : Float := Id.run do
  let pi := 3.141592653589793
  let h := pi / quad.toFloat
  let mut acc : Float := 0.0
  for i in [0:quad + 1] do
    let th := h * i.toFloat
    let w := if i == 0 || i == quad then 0.5 else 1.0     -- trapezoid endpoints
    acc := acc + w * Float.cos (nf * th - b * Float.sin th)
  return acc * h / pi

/-- Static-index FM as a build-time pole move (Jacobi–Anger). Modulating a bank by
    `sin(ω_m d)` at index `b` sprouts, from each mode `(μ, A)`, a comb of sidebands
    at pole `μ + i·n·ω_m` with amp `A·Jₙ(b)`, `n ∈ [−N, N]` — so an FM'd voice is
    STILL a modal bank (it keeps poles, so it can feed the residue calculus). `ω_m`
    (rad/s) and `b` are baked in v1 (a change relowers); the carrier's poles/amps
    stay live (each sideband adds a baked offset to `ω` and scales `A` by the real
    `Jₙ(b)`). `N` is the sideband capacity — the tail `|n| > b` decays
    superexponentially, so `N ≈ ⌈b⌉ + few` is exact to machine precision
    (`demos/modal_fm.py` D4). deg-0. -/
def besselFuse (modes : Array ModalMode) (wm b : Float) (N : Nat) : Array ModalMode := Id.run do
  let mut out : Array ModalMode := #[]
  for m in modes do
    for i in [0:2 * N + 1] do
      let nf := i.toFloat - N.toFloat
      let jn := besselJ nf b
      out := out.push
        { sigma := m.sigma
        , omega := add m.omega (litF (nf * wm))
        , cre := mul m.cre (litF jn)
        , cim := mul m.cim (litF jn)
        , deg := m.deg }
  return out

/-- Affine pole reclock: the pole-space image of the affine clock warp `d ↦ a·d+b`.
    `e^{μ(a d + b)} = e^{μb}·e^{(μa)d}`, so the pole scales `μ ↦ μa` (`σ↦σa, ω↦ωa`)
    and the amp rotates `A ↦ A·e^{μb} = A·e^{−σb}(cos ωb + i sin ωb)`. Pure `CplxE`
    over `expSig`/`sinSig`/`cosSig`, so `a`,`b` may be live. AFFINE ONLY: a nonlinear
    warp does not preserve poles (the varispeed case is deferred, per
    `design/modal-island.local.md`); `b` in seconds, `a` dimensionless. deg-0 (the
    `(a d+b)^k` binomial for `deg>0` is out of scope for v1). -/
def reclockAffine (a b : Sig) (modes : Array ModalMode) : Array ModalMode :=
  modes.map fun m =>
    let envB := expSig (neg (mul m.sigma b))
    let eMub : CplxE := (mul envB (cosSig (mul m.omega b)), mul envB (sinSig (mul m.omega b)))
    let a' := cmulE m.ampE eMub
    { sigma := mul m.sigma a, omega := mul m.omega a, cre := a'.1, cim := a'.2, deg := m.deg }

-- ── The DIVIDED-DIFFERENCE paired-mode family (WS-B2) ─────────────────────────
-- Near-degenerate composition without the `1/Δ` cancellation: each (λ, ν) coupling
-- is ONE fused paired atom `Re(c·e^{νd}·d·cexpm1((λ−ν)d))`, `c = a·r` bounded. The
-- τ·e resonance at λ=ν is the smooth series limit of `cexpm1`, no branch. Its own
-- uniform family (all one shape). Fixed-point realization validated in
-- `demos/divdiff_qdatapath.py` (qA: never form `e^{λd}−e^{νd}`; compute
-- `(e^z−1)/z` with the `−1` exact and `|z|≥thr`).

/-- A fused divided-difference paired mode: two poles `λ, ν` (`CplxE`, pole form
    `(−σ, ω)`) and a BOUNDED coeff `c = a·r` (no `1/Δ`). -/
structure PairedMode where
  lam : Sig × Sig
  nu  : Sig × Sig
  c   : Sig × Sig

/-- `voice ⋙ reverb` as fused paired modes — one per (λ, ν), `c = a·r` via `cmulE`
    only (NO `cdivE`, so no `1/Δ` is ever formed at build time; the division is
    deferred to the render's stable `cexpm1`). Inherently `m·n` (collecting would
    reintroduce the `1/Δ` that `residueComposeEC` suffers) — the stability trade for
    the factor-m saving. The near-degenerate form: a live pole sweeping through a
    room pole stays finite and reproduces the resonance (`demos/modal_divdiff.py`). -/
def residueComposeDD (voice reverb : Array ModalMode) : Array PairedMode :=
  voice.flatMap fun v => reverb.map fun r =>
    { lam := v.poleE, nu := r.poleE, c := cmulE v.ampE r.ampE }

/-- Multiply a `CplxE` by a real `Sig`. -/
def scaleRealE (s : Sig) (z : CplxE) : CplxE := (mul s z.1, mul s z.2)

/-- `(e^z − 1)/z` for a complex `z` as the Horner series `Σ_{k≥0} zᵏ/(k+1)!`, N=6
    terms — the STABLE branch for small `|z|`, where the `−1` in the direct form
    catastrophically cancels. Coeffs `1/(k+1)!`. Limit 1 at z=0 (the τ·e resonance).
    Validated: series↔direct discontinuity at the threshold is 2.5e-12. -/
def cexpm1SeriesE (z : CplxE) : CplxE :=
  let step := fun (ck : Float) (acc : CplxE) => caddE (litF ck, lit 0) (cmulE z acc)
  step 1.0 (step (1.0/2.0) (step (1.0/6.0) (step (1.0/24.0)
    (step (1.0/120.0) (step (1.0/720.0) (litF (1.0/5040.0), lit 0))))))

/-- The paired bank's coefficient columns — 7 (vs the single-pole `BankCols`'s 4):
    the ν rotator increment, the SIGNED difference (`ω_λ−ω_ν`) rotator increment,
    `σ_ν`, `σ_λ−σ_ν`, `ω_λ−ω_ν`, and the complex coeff. `poleE = (−σ, ω)`, so
    `σ_ν = −ν.1`, `σ_λ−σ_ν = ν.1−λ.1`, `ω_d = λ.2−ν.2`. -/
structure PairedBankCols where
  count : Nat
  live? : Option Sig := none
  idxId : Nat := 0
  incrNu   : Sig
  incrDiff : Sig
  sigmaNu  : Sig
  ds       : Sig
  wd       : Sig
  cre      : Sig
  cim      : Sig

structure PairedModeSym where
  incrNu   : Sig
  incrDiff : Sig
  sigmaNu  : Sig
  ds       : Sig
  wd       : Sig
  cre      : Sig
  cim      : Sig

def pairedBankCols (modes : Array PairedMode) (live? : Option Sig := none) : PairedBankCols where
  count := modes.size
  live? := live?
  incrNu   := Sig.arr (modes.map fun m => div (mul (div m.nu.2 twoPiE) (lit 4294967296)) .sampleRate)
  incrDiff := Sig.arr (modes.map fun m =>
    div (mul (div (sub m.lam.2 m.nu.2) twoPiE) (lit 4294967296)) .sampleRate)
  sigmaNu  := Sig.arr (modes.map fun m => neg m.nu.1)
  ds       := Sig.arr (modes.map fun m => sub m.nu.1 m.lam.1)
  wd       := Sig.arr (modes.map fun m => sub m.lam.2 m.nu.2)
  cre      := Sig.arr (modes.map fun m => m.c.1)
  cim      := Sig.arr (modes.map fun m => m.c.2)

def bankFoldPaired (cols : PairedBankCols) (body : PairedModeSym → Sig) : Sig :=
  let k := Sig.loopIdx cols.idxId
  Sig.bankSum cols.count
    #[cols.incrNu, cols.incrDiff, cols.sigmaNu, cols.ds, cols.wd, cols.cre, cols.cim]
    (body { incrNu := Sig.index cols.incrNu k, incrDiff := Sig.index cols.incrDiff k
          , sigmaNu := Sig.index cols.sigmaNu k, ds := Sig.index cols.ds k
          , wd := Sig.index cols.wd k, cre := Sig.index cols.cre k, cim := Sig.index cols.cim k })
    cols.live? cols.idxId

/-- The divided-difference paired-mode bank body (qA). Per mode: the ν rotator (exact
    integer phase) plus a SECOND integer-phase rotator at the signed difference
    frequency `ω_λ−ω_ν` for `e^z`; `cexpm1(z)` by a per-sample `selectE` between the
    direct `(e^z−1)/z` (`|z|²≥thr²=0.01`, guarded so the unused branch never divides
    by ~0) and the Horner series (small `|z|`); the complex weight
    `Wc = c·(e^{νd}·d·cexpm1)` lands in Q4.28 and combines with the ν oscillator in
    i64 — `modalBankSigTable`'s skeleton exactly, one float boundary scale. `envDf`
    and `e^z` stay float (never landed); `z`'s imaginary part uses raw `ω_d·dSec`
    (the divisor needs only relative precision; the rotator phase is integer-reduced,
    consistent because `e^z` is periodic). -/
def modalBankSigTableDD (modes : Array PairedMode) (clkInt anchorSamples : Sig)
    (live? : Option Sig := none) : Sig :=
  let clkRel := relClockQ clkInt anchorSamples
  let dSec := div (div (toFloatE clkRel) (lit 4294967296)) .sampleRate
  let bankQ := bankFoldPaired (pairedBankCols modes live?) fun m =>
    let phQnu := modePhaseQFromIncr (toIntE m.incrNu) clkRel
    let phQdf := modePhaseQFromIncr (toIntE m.incrDiff) clkRel
    let cosDf := div (toFloatE (fixedCosCycSig phQdf)) (lit 1073741824)   -- Q2.30 → float
    let sinDf := div (toFloatE (fixedSinCycSig phQdf)) (lit 1073741824)
    let envNu := expSig (neg (mul m.sigmaNu dSec))
    let envDf := expSig (neg (mul m.ds dSec))
    let ez : CplxE := (mul envDf cosDf, mul envDf sinDf)                  -- e^z (float)
    let z : CplxE := (neg (mul m.ds dSec), mul m.wd dSec)                 -- z = (λ−ν)d
    let zsq := add (mul z.1 z.1) (mul z.2 z.2)
    let big := gt zsq (litF 0.01)                                        -- |z|² ≥ thr² (thr=0.1)
    let zsafe : CplxE := (selectE big z.1 (lit 1), selectE big z.2 (lit 0))
    let direct := cdivE (csubE ez (lit 1, lit 0)) zsafe                  -- (e^z−1)/z
    let series := cexpm1SeriesE z
    let cx : CplxE := (selectE big direct.1 series.1, selectE big direct.2 series.2)
    let wc := cmulE (m.cre, m.cim) (scaleRealE (mul envNu dSec) cx)      -- c·(e^{νd}·d·cexpm1)
    let wCre := toIntE (mul wc.1 (lit 268435456))
    let wCim := toIntE (mul wc.2 (lit 268435456))
    rshift (sub (mul wCre (fixedCosCycSig phQnu)) (mul wCim (fixedSinCycSig phQnu))) (lit 28)
  selectE (gt clkRel (lit 0)) (fixedOutQ 30 bankQ) (lit 0)


-- ── The MODAL ISLAND (v1): a decaying-resonator bank as a term over the clock ──
-- The pole/modal island's emit path. A bank is a gated sum of decaying sinusoids
-- (`modalBankSig`) — the real part of Σ amp·e^{μd}. It needs NO new ArrowTerm
-- node: as a pure function of the warped clock it rides `arrUn … (.clk c)`, so
-- warps reclock it (affine = exact pole reclocking; nonlinear = the defined
-- varispeed case) and it is random-access by construction. The residue calculus
-- (voice ⋙ reverb) is a BUILD-TIME pass that fills the ModalMode array; the
-- runtime substrate is just this bank. Gated by `modal-bank`.

/-- Banks-as-data is the DEFAULT lowering: uniform (all deg-0) modal banks lower
    through the indexed reduction (`bankFold` — `modalBankSigTable` /
    `modalBankSigDirTable`) instead of unrolling. Sound because the banked
    render is bit-identical to the unroll (order-preserving loop, i64-modular
    sum — the `banks-as-data`/`banks-as-data-dir` gates pin it) and the
    coefficient columns are generation-buffered in FlatRuntime (no cross-column
    tear on live knob moves). `TROPICAL_BANKS_UNROLL` is the escape hatch back
    to the unrolled form (bisection ladder: the naive realization stays
    reachable). Reads the ONE shared flag (`Ir.banksEnabled`), read once at
    load, so the pure lowering may branch on it. -/
def banksTableEnabled : Bool := Tropical.Ir.banksEnabled

/-- A modal bank struck at `anchor` (samples) as a term over the clock leaf: no
    `gen`, no `.trop` instance — `{clk, +, ×, round, clamp, ldexp}` all the way
    down, and warp-reachable like any generator. Rides `arrUn … (.clk c)`, so
    warps reach the (banked or unrolled) body through the clock leaf identically. -/
def modalBankTerm (modes : Array ModalMode) (anchor : Sig) (c : Clock)
    (count? : Option Sig := none) : ArrowTerm :=
  -- A dynamic-count bank ALWAYS banks: a runtime count cannot be unrolled, so
  -- when `count?` is present the banked lowering fires regardless of
  -- TROPICAL_BANKS_UNROLL — the escape hatch governs only STATIC banks. (A
  -- non-uniform bank can't take the table lowering at all; it unrolls at
  -- capacity and the live count is dropped — graceful, and unreachable from
  -- the Playground, whose resonator banks are all deg-0.)
  let banked := bankIsUniform modes && (count?.isSome || banksTableEnabled)
  let lower := if banked
    then fun ms clk a => modalBankSigTable ms clk a count?
    else fun ms clk a => modalBankSig ms clk a
  ArrowTerm.arrUn (fun clkSig => lower modes clkSig anchor) (ArrowTerm.clk c)

-- ── The DIRECTION operator: forward↔reverse crossfade ─────────────────────────
-- A bank's reading DIRECTION crossfades between the CAUSAL tail (energy at d>0, a
-- forward ring) and the ANTI-CAUSAL tail (energy at d<0, a pre-verb blooming INTO
-- the strike). Both keep the mode's own σ AND ω — only which side of the strike the
-- energy sits on changes. (An earlier version ROTATED the pole `μ↦e^{iθ}μ`; elegant,
-- but for audio modes `ω≫σ`, so any interior angle throws ω into the damping and the
-- tail decays in microseconds — silent. The crossfade never touches the decay, so it
-- stays audible across the whole knob. The rotation's cost was invisible because the
-- gate checked "bounded", not "audible".)

/-- A modal bank read with a forward↔reverse DIRECTION crossfade `dir ∈ [0,1]`.
    Per mode: the CAUSAL ring `e^{−σd}` gated `d>0`, and the ANTI-CAUSAL ring
    `e^{+σd} = e^{−σ|d|}` gated `d<0` reading the time-mirrored oscillator (`cos`
    even, `sin` odd), blended `(1−dir)·forward + dir·reverse`. σ and ω are untouched,
    so no setting can swing the frequency into the damping. `dir=0` reduces
    bit-for-bit to the forward bank (`modalBankSig`); `dir=1` is its exact time-mirror
    (reverse reverb). `dampScale?` bends the decay clock (sway) on both sides. Pure
    `f(clk)`: no state. -/
def modalBankSigDir (modes : Array ModalMode) (clkInt : Sig) (anchorSamples : Sig)
    (dir : Sig) (dampScale? : Option Sig := none) : Sig :=
  let clkRel := relClockQ clkInt anchorSamples
  let dSec := div (div (toFloatE clkRel) (lit 4294967296)) .sampleRate
  -- Q datapath, same ledger as `modalBankSig` (Q4.28 weight landings, exact
  -- Q2.30 oscillators, i64 mode sums). The causal gates and the dir crossfade
  -- hoist OUT of the per-mode fold: the gate condition is per-bank (same
  -- clkRel for every mode) and the blend distributes over the sum — done once
  -- at the float boundary instead of per mode.
  let (fwdQ, revQ) := modes.foldl
    (fun (acc : Sig × Sig) m =>
      let phQ := modePhaseQ m.omega clkRel
      -- osc(−d) as the phase of the NEGATED relative clock (not cos-even/
      -- sin-odd), so `dir=1` is the forward tail's exact time-mirror — the
      -- fixed sine isn't bit-symmetric (one signed floor-shift), so the
      -- mirrored phase must be spelled out to match `fwd[2C−i]`. `modePhaseQ`
      -- is exact on the negated i64, so the rev side at `clkRel = −X`
      -- evaluates at bit-equal phase to the fwd side at `+X`.
      let phQN := modePhaseQ m.omega (neg clkRel)
      -- σ·d, optionally sway-bent (decay clock only, so pitch is untouched).
      let sd := mul m.sigma dSec
      let sd := match dampScale? with | none => sd | some s => mul sd s
      let envF := expSig (neg sd)
      let envF := if m.deg == 0 then envF else mul (powE dSec m.deg) envF
      let envR := expSig sd
      let envR := if m.deg == 0 then envR else mul (powE (neg dSec) m.deg) envR
      let wCreF := toIntE (mul (mul envF m.cre) (lit 268435456))
      let wCimF := toIntE (mul (mul envF m.cim) (lit 268435456))
      let wCreR := toIntE (mul (mul envR m.cre) (lit 268435456))
      let wCimR := toIntE (mul (mul envR m.cim) (lit 268435456))
      let fwdM := rshift (sub (mul wCreF (fixedCosCycSig phQ))
                              (mul wCimF (fixedSinCycSig phQ))) (lit 28)
      let revM := rshift (sub (mul wCreR (fixedCosCycSig phQN))
                              (mul wCimR (fixedSinCycSig phQN))) (lit 28)
      (add acc.1 fwdM, add acc.2 revM))
    (litI 0, litI 0)
  let fwd := selectE (gt clkRel (lit 0)) (fixedOutQ 30 fwdQ) (lit 0)
  let rev := selectE (gt (lit 0) clkRel) (fixedOutQ 30 revQ) (lit 0)
  add (mul (sub (lit 1) dir) fwd) (mul dir rev)

/-- The BANKED direction bank: `modalBankSigDir`'s value with BOTH mode sums as
    indexed reductions — the forward and reverse accumulators are two `bankFold`s
    over the SAME coefficient columns (a pair-valued fold is two scalar folds;
    each visits modes in array order, so each i64 sum agrees bit-for-bit with its
    unrolled half). This is NOT a hand table twin: both bodies are
    `modalBankSigDir`'s per-mode lambda split at the pair, written over the
    symbolic mode. Requires `deg == 0` uniformity (the `bankIsUniform` guard at
    dispatch), so the `powE` degree branches vanish, exactly as in
    `modalBankSigTable`. Same signature as `modalBankSigDir` — the dispatch in
    `modalBankTermDir` picks between them. -/
def modalBankSigDirTable (modes : Array ModalMode) (clkInt anchorSamples : Sig)
    (dir : Sig) (dampScale? : Option Sig := none) (live? : Option Sig := none) : Sig :=
  let clkRel := relClockQ clkInt anchorSamples
  let dSec := div (div (toFloatE clkRel) (lit 4294967296)) .sampleRate
  let cols := bankCols modes live?
  -- σ·d, optionally sway-bent (decay clock only, pitch untouched) — the same
  -- subterm both sides read, per symbolic mode.
  let sdOf := fun (m : ModeSym) =>
    let sd := mul m.sigma dSec
    match dampScale? with | none => sd | some s => mul sd s
  let fwdQ := bankFold cols fun m =>
    let phQ   := modePhaseQFromIncr (toIntE m.incr) clkRel
    let envF  := expSig (neg (sdOf m))
    let wCreF := toIntE (mul (mul envF m.cre) (lit 268435456))
    let wCimF := toIntE (mul (mul envF m.cim) (lit 268435456))
    rshift (sub (mul wCreF (fixedCosCycSig phQ))
                (mul wCimF (fixedSinCycSig phQ))) (lit 28)
  let revQ := bankFold cols fun m =>
    -- the mirrored phase spelled out on the NEGATED clock, as in the unrolled
    -- path (the fixed sine isn't bit-symmetric; see `modalBankSigDir`).
    let phQN  := modePhaseQFromIncr (toIntE m.incr) (neg clkRel)
    let envR  := expSig (sdOf m)
    let wCreR := toIntE (mul (mul envR m.cre) (lit 268435456))
    let wCimR := toIntE (mul (mul envR m.cim) (lit 268435456))
    rshift (sub (mul wCreR (fixedCosCycSig phQN))
                (mul wCimR (fixedSinCycSig phQN))) (lit 28)
  let fwd := selectE (gt clkRel (lit 0)) (fixedOutQ 30 fwdQ) (lit 0)
  let rev := selectE (gt (lit 0) clkRel) (fixedOutQ 30 revQ) (lit 0)
  add (mul (sub (lit 1) dir) fwd) (mul dir rev)

/-- The direction bank as a term over the clock leaf, with the LOWERING (unrolled
    or banked) supplied explicitly — shared by `modalBankTermDir` (which picks by
    flag) and the `banks-as-data-dir` equivalence gate (which builds both sides
    regardless of the flag). -/
def modalBankTermDirWith
    (lower : Array ModalMode → Sig → Sig → Sig → Option Sig → Sig)
    (modes : Array ModalMode) (anchor : Sig) (c : Clock)
    (dir : Sig) (damp? : Option (Sig × Sig) := none) : ArrowTerm :=
  ArrowTerm.arrUn
    (fun clkSig =>
      -- the sway LFO is a pure function of the SAME clock leaf the bank rides, so a
      -- master scrub reverses it coherently with the tail. Its phase is integer-
      -- reduced (`phasorPhaseSig`), never `rate·t` on unbounded float seconds.
      let dampScale? := damp?.map (fun (depth, rate) =>
        add (lit 1) (mul depth (sinSig (mul twoPiE (phasorPhaseSig rate (lit 0) clkSig)))))
      lower modes clkSig anchor dir dampScale?)
    (ArrowTerm.clk c)

/-- `modalBankTerm` with a DIRECTION crossfade. Rides the `.clk` leaf like
    `modalBankTerm`, so master warps still reach it. Dispatches to the banked
    lowering for uniform banks under the banks flag, like `modalBankTerm`. -/
def modalBankTermDir (modes : Array ModalMode) (anchor : Sig) (c : Clock)
    (dir : Sig) (damp? : Option (Sig × Sig) := none)
    (count? : Option Sig := none) : ArrowTerm :=
  -- Same dispatch rule as `modalBankTerm`: a dynamic count ALWAYS banks (a
  -- runtime count cannot be unrolled; TROPICAL_BANKS_UNROLL governs only
  -- static banks); non-uniform banks unroll at capacity, dropping the count.
  let lower : Array ModalMode → Sig → Sig → Sig → Option Sig → Sig :=
    if bankIsUniform modes && (count?.isSome || banksTableEnabled)
    then fun ms clk a d s? => modalBankSigDirTable ms clk a d s? count?
    else fun ms clk a d s? => modalBankSigDir ms clk a d s?
  modalBankTermDirWith lower modes anchor c dir damp?

/-- A bank's reading DIRECTION as data: the forward↔reverse crossfade `dir`
    (0 = forward tail, 1 = reverse/pre-verb) plus optional decay sway `(depth,
    rateHz)`. Attached to a `modalReverb` node, carried to `lowerInput`. -/
structure ModalDir where
  dir : Sig := lit 0
  damp : Option (Sig × Sig) := none

end Tropical.EmitArrow
