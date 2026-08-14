import Tropical.EmitArrow.ArenaModal.Residue

/-!
# EmitArrow.Modal.Bloom

Baked pitch-bloom atoms, certified coefficient depths, and seam classification.
-/

namespace Tropical.EmitArrow.ArenaNative

open Tropical.Ir
open Tropical.Exact (DyadicI CplxD CplxDI)

-- ── The bloom⋙reverb Γ-BRIDGE atom (WS-B3) ────────────────────────────────────
-- A pitch-bloomed mode feeding a reverb pole. The bloomed mode is the voice mode
-- on the bloom-warped clock, `Re(A·e^{μφ(d)})`, `φ(d) = d + B(1−e^{−gd})`;
-- composing with a reverb pole ν is the convolution `∫₀^d e^{ν(d−s)}e^{μφ(s)}ds`.
-- The Poisson-lattice expansion of `e^{−κe^{−gd}}` (κ = μB) is representation-
-- DEAD: its coefficients `(−κ)ⁿ/n!` are Taylor coefficients (unique — no
-- re-expansion escapes) cancelling from magnitude ~e^{|κ|}, and the shipped gong
-- runs |κ| = |μ|·B ≈ 19–178. Decay-spaced sidebands at one frequency are nearly
-- parallel atoms (Prony ill-conditioning); contrast `besselFuse`, whose
-- frequency-spaced sidebands Parseval bounds forever. The composition instead
-- has a closed form one RING EXTENSION up — the incomplete gamma `γ(a, κe^{−gd})`,
-- `a = (ν−μ)/g` — and after cancelling every power/log prefactor against the
-- carriers it is a branch-cut-free TWO-CARRIER atom (cockpit
-- `demos/modal_bloom_gamma.py`, ALL PASS: 3.7e-13 worst vs the independent
-- time-domain oracle, ≥11.8 digits over the (|a|,|κ|) box):
--
--   y(d) = Re[ c · ( K1·e^{νd} + K2(z(d))·e^{μφ(d)} ) ],   z(d) = κe^{−gd}
--   series side (|a+1| ≥ |z|):  K1 = C,       K2 = −M(1,a+1,z)/(ν−μ)
--   CF side     (|z| > |a+1|):  K1 = C − Γ★,  K2 = (Γ(a,z)·eᶻ·z^{−a})/g
--
-- C is the κ-side constant by the same rule at z = κ; the branches are bridged
-- EXACTLY by the d-constant Γ★ = Γ(a)·κ^{−a}·e^{κ}/g — the Γ(a) term the two
-- forms share (identity: `z^{−a}e^{z}e^{μφ(d)} = κ^{−a}e^{κ}e^{νd}`); its
-- e^{±π|Im a|/2} blowups cancel in the EXPONENT, so it is computed as
-- `exp(lgamma(a) − a·log κ + κ)/g` and is moderate. `z(d)` decays, so a CF pair
-- re-enters the series side at the baked `d_switch = ln(|κ|/|a+1|)/g` — a
-- per-sample `selectE`, the `cexpm1` discipline one floor up. κ→0 collapses to
-- the WS-B2 divided-difference atom: this is its κ-extension.

/-- Build-time complex `Float` — the `besselJ` tier: production BAKE-time
    numerics for the Γ-bridge constants, never per-sample. Local and minimal
    (no Mathlib). -/
structure CplxB where
  re : Float
  im : Float
deriving Inhabited, Repr

namespace CplxB
def add (a b : CplxB) : CplxB := ⟨a.re + b.re, a.im + b.im⟩
def sub (a b : CplxB) : CplxB := ⟨a.re - b.re, a.im - b.im⟩
def mul (a b : CplxB) : CplxB := ⟨a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re⟩
def neg (a : CplxB) : CplxB := ⟨-a.re, -a.im⟩
def scale (s : Float) (a : CplxB) : CplxB := ⟨s * a.re, s * a.im⟩
def normSq (a : CplxB) : Float := a.re * a.re + a.im * a.im
def div (a b : CplxB) : CplxB :=
  let d := b.normSq
  ⟨(a.re * b.re + a.im * b.im) / d, (a.im * b.re - a.re * b.im) / d⟩
end CplxB

/-- A build-time complex constant as a `CplxE` literal pair. -/
def cplxLitE (x : CplxB) : BuildM CplxE := do
  let real ← litF x.re
  let imag ← litF x.im
  pure (real, imag)

/-- The EXACT carrier's constant as a `CplxE` literal pair — `cplxLitE`'s twin
    and the one place the carrier meets the emit funnel. `litF` is UNCHANGED:
    the midpoint is taken here and the `x·10¹²` product still rounds INSIDE
    `litF`, so the emitted literal is bit-for-bit what the float path would have
    emitted from the same value. (`toDecimalMantissa` is NOT the analogue of
    `litF` and must not be substituted here — `litF` forms `x·1e12` in f64
    first, and that multiply rounds.)

    `Option`-valued because `DyadicI.toFloat` of poison is `0.0`, and a silent
    zero constant is bit-for-bit the `sigConstF? (x/0) = 0` pathology this whole
    campaign exists to delete, reintroduced one floor up. A caller that gets
    `none` must refuse the pair, never emit a fabricated zero. -/
def cplxLitD? (x : CplxDI) : BuildM (Option CplxE) := do
  if x.ok then
    let real ← litF (DyadicI.toFloat x.re)
    let imag ← litF (DyadicI.toFloat x.im)
    pure (some (real, imag))
  else pure none

/-- The `CplxE` unit — the Kummer Horner's seed and the CF's numerator. -/
def cOneE : BuildM CplxE := do
  let one ← lit 1
  let zero ← lit 0
  pure (one, zero)

/-- The fixed-depth Kummer `M(1, a+1, z)` Horner over the emitted reciprocals
    `invA[k] = 1/(a+k+1)` — THE series lane's expression, shared verbatim by the
    per-sample lane in `bloomComposedSig` (z live) and the live-pole lift's
    `mK = M(1,a+1,κ)` constant in `bloomCompose` (z = κ, a coefficient — s0), so
    the constant is computed by the SAME arithmetic the lane renders with. -/
def bloomM1E (invA : Array CplxE) (z : CplxE) : BuildM CplxE := do
  let one ← cOneE
  invA.foldrM (fun ik h => do
    let zik ← cmulE z ik
    let product ← cmulE zik h
    caddE one product) one

/-- The fixed-depth bottom-up continued fraction `CF(z) = Γ(a,z)eᶻz^{−a}` over
    the emitted constants `cfB` (b-terms, `z` added per level) and `cfN`
    (numerators) — THE CF lane's expression, shared verbatim by the per-sample
    lane in `bloomComposedSig` (z live) and the live-pole lift's `CF(κ)`
    constant in `bloomCompose` (z = κ, s0), so the constant is computed by the
    SAME arithmetic the lane renders with (WS-LP phase 2, `bloomM1E`'s twin).
    Requires `cfB.size = cfN.size + 1`. -/
def bloomCFE (cfB cfN : Array CplxE) (z : CplxE) : BuildM CplxE := do
  let kk := cfN.size
  let mut h : CplxE ← caddE z cfB[kk]!
  for jr in [0:kk] do
    let j := kk - 1 - jr
    let sum ← caddE z cfB[j]!
    let quotient ← cdivE cfN[j]! h
    h ← csubE sum quotient
  let one ← cOneE
  cdivE one h

/-- Is this `Sig` a pure s0 value — a function of knob slots and constants only
    (no `τ`/tick, no input, no bank machinery)? The live-pole lift's admission
    check: the lifted pair constants are correct (and Stage0-hoistable) only if
    the pole read is knob-invariant per sample. A GLIDED pole (`.sampleIndex`
    inside `glideExpr`) fails here — `settle` it first (the recorded WS-LP
    discipline; reverb rt60 is raw ⇒ s0, filter cutoff/resonance are glided).
    The frozen arena is classified once in child-before-parent order, so shared
    subgraphs are visited once. A dangling child or queried ID reads false and
    admission therefore fails closed. -/
def sigIsS0 (signal : Sig) : BuildM Bool := do
  let builder ← get
  let values := Id.run do
    let mut values : Array Bool := #[]
    for node in builder.exprs.nodes do
      let getS0 := fun (id : ExprId) => (values[id.idx]?).getD false
      let value := match node with
        | .num _ | .paramRef _ | .sampleRate => true
        | .unary _ arg => getS0 arg
        | .binary _ lhs rhs => getS0 lhs && getS0 rhs
        | .clamp value lo hi | .select value lo hi =>
            getS0 value && getS0 lo && getS0 hi
        | _ => false
      values := values.push value
    pure values
  pure ((values[signal.idx]?).getD false)

-- ── The EMITTED complex transcendentals (WS-LP: the live Γ★ bridge) ────────────
-- The bloom crossing's bake-time constants are lifted from build-time `CplxB` to
-- emitted `CplxE` (Sig × Sig) so a LIVE pole survives the crossing. The one new
-- op under this is `atan2E` (Numerics); everything else is mechanical CplxE.

/-- Complex log at Sig level: `⟨½·log|z|², atan2(Im, Re)⟩` — `logSig` supplies the
    modulus half, `atan2E` the phase. The live twin of `CplxB.log`. -/
def clogE (z : CplxE) : BuildM CplxE := do
  let real2 ← mul z.1 z.1
  let imag2 ← mul z.2 z.2
  let norm2 ← add real2 imag2
  let logarithm ← logSig norm2
  let half ← lit 5 1
  let real ← mul half logarithm
  let imag ← atan2E z.2 z.1
  pure (real, imag)

/-- Complex exp at Sig level: `e^{Re}·⟨cos Im, sin Im⟩`. The live twin of `CplxB.exp`. -/
def cexpE (z : CplxE) : BuildM CplxE := do
  let exponential ← expSig z.1
  let cosine ← cosSig z.2
  let real ← mul exponential cosine
  let sine ← sinSig z.2
  let imag ← mul exponential sine
  pure (real, imag)

/-- Complex log-gamma at Sig level — the EMITTED twin of `lgammaB`. Same Lanczos
    (g=7, n=9) core, same reflection for `Re z < ½` on the dominant half of
    `log sin πz`; the build-time `if z.re<½` / `if z.im<0` branches become `selectE`s
    (the unselected core lane may go non-finite off its region — the select discards
    it, the bloom's established discipline). Both `clogE`/`cexpE` reachable. -/
def lgammaE (z : CplxE) : BuildM CplxE := do
  let lanczos : Array Float := #[0.99999999999980993, 676.5203681218851,
    -1259.1392167224028, 771.32342877765313, -176.61502916214059,
    12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
    1.5056327351493116e-7]
  let core : CplxE → BuildM CplxE := fun z => do
    let one ← lit 1
    let zero ← lit 0
    let zz ← csubE z (one, zero)
    let initial ← litF lanczos[0]!
    let x ← (Array.range 8).foldlM (fun acc i => do
      let coefficient ← litF lanczos[i + 1]!
      let index ← litF (i + 1).toFloat
      let denominator ← caddE zz (index, zero)
      let quotient ← cdivE (coefficient, zero) denominator
      caddE acc quotient) (initial, zero)
    let sevenHalf ← litF 7.5
    let t ← caddE zz (sevenHalf, zero)
    let half ← lit 5 1
    let zzHalf ← caddE zz (half, zero)
    let logT ← clogE t
    let product ← cmulE zzHalf logT
    let left ← csubE product t
    let logX ← clogE x
    let normalization ← litF
      (DyadicI.toFloat ((DyadicI.log Tropical.Exact.twoPiI).shift (-1)))
    let right ← caddE logX (normalization, zero)
    caddE left right
  let pi ← lit 3141592653589793 15
  let zero ← lit 0
  let imNeg ← gt zero z.2                                    -- Im z < 0
  let piImag ← mul pi z.2
  let negativePiImag ← neg piImag
  let sReal ← selectE imNeg negativePiImag piImag
  let piReal ← mul pi z.1
  let negativePiReal ← neg piReal
  let sImag ← selectE imNeg piReal negativePiReal
  let s : CplxE := (sReal, sImag)
  let logTwo ← litF (DyadicI.toFloat Tropical.Exact.ln2I)
  let halfPi ← lit 15707963267948966 16
  let negativeHalfPi ← neg halfPi
  let logTwoImag ← selectE imNeg halfPi negativeHalfPi
  let log2i : CplxE := (logTwo, logTwoImag)
  let one ← lit 1
  let oneC : CplxE := (one, zero)
  let negativeTwo ← lit (-2)
  let scaledS ← scaleRealE negativeTwo s
  let expScaled ← cexpE scaledS
  let oneMinus ← csubE oneC expScaled
  let logOneMinus ← clogE oneMinus
  let sum ← caddE s logOneMinus
  let logsin ← csubE sum log2i
  let logPi ← litF (DyadicI.toFloat (DyadicI.log Tropical.Exact.piI))
  let reflectionLeft ← csubE (logPi, zero) logsin
  let reflectedArg ← csubE oneC z
  let reflectedCore ← core reflectedArg
  let reflected ← csubE reflectionLeft reflectedCore
  let base ← core z
  let half ← lit 5 1
  let useRefl ← gt half z.1                                -- Re z < ½
  let real ← selectE useRefl reflected.1 base.1
  let imag ← selectE useRefl reflected.2 base.2
  pure (real, imag)

/-- Γ★ = `exp(lgamma(a) − a·log κ + κ)/g` at Sig level — the live twin of
    `bloomGammaStar`, the bloom crossing's d-constant bridge between the two envelope
    branches (its `e^{±π|Im a|/2}` blowups cancel in the exponent). -/
def bloomGammaStarE (a kappa : CplxE) (g : Sig) : BuildM CplxE := do
  let one ← lit 1
  let inverseG ← div one g
  let logGamma ← lgammaE a
  let logKappa ← clogE kappa
  let product ← cmulE a logKappa
  let difference ← csubE logGamma product
  let exponent ← caddE difference kappa
  let exponential ← cexpE exponent
  scaleRealE inverseG exponential

-- ── The CERTIFIED DEPTHS: structure that cannot come from rounding ────────────
-- `bloomM1`/`bloomCF` above are iterate-until-tolerance loops, and the count
-- they return SIZES AN EMITTED ARRAY (`invA`, `cfB`/`cfN`) — so a one-ulp
-- platform difference in the stopping test is not a different last bit, it is a
-- different PROGRAM. That is the sharpest instance of the whole exact-bake
-- campaign, and it is fixed independently of the values: the same recurrences
-- run on the exact carrier, returning the count alone.
--
-- Both loops run on the POINT carrier (`CplxD`), not the enclosure. That is a
-- deliberate choice about which instrument fits: these are SELF-CORRECTING
-- recurrences — modified Lentz especially — that converge in floating point
-- because each step damps the last step's error. An interval cannot see that
-- damping; it tracks a worst case the recurrence has already forgotten, and the
-- enclosure widens about two bits per iteration until it poisons (measured: gone
-- by iteration ~100 at 128 bits, on a config the float loop settles at 108).
-- Certification is not what a depth wants anyway — a rigorous depth would need a
-- remainder bound on the TAIL, which the enclosure of the computed δ is not.
--
-- What the depth wants is REPRODUCIBILITY, and that is exactly what exact dyadic
-- arithmetic at a fixed rounding delivers: the same algorithm, the same count on
-- every platform, no libm, and 128 mantissa bits against a double's 53. The
-- enclosure stays where it earns its keep — the region thresholds below, where a
-- genuine separation question is being asked.
--
-- P2 landed the VALUES beside the depths, and the two carriers split the work
-- the way the lesson prescribes. `bloomCF`'s value collapsed INTO its depth loop
-- (`bloomCFPointD`, of which the depth is now the second projection) because
-- both belong on the point carrier. `bloomM1`'s did not: its value runs on the
-- ENCLOSURE (`bloomM1D`), summed to the count the point loop returns — the
-- series is admitted only where `|z| ≤ |a+1|`, so its terms decay monotonically
-- and an interval loses nothing following them.

/-- `bloomM1`'s term count, reproducibly. Same recurrence (`tₙ = tₙ₋₁·z/(a+n)`,
    stop when `|tₙ| ≤ tol·max(|s|,1)`), exact arithmetic. -/
def bloomM1DepthD (a z : CplxD) (tol : Dyadic) (cap : Nat := 4000) : Nat := Id.run do
  let mut s : CplxD := CplxD.one
  let mut t : CplxD := CplxD.one
  for n in [1:cap] do
    let some t' := CplxD.div (CplxD.mul t z) (CplxD.add a (CplxD.ofNat n)) | return cap
    t := t'
    s := CplxD.add s t
    if Dyadic.ble (CplxD.abs t) (tol * Dyadic.dmax (CplxD.abs s) 1) then return n
  return cap

/-- `M(1, a+1, z) = 1 + z/(a+1)(1 + z/(a+2)(…))` summed to a FIXED term count on
    the certified enclosure — the VALUE half of `bloomM1`, with the count coming
    from `bloomM1DepthD` above. The split is the campaign's two-instruments rule
    applied inside one function: the point carrier answers "how many terms, the
    same everywhere" (a size, so reproducibility), the enclosure answers "what is
    the sum, and how much of it is certified" (a value, so accuracy). Nothing
    here decides a size, so its stopping is not a cliff.

    Sound on the ENCLOSURE (unlike the continued fraction) because every value
    call site sits in the admitted region `|z| ≤ |a+1|`, where the terms decay
    monotonically and there is no cancellation for an interval to lose. The
    enclosure's WIDTH is then real information: it is the detector for the
    near-integer-`a` conditioning cliff documented at `bloomComposedSig` (a float
    Horner off by ~1e8 at `a = −0.98, |κ| = 76.8`). Nothing consumes that width
    yet — a width-based admission guard is the recorded follow-on. -/
def bloomM1D (a z : CplxDI) (n : Nat) : CplxDI := Id.run do
  let mut s : CplxDI := CplxDI.one
  let mut t : CplxDI := CplxDI.one
  for k in [1:n+1] do
    t := CplxDI.div (CplxDI.mul t z) (CplxDI.add a (CplxDI.ofNat k))
    s := CplxDI.add s t
  return s

/-- `bloomCF`'s Lentz depth, reproducibly. Same modified-Lentz iteration
    including its `tiny` renormalizations — those guard the FLOAT path against
    underflow and exact arithmetic does not need them, but they are kept verbatim
    because they participate in the count, and this function's contract is to
    reproduce `bloomCF`'s depth deterministically, not to improve on it. -/
def bloomCFPointD (a z : CplxD) (tol : Dyadic) (cap : Nat := 4000) : CplxD × Nat := Id.run do
  let tiny : Dyadic := Dyadic.ofFloat 1.0e-300
  let tinyC : CplxD := CplxD.ofDyadic tiny
  let two : CplxD := CplxD.ofNat 2
  let mut b : CplxD := CplxD.sub (CplxD.add z CplxD.one) a
  let mut c : CplxD := CplxD.ofDyadic (Dyadic.ofFloat 1.0e300)
  let mut d : CplxD := if (CplxD.normSq b).isZero then tinyC
                       else (CplxD.div CplxD.one b).getD tinyC
  let mut h : CplxD := d
  for i in [1:cap] do
    let ic := CplxD.ofNat i
    let an : CplxD := CplxD.mul ic (CplxD.sub a ic)             -- −i(i−a)
    b := CplxD.add b two
    d := CplxD.add (CplxD.mul an d) b
    if Dyadic.blt (CplxD.abs d) tiny then d := tinyC
    c := CplxD.add b ((CplxD.div an c).getD tinyC)
    if Dyadic.blt (CplxD.abs c) tiny then c := tinyC
    let some dInv := CplxD.div CplxD.one d | return (h, cap)
    d := dInv
    let delta := CplxD.mul d c
    h := CplxD.mul h delta
    if Dyadic.ble (CplxD.abs (CplxD.sub delta CplxD.one)) tol then return (h, i)
  return (h, cap)

/-- `bloomCF`'s Lentz depth — now the second component of the value loop above.
    The `h` accumulation cannot move this count: `h` never enters the stopping
    test, which reads only `delta`. -/
def bloomCFDepthD (a z : CplxD) (tol : Dyadic) (cap : Nat := 4000) : Nat :=
  (bloomCFPointD a z tol cap).2

/-- The two tolerances the depth loops stop at (the same doubles the `Float`
    twins default to). -/
def bloomM1TolD : Dyadic := Dyadic.ofFloat 1.0e-17
def bloomCFTolD : Dyadic := Dyadic.ofFloat 1.0e-15

/-- A build-time complex `Float` lifted into the exact ENCLOSURE — exactly (a
    finite double is a dyadic). The bridge the region thresholds cross while the
    values still live in `CplxB`. -/
def CplxB.toExact (a : CplxB) : CplxDI := CplxDI.ofFloats a.re a.im

/-- The same lift into the POINT carrier, for the depth loops. -/
def CplxB.toPoint (a : CplxB) : CplxD := CplxD.ofFloats a.re a.im

/-- `Γ★` on the exact carrier — the same three-term exponent, `CplxDI.lgamma`
    for the Lanczos half. Poisons (rather than fabricating a value) when `κ`
    cannot be certified away from the origin, since `log κ` does not exist
    there. -/
def bloomGammaStarD (a kappa : CplxDI) (g : DyadicI) : CplxDI :=
  CplxDI.scale (DyadicI.inv g)
    (CplxDI.exp (CplxDI.add (CplxDI.sub (CplxDI.lgamma a) (CplxDI.mul a (CplxDI.log kappa))) kappa))

-- ── The COINCIDENCE (`|a| < ½`) divided-difference constants (WS-A4, atom four) ──
-- The τ·e resonance — a room pole ON the settled partial (ν → μ, `a → 0`) — is a
-- REMOVABLE singularity: at `a = 0` the E1 numerator vanishes identically (0/0), so
-- the meaning is total there, it is only E1's `1/(ν−μ)` and the Γ★ bridge that
-- overflow. The fix is the a-divided-difference of the numerator, evaluated in the
-- SAME two-region split as the crossing: the CF branch (`bloomCF`) is already
-- coincidence-stable for large `z` (the Γ(a) pole cancels between `CF(z)` and
-- `CF(κ)` by the Γ★ identity), and for small `z` it degrades into the log region, so
-- there the series divided difference `Φ(a,z) = Σ dₙ zⁿ = (M_a(z)−eᶻ)/a` takes over
-- plus the τ·e secular `e^κ·(e^{νd}−e^{μd})/(ν−μ)` — a `residueComposeDD`-shaped
-- paired atom. Validated bit-clean over the whole (a,κ) box incl. a=0 exact and
-- lightly-damped long tails (`demos/modal_bloom_gamma.py`, `d_bg6`).

/-- `(eᶻ−1)/z` on the exact carrier. The series/direct split at `|z|² = 0.01` is
    an OVERLAP SWITCH — both branches are valid in an annulus around it — so it
    is decided from the enclosure's MIDPOINT, deterministically, and a
    near-threshold config may take either side without being wrong.

    The reciprocal factorials enter as EXACT rationals (`1/6`, `1/120`, … are not
    dyadic, so they are tight enclosures) where the float twin used their double
    roundings. That is a deliberate accuracy gain of ~1 ulp per coefficient, and
    it is why the coincident arms show a systematic ~1e-16 offset in the P2
    differential rather than agreeing to the last bit. -/
def cexpm1D (z : CplxDI) : CplxDI :=
  let recip := fun (k : Nat) => DyadicI.div DyadicI.one (DyadicI.ofNat k)
  if Dyadic.blt (DyadicI.mid (CplxDI.normSq z)) (Dyadic.ofFloat 0.01) then
    let step := fun (ck : DyadicI) (acc : CplxDI) =>
      CplxDI.add (CplxDI.ofI ck) (CplxDI.mul z acc)
    step DyadicI.one (step (recip 2) (step (recip 6) (step (recip 24)
      (step (recip 120) (step (recip 720) (CplxDI.ofI (recip 5040)))))))
  else CplxDI.div (CplxDI.sub (CplxDI.exp z) CplxDI.one) z

/-- The coincident series-DD coefficients on the exact carrier — the SAME exact
    rational recurrence, now evaluated exactly. `fₖ = 1/k!` is built by DIVISION
    rather than by multiplying a rounded reciprocal, so the enclosure stays tight
    down to `k ≈ 300`. No `1/a` is formed here either: the removable singularity
    stays removed analytically. -/
def bloomDCoefD (aC : CplxDI) (n : Nat) : Array CplxDI := Id.run do
  let mut out : Array CplxDI := #[]
  let mut dprev : CplxDI := CplxDI.zero   -- d₀
  let mut fprev : CplxDI := CplxDI.one    -- f₀ = 1/0!
  for k in [1:n+1] do
    let kI := DyadicI.ofNat k
    let kC := CplxDI.ofI kI
    let dk := CplxDI.div (CplxDI.sub (CplxDI.scale kI dprev) fprev)
                         (CplxDI.mul kC (CplxDI.add aC kC))
    out := out.push dk
    dprev := dk
    fprev := CplxDI.div fprev kC        -- fₖ = fₖ₋₁/k
  return out

/-- `Φ(a,κ)/g` on the exact carrier — the same two regimes, the same guard.

    TWO decisions are preserved VERBATIM rather than improved, on the campaign's
    own rule that a carrier cutover whose differential also contains a semantic
    change cannot tell you which one moved a value:

    * `|κ| < |a+1|` is re-decided HERE from the enclosure, even though it is the
      same predicate `classifyBloomPair` already decided in choosing
      `coincidentCrossing` vs `coincidentSubtle`. Two answers to one question is
      a coherence defect — it can only bite on a straddle, where both branches
      are analytically valid, so it is cosmetic today rather than a wrong number.
      Recorded, not fixed here: the repair is to pass the plan's region in, in
      its own commit.
    * Euler's γ stays the double literal, exact into the carrier. A certified γ
      belongs in `Exact/Const` beside π and ln 2, with its own re-derivation
      gate — a separate, cheap follow-on, and it only ever fires on the
      `|a|² < 1e-12` branch.

    The two guards use DIFFERENT instruments on purpose: the region split reads
    `certLt` (the classifier's instrument, and a genuine separation question),
    while the removable-singularity guard reads the midpoint (a pure conditioning
    switch — both arms are the same analytic function, and `certLt` there would
    silently take the UNSTABLE arm on a straddle). -/
def bloomPhiKappaOverGD (aC kappa cfK : CplxDI) (dCoef : Array CplxDI) (g : DyadicI) : CplxDI :=
  let invG := DyadicI.inv g
  if DyadicI.certLt (CplxDI.abs kappa) (CplxDI.abs (CplxDI.add aC CplxDI.one)) then
    CplxDI.scale invG
      (CplxDI.mul kappa
        (dCoef.foldr (fun dk acc => CplxDI.add dk (CplxDI.mul kappa acc)) CplxDI.zero))
  else
    let euler := DyadicI.ofFloat 0.5772156649015329
    let laOverA : CplxDI :=
      if Dyadic.blt (DyadicI.mid (CplxDI.normSq aC)) (Dyadic.ofFloat 1.0e-12)
      then CplxDI.ofI (DyadicI.neg euler)
      else CplxDI.div (CplxDI.lgamma (CplxDI.add aC CplxDI.one)) aC
    let waOverA := CplxDI.sub laOverA (CplxDI.log kappa)
    let w := CplxDI.mul aC waOverA
    CplxDI.scale invG
      (CplxDI.sub (CplxDI.mul (CplxDI.mul (CplxDI.exp kappa) (cexpm1D w)) waOverA) cfK)

-- ── The ROOM-CHAIN FOLD divided difference (WS-DDF) ───────────────────────────
-- A bloomed voice crossing a reverb CHAIN reassociates as (fold room1|>room2, then
-- cross the bloom once). The fold's collected residues carry a 1/Δ over the ROOM
-- poles (Δ = ν1−ν2); near-coincident rooms (two rooms sharing a mode region) drive
-- the residue `c/Δ` out of Q4.28 range as a VALUE (|c/Δ| > 8). MEASURED QUALIFIER
-- (the `ddfold` gate, arm A): at the BLOOMED site that does not yet break the
-- render — the bloom's per-sample K factor (≈ M(κ)/(gα) ≈ 1e-3 for rooms far from
-- the voice) divides the huge residue back down before anything lands, so the
-- collected fold stays finite to Δ ~ 1e-5 rad/s. The premise holds unqualified only
-- at the BARE-residue site (`modal_divdiff`'s D_dd2, where the residue lands with
-- no K factor in front of it). The fix below is the correct, modestly-tighter
-- form — and the transfer datum — not an urgent repair. The fix: the chain is
-- the DIVIDED DIFFERENCE of the bloom cross over the two room poles,
--   chain(d) = c·[Y0(μ,ν1;d) − Y0(μ,ν2;d)]/(ν1−ν2),
-- and since ν1≈ν2 ⟹ a1≈a2 it decomposes (cockpit `demos/modal_ddfold.py`, D_df2/3):
--   chain/c = e^{ν2 d}·[K1(a1)·d·cexpm1(Δd) + DDa(K1)/g] + DDa(K2)(z)/g·e^{μφ(d)}
-- reusing atom four's KIND of machinery one axis over — cexpm1 on the ν2 carrier,
-- an a-divided-difference on the Γ-bridge constants (SERIES side only: the rooms are
-- separated from the voice, |a|≫0, so `|a+1| ≥ |z|` throughout; no CF branch, no Γ★).

/-- The general-a divided-difference coefficients on the exact carrier (WS-DDF).
    Same stable recurrence, which never forms `M(a1) − M(a2)`. -/
def bloomFoldQCoefD (a1 a2 : CplxDI) (n : Nat) : Array CplxDI := Id.run do
  let mut out : Array CplxDI := #[]
  let mut pPrev : CplxDI := CplxDI.one    -- P₀
  let mut qPrev : CplxDI := CplxDI.zero   -- Q₀
  for k in [1:n+1] do
    let kC := CplxDI.ofNat k
    let a1k := CplxDI.add a1 kC
    let a2k := CplxDI.add a2 kC
    let qk := CplxDI.sub (CplxDI.div qPrev a2k) (CplxDI.div pPrev (CplxDI.mul a1k a2k))
    out := out.push qk
    pPrev := CplxDI.div pPrev a1k
    qPrev := qk
  return out

/-- `DDa(M(1,a+1,x)) = x·Horner(Q)` on the exact carrier. -/
def bloomFoldDDaMD (qcoef : Array CplxDI) (x : CplxDI) : CplxDI :=
  CplxDI.mul x (qcoef.foldr (fun q h => CplxDI.add q (CplxDI.mul x h)) CplxDI.zero)

/-- One composed (voice μ, reverb ν) pair of the Γ-bridge atom. Every per-pair
    constant is a `CplxE`/`Sig` (WS-LP): the BAKED path (`besselFuse` parity — B,
    g, both pole sets fold to Floats; a change relowers) fills them with
    `cplxLitE`/`litF`, byte-identical to the old `CplxB` fields; the LIVE-σ path
    (a live-rt60 reverb pole) fills them with s0 expressions of the pole's knob
    slot, so Stage0 hoists them to the coefficient kernel and turning rt60 never
    relowers. B and g stay structural Floats (the bloom's shape, not a pole).
    The amp `c = a_voice·r_reverb` is live as always and enters linearly.
    `cfN.isEmpty` ⟺ series-only (the CF branch is not emitted at all);
    otherwise `dSwitch > 0` and the per-sample select bridges at it. -/
structure BloomPair where
  muSigma  : Sig
  muOmega  : Sig
  nuSigma  : Sig
  nuOmega  : Sig
  bloomB   : Float
  gRate    : Float
  c        : CplxE
  kappa    : CplxE
  k1Ser    : CplxE          -- C  (the κ-side constant)
  k1Cf     : CplxE          -- C − Γ★  (= −CF(κ)/g)
  fSer     : CplxE          -- −1/(ν−μ)  (the series envelope's factor)
  dSwitch  : Sig            -- 0 ⇒ series-only
  invA     : Array CplxE    -- 1/(a+k), k = 1..N (series Horner reciprocals)
  cfB      : Array CplxE    -- (2j+1)−a, j = 0..K (CF b-constants; z is added per sample)
  cfN      : Array CplxE    -- (j+1)((j+1)−a), j = 0..K−1 (CF numerators)
  -- coincidence (`|a| < ½`, WS-A4). `coincident = false` ⇒ the fields below are
  -- unused and the crossing/series-only per-sample body runs. When true: the CF
  -- branch (large z, `k1Cf`/`cfB`/`cfN`) bridges to the series-DD branch (small z)
  -- across `dSwitch`, and the τ·e secular rides a straight-μ carrier.
  coincident : Bool := false
  dCoef      : Array CplxE := #[]     -- dₙ, n = 1..N (the a-divided-difference Horner coeffs)
  k1SerDD    : CplxE := default  -- Φ(a,κ)/g (series-DD e^{νd} const, via the lgamma(a+1) bridge)
  eKappa     : CplxE := default  -- e^κ (the secular coeff c·e^κ)
  -- (μ−ν as the τ·e secular's z-coefficients: (Re, −Im) — a FIELD so the baked
  -- path emits the same single literals as before the CplxE lift.)
  secCoef    : CplxE := default
deriving Inhabited

/-- The bloom crossing's region for one composed (μ, ν) pair — a TOTAL partition
    of config space (`classifyBloomPair` is total, never `Option`). Dispatch is an
    exhaustive match, so a region without a handler is a compile error, and the
    depth-cap refusal is a NAMED outcome (`excludedDepth`) the coverage gate tallies
    rather than a silent `continue`. This is the type the island's totality claim
    hangs off. The four served regions differ only in which per-pair lanes the
    realizer emits; the two axes are coincidence (`|a| < ½`, the pole on the
    settled partial, τ·e) and the CF/series boundary (`|κ|` vs `|a+1|`, whether the
    continued fraction is reached). -/
inductive SeamRegion where
  /-- `¬coincident ∧ |a+1| ≥ |κ|`: the E1 Kummer series alone (`M` Horner over
      `invA`). The CF is never reached — no CF lane emitted. -/
  | serOnly
  /-- `¬coincident ∧ |a+1| < |κ|`: the E1 series bridged to the continued fraction
      at `dSwitch > 0` — both lanes emitted, `selectE` picks per sample. -/
  | crossing
  /-- `|a| < ½ ∧ |κ| ≥ |a+1|` (τ·e, WS-A4): the CF (large z) bridged to the
      series-DD branch (small z) at `dSwitch > 0`, plus the τ·e secular. The E1
      series lane (`invA`/`k1Ser`, singular at a = 0) is NOT emitted. -/
  | coincidentCrossing
  /-- `|a| < ½ ∧ |κ| < |a+1|` (a subtle bloom, `dSwitch < 0`): the per-sample path
      starts on series-DD at d = 0 and never crosses, so the CF lane is dead (the
      `selectE` is const-true) and is NOT emitted — the tightest region, series-DD
      + secular only. -/
  | coincidentSubtle
  /-- The E1 Horner would cross to the CF while `a` lies in the measured
      ill-conditioned disc around one of `-1, …, -300`.  At the concrete
      `a = -0.98`, `|κ| ≈ 72.2` witness the f64 Horner is wrong by more than
      eight orders of magnitude.  This is a numerical refusal, not a semantic
      zero and not a depth-cap event. -/
  | excludedConditioning
  /-- Envelope depth over the shared cap. The checked composer treats this as an
      explicit whole-composition refusal; the coverage gate reports its count. -/
  | excludedDepth
deriving Inhabited, DecidableEq

/-- Short label for the coverage gate's region histogram. -/
def SeamRegion.label : SeamRegion → String
  | .serOnly            => "serOnly"
  | .crossing           => "crossing"
  | .coincidentCrossing => "coincidentCrossing"
  | .coincidentSubtle   => "coincidentSubtle"
  | .excludedConditioning => "excludedConditioning"
  | .excludedDepth      => "excludedDepth"

/-- Radius of the stop-line around the negative-integer `a` lattice.  The
    measured failure is at distance `0.02` from `-1`; `1/32 = 0.03125` is the
    next outward exact binary radius, giving a reproducible 1.5625× guard.
    This is intentionally a measured admission boundary, not a claim that the
    Horner error has an analytic bound outside it. -/
def bloomConditioningRadius : Float := 0.03125

/-- One source of truth for the emitted Horner/CF depth limit and for the
    negative-integer lattice whose denominators those arrays can materialize. -/
def bloomDepthCap : Nat := 300

/-- The inspected conditioning lattice is exactly `-1, …, -bloomDepthCap`. -/
def bloomConditioningLatticeDepth : Nat := bloomDepthCap

/-- Point diagnostic reported by the seam gate: distance from `a` to the
    nearest negative integer represented by the bounded Horner. -/
def bloomConditioningMetric (a : CplxB) : Float := Id.run do
  let mut d := Float.sqrt ((a.re + 1.0) * (a.re + 1.0) + a.im * a.im)
  for j in [1:bloomConditioningLatticeDepth] do
    let n := (j + 1).toFloat
    d := min d (Float.sqrt ((a.re + n) * (a.re + n) + a.im * a.im))
  return d

/-- Evidence-carrier version of the conditioning stop-line for a baked pair.
    A pair is refused only where the CF crossing is reached; `κ = 0` and other
    series-only pairs retain their exact degeneration.  Failure to certify that
    the point lies outside the disc fails toward refusal. -/
def bloomExcludedConditioningD (a kappa : CplxDI) : Bool := Id.run do
  let absAP1 := CplxDI.abs (CplxDI.add a CplxDI.one)
  let absKappa := CplxDI.abs kappa
  if absKappa.ok && absKappa.lo.isZero && absKappa.hi.isZero then return false
  -- Equality/overlap is not a certificate that the crossing is unreachable.
  -- Fail open only with positive evidence that `|a+1| > |κ|`; otherwise the
  -- disc test decides and uncertainty fails toward refusal.
  if DyadicI.certGt absAP1 absKappa then return false
  let radius := DyadicI.ofFloat bloomConditioningRadius
  for j in [0:bloomConditioningLatticeDepth] do
    let n := CplxDI.ofNat (j + 1)
    if !DyadicI.certGt (CplxDI.abs (CplxDI.add a n)) radius then
      return true
  return false

/-- Interval-aware stop-line for live room damping. `Re a` walks the horizontal
    segment `[reLo,reHi]`; for each negative integer the closest point is its
    projection onto that segment. A disc intersection and CF reachability are
    certified independently anywhere on the interval, so uncertainty can only
    over-refuse, never miss an edge whose two witnesses occur at different points. -/
def bloomExcludedConditioningLive (reLo reHi imA : Float) (kappa : CplxB) : Bool := Id.run do
  let lo := DyadicI.ofFloat (min reLo reHi)
  let hi := DyadicI.ofFloat (max reLo reHi)
  let im := DyadicI.ofFloat imA
  let absAt := fun (re c : DyadicI) =>
    DyadicI.sqrt (DyadicI.add (DyadicI.square (DyadicI.add re c)) (DyadicI.square im))
  let radius := DyadicI.ofFloat bloomConditioningRadius
  let absKappa := CplxDI.abs kappa.toExact
  if absKappa.ok && absKappa.lo.isZero && absKappa.hi.isZero then return false
  -- Conservative conjunction of two interval facts.  The crossing and disc
  -- witnesses need not be the same endpoint: coupling them at the disc-centre
  -- projection can miss a segment whose right disc edge enters the crossing.
  let reCross := DyadicI.max lo (DyadicI.min hi (DyadicI.neg DyadicI.one))
  if DyadicI.certGt (absAt reCross DyadicI.one) absKappa then return false
  for j in [0:bloomConditioningLatticeDepth] do
    let n := DyadicI.ofNat (j + 1)
    let re := DyadicI.max lo (DyadicI.min hi (DyadicI.neg n))
    let discDistance := absAt re n
    if !DyadicI.certGt discDistance radius then
      return true
  return false

/-- The per-pair classification + baking data for one composed (μ, ν) pair — the
    executable form of the bloom atom's region partition (the sprint's epistemics
    fix: the boundary is an apparatus output, not a comment). TOTAL: `region` is
    always one of `SeamRegion`'s constructors (`excludedDepth` for a depth-cap
    refusal, never a `none`). The checked composer inspects that region before
    materialization, and supported regions emit exactly their own lanes; the
    seam-sweep harness and the coverage gate consult the SAME classifier, so
    "which region is this pair, and does the atom promise anything there" has
    one answer, in code. `nDepth`/`kDepth` size the
    coefficient arrays (`kDepth = 0` where no CF is reached). -/
structure BloomPairPlan where
  mu      : CplxB
  nu      : CplxB
  aC      : CplxB
  kappa   : CplxB
  region  : SeamRegion
  /-- `invA`/`dCoef` length / series depth (`nRaw + 8`). -/
  nDepth  : Nat
  /-- CF depth (`kRaw + 8`); 0 where no CF lane is emitted. -/
  kDepth  : Nat
deriving Inhabited

/-- Classify one composed (μ, ν) pair into its `SeamRegion` and size its depths —
    TOTAL (replaces the `Option`-with-flags `bloomPairPlan?`: depth and measured
    conditioning exclusions are typed regions, not `none`). Away from the new
    conditioning stop-line and exact `κ = 0` identity, the depth control flow is
    preserved — same `zBnd`, same `nRaw`/`kRaw` depth caps. The
    two axes (coincidence `|a| < ½` and the CF boundary `|κ|` vs `|a+1|`) name the
    branches the old flags already encoded.

    EVERY comparison and both depths here are computed on the exact carrier: this
    function decides which lanes a pair emits and how long its coefficient arrays
    are, so its answer must be a function of the configuration alone. The region
    axes (`|a|` against ½, `|a+1|` against `|κ|`) are OVERLAP SWITCHES — the two
    schemes agree in an annulus around each threshold, so a pair whose enclosures
    straddle may take either side without being wrong, and the side is picked
    deterministically. The DEPTHS are not overlap switches, which is why they go
    through `bloomM1DepthD`/`bloomCFDepthD`. -/
def classifyBloomPair (mu nu : CplxB) (B g : Float) : BloomPairPlan := Id.run do
  let aC : CplxB := (nu.sub mu).scale (1.0 / g)
  -- WS-A4 (atom four): the `|a| < ½` coincidence is served by the coincident
  -- divided difference; `|κ| < |a+1|` (a subtle bloom — see `bloomPhiKappaOverG`)
  -- is `dSwitch < 0`, where the per-sample path starts on the series-DD lane at
  -- d = 0 and the CF lane is dead.
  let aD := aC.toExact
  let coincident := DyadicI.certLt (CplxDI.abs aD) (DyadicI.ofFloat 0.5)
  let kappa := mu.scale B
  let absAP1 := CplxDI.abs (CplxDI.add aD CplxDI.one)
  let absKappa := CplxDI.abs kappa.toExact
  -- κ=0 is the exact identity M(1,a+1,0)=1 for every a, including the
  -- negative-integer lattice where evaluating unused Horner reciprocals would
  -- divide by zero. Emit the empty Horner explicitly.
  if !coincident && absKappa.ok && absKappa.lo.isZero && absKappa.hi.isZero then
    return { mu, nu, aC, kappa, region := .serOnly, nDepth := 0, kDepth := 0 }
  if bloomExcludedConditioningD aD kappa.toExact then
    return { mu, nu, aC, kappa, region := .excludedConditioning,
             nDepth := 0, kDepth := 0 }
  let serOnly := !coincident && !DyadicI.certLt absAP1 absKappa
  let excluded : BloomPairPlan :=
    { mu, nu, aC, kappa, region := .excludedDepth, nDepth := 0, kDepth := 0 }
  -- The worst z either per-sample branch evaluates: the branch boundary. `zBnd`
  -- is not itself emitted — it is the ARGUMENT the depth loops are sized at, so
  -- it inherits their carrier and their question. That is why it runs on the
  -- POINT carrier rather than the enclosure: a depth wants reproducibility, and
  -- `|a+1|/|κ|` computed in f64 (two libm square roots and a division) is the
  -- last input to an emitted ARRAY SIZE that a platform could have moved.
  let aP := aC.toPoint
  let kP := kappa.toPoint
  let ratio := (Dyadic.divRel? .down Tropical.Exact.workingPrec
                  (CplxD.abs (CplxD.add aP CplxD.one)) (CplxD.abs kP)).getD 1
  let zBnd := if serOnly then kP else CplxD.scale ratio kP
  let nRaw := bloomM1DepthD aP zBnd bloomM1TolD
  if nRaw + 8 > bloomDepthCap then return excluded
  if serOnly then
    return { mu, nu, aC, kappa, region := .serOnly, nDepth := nRaw + 8, kDepth := 0 }
  else
    let kRaw := bloomCFDepthD aP zBnd bloomCFTolD
    if kRaw + 8 > bloomDepthCap then return excluded
    -- non-coincident here is always the CF-bridged crossing; coincident splits on
    -- the CF boundary (`dSwitch` sign = sign of `|κ| − |a+1|`): `|κ| ≥ |a+1|`
    -- reaches the CF (coincidentCrossing), else the CF lane is dead (subtle).
    let region : SeamRegion :=
      if !coincident then .crossing
      else if !DyadicI.certLt absKappa absAP1 then .coincidentCrossing
      else .coincidentSubtle
    return { mu, nu, aC, kappa, region, nDepth := nRaw + 8, kDepth := kRaw + 8 }

/-- The bloom atom's admission predicate at the pair level: both envelope depths
    within the 300 cap — exactly the region `bloomCompose` keeps a pair (the
    classifier lands in a served region, not `excludedDepth`). TOTAL over the `|a|`
    axis since WS-A4: the `|a| < ½` coincidence is served by the divided-difference
    branch, and the ½ boundary is a scheme crossover, not an exclusion edge.
    Executable data the sweep probes at its edges, not an annotation. -/
def bloomAdmitsPair (mu nu : CplxB) (B g : Float) : Bool :=
  match (classifyBloomPair mu nu B g).region with
  | .excludedConditioning | .excludedDepth => false
  | _ => true


end Tropical.EmitArrow.ArenaNative
