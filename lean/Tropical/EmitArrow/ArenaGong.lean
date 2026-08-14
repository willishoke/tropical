import Tropical.EmitArrow.ArenaPatch

/-!
# EmitArrow.Gong — the struck resonator

A gong strike is two closed-form ideas stacked on the modal island, neither
of them new machinery:

* **amplitude bloom** — a register whose energy RISES after the strike is a
  difference of exponentials `e^{−d₁d} − e^{−(d₁+d₂)d}`, i.e. a PAIR of
  ordinary modes (same ω, amps `+a`/`−a`, decays `d₁`/`d₁+d₂`). Strictly
  in-basis; the bank neither knows nor cares.
* **pitch bloom** — the strike rides sharp and settles as it decays:
  instantaneous frequency `ω(1 + β·e^{−gd})`, integrated analytically, is
  the clock warp `d ↦ d + β(1−e^{−gd})/g`. Applied as a per-strike `warp`
  around the bank term, so the slide lands it on the bank's clock leaf and
  a master scrub reverses the glide coherently with the tail.

(The former third idea — an odd-polynomial "alias-free drive" — is retired:
a static memoryless waveshaper is not warmth, and the `shaper` vocabulary
kind that exposed it is gone with it. The gong is linear per strike; its
nonlinearity is velocity coupling at build time.)

Velocity enters at build time (bank amps, bloom depth β) — the
score's strikes are baked; the live upgrade is amps/β as `paramRef`s.
-/

namespace Tropical.EmitArrow.ArenaNative

open Tropical.Ir
open Tropical.Exact (DyadicI)

-- ── The pitch-bloom clock warp ────────────────────────────────────────────

/-- The analytic pitch-bloom warp: advance the clock by the bounded offset
    `bloom(d) = scale·β·(1 − e^{−g·d⁺})/g` seconds, `d` the strike-relative
    time. Added to the UNTOUCHED integer clock (never a float round-trip of
    the absolute coordinate), so `β = 0` is the identity exactly and the
    far-field precision profile is the bank's own. `d⁺ = clamp(d, 0, ·)`
    keeps the exponential's argument bounded pre-strike (`d < 0` would blow
    `e^{−gd}` up and wrap the `toInt` — the causal gate must not depend on
    i64 overflow behavior). `W(0) = 0` and `W` is monotone, so the bank's
    own `clkRel > 0` gate fires at the strike exactly as unwarped.
    `scale` lets a stiffer register (high partials) take a fraction of the
    glide. -/
def gongBloomWarp (anchorSamples beta g : Sig) (scale : Float) : Clock → BuildM Clock :=
  fun clk => do
    let clkRel ← relClockQ clk anchorSamples
    let clkFloat ← toFloatE clkRel
    let twoPow32 ← lit 4294967296
    let scaledClock ← div clkFloat twoPow32
    let sr ← sampleRate
    let dSec ← div scaledClock sr
    let zero ← lit 0
    let horizon ← lit 1000000
    let dPos ← clampE dSec zero horizon
    let scaleSig ← litF scale
    let scaledBeta ← mul beta scaleSig
    let gd ← mul g dPos
    let negativeGd ← neg gd
    let decay ← expSig negativeGd
    let one ← lit 1
    let rise ← sub one decay
    let normalizedRise ← div rise g
    let bloom ← mul scaledBeta normalizedRise
    let samples ← mul bloom sr
    let fixed ← mul samples twoPow32
    let shift ← toIntE fixed
    add clk shift

-- ── The default strike (a bare gong node, no score data) ──────────────────

/-- A modest built-in gong voiced at `f0` — the bank a bare `gong` node
    (no `modes_*` params) strikes at its anchor. Deterministic (no build-time
    RNG): near-harmonic low partials, a small inharmonic mid cluster on
    golden-ratio spacing, and four blooming high pairs. Returns
    `(full-glide, half-glide)` register tables. This is the kind's audible
    default (every vocabulary kind sounds when dropped), and it keeps the
    master-clock slots read in the minimal patch (an EMPTY gong would emit
    no generator at all). -/
def defaultGongModes (f0 : Float) : BuildM (Array ModalMode × Array ModalMode) := do
  -- The bake layer's libm exile reaches a SERVED kind here: these literals are
  -- what a bare `gong` node emits, so they must be a function of `f0` alone and
  -- not of the host's trig. `DyadicI.pow` is `exp(y·ln x)` where libm's `pow` is
  -- separately rounded, so the amplitudes move by ~1e-16 relative — the carrier
  -- being deterministic, not better, is the claim.
  let ex := DyadicI.toFloat
  let mode := fun (f sigma amp ph : Float) => do
    let phD := DyadicI.ofFloat ph
    let ampD := DyadicI.ofFloat amp
    let sigma ← litF sigma
    let omega ← litF (ex (DyadicI.mul Tropical.Exact.twoPiI (DyadicI.ofFloat f)))
    let cre ← litF (ex (DyadicI.mul ampD (DyadicI.cos phD)))
    let cim ← litF (ex (DyadicI.mul ampD (DyadicI.sin phD)))
    pure ({ sigma, omega, cre, cim } : ModalMode)
  -- `c / x^p` as one certified quotient (`pow` needs a certifiably positive
  -- base, which every ratio here is)
  let rolloff := fun (c x p : Float) =>
    ex (DyadicI.div (DyadicI.ofFloat c)
                    (DyadicI.pow (DyadicI.ofFloat x) (DyadicI.ofFloat p)))
  let lowRatios : Array Float := #[1.0, 1.51, 2.07, 2.63, 3.21]
  let mut full : Array ModalMode := #[]
  for i in [0:lowRatios.size] do
    full := full.push (← mode (f0 * lowRatios[i]!) (0.2 + 0.12 * i.toFloat)
      (rolloff 1.0 (i.toFloat + 1.0) 0.7) (2.399963 * i.toFloat))
  for j in [0:8] do
    let r := 3.0 + 0.9 * j.toFloat
    full := full.push (← mode (f0 * r) (0.5 + 0.09 * j.toFloat)
      (rolloff 0.3 r 0.8) (2.399963 * (j.toFloat + 5.0)))
  let mut half : Array ModalMode := #[]
  for j in [0:4] do
    let r := 8.0 + 3.5 * j.toFloat
    let a := rolloff 0.18 r 0.5
    let ph := 2.399963 * (j.toFloat + 13.0)
    let d1 := 0.5 + 0.1 * j.toFloat
    half := half.push (← mode (f0 * r) d1 a ph)
    half := half.push (← mode (f0 * r) (d1 + 2.5) (-a) ph)
  return (full, half)

-- ── The strike, composed from existing node kinds ─────────────────────────

/-- One gong strike as patch nodes: two anchored modal banks (the full-glide
    registers and the stiff half-glide register), each read through its own
    pitch-bloom warp, mixed. NOTHING here is a new lowering — the banks are
    `modalSource`s, the warps are `warpFx`s, the mix is `mix`; the slide
    composes the warp into each bank's clock leaf as for any effect. Helper
    ids carry the `__` prefix (skipped by taps). Empty registers are simply
    omitted; both empty ⇒ `mix #[]` (the graceful-silence contract). -/
def gongStrikeNodes (id : String) (clk : Clock) (anchorSamples beta g : Sig)
    (full half : Array ModalMode) : Node × Array PatchNode := Id.run do
  let mut extras : Array PatchNode := #[]
  let mut ins : Array String := #[]
  let register := fun (extras : Array PatchNode) (ins : Array String)
      (tag : String) (modes : Array ModalMode) (scale : Float) =>
    if modes.isEmpty then (extras, ins) else
    let src := s!"__gong_{id}_{tag}"
    let wf := s!"__gong_{id}_{tag}w"
    ( extras.push { id := src, node := .modalSource modes anchorSamples clk none none }
        |>.push { id := wf, node := .warpFx src (gongBloomWarp anchorSamples beta g scale) }
    , ins.push wf )
  let (e1, i1) := register extras ins "a" full 1.0
  let (e2, i2) := register e1 i1 "b" half 0.5
  extras := e2; ins := i2
  return (.mix ins, extras)

end Tropical.EmitArrow.ArenaNative
