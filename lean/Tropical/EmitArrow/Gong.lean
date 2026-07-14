import Tropical.EmitArrow.Patch

/-!
# EmitArrow.Gong — the struck nonlinear resonator

A gong strike is three closed-form ideas stacked on the modal island, none
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
* **warmth without aliasing** — an odd POLYNOMIAL fit of the tanh drive
  curve. A degree-p polynomial multiplies bandwidth by exactly p, so with
  the mode bank capped at `f_max` and `p·f_max < rate/2` the shaper's
  output is bandlimited BY CONSTRUCTION — there is no component to fold.
  (tanh's order-7+ products fold back in-band and, under the pitch bloom,
  glide the wrong way — low energy, high salience.)

Velocity enters at build time (bank amps, bloom depth β, drive) — the
score's strikes are baked; the live upgrade is amps/β as `paramRef`s.
-/

namespace Tropical.EmitArrow

open Tropical.Ir

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
def gongBloomWarp (anchorSamples beta g : Sig) (scale : Float) : Clock → Clock :=
  fun clk =>
    let clkRel := relClockQ clk anchorSamples
    let dSec := div (div (toFloatE clkRel) (lit 4294967296)) .sampleRate
    let dPos := clampE dSec (lit 0) (lit 1000000)
    let bloom := mul (mul beta (litF scale))
      (div (sub (lit 1) (expSig (neg (mul g dPos)))) g)
    add clk (toIntE (mul (mul bloom .sampleRate) (lit 4294967296)))

-- ── The alias-free drive: odd-polynomial fit of the tanh curve ────────────

/-- Least-squares odd-polynomial (degree 5) fit of `tanh(k·u)/tanh(k)` on
    `u ∈ [−1, 1]` — the same curve the tanh drive shapes, truncated to the
    orders a bandlimited input keeps under Nyquist. Build-time Float math:
    normal equations over the `{u, u³, u⁵}` basis on a 4001-point grid,
    solved by Cramer. Returns `(c₁, c₃, c₅)`. -/
def polyFitTanhOdd (k : Float) : Float × Float × Float := Id.run do
  let n : Nat := 4001
  let tk := Float.tanh k
  -- normal equations G·c = b over the odd basis powers (1, 3, 5)
  let mut gm : Array Float := Array.replicate 9 0.0
  let mut bv : Array Float := Array.replicate 3 0.0
  for i in [0:n] do
    let u := -1.0 + 2.0 * i.toFloat / (n - 1).toFloat
    let t := Float.tanh (k * u) / tk
    let p : Array Float := #[u, u * u * u, u * u * u * u * u]
    for r in [0:3] do
      bv := bv.set! r (bv[r]! + p[r]! * t)
      for c in [0:3] do
        gm := gm.set! (r * 3 + c) (gm[r * 3 + c]! + p[r]! * p[c]!)
  let det := fun (m : Array Float) =>
      m[0]! * (m[4]! * m[8]! - m[5]! * m[7]!)
    - m[1]! * (m[3]! * m[8]! - m[5]! * m[6]!)
    + m[2]! * (m[3]! * m[7]! - m[4]! * m[6]!)
  let d := det gm
  let col := fun (j : Nat) => det (Array.range 9 |>.map fun i =>
    if i % 3 == j then bv[i / 3]! else gm[i]!)
  (col 0 / d, col 1 / d, col 2 / d)

/-- The shaper body: `peak · P(s/peak)` with `P(u) = c₁u + c₃u³ + c₅u⁵`
    (Horner in `u²`). `peak` is a FIXED reference (the vel-1.0 strike's
    measured peak), not a per-render normalization — that's what keeps
    velocity dynamics: a soft strike stays in the polynomial's linear
    region instead of being renormalized into the knee. -/
def polyShapeSig (c1 c3 c5 peak : Float) (s : Sig) : Sig :=
  let u := mul s (litF (1.0 / peak))
  let u2 := mul u u
  mul (litF peak)
    (mul u (add (litF c1) (mul u2 (add (litF c3) (mul u2 (litF c5))))))

-- ── The default strike (a bare gong node, no score data) ──────────────────

/-- A modest built-in gong voiced at `f0` — the bank a bare `gong` node
    (no `modes_*` params) strikes at its anchor. Deterministic (no build-time
    RNG): near-harmonic low partials, a small inharmonic mid cluster on
    golden-ratio spacing, and four blooming high pairs. Returns
    `(full-glide, half-glide)` register tables. This is the kind's audible
    default (every vocabulary kind sounds when dropped), and it keeps the
    master-clock slots read in the minimal patch (an EMPTY gong would emit
    no generator at all). -/
def defaultGongModes (f0 : Float) : Array ModalMode × Array ModalMode := Id.run do
  let mode := fun (f sigma amp ph : Float) =>
    ({ sigma := litF sigma, omega := litF (6.283185307179586 * f),
       cre := litF (amp * Float.cos ph), cim := litF (amp * Float.sin ph) }
     : ModalMode)
  let lowRatios : Array Float := #[1.0, 1.51, 2.07, 2.63, 3.21]
  let mut full : Array ModalMode := #[]
  for i in [0:lowRatios.size] do
    full := full.push (mode (f0 * lowRatios[i]!) (0.2 + 0.12 * i.toFloat)
      (1.0 / Float.pow (i.toFloat + 1.0) 0.7) (2.399963 * i.toFloat))
  for j in [0:8] do
    let r := 3.0 + 0.9 * j.toFloat
    full := full.push (mode (f0 * r) (0.5 + 0.09 * j.toFloat)
      (0.3 / Float.pow r 0.8) (2.399963 * (j.toFloat + 5.0)))
  let mut half : Array ModalMode := #[]
  for j in [0:4] do
    let r := 8.0 + 3.5 * j.toFloat
    let a := 0.18 / Float.pow r 0.5
    let ph := 2.399963 * (j.toFloat + 13.0)
    let d1 := 0.5 + 0.1 * j.toFloat
    half := half.push (mode (f0 * r) d1 a ph)
    half := half.push (mode (f0 * r) (d1 + 2.5) (-a) ph)
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

end Tropical.EmitArrow
