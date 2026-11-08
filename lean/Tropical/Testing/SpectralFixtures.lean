import Tropical.EmitArrow
import Tropical.EmitArrow.Stdlib

/-!
# Spectral test fixtures

Programs built from the partial-family term (`EmitArrow.Spectral`), for the
spectral gates in `tropicaltest` and, registered under `--fixtures`, for the
wasm≡JIT suite and the cost benchmark (`benchmarks/spectral/`). Every control
is a program input, so a patch wires any signal into it; nothing is baked in
except the family's static capacity (its topology).

* `SpectralSaw` — a 200-partial saw family (`1/k` coefficient table, sine
  phase convention). Inputs: `f0` (Hz, uniform), `phase` (cycles),
  `partials` (live count, at most 200), `clk`.
* `SpectralLowpassSaw` — the same family under a complex SVF lowpass envelope,
  zero-phase cosine convention. Inputs: `f0`, `cutoff` (Hz, any signal),
  `q` (any signal), `clk`.
* `ExpCutoffLfo` — `c0·2^(octaves·sin(2π·rate·τ))`, the cutoff modulator of
  the cost benchmark. Inputs: `c0`, `octaves`, `rate` (Hz), `clk`.

`chirpCutoffSig` is the filter-FM listening set's cutoff path (built into the
oracle gate's carrier directly, not registered).
-/

namespace Tropical.Testing.SpectralFixtures

open Tropical.Ir
open Tropical.EmitArrow

/-- The fixtures' capacity: 200 partials. -/
def fixtureCount : Nat := 200

/-- Seconds since τ = 0 as a float: `τ / (2^F · rate)`. -/
def secondsOf (clock : Clock) : BuildM Sig := do
  let samples ← div (← toFloatE clock) (← clockOneLit)
  div samples (← sampleRate)

/-- `sin(2π·cycles)` through the exact fixed-point sine: the fractional cycle
    lands on the Q32 rail (`⌊frac·2³²⌋`) and `fixedSinCycSig` evaluates it, so
    the result's error is the rail's (≈1e-9), independent of how many cycles
    have elapsed. -/
def sinCyclesSig (cycles : Sig) : BuildM Sig := do
  let whole ← unary .floor cycles
  let frac ← sub cycles whole
  let phaseQ ← bitAnd (← toIntE (← mul frac (← lit 4294967296))) (← lit 4294967295)
  div (← toFloatE (← fixedSinCycSig phaseQ)) (← lit 1073741824)

/-- `exp x` to double precision, for long-running control phases: Cody–Waite
    reduction `x = n·ln2 + r` (`ln2` split so `n·ln2_hi` is exact), `|r| ≤ ln2/2`,
    then the Taylor series through `r¹³/13!` (truncation < 1e-17) and
    `ldexp(·, n)`. The core `expSig` is a minimax fit good to ~1e-9 relative,
    which is plenty for an envelope but not for the phase of an exponential
    chirp: at 40 s the sweep's phase is ~1.5e4 rad, so a 1e-9 relative error is
    a 1.5e-5 rad phase error. -/
def expPreciseSig (x : Sig) : BuildM Sig := do
  let clamped ← clampE x (← lit (-700)) (← lit 700)
  let n ← roundE (← mul clamped (← litExact 1.4426950408889634))
  -- ln2 = ln2Hi + ln2Lo, ln2Hi with trailing zero bits so n·ln2Hi is exact
  let r ← sub (← sub clamped (← mul n (← litExact 0.6931471803691238)))
    (← mul n (← litExact 1.9082149292705877e-10))
  let mut acc ← litExact (1.0 / 6227020800.0)
  for k in (List.range 13).reverse do
    let fact := ((List.range k).map (·.toFloat + 1.0)).foldl (· * ·) 1.0
    let coeff := 1.0 / fact
    acc ← add (← litExact coeff) (← mul r acc)
  ldexpE acc n

/-- `2^x` as `exp(x·ln 2)`. -/
def exp2Sig (x : Sig) : BuildM Sig := do
  expSig (← mul x (← lit 6931471805599453 16))

/-- `c0·2^(octaves·sin(2π·rate·τ))`, the LFO phase from the exact phasor. -/
def expCutoffLfoSig (c0 octaves rate : Sig) (clock : Clock) : BuildM Sig := do
  let phase ← phasorPhaseSig rate (← lit 0) clock
  let s ← sinCyclesSig phase
  mul c0 (← exp2Sig (← mul octaves s))

/-- The listening set's sweep: `fm(t) = fa·(fb/fa)^(t/T)`, whose phase is the
    closed form `φm(t)/2π = fa·T/ln(fb/fa)·((fb/fa)^(t/T) − 1)` cycles, and the
    cutoff `c0·2^(octaves·sin φm)`. The sweep's shape (`fa`, `fb`, `T`) is a
    build-time constant, so `ln(fb/fa)` is folded at assemble time. -/
def chirpCutoffSig (c0 octaves : Sig) (fa fb dur : Float) (clock : Clock) : BuildM Sig := do
  let lr := Float.log (fb / fa)
  let t ← secondsOf clock
  let growth ← expPreciseSig (← mul (← litExact (lr / dur)) t)
  let cycles ← mul (← litExact (fa * dur / lr)) (← sub growth (← lit 1))
  let s ← sinCyclesSig cycles
  mul c0 (← exp2Sig (← mul octaves s))

/-- A saw family: `1/k` weights, `count` partials. -/
def sawFamily (freq : Sig) (phase : Option Sig) (clk : Clock) (count : Nat)
    (wave : PartialWave := .sine) : BuildM PartialFamily := do
  pure { name := "saw", freq, phase, clk, count, weights := ← sawWeights count, wave }

private def floatIn (name : String) (mantissa : Int) (exponent : Nat := 0) :
    BuildM AInputDecl := do
  pure (floatDecl name (← lit mantissa exponent))

private def clockIn (name : String) : BuildM AInputDecl := do
  pure (clockInDecl name (← tau))

def buildSpectralSaw (arena : Arena) : Except String (Arena × ProgramIdx) :=
  assemble arena "SpectralSaw" #[{ name := "out", type? := some (.scalar .float) }] #[] do
    let f0 ← inputRef ⟨0⟩
    let phase ← inputRef ⟨1⟩
    let partials ← inputRef ⟨2⟩
    let clk ← inputRef ⟨3⟩
    let family ← sawFamily f0 (some phase) clk fixtureCount
    let out ← emitTerm (normalize (.family { family with liveCount := some partials }))
    pure {
      inputs := #[← floatIn "f0" 110, ← floatIn "phase" 0,
        ← floatIn "partials" fixtureCount, ← clockIn "clk"]
      assigns := #[(.port ⟨0⟩, out)] }

def buildSpectralLowpassSaw (arena : Arena) : Except String (Arena × ProgramIdx) :=
  assemble arena "SpectralLowpassSaw" #[{ name := "out", type? := some (.scalar .float) }] #[] do
    let f0 ← inputRef ⟨0⟩
    let cutoff ← inputRef ⟨1⟩
    let q ← inputRef ⟨2⟩
    let clk ← inputRef ⟨3⟩
    let family ← sawFamily f0 none clk fixtureCount (wave := .cosine)
    let term := svfEnvelope .lowpass (.konst cutoff) (.konst q) (.family family)
    let out ← emitTerm (normalize term)
    pure {
      inputs := #[← floatIn "f0" 110, ← floatIn "cutoff" 800, ← floatIn "q" 7 1,
        ← clockIn "clk"]
      assigns := #[(.port ⟨0⟩, out)] }

def buildExpCutoffLfo (arena : Arena) : Except String (Arena × ProgramIdx) :=
  assemble arena "ExpCutoffLfo" #[{ name := "out", type? := some (.scalar .float) }] #[] do
    let out ← expCutoffLfoSig (← inputRef ⟨0⟩) (← inputRef ⟨1⟩) (← inputRef ⟨2⟩)
      (← inputRef ⟨3⟩)
    pure {
      inputs := #[← floatIn "c0" 800, ← floatIn "octaves" 2, ← floatIn "rate" 1,
        ← clockIn "clk"]
      assigns := #[(.port ⟨0⟩, out)] }

/-- The roster `--fixtures` registers, in order. -/
def spectralFixtures : Array (String × (Arena → Except String (Arena × ProgramIdx))) := #[
  ("SpectralSaw", buildSpectralSaw),
  ("SpectralLowpassSaw", buildSpectralLowpassSaw),
  ("ExpCutoffLfo", buildExpCutoffLfo)
  ]

end Tropical.Testing.SpectralFixtures
