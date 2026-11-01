import Tropical.Ir.ClockFormat
import Tropical.Kernel
import Tropical.Session.Compile
import Tropical.StagedLoad
import Tropical.Tropicaltest.ArrowLaws
import Tropical.Tropicaltest.Reversibility
import Tropical.Tropicaltest.Gates

/-!
# The clock rail's horizon

τ is a signed fixed-point `i64` with `clockFracBits` fraction bits
(`Tropical.Ir.ClockFormat`), so a coordinate stays non-negative for
`2^(63 − clockFracBits)` samples. The free-running phasors reduce on ℤ/2³²
and do not notice the sign bit; the gate covers the consumers that do. Each
leg renders through the host's τ fill (`Runtime.renderTau`) at far
coordinates and compares against the same closed form near the origin, or
against the exact expected value:

* **warp**: `ScrubClock(½)` feeding `FixedSinOsc` at τ = 2(B+k)+1 ≡ the
  default clock at τ = B+k+½. A non-integer speed is not a homomorphism of
  the wrapped ring, so this is the leg that sees the sign bit.
* **strike**: a modal bank struck at A+200 and read on [A, A+N) ≡ struck at
  200 and read on [0, N). The anchor lands on the rail as
  `toInt(anchor · 2^F)`, which must stay in range.
* **index**: the derived `sampleIndex` at τ = B+k+½ is exactly B+k, both
  as an arrow builder spells it and as a session wire does (`VCA` with
  `audio = sampleIndex`, `cv = 1`): the wire lowering is a separate site.

The gate also pins the manifest's name for the format, `i64q16`, to
`clockFracBits`, so a host reading the element type reads the split.

B runs across 2³¹ samples (the Q32.32 sign bit, ≈ 12.4 h at 48 kHz) and at
2⁴⁰ samples (≈ 265 days at 48 kHz).
-/

namespace Tropical.Tropicaltest.LongHorizon

open Tropical.Ir (Arena ProgramIdx clockOne)

private def frames : Nat := 512

/-- The far origins: just before the Q32.32 sign bit, and 2⁴⁰ samples. -/
private def origins : Array Nat := #[2 ^ 31 - 256, 2 ^ 40]

/-- Sample `n` plus `num / den` of a sample, as the rail's bit pattern. -/
private def at_ (n : Nat) (num : Nat := 0) (den : Nat := 1) : UInt64 :=
  (n * clockOne + num * clockOne / den).toUInt64

private def patch (decls assign : String) : String :=
  "{\"schema\":\"tropical_program_2\",\"name\":\"long_horizon\",\"body\":{\"op\":\"block\",\"decls\":[" ++
  decls ++ "],\"assigns\":[{\"op\":\"outputAssign\",\"name\":\"dac.out\",\"expr\":" ++ assign ++ "}]}}"

/-- The warp leg's oscillator. A clock that is wrong by half the ring
    (2⁶³, what `v = ½` does to a wrapped τ) moves the phase by
    `inc · 2^(63 − clockFracBits)` mod 2³², which is visible only when the
    phase step `inc = ⌊f·2³²/SR⌋` is odd; at 44.1 kHz, 223 Hz has
    `inc = 21718315`. -/
private def oscAt (clkInput : String) : String :=
  patch ("{\"op\":\"instanceDecl\",\"name\":\"clock\",\"program\":\"ScrubClock\",\"inputs\":{\"velocity\":0.5}}," ++
    "{\"op\":\"instanceDecl\",\"name\":\"osc\",\"program\":\"FixedSinOsc\",\"inputs\":{\"freq\":223" ++ clkInput ++ "}}")
    "{\"op\":\"ref\",\"instance\":\"osc\",\"output\":\"sine\"}"

private def loadPatch (text : String) : IO (Except String Tropical.Ffi.Runtime) := do
  let registry ← match Tropical.Session.bootRegistry with
    | .error f => return .error s!"stdlib boot failed: {f.toJson.compress}"
    | .ok st => pure st
  match (Tropical.Session.compilePatch text).runOn registry with
  | (.error f, _) => return .error s!"compile: {f.toJson.compress}"
  | (.ok (plan, blocks), _) =>
    let rt ← Tropical.Ffi.Runtime.new frames.toUInt32
    Tropical.StagedLoad.loadTyped rt plan blocks
    return .ok rt

private def loadPlan (plan : Tropical.Plan.FlatPlan) : IO Tropical.Ffi.Runtime := do
  let rt ← Tropical.Ffi.Runtime.new frames.toUInt32
  Tropical.StagedLoad.load rt plan
  pure rt

private def render (rt : Tropical.Ffi.Runtime) (tau : Array UInt64) : IO (Array Float) := do
  pure (decodeF64LE (← rt.renderTau tau #[0]))

private def energy (xs : Array Float) : Float := xs.foldl (fun acc x => acc + x * x) 0.0

open Tropical.EmitArrow in
private def struckBank (arena : Arena) (anchor : Nat) :
    Except String Tropical.Plan.FlatPlan :=
  let mode (frequency sigma amplitude : Int) : BuildM ModalMode := do
    ModalMode.hz (← lit frequency) (← lit sigma 1) (← lit amplitude 1)
  buildAndFinish (buildExprCarrier s!"long_horizon_strike_{anchor}" (do
    let modes ← modeArray #[mode 221 30 6, mode 331 40 4, mode 443 55 3]
    modalBankSig modes (← clockLit) (← lit anchor)) arena)

open Tropical.EmitArrow in
/-- The derived integer sample index, `τ >> clockFracBits`, as the output. -/
private def indexCarrier (arena : Arena) : Except String Tropical.Plan.FlatPlan :=
  buildAndFinish (buildExprCarrier "long_horizon_index" (do
    toFloatE (← sampleIndex)) arena)

def runLongHorizon (arena : Arena) (_resolved : Array (String × ProgramIdx)) : IO Bool := do
  let label := "long-horizon"
  let elemName := Tropical.Kernel.Elem.wire .i64q16
  let elemOk := elemName == s!"i64q{Tropical.Ir.clockFracBits}"
  IO.println s!"        τ's element type {elemName} names {Tropical.Ir.clockFracBits} fraction bits: {elemOk}"
  let ks := Array.range frames
  let half ← match ← loadPatch (oscAt ",\"clk\":{\"op\":\"ref\",\"instance\":\"clock\",\"output\":\"clk\"}") with
    | .error e => return ← failGate label s!"half-speed patch: {e}"
    | .ok rt => pure rt
  let full ← match ← loadPatch (oscAt "") with
    | .error e => return ← failGate label s!"full-speed patch: {e}"
    | .ok rt => pure rt
  let index ← match indexCarrier arena with
    | .error e => return ← failGate label s!"sampleIndex build: {e}"
    | .ok plan => loadPlan plan
  let wireIndex ← match ← loadPatch (patch
      "{\"op\":\"instanceDecl\",\"name\":\"g\",\"program\":\"VCA\",\"inputs\":{\"audio\":{\"op\":\"sampleIndex\"},\"cv\":1}}"
      "{\"op\":\"ref\",\"instance\":\"g\",\"output\":\"out\"}") with
    | .error e => return ← failGate label s!"wire sampleIndex patch: {e}"
    | .ok rt => pure rt
  let near ← match struckBank arena 200 with
    | .error e => return ← failGate label s!"strike build: {e}"
    | .ok plan => loadPlan plan
  let nearStrike ← render near (ks.map (at_ ·))
  let mut ok : Bool := elemOk && energy nearStrike > 1e-6
  for b in origins do
    let warped ← render half (ks.map fun k => at_ (2 * (b + k) + 1))
    let direct ← render full (ks.map fun k => at_ (b + k) 1 2)
    let warpDiff := bitDiffCount warped direct
    let far ← match struckBank arena (b + 200) with
      | .error e => return ← failGate label s!"strike build at {b}: {e}"
      | .ok plan => loadPlan plan
    let farStrike ← render far (ks.map fun k => at_ (b + k))
    let strikeDiff := bitDiffCount nearStrike farStrike
    let indices ← render index (ks.map fun k => at_ (b + k) 1 2)
    let wired ← render wireIndex (ks.map fun k => at_ (b + k) 1 2)
    let indexMiss := (ks.filter fun k =>
      indices[k]? != some (b + k).toFloat || wired[k]? != some (b + k).toFloat).size
    IO.println s!"        B={b}: warp bitDiff={warpDiff}/{frames} (E={energy direct}) · strike bitDiff={strikeDiff}/{frames} · index misses={indexMiss}/{frames}"
    ok := ok && warpDiff == 0 && energy direct > 1.0 && strikeDiff == 0 && indexMiss == 0
  if ok then
    passGate label s!"scrub(½), a struck modal bank and sampleIndex are exact past 2³¹ samples and at 2⁴⁰ (clock fraction {Tropical.Ir.clockFracBits} bits)"
  else
    failGate label s!"a far-coordinate render differs from its closed form (clock fraction {Tropical.Ir.clockFracBits} bits)"

end Tropical.Tropicaltest.LongHorizon
