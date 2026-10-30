import Tropical.Session.Compile
import Tropical.StagedLoad
import Tropical.Tropicaltest.Reversibility
import Tropical.Tropicaltest.Gates

/-!
# Fractional τ at the port

`design/typed-kernel.md`: τ is the kernel's Q32.32 time coordinate, passed as
an array, and the kernel is a closed form `f(τ, params)`. The IR's time leaf
is τ itself (`ENode.tau`), so a fractional coordinate evaluates the closed
form at that fractional instant — nothing between the port and the
oscillator's phase rounds it.

The gate is an exact identity between two renders of emitted kernels through
the runtime's host-side τ fill (`Runtime.renderTau`):

* **half speed on the integer grid**: `ScrubClock(velocity = ½)` feeding
  `FixedSinOsc`, rendered at τ = 2k+1 for k < N; the scrub clock is
  `base + velocity · τ` on the rail, exactly (`scaleClockQ`), so the
  oscillator sees the clock `(k + ½) · 2³²`;
* **full speed at half-sample instants**: `FixedSinOsc` on the default clock
  (τ itself), rendered at τ = k + ½, i.e. the i64 `(2k+1) <<< 31`.

Byte-identical output is the statement that (a) the port honours the fraction
of τ and (b) the scrub clock's product is exact when it *produces* a fraction.
Its input τ is whole there, so a third leg feeds it fractional τ:

* **dense velocity at dense τ**: `ScrubClock` at velocity
  `S / 2³²` with `S = 7331967795` (every fraction bit significant, exact in
  f64 and in its decimal spelling), rendered at `τ_k = k·2³² + h(k)` with
  `h` a 32-bit hash, so every word of `scaleClockQ`'s split product is live,
  against the default clock at `⌊S·τ_k / 2³²⌋` computed here in `Nat`. That
  floor is `scaleClockQ`'s contract; the two nested 16-bit floors collapse
  to it exactly. The leg also checks that it can see an error of 2⁻¹⁶
  sample, the weight of the low partial product.

A second check
confirms the fraction is not simply dropped: the full-speed kernel at k + ½
differs from itself at k.
-/

namespace Tropical.Tropicaltest.FractionalTau

private def frames : Nat := 512

private def oscAt (clkInput : String) (velocity : String := "0.5") : String :=
  "{\"schema\":\"tropical_program_2\",\"name\":\"frac_tau\",\"body\":{\"op\":\"block\",\"decls\":[" ++
  "{\"op\":\"instanceDecl\",\"name\":\"clock\",\"program\":\"ScrubClock\",\"inputs\":{\"velocity\":" ++ velocity ++ "}}," ++
  "{\"op\":\"instanceDecl\",\"name\":\"osc\",\"program\":\"FixedSinOsc\",\"inputs\":{\"freq\":220" ++ clkInput ++ "}}]," ++
  "\"assigns\":[{\"op\":\"outputAssign\",\"name\":\"dac.out\",\"expr\":{\"op\":\"ref\",\"instance\":\"osc\",\"output\":\"sine\"}}]}}"

/-- The oscillator driven by the half-speed scrub clock. -/
private def halfSpeedPatch : String :=
  oscAt ",\"clk\":{\"op\":\"ref\",\"instance\":\"clock\",\"output\":\"clk\"}"

/-- `S`, the dense velocity's Q32.32 image, and its exact decimal spelling. -/
private def denseScaleQ : Nat := 7331967795
private def denseVelocity : String := "1.70710678095929324626922607421875"

/-- The oscillator driven by the scrub clock at the dense velocity. -/
private def denseScrubPatch : String :=
  oscAt ",\"clk\":{\"op\":\"ref\",\"instance\":\"clock\",\"output\":\"clk\"}" denseVelocity

/-- The oscillator on its default clock, τ itself. (The unreferenced scrub
    clock instance is kept so both patches lower the same instance set.) -/
private def fullSpeedPatch : String := oscAt ""

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

def runFractionalTau : IO Bool := do
  let label := "fractional-tau"
  let half ← match ← loadPatch halfSpeedPatch with
    | .error e => return ← failGate label s!"half-speed patch: {e}"
    | .ok rt => pure rt
  let full ← match ← loadPatch fullSpeedPatch with
    | .error e => return ← failGate label s!"full-speed patch: {e}"
    | .ok rt => pure rt
  -- Three τ images over k < N: the odd integers 2k+1, the half-samples k + ½,
  -- and the integers k.
  let odd : Array UInt64 := (Array.range frames).map fun k => (2 * k + 1).toUInt64 <<< (32 : UInt64)
  let halves : Array UInt64 := (Array.range frames).map fun k => (2 * k + 1).toUInt64 <<< (31 : UInt64)
  let ints : Array UInt64 := (Array.range frames).map fun k => k.toUInt64 <<< (32 : UInt64)
  let onGrid := decodeF64LE (← half.renderTau odd #[0])
  let atHalves := decodeF64LE (← full.renderTau halves #[0])
  let atInts := decodeF64LE (← full.renderTau ints #[0])
  if onGrid.size != frames || atHalves.size != frames || atInts.size != frames then
    return ← failGate label s!"render sizes {onGrid.size}/{atHalves.size}/{atInts.size}, want {frames}"
  let identity := bitDiffCount onGrid atHalves
  let fractionLive := bitDiffCount atHalves atInts
  let energy := atHalves.foldl (fun acc x => acc + x * x) 0.0
  IO.println s!"        half-speed@2k+1 ≡ full-speed@k+½ bitDiff={identity}/{frames} · k+½ vs k bitDiff={fractionLive}/{frames} · E={energy}"
  -- The dense leg: every fraction bit of τ and of the velocity is live.
  let dense ← match ← loadPatch denseScrubPatch with
    | .error e => return ← failGate label s!"dense-velocity patch: {e}"
    | .ok rt => pure rt
  let denseTau : Array UInt64 := (Array.range frames).map fun k =>
    ((k <<< 32) + (k * 2654435761 + 2654435769) % 4294967296).toUInt64
  let expected : Array UInt64 := denseTau.map fun t => (denseScaleQ * t.toNat / 4294967296).toUInt64
  let scrubbed := decodeF64LE (← dense.renderTau denseTau #[0])
  let direct := decodeF64LE (← full.renderTau expected #[0])
  let nudged := decodeF64LE (← full.renderTau (expected.map (· + 65536)) #[0])
  if scrubbed.size != frames || direct.size != frames || nudged.size != frames then
    return ← failGate label s!"dense render sizes {scrubbed.size}/{direct.size}/{nudged.size}, want {frames}"
  let denseIdentity := bitDiffCount scrubbed direct
  let lowWordVisible := bitDiffCount direct nudged
  IO.println s!"        scrub(S/2³²)@τ_k ≡ τ@⌊S·τ_k/2³²⌋ bitDiff={denseIdentity}/{frames} · a 2⁻¹⁶-sample error is visible in {lowWordVisible}/{frames}"
  if identity == 0 && fractionLive > frames / 2 && energy > 1.0
      && denseIdentity == 0 && lowWordVisible > frames / 2 then
    passGate label "scrub(½)·τ on the integer grid ≡ τ at half-sample instants, and scrub(S)·τ ≡ ⌊S·τ/2³²⌋ at dense fractional τ, byte-exact"
  else
    failGate label s!"identity bitDiff={identity} fractionLive={fractionLive} energy={energy} denseIdentity={denseIdentity} lowWordVisible={lowWordVisible}"

end Tropical.Tropicaltest.FractionalTau
