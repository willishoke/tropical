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
of τ and (b) the scrub clock's product is exact at fractional τ. A second check
confirms the fraction is not simply dropped: the full-speed kernel at k + ½
differs from itself at k.
-/

namespace Tropical.Tropicaltest.FractionalTau

private def frames : Nat := 512

private def oscAt (clkInput : String) : String :=
  "{\"schema\":\"tropical_program_2\",\"name\":\"frac_tau\",\"body\":{\"op\":\"block\",\"decls\":[" ++
  "{\"op\":\"instanceDecl\",\"name\":\"clock\",\"program\":\"ScrubClock\",\"inputs\":{\"velocity\":0.5}}," ++
  "{\"op\":\"instanceDecl\",\"name\":\"osc\",\"program\":\"FixedSinOsc\",\"inputs\":{\"freq\":220" ++ clkInput ++ "}}]," ++
  "\"assigns\":[{\"op\":\"outputAssign\",\"name\":\"dac.out\",\"expr\":{\"op\":\"ref\",\"instance\":\"osc\",\"output\":\"sine\"}}]}}"

/-- The oscillator driven by the half-speed scrub clock. -/
private def halfSpeedPatch : String :=
  oscAt ",\"clk\":{\"op\":\"ref\",\"instance\":\"clock\",\"output\":\"clk\"}"

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
  if identity == 0 && fractionLive > frames / 2 && energy > 1.0 then
    passGate label "scrub(½)·τ on the integer grid ≡ τ at half-sample instants, byte-exact; the fraction of τ reaches the phase"
  else
    failGate label s!"identity bitDiff={identity} fractionLive={fractionLive} energy={energy}"

end Tropical.Tropicaltest.FractionalTau
