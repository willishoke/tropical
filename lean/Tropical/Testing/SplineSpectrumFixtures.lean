import Tropical.EmitArrow
import Tropical.EmitArrow.Stdlib

/-!
# Band-limited spline fixtures

The periodic spline rendered as its harmonics below Nyquist
(`EmitArrow.SplineSpectrum`), for the `tropicaltest` spline-spectrum gates and,
registered under `--fixtures`, the wasm≡JIT suite. K = 8 control values,
capacity 256 partials.

* `SplineSpectrumCatmullRom8`, `SplineSpectrumBSpline8` — inputs `f0` (Hz,
  uniform), `c0 … c7` (live when wired from `param:` slots), `clk`.
-/

namespace Tropical.Testing.SplineSpectrumFixtures

open Tropical.Ir
open Tropical.EmitArrow

/-- The fixtures' partial capacity. -/
def fixtureCount : Nat := 256

/-- The fixtures' control count. -/
def fixtureControls : Nat := 8

private def build (name : String) (basis : SplineBasis) (arena : Arena) :
    Except String (Arena × ProgramIdx) :=
  assemble arena name #[{ name := "out", type? := some (.scalar .float) }] #[] do
    let k := fixtureControls
    let f0 ← inputRef ⟨0⟩
    let controls ← (Array.range k).mapM fun j => inputRef ⟨j + 1⟩
    let clk ← inputRef ⟨k + 1⟩
    let term := splineSpectrum basis fixtureCount clk (.konst f0) (controls.map .konst)
    let out ← emitTerm (normalize term)
    let zero ← lit 0
    pure {
      inputs := #[floatDecl "f0" (← lit 110)] ++
        (Array.range k).map (fun j => floatDecl s!"c{j}" zero) ++
        #[clockInDecl "clk" (← tau)]
      assigns := #[(.port ⟨0⟩, out)] }

def buildSplineSpectrumCatmullRom8 : Arena → Except String (Arena × ProgramIdx) :=
  build "SplineSpectrumCatmullRom8" catmullRomBasis

def buildSplineSpectrumBSpline8 : Arena → Except String (Arena × ProgramIdx) :=
  build "SplineSpectrumBSpline8" uniformBSplineBasis

/-- The roster `--fixtures` registers, in order. -/
def splineSpectrumFixtures : Array (String × (Arena → Except String (Arena × ProgramIdx))) := #[
  ("SplineSpectrumCatmullRom8", buildSplineSpectrumCatmullRom8),
  ("SplineSpectrumBSpline8", buildSplineSpectrumBSpline8)]

end Tropical.Testing.SplineSpectrumFixtures
