import Tropical.EmitArrow.Stdlib
import Tropical.Family

/-!
# EmitArrow.Families — the stdlib's type families

Each family is a builder from a size to a program (`Tropical.Family`). The
registry boots them beside the concrete stdlib; a member is assembled when an
instance first names it.

* `PeriodicCatmullRom<K>`, `PeriodicBSpline<K>`: the periodic spline of `K`
  control points under each basis (`splineProgram`, `Spline.lean`). Inputs
  `phase` (int, Q0.32) and `c0 … c(K−1)` (float), output `out`. The size-8
  members are also registered at boot under the same names, by the same call
  (`stdlibBuilders`), so patches written before the families load unchanged.
-/

namespace Tropical.EmitArrow

open Tropical.Ir

/-- The size range of a periodic spline. The fewest control points is the
    boundary's minimum (`SplineBoundary.minControls`, 1: one point is a
    constant curve, two or more wrap through the window); this is the same for
    both bases. The most is the largest `K` whose `p · K` the index arithmetic
    holds (`splineSig`: fewer than 2³¹ segments, and a periodic spline has
    `K` segments). -/
def periodicSplineSizes : Nat × Nat :=
  (SplineBoundary.wrap.minControls, 2 ^ 31 - 1)

private def periodicSplineFamily (name : String) (basis : SplineBasis) :
    TypeFamily where
  name
  sizeMin := periodicSplineSizes.1
  sizeMax := periodicSplineSizes.2
  sizeMeaning := "control points"
  ports := s!"inputs phase (int, Q0.32 phase) and c0 … c<size−1> (float control values, {basis.name} basis); output out (float)"
  build := fun member k arena _ => splineProgram arena member basis .wrap k

def periodicCatmullRomFamily : TypeFamily :=
  periodicSplineFamily "PeriodicCatmullRom" catmullRomBasis

def periodicBSplineFamily : TypeFamily :=
  periodicSplineFamily "PeriodicBSpline" uniformBSplineBasis

def stdlibFamilies : Array TypeFamily := #[periodicCatmullRomFamily, periodicBSplineFamily]

end Tropical.EmitArrow
