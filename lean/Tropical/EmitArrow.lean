import Tropical.EmitArrow.Sig
import Tropical.EmitArrow.Inspect
import Tropical.EmitArrow.ClockAlgebra
import Tropical.EmitArrow.BankOrder
import Tropical.EmitArrow.Numerics
import Tropical.EmitArrow.Term
import Tropical.EmitArrow.Modal
import Tropical.EmitArrow.Patch
import Tropical.EmitArrow.Gong

/-!
# EmitArrow — realization-by-emission of the post-strata (scalar) IR

`EmitArrow` is the combinator library that builds the resolved IR `Program`
directly in the post-strata, scalar shape and reuses the backend
(`elaborate`-linked, then strata/`compileResolved`) to emit. It is named
verb-first: it is a *realization by emission* of the existing scalar IR —
the "warp" combinators are the clock axis of that realization.

The lowered IR is scalar by definition (arrays survive only as literals,
sums and generics away), so EmitArrow stays scalar: the richness lives in
the typed elaborator upstream. There is no `Warp` type and no separate clock
algebra — the clock is a first-class expression and a warp is any operation
applied to it, drawn from the same operation set used on values.

The modules:

* `Sig` — the ID-valued builder and immediate-interning smart
  constructors.
* `ClockAlgebra` — the clock-rail judgment and laws stated
  directly on `ExprId` roots and frozen `ExprArena` dereference evidence.
* `Numerics` / `Term` — ID-valued scalar kernels,
  voices, cartesian morphisms, and reified-arrow emitter.
* `Modal` / `Patch` / `Gong` — the ID-valued modal
  compiler, graph lowering seam, and struck-register authoring.
The recursive authoring representation has been retired; this is the single
production authoring surface.
-/
