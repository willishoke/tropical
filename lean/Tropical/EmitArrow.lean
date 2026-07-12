import Tropical.EmitArrow.Sig
import Tropical.EmitArrow.Term
import Tropical.EmitArrow.Numerics
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

The post-strata IR is scalar by definition (strata's job is to lower arrays,
sums and generics away), so EmitArrow stays scalar: the richness lives in
the typed elaborator upstream. There is no `Warp` type and no separate clock
algebra — the clock is a first-class expression and a warp is any operation
applied to it, drawn from the same operation set used on values.

The modules:

* `Sig` — the authoring substrate: the build-time expression tree, its
  arena lowering, `assemble`, and the smart-constructor op set.
* `Term` — `Voice`, the cartesian combinator surface (`Mor`), and
  `ArrowTerm` + the slide (`normalize`/`emitTermC`): downstream-presented
  warps pushed up to generator clocks.
* `Numerics` — the closed-form scalar kernels (fixed-point sine, integer
  phasor, polynomial transcendentals).
* `Modal` — the pole island: modal banks, banked coefficient-column
  reductions, and the symbolic residue calculus.
* `Patch` — the patcher lowering: downstream-only patch DAG → arrow term.
* `Gong` — the struck nonlinear resonator: amplitude-bloom mode pairs,
  the analytic pitch-bloom clock warp, and the alias-free polynomial
  drive, all composed from existing node kinds.

The byte-gate carriers that certify all of this against the hand-written
stdlib programs live in `Tropical.Testing.ArrowFixtures`, outside the
production import graph.
-/
