import Tropical.Exact.Dyadic
import Tropical.Exact.Interval
import Tropical.Exact.Const
import Tropical.Exact.Elementary
import Tropical.Exact.Cplx
import Tropical.Exact.Gamma

/-!
# Exact — the bake layer's numeric carrier

The compiler computes constants before it emits them, and it makes DECISIONS
from those constants: how deep a series is emitted, which lanes a pair gets,
what exponent a bank lands at, whether a configuration is served at all. Those
decisions change the emitted program's SHAPE. A platform `libm` that answers
one ulp differently therefore does not produce a slightly different sound — it
produces a different program.

That is the hole this closes. `Tropical.Exact` is exact dyadic arithmetic with
directed rounding, wrapped in enclosures that either separate two values or say
plainly that they do not:

* `Dyadic` — core Lean's verified dyadic rational (`CommRing`, `OrderedRing`,
  proved directed rounding), extended here with the RELATIVE rounding a
  wide-dynamic-range computation needs, exact `Float` conversion both ways, and
  the exact decimal quantization the emit boundary means.
* `Interval` — `DyadicI`, outward-rounding arithmetic and `cmp`, whose three
  answers are `lt`, `gt`, and `overlap`. `overlap` is the honest verdict a
  `Float` comparison never gives: at this precision these two values are not
  separated, and whoever asked has to say what they do about that.
* `Const` — π and ln 2, gate-covered rather than trusted.
* `Elementary` — exp, log, sin, cos, atan2, pow: argument reduction plus a
  truncated series, each truncation admitted as a widening.
* `Cplx` / `Gamma` — the complex mirror of the bake layer's `CplxB`, and
  Lanczos log-gamma over the same coefficients.

The runtime already ran this purge one floor down: the audio kernel's `sin`,
`exp` and `log` are polynomials precisely so that wasm and the JIT cannot
disagree. This is that same refusal applied to the compiler that BUILDS the
kernel.

What the carrier does not claim: it bounds the ROUNDING error of the formulas
it evaluates, not their distance from the true transcendental function. The
Lanczos approximation is still an approximation. That bridge stays where it
was — under the audio gates and the developer's ear.
-/
