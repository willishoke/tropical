import Tropical.Exact.Cplx

/-!
# Tropical.Exact.Gamma — complex log-gamma on the certified carrier

The exact twin of the bake layer's `lgammaB`: the SAME Lanczos approximation
(g = 7, n = 9) with the SAME coefficients, the same `Re z ≥ ½` split and the
same reflection through the dominant half of `log sin πz`. Only the arithmetic
changes.

Keeping the algorithm identical is the point. This module does not claim to
compute Γ better than the float version did — it claims to compute THE SAME
FUNCTION deterministically, so that a decision made from its value is the same
decision on every platform, and so that the value/decision differential against
the float path has exactly one variable in it.

Two consequences worth stating plainly:

* The Lanczos formula carries its own APPROXIMATION error (~1e-15 relative over
  the shipped a-range, gated at 1.8e-15 against mpmath by the cockpit's D_bg5).
  The enclosure here bounds the ROUNDING error of evaluating that formula, not
  the distance to the true Γ. The analytic bridge stays gate-covered, as the
  campaign scoped it.
* The two branch conditions (`Re z < ½`, `Im z < 0`) are OVERLAP SWITCHES: both
  arms compute the same analytic function, so a near-threshold config may take
  either arm without being wrong. They are therefore decided from the
  enclosure's midpoint — deterministic, and no cliff.
-/

namespace Tropical.Exact

namespace CplxDI

open DyadicI

/-- The Lanczos (g = 7, n = 9) coefficients, byte-identical to the float bake
    path's table (`lgammaB`) and to its emitted twin (`lgammaE`). Each enters
    the carrier EXACTLY — a finite double is a dyadic — so the port changes the
    arithmetic and not one bit of the input data. -/
def lanczos : Array Float :=
  #[0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]

/-- The Lanczos core, valid for `Re z ≥ ½`. -/
private def lgammaCore (z : CplxDI) : CplxDI :=
  let zz := sub z one
  let x := (Array.range 8).foldl
    (fun acc i => add acc (div (ofI (DyadicI.ofFloat lanczos[i+1]!))
                               (add zz (ofNat (i + 1)))))
    (ofI (DyadicI.ofFloat lanczos[0]!))
  let t := add zz (ofI (DyadicI.ofFloat 7.5))
  add (sub (mul (add zz (ofI (DyadicI.ofFloat 0.5))) (log t)) t)
      (add (log x) (ofI ((DyadicI.log (DyadicI.mul (DyadicI.ofInt 2) piI)).shift (-1))))

/-- Complex log-gamma. `Re z ≥ ½` takes the Lanczos core; below it the
    reflection `lgamma z = log π − log sin(πz) − lgamma(1 − z)`, with
    `log sin πz` taken on its DOMINANT exponential (`s + log(1 − e^{−2s}) −
    log 2i`) so a large `|Im z|` never overflows.

    Both branch picks read the enclosure's MIDPOINT (`Re z` against ½, `Im z`
    against 0). That is deterministic and, on the `Re z` axis, harmless — the
    two arms agree analytically. On the `Im z` axis the choice selects which
    half of `sin πz` is dominant; at `Im z = 0` neither dominates and the
    subtraction `1 − e^{−2s}` is ill-conditioned for BOTH arms, which is a
    property of the algorithm the float path already had, carried over
    unchanged rather than papered over. -/
def lgamma (z : CplxDI) : CplxDI :=
  if !z.ok then poison
  else if Dyadic.blt (DyadicI.mid z.re) (Dyadic.ofFloat 0.5) then
    -- s = ∓iπz picked so e^{s} is the dominant half of sin πz
    let imNeg := (DyadicI.mid z.im).isNeg
    let pz_re := DyadicI.mul piI z.re
    let pz_im := DyadicI.mul piI z.im
    let s : CplxDI :=
      if imNeg then ⟨DyadicI.neg pz_im, pz_re⟩ else ⟨pz_im, DyadicI.neg pz_re⟩
    let log2i : CplxDI :=
      ⟨DyadicI.log (DyadicI.ofInt 2),
       if imNeg then piHalfI else DyadicI.neg piHalfI⟩
    let logsin := sub (add s (log (sub one (exp (scale (DyadicI.ofInt (-2)) s))))) log2i
    sub (sub (ofI (DyadicI.log piI)) logsin) (lgammaCore (sub one z))
  else lgammaCore z

end CplxDI

end Tropical.Exact
