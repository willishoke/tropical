import Tropical.Exact.Interval

/-!
# Tropical.Exact.Const — the certified transcendental constants

π and ln 2 as `DyadicI` enclosures of width `2^{−300}`, held as literal
integer mantissas so they cost nothing at bake time.

The literals are DATA, so they are gate-covered rather than trusted: the
`exact-constants` gate (`Tropicaltest/Exact.lean`) recomputes both from
scratch inside the carrier — π by Machin's `16·atan(1/5) − 4·atan(1/239)`,
ln 2 by `2·atanh(1/3)`, each summed with a rigorous remainder bound — and
checks that the recomputed enclosure CONTAINS the literal one. A wrong digit
is a red gate, not a silent 1-ulp drift; and the recomputation is the same
apparatus, so the gate cannot pass by sharing a bug with the literal.

(The literals themselves were produced by two independent integer methods —
a guard-bit Machin/atanh series and a 140-digit decimal evaluation — which
agreed to the last bit.)
-/

namespace Tropical.Exact

/-- `⌊π·2³⁰⁰⌋`. -/
def piFloorMantissa : Int :=
  6399537258350533404498902296276095619968599404742382956244570183135712832137875919436931981

/-- `⌊ln2·2³⁰⁰⌋`. -/
def ln2FloorMantissa : Int :=
  1411965743695424507508857192728570223734551239724995589061361292466257174357546532574725854

/-- A one-ulp-wide enclosure `[m·2^{−prec}, (m+1)·2^{−prec}]`. -/
private def ulpEnclosure (m : Int) (prec : Nat) : DyadicI :=
  ⟨Dyadic.ofIntWithPrec m (prec : Int), Dyadic.ofIntWithPrec (m + 1) (prec : Int), true⟩

/-- π, enclosed to `2^{−300}`. -/
def piI : DyadicI := ulpEnclosure piFloorMantissa constPrec

/-- `ln 2`, enclosed to `2^{−300}`. -/
def ln2I : DyadicI := ulpEnclosure ln2FloorMantissa constPrec

/-- `π/2`, exact halving of the π enclosure (no rounding — a power of two). -/
def piHalfI : DyadicI := piI.shift (-1)

/-- `2π`. -/
def twoPiI : DyadicI := piI.shift 1

/-- `π/4`. -/
def piQuarterI : DyadicI := piI.shift (-2)

end Tropical.Exact
