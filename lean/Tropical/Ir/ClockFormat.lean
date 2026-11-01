/-!
# The clock rail's format

τ and every clock are signed fixed-point `i64` values with `clockFracBits`
fraction bits: the integer sample count in the high bits, the sub-sample
fraction in the low ones (`design/fixed-carrier.md` is normative). This is
the one definition of the split; every builder, emitter, host fill and gate
that scales samples onto the rail reads it from here.

The split trades time resolution for horizon. With `F` fraction bits a
coordinate resolves `2⁻ᶠ` of a sample and stays non-negative for `2⁶³⁻ᶠ`
samples. Phase (Q0.32, one turn = 2³²) and sample values (Q2.30) are
separate formats and do not change with it.
-/

namespace Tropical.Ir

/-- Fraction bits of the clock rail. -/
def clockFracBits : Nat := 16

/-- One sample on the clock rail, `2^clockFracBits`. -/
def clockOne : Nat := 2 ^ clockFracBits

/-- The fraction field of a clock, `clockOne − 1`. -/
def clockFracMask : Nat := clockOne - 1

end Tropical.Ir
