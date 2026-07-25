import Tropical.Exact.Elementary

/-!
# Tropical.Exact.Cplx — complex arithmetic over the certified enclosure

`CplxDI` mirrors the bake layer's `CplxB` method for method — `add`, `sub`,
`mul`, `neg`, `scale`, `normSq`, `abs`, `div`, `exp`, `log` — so porting a
build-time computation onto the exact carrier is a type swap and nothing else.
That mirror is deliberate: it is what makes the value-site cutover mechanical
and therefore reviewable.

The enclosure is a RECTANGLE (independent real and imaginary intervals), not a
disc. Rectangles overestimate a rotation, so `abs` of a rotated product is a
little wider than a disc enclosure would be — irrelevant at 128 mantissa bits,
and rectangles keep every operation a handful of `Int` multiplies.

`poison` propagates through both components: a complex value is defined only if
both of its parts are.
-/

namespace Tropical.Exact

-- ── The POINT carrier: reproducible arithmetic without an enclosure ──────────
-- `DyadicI` answers "where is the value, certainly". Some bake-time questions
-- do not want that answer. A SELF-CORRECTING RECURRENCE — modified Lentz for a
-- continued fraction is the example — converges in floating point precisely
-- because each step damps the previous step's error; interval arithmetic cannot
-- see that damping (it tracks a worst case that the recurrence has already
-- forgotten) and the enclosure widens about two bits per iteration until it is
-- useless. Tracking it is the wrong instrument, not a precision shortfall.
--
-- What such a loop actually needs is REPRODUCIBILITY: the same iteration count
-- on every platform. `Dyadic` with a fixed round-to-nearest at the working
-- precision gives exactly that — the same algorithm, no `libm`, no
-- fused-multiply-add, no reassociation, and 128 mantissa bits against a
-- double's 53. That is the point carrier below.
--
-- The division is `Option`-valued for the same reason `DyadicI.inv` poisons: a
-- reciprocal that does not exist is refused, never fabricated as zero.

/-- Round a `Dyadic` to the working precision, to nearest — the point carrier's
    one rounding, applied after every operation so mantissas stay bounded. -/
private def rn (a : Dyadic) : Dyadic := Dyadic.roundNearestRel workingPrec a

/-- A complex number in exact, reproducibly-rounded arithmetic — no enclosure.
    The deterministic stand-in for `CplxB`, for recurrences whose convergence an
    interval cannot follow. -/
structure CplxD where
  re : Dyadic
  im : Dyadic

namespace CplxD

def zero : CplxD := ⟨0, 0⟩

instance : Inhabited CplxD := ⟨zero⟩
def one  : CplxD := ⟨1, 0⟩
def ofNat (n : Nat) : CplxD := ⟨Dyadic.ofInt (n : Int), 0⟩
def ofDyadic (d : Dyadic) : CplxD := ⟨d, 0⟩

/-- A build-time `Float` pair enters EXACTLY. -/
def ofFloats (re im : Float) : CplxD := ⟨Dyadic.ofFloat re, Dyadic.ofFloat im⟩

def add (a b : CplxD) : CplxD := ⟨rn (a.re + b.re), rn (a.im + b.im)⟩
def sub (a b : CplxD) : CplxD := ⟨rn (a.re - b.re), rn (a.im - b.im)⟩
def neg (a : CplxD) : CplxD := ⟨-a.re, -a.im⟩

def mul (a b : CplxD) : CplxD :=
  ⟨rn (a.re * b.re - a.im * b.im), rn (a.re * b.im + a.im * b.re)⟩

def normSq (a : CplxD) : Dyadic := rn (a.re * a.re + a.im * a.im)

def abs (a : CplxD) : Dyadic :=
  (Dyadic.sqrtRel? .down workingPrec (normSq a)).getD 0

/-- `a / b`; `none` when `b` is exactly zero. -/
def div (a b : CplxD) : Option CplxD := do
  let d := normSq b
  let re ← Dyadic.divRel? .down workingPrec (a.re * b.re + a.im * b.im) d
  let im ← Dyadic.divRel? .down workingPrec (a.im * b.re - a.re * b.im) d
  pure ⟨re, im⟩

def toFloats (a : CplxD) : Float × Float := (a.re.toFloat, a.im.toFloat)

end CplxD

/-- A complex number as a pair of certified enclosures. The exact twin of the
    bake layer's `CplxB`. -/
structure CplxDI where
  re : DyadicI
  im : DyadicI

namespace CplxDI

open DyadicI

instance : Inhabited CplxDI := ⟨⟨poison, poison⟩⟩

def ok (a : CplxDI) : Bool := a.re.ok && a.im.ok
def poison : CplxDI := ⟨DyadicI.poison, DyadicI.poison⟩

def mkI (re im : DyadicI) : CplxDI := ⟨re, im⟩
def ofI (re : DyadicI) : CplxDI := ⟨re, DyadicI.zero⟩
def ofInt (n : Int) : CplxDI := ⟨DyadicI.ofInt n, DyadicI.zero⟩
def ofNat (n : Nat) : CplxDI := ⟨DyadicI.ofNat n, DyadicI.zero⟩

/-- A build-time `Float` pair enters EXACTLY — the bridge every literal
    coefficient (the Lanczos table, an authored pole) crosses unchanged. -/
def ofFloats (re im : Float) : CplxDI := ⟨DyadicI.ofFloat re, DyadicI.ofFloat im⟩

def zero : CplxDI := ⟨DyadicI.zero, DyadicI.zero⟩
def one  : CplxDI := ⟨DyadicI.one, DyadicI.zero⟩

def add (a b : CplxDI) : CplxDI := ⟨DyadicI.add a.re b.re, DyadicI.add a.im b.im⟩
def sub (a b : CplxDI) : CplxDI := ⟨DyadicI.sub a.re b.re, DyadicI.sub a.im b.im⟩
def neg (a : CplxDI) : CplxDI := ⟨DyadicI.neg a.re, DyadicI.neg a.im⟩

def mul (a b : CplxDI) : CplxDI :=
  ⟨DyadicI.sub (DyadicI.mul a.re b.re) (DyadicI.mul a.im b.im),
   DyadicI.add (DyadicI.mul a.re b.im) (DyadicI.mul a.im b.re)⟩

/-- Multiply by a real enclosure. -/
def scale (s : DyadicI) (a : CplxDI) : CplxDI :=
  ⟨DyadicI.mul s a.re, DyadicI.mul s a.im⟩

def normSq (a : CplxDI) : DyadicI :=
  DyadicI.add (DyadicI.mul a.re a.re) (DyadicI.mul a.im a.im)

def abs (a : CplxDI) : DyadicI := DyadicI.sqrt (normSq a)

/-- `a / b`. Poison when `|b|²` cannot be certified away from zero — the
    complex counterpart of refusing a reciprocal across zero, and the place the
    old `sigConstF?`-style "divide by zero, get zero" convention is replaced by
    an honest refusal. -/
def div (a b : CplxDI) : CplxDI :=
  let d := normSq b
  if DyadicI.straddlesZero d then poison
  else
    ⟨DyadicI.div (DyadicI.add (DyadicI.mul a.re b.re) (DyadicI.mul a.im b.im)) d,
     DyadicI.div (DyadicI.sub (DyadicI.mul a.im b.re) (DyadicI.mul a.re b.im)) d⟩

/-- `e^z = e^{Re z}·(cos Im z, sin Im z)`. -/
def exp (z : CplxDI) : CplxDI :=
  let e := DyadicI.exp z.re
  ⟨DyadicI.mul e (DyadicI.cos z.im), DyadicI.mul e (DyadicI.sin z.im)⟩

/-- `log z = (½·ln|z|², atan2(Im z, Re z))` — the principal branch, matching
    the bake layer's `CplxB.log` term for term. -/
def log (z : CplxDI) : CplxDI :=
  ⟨(DyadicI.log (normSq z)).shift (-1), DyadicI.atan2 z.im z.re⟩

/-- `z^n` for a natural exponent — exact repeated multiplication. -/
def powNat (z : CplxDI) : Nat → CplxDI
  | 0 => one
  | 1 => z
  | n + 1 => mul (powNat z n) z

/-- The midpoint pair as `Float`s — what reaches `litF`. -/
def toFloats (a : CplxDI) : Float × Float := (a.re.toFloat, a.im.toFloat)

def render (a : CplxDI) : String := s!"({a.re.render}) + ({a.im.render})i"

end CplxDI

end Tropical.Exact
