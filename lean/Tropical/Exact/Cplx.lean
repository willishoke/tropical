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
