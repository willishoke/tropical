import Tropical.EmitArrow.Modal.Residue

/-!
# Per-kernel oriented modal convolution

This module records the algebra required when a room's direction orients only
that room kernel.  It is deliberately independent of patch lowering: a caller
convolves the already-oriented prefix with the future/past pieces of the next
room instead of attaching one direction to the complete prefix.

The formulas below are for complex analytic atoms.  Conjugate completion and
taking the real part remain the surrounding `ModalMode` convention.  For a
decaying pole `lam`, amplitude `a`, and degree `p`, write

```text
F(lam,a,p)(t) = 1_{t>0} a t^p exp(lam t)
P(lam,a,p)(t) = 1_{t<0} a (-t)^p exp(-lam t).
```

Both use the same left-half-plane physical pole.  `P` is the exact time mirror
of `F`.  Axis gates are strict, matching modal realization.  A convolution of
oppositely oriented atoms is nevertheless generally nonzero at exactly zero,
so `Expansion.atZero` carries that value explicitly.

For two same-side atoms, put `delta = lam - nu`, `m = p+1`, and `n = q+1`.
When `delta != 0`, partial fractions give, for `0 <= r <= p`,

```text
 [F/P lam,r] : a*b*p!*q!*(-1)^(p-r)*choose(q+p-r,p-r)
               / (r!*delta^(q+p-r+1))
```

and, for `0 <= s <= q`,

```text
 [F/P nu,s]  : a*b*p!*q!*(-1)^(p+1)*choose(p+q-s,q-s)
               / (s!*delta^(p+q-s+1)).
```

At `lam = nu` the finite beta-function limit is the single atom

```text
 a*b*p!*q!/(p+q+1)! * F/P(lam, p+q+1).
```

For `F(lam,a,p) * P(nu,b,q)`, let `rho = -(lam+nu)`.  Stable physical poles
make `Re rho > 0`, so this mixed case has no admissible coincident-pole
singularity.  Its future arm has, for `0 <= i <= p`,

```text
 [F lam,p-i] : a*b*choose(p,i)*(q+i)! / rho^(q+i+1),
```

its past arm has, for `0 <= j <= q`,

```text
 [P nu,q-j]  : a*b*choose(q,j)*(p+j)! / rho^(p+j+1),
```

and both one-sided limits agree at
`a*b*(p+q)!/rho^(p+q+1)`.  `P*F` is this formula with the operands swapped.
-/

namespace Tropical.EmitArrow.Oriented

open Tropical.Ir

/-- Which strict half-axis supports an analytic modal atom. -/
inductive Orientation where
  | future
  | past
deriving BEq, Repr

/-- One analytic exponential-polynomial atom.  `pole` is always the physical
decaying pole: the past interpretation negates it in the exponential. -/
structure Atom where
  pole : CplxE
  amp : CplxE
  deg : Nat := 0

/-- An atom with its half-axis support attached. -/
structure DirectedAtom extends Atom where
  orientation : Orientation

/-- A two-sided convolution result.  The arrays are evaluated only on their
strict half-axes; `atZero` is the exact point value. -/
structure Expansion where
  future : Array Atom := #[]
  past : Array Atom := #[]
  atZero : CplxE := (lit 0, lit 0)

/-- Production-facing oriented carrier.  It deliberately keeps the two axes
separate because a later room must convolve against both, not realize the
prefix and direction-crossfade it as one signal. -/
structure Bank where
  future : Array ModalMode := #[]
  past : Array ModalMode := #[]
  atZero : CplxE := (lit 0, lit 0)

/-- The caller's certified route for a same-side pole pair.  The `.distinct`
route requires `lam - nu != 0`; `.coincident` requires `lam = nu`.  Keeping the
certificate outside `CplxE` is intentional: live `Sig` expressions cannot be
soundly classified by syntactic equality. -/
inductive SameSideRoute where
  | distinct
  | coincident
deriving BEq, Repr

/-- Embed an exact natural scalar as a complex `Sig` expression. -/
def natE (n : Nat) : CplxE := (lit (Int.ofNat n), lit 0)

/-- Exact natural factorial; kept local because the project intentionally does
not depend on a general-purpose mathematics library. -/
def factorial : Nat → Nat
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Exact binomial coefficient, with `choose n k = 0` for `k > n`. -/
def choose : Nat → Nat → Nat
  | _, 0 => 1
  | 0, _ + 1 => 0
  | n + 1, k + 1 => choose n k + choose n (k + 1)

/-- Complex exponentiation by a small natural degree. -/
def cpowE (base : CplxE) : Nat → CplxE
  | 0 => natE 1
  | n + 1 => cmulE (cpowE base n) base

/-- Scale an amplitude by an exact signed natural divided by
`factorial * pole^power`. -/
def scaledPoleQuotient (amp : CplxE) (numerator factorial : Nat)
    (pole : CplxE) (power : Nat) (negative : Bool := false) : CplxE :=
  let signed := if negative then cnegE (natE numerator) else natE numerator
  cdivE (cmulE amp signed) (cmulE (natE factorial) (cpowE pole power))

/-- Scale an amplitude by an exact rational natural. -/
def scaledNatQuotient (amp : CplxE) (numerator denominator : Nat) : CplxE :=
  cdivE (cmulE amp (natE numerator)) (natE denominator)

/-- View an existing modal mode as an analytic atom.  Admission/range metadata
is intentionally not copied: this module states algebra, not a lowering route. -/
def Atom.ofMode (mode : ModalMode) : Atom :=
  { pole := mode.poleE, amp := mode.ampE, deg := mode.deg }

/-- Return an analytic atom to the existing rectangular carrier. -/
def Atom.toMode (atom : Atom) : ModalMode :=
  modeOfE atom.pole atom.amp atom.deg

/-- A source/modal prefix starts on the future axis. -/
def Bank.ofFuture (modes : Array ModalMode) : Bank :=
  { future := modes }

/-- Multiply a mode's analytic amplitude while preserving its pole, degree,
and admission metadata. -/
def scaleModeAmp (scale : CplxE) (mode : ModalMode) : ModalMode :=
  let amp := cmulE mode.ampE scale
  { mode with cre := amp.1, cim := amp.2 }

/-- Orient only this kernel: `(1-direction)` scales its future half and
`direction` scales its past half.  No prefix direction is stored or replaced. -/
def Bank.kernel (modes : Array ModalMode) (direction : Sig) : Bank :=
  let futureWeight : CplxE := (sub (lit 1) direction, lit 0)
  let pastWeight : CplxE := (direction, lit 0)
  { future := modes.map (scaleModeAmp futureWeight)
    past := modes.map (scaleModeAmp pastWeight) }

/-- Split one room atom into the two pieces selected by its own direction
control.  Applying this to each room at its convolution step is the local-room
orientation contract. -/
def Atom.orientKernel (atom : Atom) (direction : Sig) : Array DirectedAtom :=
  let forwardWeight : CplxE := (sub (lit 1) direction, lit 0)
  let pastWeight : CplxE := (direction, lit 0)
  #[{ atom with
        amp := cmulE atom.amp forwardWeight
        orientation := .future },
    { atom with
        amp := cmulE atom.amp pastWeight
        orientation := .past }]

private def onSide (orientation : Orientation) (atoms : Array Atom)
    (atZero : CplxE) : Expansion :=
  match orientation with
  | .future => { future := atoms, atZero }
  | .past => { past := atoms, atZero }

/-- Sum two analytic expansions in stable left-to-right atom order. -/
def Expansion.add (left right : Expansion) : Expansion :=
  { future := left.future ++ right.future
    past := left.past ++ right.past
    atZero := caddE left.atZero right.atZero }

/-- Flatten analytic atoms into the production rectangular carrier. -/
def Expansion.toBank (expansion : Expansion) : Bank :=
  { future := expansion.future.map Atom.toMode
    past := expansion.past.map Atom.toMode
    atZero := expansion.atZero }

/-- Distinct-pole `FF` or `PP` convolution in finite partial-fraction form. -/
def convolveSameSideDistinct (orientation : Orientation) (left right : Atom) :
    Expansion :=
  let p := left.deg
  let q := right.deg
  let ab := cmulE left.amp right.amp
  let delta := csubE left.pole right.pole
  let atLeft := (Array.range (p + 1)).map fun r =>
    let distance := p - r
    let numerator :=
      factorial p * factorial q * choose (q + p - r) distance
    let amp := scaledPoleQuotient ab numerator (factorial r) delta
      (q + p - r + 1) (distance % 2 == 1)
    ({ pole := left.pole, amp, deg := r } : Atom)
  let atRight := (Array.range (q + 1)).map fun s =>
    let distance := q - s
    let numerator :=
      factorial p * factorial q * choose (p + q - s) distance
    let amp := scaledPoleQuotient ab numerator (factorial s) delta
      (p + q - s + 1) ((p + 1) % 2 == 1)
    ({ pole := right.pole, amp, deg := s } : Atom)
  onSide orientation (atLeft ++ atRight) (natE 0)

/-- Coincident-pole beta-function limit of `FF` or `PP`.  The caller certifies
that the two physical poles are equal. -/
def convolveSameSideCoincident (orientation : Orientation)
    (left right : Atom) : Expansion :=
  let p := left.deg
  let q := right.deg
  let ab := cmulE left.amp right.amp
  let amp := scaledNatQuotient ab (factorial p * factorial q)
    (factorial (p + q + 1))
  onSide orientation #[{ pole := left.pole, amp, deg := p + q + 1 }] (natE 0)

/-- Certified same-side dispatch. -/
def convolveSameSide (route : SameSideRoute) (orientation : Orientation)
    (left right : Atom) : Expansion :=
  match route with
  | .distinct => convolveSameSideDistinct orientation left right
  | .coincident => convolveSameSideCoincident orientation left right

/-- Exact mixed `F * P` expansion.  For stable physical poles,
`rho = -(lam + nu)` has positive real part and every improper integral
converges. -/
def convolveFuturePast (future past : Atom) : Expansion :=
  let p := future.deg
  let q := past.deg
  let ab := cmulE future.amp past.amp
  let rho := cnegE (caddE future.pole past.pole)
  let futureTerms := (Array.range (p + 1)).map fun i =>
    let numerator := choose p i * factorial (q + i)
    let amp := scaledPoleQuotient ab numerator 1 rho (q + i + 1)
    ({ pole := future.pole, amp, deg := p - i } : Atom)
  let pastTerms := (Array.range (q + 1)).map fun j =>
    let numerator := choose q j * factorial (p + j)
    let amp := scaledPoleQuotient ab numerator 1 rho (p + j + 1)
    ({ pole := past.pole, amp, deg := q - j } : Atom)
  let atZero := scaledPoleQuotient ab (factorial (p + q)) 1 rho (p + q + 1)
  { future := futureTerms, past := pastTerms, atZero }

/-- Exact mixed `P * F` expansion, definitionally reduced by convolution
commutativity to `F * P` with swapped operands. -/
def convolvePastFuture (past future : Atom) : Expansion :=
  convolveFuturePast future past

/-- Convolve two directed analytic atoms.  `sameSideRoute` is inspected only
for `FF` and `PP`; stable mixed pairs need no coincidence route. -/
def convolve (sameSideRoute : SameSideRoute) (left right : DirectedAtom) :
    Expansion :=
  match left.orientation, right.orientation with
  | .future, .future => convolveSameSide sameSideRoute .future left.toAtom right.toAtom
  | .past, .past => convolveSameSide sameSideRoute .past left.toAtom right.toAtom
  | .future, .past => convolveFuturePast left.toAtom right.toAtom
  | .past, .future => convolvePastFuture left.toAtom right.toAtom

/-- A production classifier supplies the certified same-side route per mode
pair.  It may use the existing pole-region admission machinery; this module
does not guess equality from live `Sig` syntax. -/
abbrev SameSideClassifier :=
  Orientation → ModalMode → ModalMode → SameSideRoute

/-- Compose one production same-side pair.  The incumbent degree-zero
distinct-pole residue transform is reused exactly where it is sound.  General
polynomial degrees and the certified coincidence route use the closed forms
above. -/
private def convolveSameSideModes (classify : SameSideClassifier)
    (orientation : Orientation) (left right : ModalMode) : Expansion :=
  match classify orientation left right with
  | .distinct =>
      if left.deg == 0 && right.deg == 0 then
        onSide orientation
          ((residueComposeE #[left] #[right]).map Atom.ofMode) (natE 0)
      else
        convolveSameSideDistinct orientation (Atom.ofMode left) (Atom.ofMode right)
  | .coincident =>
      convolveSameSideCoincident orientation (Atom.ofMode left) (Atom.ofMode right)

private def convolvePairs (left right : Array ModalMode)
    (pair : ModalMode → ModalMode → Expansion) : Expansion :=
  left.foldl (fun accumulated leftMode =>
    right.foldl (fun accumulated rightMode =>
      accumulated.add (pair leftMode rightMode)) accumulated)
    {}

private def sumAtDifference (pole : CplxE) (modes : Array ModalMode) : CplxE :=
  modes.foldl (fun total mode =>
    caddE total (cdivE mode.ampE (csubE pole mode.poleE))) (natE 0)

private def sumAtPhysicalSum (pole : CplxE) (modes : Array ModalMode) : CplxE :=
  modes.foldl (fun total mode =>
    caddE total (cdivE mode.ampE (caddE pole mode.poleE))) (natE 0)

/-- The mixed-pair value at exactly zero, summed without materializing its two
axis modes. -/
private def mixedAtZero (future past : Array ModalMode) : CplxE :=
  future.foldl (fun total futureMode =>
    past.foldl (fun total pastMode =>
      csubE total (cdivE (cmulE futureMode.ampE pastMode.ampE)
        (caddE futureMode.poleE pastMode.poleE))) total) (natE 0)

/-- Collected degree-zero formula.  Its future poles are definitionally the
input-future/kernel-future union and its past poles the input-past/kernel-past
union; mixed pairs alter residues but introduce no poles. -/
private def convolveKernelDegZeroCollected (input kernel : Bank) : Bank :=
  let inputFuture := input.future.map fun mode =>
    let gain := csubE
      (sumAtDifference mode.poleE kernel.future)
      (sumAtPhysicalSum mode.poleE kernel.past)
    scaleModeAmp gain mode
  let kernelFuture := kernel.future.map fun mode =>
    let gain := csubE
      (sumAtDifference mode.poleE input.future)
      (sumAtPhysicalSum mode.poleE input.past)
    scaleModeAmp gain mode
  let inputPast := input.past.map fun mode =>
    let gain := csubE
      (sumAtDifference mode.poleE kernel.past)
      (sumAtPhysicalSum mode.poleE kernel.future)
    scaleModeAmp gain mode
  let kernelPast := kernel.past.map fun mode =>
    let gain := csubE
      (sumAtDifference mode.poleE input.past)
      (sumAtPhysicalSum mode.poleE input.future)
    scaleModeAmp gain mode
  { future := inputFuture ++ kernelFuture
    past := inputPast ++ kernelPast
    atZero := caddE (mixedAtZero input.future kernel.past)
      (mixedAtZero kernel.future input.past) }

private def modesDegreeZero (modes : Array ModalMode) : Bool :=
  modes.all fun mode => mode.deg == 0

private def sameSidePairsDistinct (classify : SameSideClassifier)
    (orientation : Orientation) (left right : Array ModalMode) : Bool :=
  left.all fun leftMode =>
    right.all fun rightMode =>
      classify orientation leftMode rightMode == .distinct

private def canCollectDegreeZero (classify : SameSideClassifier)
    (input kernel : Bank) : Bool :=
  modesDegreeZero input.future && modesDegreeZero input.past &&
    modesDegreeZero kernel.future && modesDegreeZero kernel.past &&
    sameSidePairsDistinct classify .future input.future kernel.future &&
    sameSidePairsDistinct classify .past input.past kernel.past

/-- General-degree/certified-coincidence oracle path.  It keeps the four cases
literal and is also the foundation against which the collected path is checked. -/
private def convolveKernelPairwise (classify : SameSideClassifier)
    (input kernel : Bank) : Bank :=
  let ff := convolvePairs input.future kernel.future
    (convolveSameSideModes classify .future)
  let pp := convolvePairs input.past kernel.past
    (convolveSameSideModes classify .past)
  let fp := convolvePairs input.future kernel.past fun future past =>
    convolveFuturePast (Atom.ofMode future) (Atom.ofMode past)
  let pf := convolvePairs input.past kernel.future fun past future =>
    convolvePastFuture (Atom.ofMode past) (Atom.ofMode future)
  (ff.add pp |>.add fp |>.add pf).toBank

/-- Convolve an oriented prefix with one newly authored room kernel.

For certified-distinct degree-zero banks this uses a collected `m+n` carrier:

* at an input future pole `lam`, the gain is
  `sum Bf/(lam-nu) - sum Bp/(lam+mu)`;
* at a kernel future pole `nu`, it is
  `sum Af/(nu-lam) - sum Ap/(nu+mu)`;
* the two past origins use the reflected formulas.

Thus repeated room application does not multiply the mode count.  General
degrees or a certified same-side coincidence route fall back to the exact
pairwise expansion.  Mixed `atZero` values are summed in either route.
`input.atZero` is a point value, not a Dirac mass, so it has measure zero in the
next continuous convolution and correctly does not enter either fold. -/
def Bank.convolveKernel (input : Bank) (room : Array ModalMode)
    (direction : Sig) (classify : SameSideClassifier) : Bank :=
  let kernel := Bank.kernel room direction
  if canCollectDegreeZero classify input kernel then
    convolveKernelDegZeroCollected input kernel
  else
    convolveKernelPairwise classify input kernel

@[simp] theorem Bank.ofFuture_future (modes : Array ModalMode) :
    (Bank.ofFuture modes).future = modes := rfl

@[simp] theorem Bank.ofFuture_past (modes : Array ModalMode) :
    (Bank.ofFuture modes).past = #[] := rfl

@[simp] theorem Bank.ofFuture_atZero (modes : Array ModalMode) :
    (Bank.ofFuture modes).atZero = natE 0 := rfl

@[simp] theorem Bank.kernel_atZero (modes : Array ModalMode) (direction : Sig) :
    (Bank.kernel modes direction).atZero = natE 0 := rfl

@[simp] theorem convolveSameSideDistinct_atZero
    (orientation : Orientation) (left right : Atom) :
    (convolveSameSideDistinct orientation left right).atZero = natE 0 := by
  cases orientation <;> rfl

@[simp] theorem convolveSameSideCoincident_atZero
    (orientation : Orientation) (left right : Atom) :
    (convolveSameSideCoincident orientation left right).atZero = natE 0 := by
  cases orientation <;> rfl

@[simp] theorem convolvePastFuture_eq_swapped
    (past future : Atom) :
    convolvePastFuture past future = convolveFuturePast future past := rfl

/-- The strict-gate zero convention for a mixed pair is the shared continuous
one-sided limit, not zero. -/
theorem convolveFuturePast_atZero (future past : Atom) :
    (convolveFuturePast future past).atZero =
      let ab := cmulE future.amp past.amp
      let rho := cnegE (caddE future.pole past.pole)
      scaledPoleQuotient ab (factorial (future.deg + past.deg)) 1 rho
        (future.deg + past.deg + 1) := rfl

end Tropical.EmitArrow.Oriented
