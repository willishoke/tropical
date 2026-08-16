import Std

/-!
# Independent EmitArrow numeric oracles

Float/complex reference mathematics used by tropicaltest.  Nothing in this
module authors expressions or calls the arena-native modal implementation under
test.
-/

namespace Tropical.Testing.ArrowOracles

structure Cplx where
  re : Float
  im : Float
deriving Inhabited

namespace Cplx

def ofReal (x : Float) : Cplx := ⟨x, 0.0⟩
def add (a b : Cplx) : Cplx := ⟨a.re + b.re, a.im + b.im⟩
def sub (a b : Cplx) : Cplx := ⟨a.re - b.re, a.im - b.im⟩
def mul (a b : Cplx) : Cplx :=
  ⟨a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re⟩
def neg (a : Cplx) : Cplx := ⟨-a.re, -a.im⟩
def normSq (a : Cplx) : Float := a.re * a.re + a.im * a.im
def div (a b : Cplx) : Cplx :=
  let denominator := b.normSq
  ⟨(a.re * b.re + a.im * b.im) / denominator,
    (a.im * b.re - a.re * b.im) / denominator⟩
def abs (a : Cplx) : Float := Float.sqrt a.normSq
def arg (a : Cplx) : Float := Float.atan2 a.im a.re
def powNat (a : Cplx) : Nat → Cplx
  | 0 => ⟨1.0, 0.0⟩
  | n + 1 => (powNat a n).mul a

end Cplx

structure CMode where
  pole : Cplx
  amp : Cplx
  deg : Nat := 0

def residueCompose (voice reverb : Array (Cplx × Cplx)) : Array CMode :=
  let tolerance := 1.0e-6
  voice.foldl (fun accumulated (pole, amplitude) =>
    let transfer := reverb.foldl (fun sum (roomPole, residue) =>
      if (pole.sub roomPole).normSq < tolerance then sum
      else sum.add (residue.div (pole.sub roomPole))) (Cplx.ofReal 0.0)
    let accumulated := accumulated.push { pole, amp := amplitude.mul transfer }
    reverb.foldl (fun result (roomPole, residue) =>
      if (pole.sub roomPole).normSq < tolerance then
        result.push { pole := roomPole, amp := amplitude.mul residue, deg := 1 }
      else
        result.push {
          pole := roomPole
          amp := ((amplitude.mul residue).div (pole.sub roomPole)).neg
        }) accumulated) #[]

def cmodeMoment (mode : CMode) (order : Nat) : Cplx :=
  if order < mode.deg then ⟨0.0, 0.0⟩
  else
    let fallingFactorial := (List.range mode.deg).foldl
      (fun product index => product * (order - index).toFloat) 1.0
    (mode.amp.mul (mode.pole.powNat (order - mode.deg))).mul
      (Cplx.ofReal fallingFactorial)

def residueMomentError (voice reverb : Array (Cplx × Cplx)) (maxOrder : Nat) : Float :=
  let modes := residueCompose voice reverb
  let roomMoment (order : Nat) : Cplx :=
    reverb.foldl (fun sum (pole, residue) =>
      sum.add (residue.mul (pole.powNat order))) ⟨0.0, 0.0⟩
  let modeMoment (order : Nat) : Cplx :=
    modes.foldl (fun sum mode => sum.add (cmodeMoment mode order)) ⟨0.0, 0.0⟩
  let convolutionJet (order : Nat) : Cplx :=
    if order == 0 then ⟨0.0, 0.0⟩ else
    voice.foldl (fun sum (pole, amplitude) =>
      let jet := (List.range order).foldl (fun total index =>
        total.add ((roomMoment index).mul (pole.powNat (order - 1 - index))))
        ⟨0.0, 0.0⟩
      sum.add (amplitude.mul jet)) ⟨0.0, 0.0⟩
  let scale (order : Nat) : Float :=
    modes.foldl (fun total mode => total + (cmodeMoment mode order).abs) 0.0
  (List.range (maxOrder + 1)).foldl (fun maximum order =>
    let error := ((modeMoment order).sub (convolutionJet order)).abs /
      (scale order + 1.0e-300)
    max maximum error) 0.0

/-- Independent Float complex arithmetic for the exact-vs-libm and bloom
    differential gates.  This intentionally does not reuse the compiler's
    bake-time complex type or any of its transcendental implementation. -/
structure CplxB where
  re : Float
  im : Float
deriving Inhabited, Repr

namespace CplxB

def add (a b : CplxB) : CplxB := ⟨a.re + b.re, a.im + b.im⟩
def sub (a b : CplxB) : CplxB := ⟨a.re - b.re, a.im - b.im⟩
def mul (a b : CplxB) : CplxB :=
  ⟨a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re⟩
def neg (a : CplxB) : CplxB := ⟨-a.re, -a.im⟩
def scale (s : Float) (a : CplxB) : CplxB := ⟨s * a.re, s * a.im⟩
def normSq (a : CplxB) : Float := a.re * a.re + a.im * a.im
def div (a b : CplxB) : CplxB :=
  let denominator := b.normSq
  ⟨(a.re * b.re + a.im * b.im) / denominator,
    (a.im * b.re - a.re * b.im) / denominator⟩
def abs (a : CplxB) : Float := Float.sqrt a.normSq
def exp (z : CplxB) : CplxB :=
  let magnitude := Float.exp z.re
  ⟨magnitude * Float.cos z.im, magnitude * Float.sin z.im⟩
def log (z : CplxB) : CplxB :=
  ⟨0.5 * Float.log z.normSq, Float.atan2 z.im z.re⟩

end CplxB

/-- Independent diagnostic distance to the negative-integer conditioning
    lattice used by the bloom safety gate. This deliberately stays in the
    Float oracle tier, outside production classification. -/
def bloomConditioningMetric (a : CplxB) (depth : Nat := 300) : Float := Id.run do
  let mut distance := CplxB.abs (a.add ⟨1, 0⟩)
  for index in [1:depth] do
    distance := min distance (CplxB.abs (a.add ⟨(index + 1).toFloat, 0⟩))
  return distance

/-- Independent complex log-gamma oracle: Lanczos on `Re z ≥ 1/2`, with a
    stable reflection formula below the boundary. -/
def lgammaB (z : CplxB) : CplxB :=
  let core : CplxB → CplxB := fun z =>
    let lanczos : Array Float := #[0.99999999999980993, 676.5203681218851,
      -1259.1392167224028, 771.32342877765313, -176.61502916214059,
      12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
      1.5056327351493116e-7]
    let shifted := z.sub ⟨1, 0⟩
    let coefficient := (Array.range 8).foldl
      (fun accumulated index => accumulated.add
        (CplxB.div ⟨lanczos[index + 1]!, 0⟩
          (shifted.add ⟨(index + 1).toFloat, 0⟩)))
      ⟨lanczos[0]!, 0⟩
    let t := shifted.add ⟨7.5, 0⟩
    (((shifted.add ⟨0.5, 0⟩).mul (CplxB.log t)).sub t).add
      ((CplxB.log coefficient).add
        ⟨0.5 * Float.log (2.0 * 3.141592653589793), 0⟩)
  if z.re < 0.5 then
    let pi := 3.141592653589793
    let dominant : CplxB :=
      if z.im < 0 then ⟨-pi * z.im, pi * z.re⟩ else ⟨pi * z.im, -pi * z.re⟩
    let logTwoI : CplxB :=
      if z.im < 0 then ⟨Float.log 2.0, pi / 2.0⟩ else ⟨Float.log 2.0, -pi / 2.0⟩
    let logSin := (dominant.add
      (CplxB.log (CplxB.sub ⟨1, 0⟩ (CplxB.exp (dominant.scale (-2.0)))))).sub logTwoI
    (CplxB.sub ⟨Float.log pi, 0⟩ logSin).sub (core (CplxB.sub ⟨1, 0⟩ z))
  else core z

def bloomM1 (a z : CplxB) (tol : Float := 1e-17) (cap : Nat := 4000) :
    CplxB × Nat := Id.run do
  let mut sum : CplxB := ⟨1, 0⟩
  let mut term : CplxB := ⟨1, 0⟩
  for n in [1:cap] do
    term := (term.mul z).div (a.add ⟨n.toFloat, 0⟩)
    sum := sum.add term
    if term.abs ≤ tol * max sum.abs 1.0 then return (sum, n)
  return (sum, cap)

def bloomCF (a z : CplxB) (tol : Float := 1e-15) (cap : Nat := 4000) :
    CplxB × Nat := Id.run do
  let tiny := 1e-300
  let mut b := (z.add ⟨1, 0⟩).sub a
  let mut c : CplxB := ⟨1.0 / tiny, 0⟩
  let mut d : CplxB := if b.normSq == 0.0 then ⟨tiny, 0⟩ else CplxB.div ⟨1, 0⟩ b
  let mut h := d
  for i in [1:cap] do
    let index : CplxB := ⟨i.toFloat, 0⟩
    let an := index.mul (a.sub index)
    b := b.add ⟨2, 0⟩
    d := (an.mul d).add b
    if d.abs < tiny then d := ⟨tiny, 0⟩
    c := b.add (an.div c)
    if c.abs < tiny then c := ⟨tiny, 0⟩
    d := CplxB.div ⟨1, 0⟩ d
    let delta := d.mul c
    h := h.mul delta
    if (delta.sub ⟨1, 0⟩).abs ≤ tol then return (h, i)
  return (h, cap)

def bloomGammaStar (a kappa : CplxB) (g : Float) : CplxB :=
  (CplxB.exp (((lgammaB a).sub (a.mul (CplxB.log kappa))).add kappa)).scale
    (1.0 / g)

def cexpm1B (z : CplxB) : CplxB :=
  if z.normSq < 0.01 then
    let step := fun (coefficient : Float) (accumulated : CplxB) =>
      (⟨coefficient, 0⟩ : CplxB).add (z.mul accumulated)
    step 1.0 (step (1.0 / 2.0) (step (1.0 / 6.0) (step (1.0 / 24.0)
      (step (1.0 / 120.0) (step (1.0 / 720.0) ⟨1.0 / 5040.0, 0⟩)))))
  else ((CplxB.exp z).sub ⟨1, 0⟩).div z

def bloomDCoef (a : CplxB) (depth : Nat) : Array CplxB := Id.run do
  let mut coefficients : Array CplxB := #[]
  let mut previous : CplxB := ⟨0, 0⟩
  let mut factorial : CplxB := ⟨1, 0⟩
  for index in [1:depth + 1] do
    let value := index.toFloat
    let coefficient := ((previous.scale value).sub factorial).div
      ((⟨value, 0⟩ : CplxB).mul (a.add ⟨value, 0⟩))
    coefficients := coefficients.push coefficient
    previous := coefficient
    factorial := factorial.scale (1.0 / value)
  return coefficients

def bloomPhiKappaOverG (a kappa continuedFraction : CplxB)
    (coefficients : Array CplxB) (g : Float) : CplxB :=
  if kappa.abs < (a.add ⟨1, 0⟩).abs then
    (kappa.mul (coefficients.foldr
      (fun coefficient accumulated => coefficient.add (kappa.mul accumulated))
      ⟨0, 0⟩)).scale (1.0 / g)
  else
    let euler := 0.5772156649015329
    let logGammaOverA : CplxB :=
      if a.normSq < 1e-12 then ⟨-euler, 0⟩
      else (lgammaB (a.add ⟨1, 0⟩)).div a
    let wOverA := logGammaOverA.sub (CplxB.log kappa)
    let w := a.mul wOverA
    ((((CplxB.exp kappa).mul (cexpm1B w)).mul wOverA).sub continuedFraction).scale
      (1.0 / g)

def bloomFoldQCoef (a1 a2 : CplxB) (depth : Nat) : Array CplxB := Id.run do
  let mut coefficients : Array CplxB := #[]
  let mut p : CplxB := ⟨1, 0⟩
  let mut q : CplxB := ⟨0, 0⟩
  for index in [1:depth + 1] do
    let value := index.toFloat
    let a1n := a1.add ⟨value, 0⟩
    let a2n := a2.add ⟨value, 0⟩
    let next := (q.div a2n).sub (p.div (a1n.mul a2n))
    coefficients := coefficients.push next
    p := p.div a1n
    q := next
  return coefficients

def bloomFoldDDaM (coefficients : Array CplxB) (x : CplxB) : CplxB :=
  x.mul (coefficients.foldr
    (fun coefficient accumulated => coefficient.add (x.mul accumulated)) ⟨0, 0⟩)

end Tropical.Testing.ArrowOracles
