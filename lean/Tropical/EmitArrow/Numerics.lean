import Tropical.EmitArrow.Sig

/-!
# EmitArrow.Numerics — ID-native closed-form scalar kernels

ID-valued counterparts of the recursive numeric authoring helpers.  Every
node-producing helper runs in `BuildM`; intermediate IDs are bound once and
reused explicitly so Horner forms and fixed-point datapaths remain shared DAGs
without leaving the active `BuildM`/`ExprArena` authoring model.
-/

namespace Tropical.EmitArrow

open Tropical.Ir

def fixedSinCycSig (phaseQ : Sig) : BuildM Sig := do
  let quarter ← lit 1073741824
  let phaseRounded ← add phaseQ quarter
  let shift31 ← lit 31
  let n ← rshift phaseRounded shift31
  let nHalfTurn ← lshift n shift31
  let r ← sub phaseQ nHalfTurn
  let oneI ← litI 1
  let twoI ← litI 2
  let one ← lit 1
  let parity ← bitAnd n one
  let twiceParity ← mul twoI parity
  let sign ← sub oneI twiceParity
  let rr ← mul r r
  let shift30 ← lit 30
  let z ← rshift rr shift30
  let zShift ← rshift z shift30
  let c6 ← lit 61
  let acc6 ← sub c6 zShift
  let acc6z ← mul acc6 z
  let acc6zShift ← rshift acc6z shift30
  let c5 ← lit 3864
  let acc5 ← sub c5 acc6zShift
  let acc5z ← mul acc5 z
  let acc5zShift ← rshift acc5z shift30
  let c4 ← lit 172272
  let acc4 ← sub c4 acc5zShift
  let acc4z ← mul acc4 z
  let acc4zShift ← rshift acc4z shift30
  let c3 ← lit 5026995
  let acc3 ← sub c3 acc4zShift
  let acc3z ← mul acc3 z
  let acc3zShift ← rshift acc3z shift30
  let c2 ← lit 85569306
  let acc2 ← sub c2 acc3zShift
  let acc2z ← mul acc2 z
  let acc2zShift ← rshift acc2z shift30
  let c1 ← lit 693598668
  let acc1 ← sub c1 acc2zShift
  let acc1z ← mul acc1 z
  let acc1zShift ← rshift acc1z shift30
  let c0 ← lit 1686629713
  let acc0 ← sub c0 acc1zShift
  let racc ← mul r acc0
  let scaled ← rshift racc shift30
  mul sign scaled

def fixedCosCycSig (phaseQ : Sig) : BuildM Sig := do
  let quarter ← lit 1073741824
  let shifted ← add phaseQ quarter
  let mask ← lit 4294967295
  let wrapped ← bitAnd shifted mask
  fixedSinCycSig wrapped

def fixedOutQ (fracBits : Nat) (x : Sig) : BuildM Sig := do
  let xf ← toFloatE x
  let scale ← lit (Int.pow 2 fracBits)
  div xf scale

def phasorPhaseSig (freqE offsetE clkSig : Sig) : BuildM Sig := do
  let twoPow32 ← lit 4294967296
  let scaledFreq ← mul freqE twoPow32
  let sr ← sampleRate
  let freqPerSample ← div scaledFreq sr
  let inc ← toIntE freqPerSample
  let scaledOffset ← mul offsetE twoPow32
  let off ← toIntE scaledOffset
  let shift32 ← lit 32
  let thi ← rshift clkSig shift32
  let mask ← lit 4294967295
  let tlo ← bitAnd clkSig mask
  let highProduct ← mul inc thi
  let lowProduct ← mul inc tlo
  let lowHigh ← rshift lowProduct shift32
  let accumulated ← add highProduct lowHigh
  let acc ← add accumulated off
  let wrapped ← bitAnd acc mask
  let wrappedFloat ← toFloatE wrapped
  div wrappedFloat twoPow32

def sinSig (x : Sig) : BuildM Sig := do
  let invPi ← lit 3183098861837907 16
  let scaled ← mul x invPi
  let n ← roundE scaled
  let pi ← lit 3141592653589793 15
  let nPi ← mul n pi
  let r ← sub x nPi
  let one ← lit 1
  let two ← lit 2
  let parity ← bitAnd n one
  let twiceParity ← mul two parity
  let sign ← sub one twiceParity
  let r2 ← mul r r
  let c5 ← lit (-2505210838544172) 23
  let p5 ← mul c5 r2
  let c4 ← lit 27557319223985893 22
  let a4 ← add c4 p5
  let p4 ← mul a4 r2
  let c3 ← lit (-1984126984126984) 19
  let a3 ← add c3 p4
  let p3 ← mul a3 r2
  let c2 ← lit 8333333333333333 18
  let a2 ← add c2 p3
  let p2 ← mul a2 r2
  let c1 ← lit (-16666666666666666) 17
  let a1 ← add c1 p2
  let p1 ← mul a1 r2
  let poly ← add one p1
  let rpoly ← mul r poly
  mul sign rpoly

def expSig (x : Sig) : BuildM Sig := do
  let lo ← lit (-87)
  let hi ← lit 88
  let clamped ← clampE x lo hi
  let log2e ← lit 14426950408889634 16
  let scaled ← mul clamped log2e
  let n ← roundE scaled
  let ln2hi ← lit 693145751953125 15
  let nHi ← mul n ln2hi
  let rHi ← sub clamped nHi
  let ln2lo ← lit 14286068203094173 22
  let nLo ← mul n ln2lo
  let r ← sub rHi nLo
  let c5 ← lit 198756915 12
  let c5r ← mul c5 r
  let c4 ← lit 13981999507 13
  let a4 ← add c4 c5r
  let a4r ← mul a4 r
  let c3 ← lit 83334519073 13
  let a3 ← add c3 a4r
  let a3r ← mul a3 r
  let c2 ← lit 41665795894 12
  let a2 ← add c2 a3r
  let a2r ← mul a2 r
  let c1 ← lit 16666665459 11
  let a1 ← add c1 a2r
  let a1r ← mul a1 r
  let c0 ← lit 50000001201 11
  let p ← add c0 a1r
  let rp ← mul r p
  let one ← lit 1
  let inner ← add one rp
  let rinner ← mul r inner
  let expR ← add one rinner
  ldexpE expR n

def logSig (x : Sig) : BuildM Sig := do
  let e0 ← unary .floatExponent x
  let negE0 ← neg e0
  let m0 ← ldexpE x negE0
  let sqrtTwo ← lit 14142135623730951 16
  let big ← gt m0 sqrtTwo
  let half ← lit 5 1
  let halfM0 ← mul m0 half
  let m ← selectE big halfM0 m0
  let one ← lit 1
  let e0PlusOne ← add e0 one
  let e ← selectE big e0PlusOne e0
  let numerator ← sub m one
  let denominator ← add m one
  let s ← div numerator denominator
  let s2 ← mul s s
  let c4 ← lit 1111111111111111 16
  let c4s ← mul s2 c4
  let c3 ← lit 14285714285714285 17
  let a3 ← add c3 c4s
  let a3s ← mul s2 a3
  let c2 ← lit 2 1
  let a2 ← add c2 a3s
  let a2s ← mul s2 a2
  let c1 ← lit 3333333333333333 16
  let a1 ← add c1 a2s
  let a1s ← mul s2 a1
  let p ← add one a1s
  let ln2 ← lit 6931471805599453 16
  let exponentPart ← mul e ln2
  let two ← lit 2
  let twiceS ← mul two s
  let mantissaPart ← mul twiceS p
  add exponentPart mantissaPart

private def atanUnit (a : Sig) : BuildM Sig := do
  let tanPi12 ← lit 2679491924311227 16
  let hi ← gt a tanPi12
  let tanPi6 ← lit 5773502691896257 16
  let numerator ← sub a tanPi6
  let tanProduct ← mul tanPi6 a
  let one ← lit 1
  let denominator ← add one tanProduct
  let reduced ← div numerator denominator
  let ar ← selectE hi reduced a
  let pi6 ← lit 5235987755982988 16
  let zero ← lit 0
  let bias ← selectE hi pi6 zero
  let s ← mul ar ar
  let c5Raw ← lit 9090909090909091 17
  let c5 ← neg c5Raw
  -- Preserve the recursive helper's `foldr ... (lit 0)` terminal step.  The
  -- `s * 0` is intentionally observable in byte-identity gates.
  let terminalProduct ← mul s zero
  let terminal ← add c5 terminalProduct
  let c5s ← mul s terminal
  let c4 ← lit 1111111111111111 16
  let a4 ← add c4 c5s
  let a4s ← mul s a4
  let c3Raw ← lit 14285714285714285 17
  let c3 ← neg c3Raw
  let a3 ← add c3 a4s
  let a3s ← mul s a3
  let c2 ← lit 2 1
  let a2 ← add c2 a3s
  let a2s ← mul s a2
  let c1Raw ← lit 3333333333333333 16
  let c1 ← neg c1Raw
  let a1 ← add c1 a2s
  let a1s ← mul s a1
  let p ← add one a1s
  let arP ← mul ar p
  add bias arP

def atan2E (y x : Sig) : BuildM Sig := do
  let ax ← unary .abs x
  let ay ← unary .abs y
  let swap ← gt ay ax
  let numerator ← selectE swap ax ay
  let denominator ← selectE swap ay ax
  let floor ← lit 1 30
  let ceiling ← lit (10^30)
  let safeDenominator ← clampE denominator floor ceiling
  let a ← div numerator safeDenominator
  let r0 ← atanUnit a
  let halfPi ← lit 15707963267948966 16
  let complemented ← sub halfPi r0
  let r1 ← selectE swap complemented r0
  let zero ← lit 0
  let xNegative ← gt zero x
  let pi ← lit 3141592653589793 15
  let reflected ← sub pi r1
  let r2 ← selectE xNegative reflected r1
  let yNegative ← gt zero y
  let negated ← neg r2
  selectE yNegative negated r2

def twoPiE : BuildM Sig := lit 6283185307179586 15

def halfPiE : BuildM Sig := lit 15707963267948966 16

def cosSig (x : Sig) : BuildM Sig := do
  let halfPi ← halfPiE
  let shifted ← add x halfPi
  sinSig shifted

end Tropical.EmitArrow
