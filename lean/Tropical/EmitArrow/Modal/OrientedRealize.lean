import Tropical.EmitArrow.Modal.Oriented
import Tropical.EmitArrow.Modal.Realize

/-!
# Realization of oriented modal banks

The algebraic carrier stays two-sided until the actual `Modal → Sig` seam.
This module is intentionally small: room composition remains in `Oriented`,
while this file owns the one terminal read and the whole-value gauge adapter.
-/

namespace Tropical.EmitArrow.Oriented

open Tropical.Ir
open Tropical.EmitArrow

/-- Read both strict half-axis banks and supply the continuous mixed-orientation
    convolution value at the strike itself. -/
def Bank.realizeSig (bank : Bank) (clkInt anchorSamples : Sig) : Sig :=
  let sides := add
    (modalBankSig bank.future clkInt anchorSamples)
    (modalBankSigDir bank.past clkInt anchorSamples (lit 1))
  let atStrike := Sig.binary .eq (relClockQ clkInt anchorSamples) (lit 0)
  selectE atStrike bank.atZero.1 sides

/-- The bilateral transfer value at `s = i·omega`.  Future atoms contribute
    `A·p!/(s-λ)^(p+1)`; mirrored past atoms contribute
    `A·p!·(-1)^(p+1)/(s+λ)^(p+1)`. -/
def Bank.transferAt (bank : Bank) (omega : Sig) : CplxE :=
  let s : CplxE := (lit 0, omega)
  let future := bank.future.foldl (fun total mode =>
    let denominator := cpowE (csubE s mode.poleE) (mode.deg + 1)
    caddE total (cdivE (cmulE mode.ampE (natE (factorial mode.deg))) denominator))
    (natE 0)
  bank.past.foldl (fun total mode =>
    let numerator := cmulE mode.ampE (natE (factorial mode.deg))
    let numerator := if (mode.deg + 1) % 2 == 1 then cnegE numerator else numerator
    let denominator := cpowE (caddE s mode.poleE) (mode.deg + 1)
    caddE total (cdivE numerator denominator)) future

/-- One authored gauge scalar for the complete current modal value.  Both arms
    are sampled in one p=8 norm and receive one scalar; they are never normalized
    independently.  Unlike the older causal-only adapter this expression is the
    current static universe, so it deliberately does not settle live controls to
    a target value before measuring them. -/
def Bank.gaugeScale (g : Sig) (bank : Bank) : Sig :=
  let frequencies := bank.future.map (fun mode => mode.omega) ++
    bank.past.map (fun mode => neg mode.omega)
  let energy8 := frequencies.foldl (fun total omega =>
    let h := bank.transferAt omega
    let h2 := add (mul h.1 h.1) (mul h.2 h.2)
    add total (mul (mul h2 h2) (mul h2 h2))) (lit 0)
  expSig (mul (mul (neg g) (lit 125 3))
    (logSig (clampE energy8 (lit 1 30) (lit (10^30)))))

/-- Gauge a complete oriented modal value in place.  The exact strike seam is
    scaled with the same value as both exponential-polynomial arms. -/
def Bank.gauge (bank : Bank) (g : Sig) : Bank :=
  if bank.future.isEmpty && bank.past.isEmpty then bank else
  let scale : CplxE := (bank.gaugeScale g, lit 0)
  { future := bank.future.map (scaleModeAmp scale)
    past := bank.past.map (scaleModeAmp scale)
    atZero := cmulE scale bank.atZero }

/-- Sway changes only this room kernel's physical damping before convolution.
    Pitch, the source prefix, and every other room remain untouched. -/
def swayKernel (modes : Array ModalMode) (sway rate responseClock : Sig) : Array ModalMode :=
  let phase := phasorPhaseSig rate (lit 0) responseClock
  let scale := add (lit 1) (mul sway (sinSig (mul twoPiE phase)))
  modes.map fun mode => { mode with sigma := mul mode.sigma scale }

/-- Syntactically identical physical poles take the exact beta-limit route.
    All other pairs use the distinct expression.  Interval-wide coincidence
    admission is a separate compiler gate; this classifier never guesses that
    two different live expressions are equal. -/
def syntacticSameSideClassifier : SameSideClassifier := fun _ left right =>
  if left.sigma == right.sigma && left.omega == right.omega
  then .coincident
  else .distinct

end Tropical.EmitArrow.Oriented
