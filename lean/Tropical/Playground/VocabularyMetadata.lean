import Tropical.Exact
import Lean.Data.Json

/-!
# Playground vocabulary metadata

Pure JSON readers and the single served port-spec table.  This module contains
no expression authoring API, so arena-native production decoding can depend on
the public vocabulary contract without importing the recursive `Sig` builder.
-/

namespace Tropical.Playground
namespace Metadata

open Lean (Json JsonNumber)
open Tropical.Exact (DyadicI)

def jNum? (obj : Json) (key : String) : Option JsonNumber :=
  match obj.getObjVal? key with
  | .ok value => value.getNum?.toOption
  | .error _ => none

def jInt (obj : Json) (key : String) (dflt : Int) : Int :=
  match jNum? obj key with
  | some number =>
      if number.exponent == 0 then number.mantissa
      else number.mantissa / ((10 : Int) ^ number.exponent)
  | none => dflt

def jStr (obj : Json) (key : String) (dflt : String) : String :=
  match obj.getObjVal? key with
  | .ok (.str value) => value
  | _ => dflt

def jFloat (obj : Json) (key : String) (dflt : Float) : Float :=
  ((jNum? obj key).map (·.toFloat)).getD dflt

def jDec (obj : Json) (key : String) (dflt : Int × Nat) : Int × Nat :=
  match jNum? obj key with
  | some number => (number.mantissa, number.exponent)
  | none => dflt

def decD (decimal : Int × Nat) : DyadicI :=
  DyadicI.ofJsonNumber ⟨decimal.1, decimal.2⟩

def jExactD (obj : Json) (key : String) (dflt : Int × Nat) : DyadicI :=
  decD (jDec obj key dflt)

def twoPiD : DyadicI := decD (6283185307179586, 15)
def piD : DyadicI := decD (3141592653589793, 15)
def goldenRatioD : DyadicI := decD (6180339887, 10)
def goldenAngleD : DyadicI := decD (2399963, 6)
def ln80D : DyadicI := DyadicI.log (DyadicI.ofNat 80)

def portSources (inObj : Json) (port : String) : Array String :=
  match (inObj.getObjVal? port).toOption.bind (·.getArr?.toOption) with
  | some values => values.filterMap (·.getStr?.toOption)
  | none => #[]

inductive Discipline where
  | raw | glide | anchor
deriving BEq, Repr

inductive PortDomain where
  | signal | modal | control
deriving BEq, Repr

structure KnobMeta where
  min : Float
  max : Float
  log : Bool := false
  unit : String := ""
deriving Repr

structure PortSpec where
  name : String
  accepts : Array PortDomain := #[]
  multi : Bool := false
  knob : Option (Int × Nat) := none
  discipline : Discipline := .raw
  display : Option KnobMeta := none
  ownerPort : Option String := none
deriving Repr

def sigIn : Array PortDomain := #[.signal, .modal, .control]
def modalIn : Array PortDomain := #[.modal]
def ctrlIn : Array PortDomain := #[.control]

def modalPhaserRatios : Array Float := #[
  0.42044820762685725,
  0.5946035575013605,
  0.8408964152537145,
  1.189207115002721,
  1.681792830507429,
  2.378414230005442]

/-- The single served port-spec table. Declaration order defines `ParamIdx`
    scan order and is therefore semantically significant. -/
def portSpecs : String → Array PortSpec
  | "knob" => #[
      { name := "value", knob := some (0, 0),
        display := some { min := 0, max := 1000 } }]
  | "source" => #[
      { name := "freq", accepts := ctrlIn, knob := some (220, 0), discipline := .anchor,
        display := some { min := 0.02, max := 2000, log := true, unit := "Hz" } },
      { name := "morph", knob := some (0, 0), discipline := .glide,
        display := some { min := 0, max := 1 } },
      { name := "pm", accepts := sigIn }]
  | "pluck" => #[
      { name := "freq", accepts := ctrlIn, knob := some (110, 0), discipline := .anchor,
        display := some { min := 20, max := 2000, log := true, unit := "Hz" } },
      { name := "morph", knob := some (0, 0), discipline := .glide,
        display := some { min := 0, max := 1 } },
      { name := "event_rate", knob := some (2, 0), discipline := .glide,
        display := some { min := 0.1, max := 20, log := true, unit := "Hz" } }]
  | "comb" => #[
      { name := "in", accepts := sigIn, multi := true },
      { name := "delay", knob := some (12, 3), discipline := .glide,
        display := some { min := 0.0005, max := 0.05, log := true, unit := "s" } },
      { name := "decay", knob := some (7, 1), discipline := .glide,
        display := some { min := 0, max := 0.95 } }]
  | "flange" => #[
      { name := "in", accepts := sigIn, multi := true },
      { name := "depth", knob := some (7, 4), discipline := .glide,
        display := some { min := 0.0001, max := 0.01, log := true, unit := "s" } }]
  | "sflange" => #[
      { name := "in", accepts := sigIn, multi := true },
      { name := "mod", accepts := sigIn },
      { name := "depth", knob := some (2, 3), discipline := .glide,
        display := some { min := 0.0002, max := 0.02, log := true, unit := "s" } },
      { name := "rate", knob := some (3, 1), ownerPort := some "mod",
        display := some { min := 0.02, max := 12, log := true, unit := "Hz" } }]
  | "fm" => #[
      { name := "in", accepts := sigIn, multi := true },
      { name := "carrier", knob := some (330, 0), discipline := .anchor,
        display := some { min := 20, max := 2000, log := true, unit := "Hz" } },
      { name := "depth", knob := some (8, 0), discipline := .glide,
        display := some { min := 1, max := 400, log := true } }]
  | "delay" => #[
      { name := "in", accepts := sigIn, multi := true },
      { name := "amount", knob := some (4, 3), discipline := .glide,
        display := some { min := 0.0001, max := 0.02, log := true, unit := "s" } }]
  | "reverse" => #[{ name := "in", accepts := sigIn, multi := true }]
  | "mix" => #[{ name := "in", accepts := sigIn, multi := true }]
  | "ring" => #[{ name := "in", accepts := sigIn, multi := true }]
  | "resonator" => #[
      { name := "addr", accepts := sigIn },
      { name := "freq", knob := some (220, 0),
        display := some { min := 20, max := 2000, log := true, unit := "Hz" } },
      { name := "decay", knob := some (4, 0),
        display := some { min := 0.5, max := 50, log := true } }]
  | "reverb" => #[
      { name := "in", accepts := modalIn, multi := true },
      { name := "rt60", accepts := sigIn, knob := some (2, 0), discipline := .glide,
        display := some { min := 0.2, max := 12, log := true, unit := "sec" } },
      { name := "dir", accepts := sigIn, knob := some (0, 0), discipline := .glide,
        display := some { min := 0, max := 1 } },
      { name := "sway", accepts := sigIn, knob := some (0, 0), discipline := .glide,
        display := some { min := 0, max := 0.9 } },
      { name := "rate", accepts := sigIn, knob := some (3, 1), discipline := .glide,
        display := some { min := 0.05, max := 8, log := true, unit := "Hz" } }]
  | "filter" => #[
      { name := "in", accepts := modalIn, multi := true },
      { name := "cutoff", knob := some (800, 0), discipline := .glide,
        display := some { min := 20, max := 8000, log := true, unit := "Hz" } },
      { name := "resonance", knob := some (5, 1), discipline := .glide,
        display := some { min := 0, max := 1 } }]
  | "phaser" => #[
      { name := "in", accepts := modalIn, multi := true },
      { name := "center", knob := some (700, 0), discipline := .glide,
        display := some { min := 40, max := 4000, log := true, unit := "Hz" } },
      { name := "sweep", knob := some (15, 1), discipline := .glide,
        display := some { min := 0, max := 3, unit := "oct" } },
      { name := "rate", knob := some (2, 1), discipline := .glide,
        display := some { min := 0.02, max := 8, log := true, unit := "Hz" } },
      { name := "mix", knob := some (5, 1), discipline := .glide,
        display := some { min := 0, max := 1 } }]
  | "modal_allpass_tail" => #[
      { name := "in", accepts := modalIn, multi := true },
      { name := "center", knob := some (700, 0), discipline := .glide },
      { name := "sweep", knob := some (15, 1), discipline := .glide },
      { name := "rate", knob := some (2, 1), discipline := .glide }]
  | "modalblend" => #[
      { name := "dry", accepts := modalIn },
      { name := "wet", accepts := modalIn },
      { name := "mix", knob := some (5, 1), discipline := .glide }]
  | "modalmix" => #[{ name := "in", accepts := modalIn, multi := true }]
  | "gauge" => #[
      { name := "in", accepts := modalIn, multi := true },
      { name := "g", accepts := sigIn, knob := some (0, 0), discipline := .glide,
        display := some { min := 0, max := 1 } }]
  | "gong" => #[
      { name := "beta", knob := some (5, 2),
        display := some { min := 0, max := 0.5 } }]
  | "string" => #[{ name := "addr", accepts := sigIn }]
  | "out" => #[{ name := "in", accepts := sigIn, multi := true }]
  | _ => #[]

def outletOf : String → Option PortDomain
  | "knob" => some .control
  | "resonator" | "reverb" | "filter" | "phaser" | "modalmix" | "string" | "gauge"
  | "modal_allpass_tail" | "modalblend" => some .modal
  | "out" => none
  | _ => some .signal

def vocabularyKinds : Array String := #[
  "source", "pluck", "comb", "flange", "delay", "reverse", "fm", "sflange",
  "mix", "ring", "gong", "string", "resonator", "reverb", "filter", "phaser",
  "modalmix", "gauge", "knob", "out"]

def hierarchyAtomKinds : Array String := #["modal_allpass_tail", "modalblend"]

def buildNodeKinds : Array String := #[
  "knob", "source", "pluck", "comb", "flange", "sflange", "fm", "delay",
  "reverse", "mix", "ring", "resonator", "reverb", "filter", "phaser", "modalmix",
  "modal_allpass_tail", "modalblend", "gauge", "gong", "bloomgong", "string"]

def withheldKinds : Array String := #["bloomgong"]

def portOf (kind kname : String) : Option PortSpec :=
  (portSpecs kind).find? (·.name == kname)

def disciplineOf (kind kname : String) : Discipline :=
  ((portOf kind kname).map (·.discipline)).getD .raw

def isGlided (kind kname : String) : Bool := disciplineOf kind kname == .glide
def isAnchored (kind kname : String) : Bool := disciplineOf kind kname == .anchor

def displayRangeOf (kind kname : String) : Option (Float × Float) :=
  ((portOf kind kname).bind (·.display)).map fun display =>
    (display.min, display.max)

end Metadata
end Tropical.Playground
