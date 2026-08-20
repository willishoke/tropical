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

/-- Deterministic qualification-only logarithmic voicings. They are selected
    by Decode only when tile-time staging is explicitly enabled and the hidden
    `_benchmark_stages` field is present; they are not factory voicings.

    These are the exact binary64 results of
    `2 ^ (-1.25 + 2.5 * i / (count - 1))`. Keeping the complete admitted input
    table literal prevents a benchmark-only `Float.pow` from entering the
    production bake closure and makes the hidden fixture platform-independent. -/
def modalPhaserBenchmarkRatios : Nat → Array Float
  | 2 => #[0.42044820762685725, 2.378414230005442]
  | 3 => #[0.42044820762685725, 1.0, 2.378414230005442]
  | 4 => #[0.42044820762685725, 0.7491535384383408,
      1.3348398541700344, 2.378414230005442]
  | 5 => #[0.42044820762685725, 0.6484197773255048, 1.0,
      1.5422108254079407, 2.378414230005442]
  | 6 => modalPhaserRatios
  | 7 => #[0.42044820762685725, 0.5612310241546865,
      0.7491535384383408, 1.0, 1.3348398541700344,
      1.7817974362806788, 2.378414230005442]
  | 8 => #[0.42044820762685725, 0.5385465128844868,
      0.6898170601727025, 0.8835774907475349, 1.1317626472738322,
      1.449659710865429, 1.8568498283350516, 2.378414230005442]
  | 9 => #[0.42044820762685725, 0.5221368912137069,
      0.6484197773255048, 0.8052451659746271, 1.0,
      1.241857812073484, 1.5422108254079407, 1.9152065613971474,
      2.378414230005442]
  | 10 => #[0.42044820762685725, 0.5097203218510724,
      0.6179472329646446, 0.7491535384383408, 0.9082183626944033,
      1.101056795453143, 1.3348398541700344, 1.618261150231923,
      1.96186017533783, 2.378414230005442]
  | 11 => #[0.42044820762685725, 0.5, 0.5946035575013605,
      0.7071067811865476, 0.8408964152537145, 1.0,
      1.189207115002721, 1.4142135623730951, 1.681792830507429,
      2.0, 2.378414230005442]
  | 12 => #[0.42044820762685725, 0.49218504495292786,
      0.5761616153452749, 0.674466261015763, 0.7895436369463346,
      0.9242555049434509, 1.0819519003689173, 1.266554441332253,
      1.4826538520903552, 1.735624125881299, 2.031756166211103,
      2.378414230005442]
  | 13 => #[0.42044820762685725, 0.48576597057680293,
      0.5612310241546865, 0.6484197773255048, 0.7491535384383408,
      0.8655365610061431, 1.0, 1.155352696872273,
      1.3348398541700344, 1.5422108254079407, 1.7817974362806788,
      2.0586044732869837, 2.378414230005442]
  | 14 => #[0.42044820762685725, 0.48039987884286783,
      0.548900053337985, 0.6271676614077392, 0.7165954405062778,
      0.8187747183931087, 0.9355237301064594, 1.0689199726512586,
      1.2213371731390976, 1.3954875282118677, 1.594469966380923,
      1.8218252920887412, 2.0815991927572615, 2.378414230005442]
  | 15 => #[0.42044820762685725, 0.4758475765053098,
      0.5385465128844868, 0.6095068271022377, 0.6898170601727025,
      0.7807091821557102, 0.8835774907475349, 1.0,
      1.1317626472738322, 1.2808866897642726, 1.449659710865429,
      1.6406707120152757, 1.8568498283350516, 2.1015132773064393,
      2.378414230005442]
  | 16 => #[0.42044820762685725, 0.47193715634084676,
      0.5297315471796477, 0.5946035575013605, 0.6674199270850172,
      0.7491535384383408, 0.8408964152537145, 0.9438743126816935,
      1.0594630943592953, 1.189207115002721, 1.3348398541700344,
      1.4983070768766815, 1.681792830507429, 1.8877486253633868,
      2.1189261887185906, 2.378414230005442]
  | 17 => #[0.42044820762685725, 0.468541908527575,
      0.5221368912137069, 0.5818624293887887, 0.6484197773255048,
      0.7225904034885233, 0.8052451659746271, 0.8973545375015536,
      1.0, 1.1143867425958924, 1.241857812073484,
      1.383909881963832, 1.5422108254079407, 1.718619298122478,
      1.9152065613971474, 2.1342808013536474, 2.378414230005442]
  | 18 => #[0.42044820762685725, 0.46556639223589835,
      0.5155261971574755, 0.5708471753712547, 0.6321046329480693,
      0.6999356118992076, 0.775045515043657, 0.8582154417880785,
      0.9503103111073667, 1.052287856200078, 1.1652085843579272,
      1.290246805626211, 1.4287028449468326, 1.5820165647831208,
      1.7517823388539016, 1.939765632694966, 2.1479213634761436,
      2.378414230005442]
  | _ => #[]

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
        display := some { min := 0, max := 1 } }]
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
