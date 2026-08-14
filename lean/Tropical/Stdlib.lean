import Tropical.EmitArrow.ArenaStdlib

/-!
# Tropical.Stdlib — arena-native production standard library

The implementation lives in `Tropical.EmitArrow.ArenaNative` during the
phased `Sig` cutover.  These stable outer names keep the production boot chain
and its callers unchanged without transporting expressions between authoring
representations.
-/

namespace Tropical.EmitArrow

open Tropical.Ir

abbrev StdBuilder := ArenaNative.StdBuilder

def stdlibBuilders : Array (String × StdBuilder) :=
  ArenaNative.stdlibBuilders

def buildStdlibChain : Except String (Arena × Array (String × ProgramIdx)) :=
  ArenaNative.buildStdlibChain

end Tropical.EmitArrow
