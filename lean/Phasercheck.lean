import Tropical.Tropicaltest.Phaser
import Tropical.Tropicaltest.Modal
import Tropical.Tropicaltest.OrientedPatch

def main : IO UInt32 := do
  match Tropical.EmitArrow.buildStdlibChain with
  | .error error =>
      IO.eprintln error
      return 1
  | .ok (arena, resolved) =>
      let oriented ← Tropical.Tropicaltest.OrientedPatch.runOrientedPatch arena
      let twoRoom ← runModalUniverseHistory arena resolved
      let phaser ← Tropical.Tropicaltest.Phaser.runPhaser arena resolved
      return if oriented && twoRoom && phaser then 0 else 1
