/-!
# Absolute-time tile-stage feature controls

The prototype is intentionally explicit and process-scoped.  Qualification
launches a fresh compiler process for each interval, so these values are read
once and cannot change halfway through authoring an arena.
-/

namespace Tropical.Ir

/-- Explicit admission flag for the isolated phaser terminal. -/
initialize phaserTimeStagingEnabled : Bool ← do
  return (← IO.getEnv "TROPICAL_PHASER_TIME_STAGING") == some "1"

/-- Falsification-only first-order mixed DD image.  The ordinary staged path
    remains the default experiment because the dense-tail qualification data
    rejects this representation; the extra flag makes that result reproducible
    without silently changing the original prototype. -/
initialize phaserTimeStagingMixedDDEnabled : Bool ← do
  return (← IO.getEnv "TROPICAL_PHASER_TIME_STAGING_MIXED_DD") == some "1"

/-- Whole-tail Newton/phase-type image.  This remains an explicit research
    flag pending the manual matched-level listening/product-integration gate. -/
initialize phaserTimeStagingHigherOrderEnabled : Bool ← do
  return (← IO.getEnv "TROPICAL_PHASER_TIME_STAGING_HIGHER_ORDER") == some "1"

/-- Endpoint separation in absolute source frames. -/
initialize phaserTimeStagingInterval : Nat ← do
  let raw ← IO.getEnv "TROPICAL_PHASER_TIME_STAGING_INTERVAL"
  return match raw.bind String.toNat? with
    | some frames => if frames > 0 then frames else 128
    | none => 128

end Tropical.Ir
