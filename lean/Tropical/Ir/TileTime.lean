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

/-- Endpoint separation in absolute source frames. -/
initialize phaserTimeStagingInterval : Nat ← do
  let raw ← IO.getEnv "TROPICAL_PHASER_TIME_STAGING_INTERVAL"
  return match raw.bind String.toNat? with
    | some frames => if frames > 0 then frames else 128
    | none => 128

end Tropical.Ir
