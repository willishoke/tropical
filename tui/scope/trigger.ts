// Pure scope-trigger helpers (no engine, unit-testable).

/**
 * Index of the first *armed* rising crossing of `level` within
 * samples[1..searchLen). Hysteresis (`hyst`): the trigger only arms after the
 * signal has dipped below `level − hyst`, so wiggles near the level don't
 * double-trigger and jump the lock. Falls back to the first bare crossing if no
 * clean edge is found. Returns -1 if none. Locking the displayed window here
 * keeps the trace from sliding as the render window advances — the scope's
 * phase lock.
 */
export function findTrigger(samples: number[], level: number, searchLen: number, hyst = 0.05): number {
  const lim = Math.min(searchLen, samples.length)
  let armed = (samples[0] ?? 0) < level - hyst
  for (let i = 1; i < lim; i++) {
    const v = samples[i]
    if (v < level - hyst) armed = true
    else if (armed && samples[i - 1] < level && v >= level) return i
  }
  for (let i = 1; i < lim; i++)
    if (samples[i - 1] < level && samples[i] >= level) return i
  return -1
}

/**
 * Period-aligned trigger: the offset (into a window starting at absolute sample
 * `startIndex`) of the first phasor wrap for an oscillator at `freq`. The phasor
 * phase is the oscillator's *timebase* — a clean ramp with exactly one wrap per
 * period regardless of the output waveform — so locking to it is rock-solid
 * where a level trigger jitters (mid-morph a saw↔sine blend has two rising
 * zero-crossings per period). Matches FixedPhasor exactly: inc = trunc(freq·2³²
 * /sr), phase(n) = (inc·n mod 2³²)/2³². Returns 0 if no wrap in [0, count).
 */
export function findPhaseTrigger(freq: number, sr: number, startIndex: number, count: number): number {
  const TWO32 = 4294967296n
  const inc = BigInt(Math.trunc((freq * 4294967296) / sr))
  const ph = (n: number) => (inc * BigInt(n)) % TWO32
  let prev = ph(startIndex)
  for (let i = 1; i < count; i++) {
    const cur = ph(startIndex + i)
    if (cur < prev) return i // wrapped → start of a new period
    prev = cur
  }
  return 0
}

/** Slice `count` samples starting at `offset`, applying a vertical shift. */
export function slice(samples: number[], offset: number, count: number, vshift = 0): number[] {
  const out = new Array<number>(count)
  for (let i = 0; i < count; i++) out[i] = (samples[offset + i] ?? 0) + vshift
  return out
}
