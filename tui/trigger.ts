// Pure scope-trigger helpers (no engine, unit-testable).

/**
 * Index of the first rising crossing of `level` within samples[1..searchLen):
 * the i where samples[i-1] < level <= samples[i]. Returns -1 if none. Locking
 * the displayed window to start at this point keeps the trace from sliding as
 * the render window advances with playback — the "phase lock" of a real scope.
 */
export function findTrigger(samples: number[], level: number, searchLen: number): number {
  const lim = Math.min(searchLen, samples.length)
  for (let i = 1; i < lim; i++)
    if (samples[i - 1] < level && samples[i] >= level) return i
  return -1
}

/** Slice `count` samples starting at `offset`, applying a vertical shift. */
export function slice(samples: number[], offset: number, count: number, vshift = 0): number[] {
  const out = new Array<number>(count)
  for (let i = 0; i < count; i++) out[i] = (samples[offset + i] ?? 0) + vshift
  return out
}
