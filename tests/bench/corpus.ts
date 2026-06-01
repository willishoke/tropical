/**
 * Shared patch corpus for the benchmark scripts. Every patch in
 * `patches/` compiles through `compileSession`; this is the runnable
 * list the bench scripts iterate.
 *
 * (The dead patches that used to be parked in a BLOCKED list — broken
 * files and ones referencing never-built stdlib types like Reverb /
 * Compressor — were deleted; array-input patches were unblocked by the
 * root-program lowering's array session-slot support.)
 */

export const COMPILES: string[] = [
  'acid_noise.json',
  'bubble_cloud.json',
  'bubble_drip.json',
  'cross_fm_4.json',
  'cross_fm_evolved.json',
  'harmonic_lfo_ring.json',
  'odd_harmonics.json',
  'sequencer_demo.json',
]
