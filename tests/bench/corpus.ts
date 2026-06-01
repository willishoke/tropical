/**
 * Shared patch corpus for the benchmark scripts. Lists the patches in
 * `patches/` that currently compile through `compileSession`. Patches
 * blocked on real issues (missing stdlib programs, malformed patch
 * files) are excluded.
 *
 * When a blocker is resolved, move the entry from BLOCKED to COMPILES.
 *
 * Note: array-typed instance inputs (Clock `ratios_in`, Sequencer
 * `values`, …) are NO LONGER a blocker — they compile through the
 * root-program lowering's array session-slot path. The patches that
 * used to be parked here on that reason are now in COMPILES.
 */

export const COMPILES: string[] = [
  'bubble_cloud.json',
  'cross_fm_4.json',
  'cross_fm_evolved.json',
  // Array-input patches (Clock ratios_in / Sequencer values) — unblocked
  // by the root-program array session-slot support.
  'acid_noise.json',
  'bubble_drip.json',
  'odd_harmonics.json',
  'sequencer_demo.json',
]

export const BLOCKED: { patch: string; reason: string }[] = [
  { patch: 'arp_transpose.json',        reason: 'malformed patch: an inter-instance cycle is not broken by a delay (CycleViolation)' },
  { patch: 'int_seq_test.json',         reason: "malformed patch: 'sequence' input is declared scalar but indexed as an array (should be float[N])" },
  { patch: 'compressor_harmonics.json', reason: "missing stdlib program 'Compressor'" },
  { patch: 'melancholy_house.json',     reason: "missing stdlib program 'Compressor'" },
  { patch: '31tet_otonal_seq.json',     reason: "missing stdlib program 'Reverb'" },
]
