/**
 * Shared patch corpus for the benchmark scripts. Lists the patches in
 * `patches/` that currently compile through `compileSession`. Patches
 * blocked on known compile limitations (array-typed instance inputs,
 * missing stdlib programs) are excluded.
 *
 * When a blocker is resolved, move the entry from BLOCKED to COMPILES.
 */

export const COMPILES: string[] = [
  'bubble_cloud.json',
  'cross_fm_4.json',
  'cross_fm_evolved.json',
]

export const BLOCKED: { patch: string; reason: string }[] = [
  { patch: 'bubble_drip.json',         reason: 'array-shaped input expression at clk.ratios_in (sequencer ratios)' },
  { patch: 'sequencer_demo.json',      reason: 'array-shaped input expression at clk.ratios_in' },
  { patch: 'acid_noise.json',          reason: 'array-shaped input expression at ClockBeat.ratios_in' },
  { patch: 'odd_harmonics.json',       reason: 'array-shaped input expression at Clock1.ratios_in' },
  { patch: 'arp_transpose.json',       reason: 'array-shaped input expression at clock1.ratios_in' },
  { patch: 'int_seq_test.json',        reason: 'index op with non-array operand (array-typed input in slotted path)' },
  { patch: 'compressor_harmonics.json', reason: "missing stdlib program 'Compressor'" },
  { patch: 'melancholy_house.json',    reason: "missing stdlib program 'Compressor'" },
  { patch: '31tet_otonal_seq.json',    reason: "missing stdlib program 'Reverb'" },
]
