/**
 * flat_plan_mode.test.ts — round-trip coverage for the `compilation_mode`
 * field added in Phase 1 of the microkernel-mode spike.
 *
 * Pure-TS test: no FFI, no native runtime. The compileSession →
 * compileSessionSlotted threading is covered transitively by the
 * cross-mode equivalence suite (Phase 6); this file pins the
 * data-format contract.
 *
 * Three properties under test:
 *   1. `parseCompilationMode` accepts 'fused' / 'microkernel' / undefined,
 *      fails closed on anything else.
 *   2. `toWirePlan` / `parseWirePlan` round-trip the mode losslessly; the
 *      legacy default 'fused' is *omitted* from the wire format so
 *      pre-existing golden fixtures don't gain a spurious key.
 *   3. `parseWirePlan` on legacy wire plans (no `compilation_mode`
 *      field) yields `'fused'`.
 */

import { describe, test, expect } from 'bun:test'

import {
  toWirePlan, parseWirePlan, parseCompilationMode,
  type FlatPlan, type WireFlatPlan,
} from './flat_plan.js'

// Smallest plan that satisfies the schema. The wire→branded parser
// touches every field, so we need them all populated even though the
// content is empty.
function makeEmptyWirePlan(): WireFlatPlan {
  return {
    schema: 'tropical_plan_5',
    config: { sampleRate: 44100 },
    state_init:       [],
    register_names:   [],
    register_types:   [],
    array_slot_names: [],
    register_count:   0,
    array_slot_count: 0,
    array_slot_sizes: [],
    instance_functions: [],
    scheduler_function: {
      preamble:       [],
      postamble:      [],
      output_targets: [],
      outputs:        [],
    },
    slot_count:    0,
    slot_names:    [],
    slot_defaults: [],
  }
}

describe('parseCompilationMode', () => {
  test('undefined → fused (legacy default)', () => {
    expect(parseCompilationMode(undefined)).toBe('fused')
  })

  test('explicit fused / microkernel pass through', () => {
    expect(parseCompilationMode('fused')).toBe('fused')
    expect(parseCompilationMode('microkernel')).toBe('microkernel')
  })

  test('unknown strings fail closed', () => {
    expect(() => parseCompilationMode('FUSED')).toThrow(/unknown compilation_mode/)
    expect(() => parseCompilationMode('')).toThrow(/unknown compilation_mode/)
    expect(() => parseCompilationMode('hybrid')).toThrow(/unknown compilation_mode/)
  })
})

describe('parseWirePlan: legacy compatibility', () => {
  test('wire plan without compilation_mode parses as fused', () => {
    const wire = makeEmptyWirePlan()
    expect(wire.compilation_mode).toBeUndefined()
    const plan = parseWirePlan(wire)
    expect(plan.compilation_mode).toBe('fused')
  })

  test("wire plan with explicit compilation_mode: 'fused' parses as fused", () => {
    const wire: WireFlatPlan = { ...makeEmptyWirePlan(), compilation_mode: 'fused' }
    const plan = parseWirePlan(wire)
    expect(plan.compilation_mode).toBe('fused')
  })

  test("wire plan with compilation_mode: 'microkernel' parses as microkernel", () => {
    const wire: WireFlatPlan = { ...makeEmptyWirePlan(), compilation_mode: 'microkernel' }
    const plan = parseWirePlan(wire)
    expect(plan.compilation_mode).toBe('microkernel')
  })
})

describe('toWirePlan: serialization', () => {
  test('fused mode is OMITTED from the wire (golden-fixture compatibility)', () => {
    const plan: FlatPlan = parseWirePlan(makeEmptyWirePlan())
    expect(plan.compilation_mode).toBe('fused')
    const wire = toWirePlan(plan)
    // The key must be absent, not `undefined`-as-value — JSON.stringify
    // would emit `"compilation_mode":null` for the latter on some
    // runtimes, but on Node/Bun it correctly drops `undefined` values.
    // We assert the property does not exist at all.
    expect('compilation_mode' in wire).toBe(false)
  })

  test('microkernel mode is emitted explicitly', () => {
    const wire0 = makeEmptyWirePlan()
    const plan: FlatPlan = { ...parseWirePlan(wire0), compilation_mode: 'microkernel' }
    const wire = toWirePlan(plan)
    expect(wire.compilation_mode).toBe('microkernel')
  })
})

describe('full round-trip', () => {
  test('fused: parseWirePlan ∘ toWirePlan = identity (on the mode field)', () => {
    const plan: FlatPlan = parseWirePlan(makeEmptyWirePlan())
    const roundTripped = parseWirePlan(toWirePlan(plan))
    expect(roundTripped.compilation_mode).toBe('fused')
  })

  test('microkernel: parseWirePlan ∘ toWirePlan = identity', () => {
    const wire0 = makeEmptyWirePlan()
    const plan: FlatPlan = { ...parseWirePlan(wire0), compilation_mode: 'microkernel' }
    const roundTripped = parseWirePlan(toWirePlan(plan))
    expect(roundTripped.compilation_mode).toBe('microkernel')
  })

  test('JSON-string round trip preserves microkernel mode', () => {
    const wire0 = makeEmptyWirePlan()
    const plan: FlatPlan = { ...parseWirePlan(wire0), compilation_mode: 'microkernel' }
    const json = JSON.stringify(toWirePlan(plan))
    const wire = JSON.parse(json) as WireFlatPlan
    expect(wire.compilation_mode).toBe('microkernel')
    expect(parseWirePlan(wire).compilation_mode).toBe('microkernel')
  })

  test('JSON-string round trip preserves fused-mode absence', () => {
    const plan: FlatPlan = parseWirePlan(makeEmptyWirePlan())
    const json = JSON.stringify(toWirePlan(plan))
    expect(json).not.toContain('compilation_mode')
    const wire = JSON.parse(json) as WireFlatPlan
    expect(parseWirePlan(wire).compilation_mode).toBe('fused')
  })
})
