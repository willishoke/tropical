/**
 * compile_session_slotted.test.ts — tests for the M4 slot-mode compile path.
 *
 * Verifies the slot-mode FlatPlan carries correct slot allocation
 * metadata and is otherwise equivalent to the legacy plan. Audio
 * behavior IS the legacy path under M4 — the engine doesn't yet
 * consume the slot fields. M5–M7 add real engine handling.
 */
import { describe, expect, test } from 'bun:test'
import { makeSession, allocateOutputSlots, allocateParamSlot } from '../session.ts'
import { loadStdlib, loadProgramAsSession } from '../program.ts'
import { instantiate } from '../program_types.ts'
import { compileSessionLegacy } from './compile_session.ts'
import { compileSessionSlotted, slotModeEnabled } from './compile_session_slotted.ts'

describe('M4: compileSessionSlotted', () => {
  test('empty session → empty slot fields', () => {
    const s = makeSession()
    const plan = compileSessionSlotted(s)
    expect(plan.slot_count).toBe(0)
    expect(plan.slot_names).toEqual([])
    expect(plan.slot_defaults).toEqual([])
  })

  test('session with one OnePole instance → one output slot', () => {
    const s = makeSession()
    loadStdlib(s)
    const onePole = s.typeRegistry.get('OnePole')!
    const inst = instantiate(onePole, 'lp1')
    s.instanceRegistry.set('lp1', inst)
    allocateOutputSlots(s, 'lp1', onePole)

    const plan = compileSessionSlotted(s)
    expect(plan.slot_count).toBe(1)
    expect(plan.slot_names).toEqual(['lp1.out'])
    expect(plan.slot_defaults).toEqual([0])
  })

  test('output slot names match outputSlotRegistry indices', () => {
    const s = makeSession()
    loadStdlib(s)
    const onePole = s.typeRegistry.get('OnePole')!
    s.instanceRegistry.set('a', instantiate(onePole, 'a'))
    s.instanceRegistry.set('b', instantiate(onePole, 'b'))
    allocateOutputSlots(s, 'a', onePole)
    allocateOutputSlots(s, 'b', onePole)

    const plan = compileSessionSlotted(s)
    expect(plan.slot_count).toBe(2)
    expect(plan.slot_names![s.outputSlotRegistry.get('a.out')!]).toBe('a.out')
    expect(plan.slot_names![s.outputSlotRegistry.get('b.out')!]).toBe('b.out')
  })

  test('param slots populate with prefix and default value', () => {
    const s = makeSession()
    loadProgramAsSession(
      {
        name: 'P', ports: { inputs: [], outputs: [] },
        body: {
          op: 'block',
          decls: [
            { op: 'paramDecl', name: 'cutoff', type: 'param', value: 1500, time_const: 0.005 },
            { op: 'paramDecl', name: 'fire',   type: 'trigger' },
          ],
          assigns: [],
        },
      } as any,
      {} as any,
      s,
    )
    const plan = compileSessionSlotted(s)
    const cutoffIdx = s.paramSlotRegistry.get('cutoff')!
    const fireIdx   = s.paramSlotRegistry.get('fire')!
    expect(plan.slot_names![cutoffIdx]).toBe('param:cutoff')
    expect(plan.slot_names![fireIdx]).toBe('param:fire')
    expect(plan.slot_defaults![cutoffIdx]).toBe(1500)
    expect(plan.slot_defaults![fireIdx]).toBe(0)  // trigger
  })

  test('default-mode slot plan has identical instruction stream to legacy plan', () => {
    // M4 invariant under DEFAULT mode (no TROPICAL_SLOT_OPS): the
    // slot-mode plan is byte-equal to legacy plus the slot metadata
    // fields. When TROPICAL_SLOT_OPS=1, the per-instance path
    // produces a different (slot-operand) instruction stream by
    // design — that's the M9 work; tested separately under M9 tests.
    const prevEnv = process.env.TROPICAL_SLOT_OPS
    delete process.env.TROPICAL_SLOT_OPS
    try {
      const s = makeSession()
      loadStdlib(s)
      const onePole = s.typeRegistry.get('OnePole')!
      s.instanceRegistry.set('lp1', instantiate(onePole, 'lp1'))
      allocateOutputSlots(s, 'lp1', onePole)
      s.graphOutputs.push({ instance: 'lp1', output: 'out' })

      const legacy  = compileSessionLegacy(s)
      const slotted = compileSessionSlotted(s)

      // Same instruction count and same instruction stream
      expect(slotted.instructions.length).toBe(legacy.instructions.length)
      expect(JSON.stringify(slotted.instructions)).toBe(JSON.stringify(legacy.instructions))
      expect(slotted.register_count).toBe(legacy.register_count)
      expect(slotted.output_targets).toEqual(legacy.output_targets)

      // Slotted-only fields: present and populated
      expect(slotted.slot_count).toBe(1)
      expect(slotted.slot_names).toEqual(['lp1.out'])
      expect(legacy.slot_count).toBeUndefined()
    } finally {
      if (prevEnv === undefined) delete process.env.TROPICAL_SLOT_OPS
      else process.env.TROPICAL_SLOT_OPS = prevEnv
    }
  })
})

describe('M8: slot mode is default; slotModeEnabled is deprecated stub', () => {
  test('always returns true (env var dispatch removed in M8)', () => {
    // Slot mode is unconditionally enabled as of M8; the env var is no
    // longer consulted. Existing callers get a stable `true` so they
    // can unblock without changing behavior.
    expect(slotModeEnabled()).toBe(true)
    expect(slotModeEnabled(makeSession())).toBe(true)
    expect(slotModeEnabled(makeSession(), false)).toBe(true)
    expect(slotModeEnabled(makeSession(), true)).toBe(true)
  })

  test('compileSession defaults to slot-mode plan', () => {
    // Verifies the M8 dispatch flip: the public compileSession entry
    // point now produces a plan with slot fields populated, without
    // any env-var setup.
    const { compileSession } = require('./compile_session.ts')
    const s = makeSession()
    const plan = compileSession(s)
    expect(plan.slot_count).toBe(0)        // empty session, no slots
    expect(plan.slot_names).toEqual([])
    expect(plan.slot_defaults).toEqual([])
  })
})
