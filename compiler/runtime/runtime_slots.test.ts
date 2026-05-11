/**
 * runtime_slots.test.ts — M5 end-to-end FFI tests for slot accessors.
 *
 * Loads a real slot-mode plan into the native FlatRuntime and verifies
 * that the C API slot helpers (slot_index / set_slot / get_slot) work
 * round-trip. Also confirms that legacy plans without slot fields keep
 * working (slot_count = 0; slot_index returns -1 sentinel).
 */
import { describe, expect, test } from 'bun:test'
import { Runtime } from './runtime.ts'
import { makeSession, allocateOutputSlots, allocateParamSlot } from '../session.ts'
import { loadStdlib } from '../program.ts'
import { instantiate } from '../program_types.ts'
import { compileSessionLegacy } from '../ir/compile_session.ts'
import { compileSessionSlotted } from '../ir/compile_session_slotted.ts'

describe('M5: Runtime slot accessors via FFI', () => {
  test('legacy plan: slot_index returns -1 for any name', () => {
    const s = makeSession()
    loadStdlib(s)
    const onePole = s.typeRegistry.get('OnePole')!
    s.instanceRegistry.set('lp1', instantiate(onePole, 'lp1'))
    // Wire to dac so the legacy plan is non-trivial
    s.graphOutputs.push({ instance: 'lp1', output: 'out' })

    const plan = compileSessionLegacy(s)
    const rt = new Runtime(64)
    rt.loadPlan(JSON.stringify(plan))

    expect(rt.slotIndex('lp1.out')).toBe(-1)   // no slot table in legacy plan
    expect(rt.slotIndex('anything')).toBe(-1)
    rt.dispose()
  })

  test('slot-mode plan: slot_index resolves names from session registries', () => {
    const s = makeSession()
    loadStdlib(s)
    const onePole = s.typeRegistry.get('OnePole')!
    s.instanceRegistry.set('lp1', instantiate(onePole, 'lp1'))
    s.instanceRegistry.set('lp2', instantiate(onePole, 'lp2'))
    allocateOutputSlots(s, 'lp1', onePole)
    allocateOutputSlots(s, 'lp2', onePole)
    s.graphOutputs.push({ instance: 'lp1', output: 'out' })

    const plan = compileSessionSlotted(s)
    const rt = new Runtime(64)
    rt.loadPlan(JSON.stringify(plan))

    expect(rt.slotIndex('lp1.out')).toBeGreaterThanOrEqual(0)
    expect(rt.slotIndex('lp2.out')).toBeGreaterThanOrEqual(0)
    expect(rt.slotIndex('lp1.out')).not.toBe(rt.slotIndex('lp2.out'))
    expect(rt.slotIndex('does-not-exist')).toBe(-1)
    rt.dispose()
  })

  test('slot-mode plan: set/get round-trip', () => {
    const s = makeSession()
    loadStdlib(s)
    const onePole = s.typeRegistry.get('OnePole')!
    s.instanceRegistry.set('lp1', instantiate(onePole, 'lp1'))
    allocateOutputSlots(s, 'lp1', onePole)
    s.graphOutputs.push({ instance: 'lp1', output: 'out' })

    const plan = compileSessionSlotted(s)
    const rt = new Runtime(64)
    rt.loadPlan(JSON.stringify(plan))

    const idx = rt.slotIndex('lp1.out')
    expect(idx).toBeGreaterThanOrEqual(0)
    expect(rt.getSlot(idx)).toBe(0)            // initial: from slot_defaults
    rt.setSlot(idx, 0.42)
    expect(rt.getSlot(idx)).toBe(0.42)
    rt.dispose()
  })

  test('slot defaults from session paramRegistry', () => {
    const s = makeSession()
    loadStdlib(s)
    // Manually allocate a param slot with a known default value.
    s.paramRegistry.set('cutoff', { value: 1234.5 } as any)
    const idx = allocateParamSlot(s, 'cutoff')
    expect(idx).toBe(0)

    const plan = compileSessionSlotted(s)
    const rt = new Runtime(64)
    rt.loadPlan(JSON.stringify(plan))

    const slotIdx = rt.slotIndex('param:cutoff')
    expect(slotIdx).toBe(idx)
    expect(rt.getSlot(slotIdx)).toBe(1234.5)   // initialized from slot_defaults
    rt.dispose()
  })

  test('slot values persist across hot-swap when names match', () => {
    // Load plan A with one slot, write a value, then load plan B that
    // also has that slot — the value should transfer (M5 hot-swap rule).
    const s = makeSession()
    loadStdlib(s)
    const onePole = s.typeRegistry.get('OnePole')!
    s.instanceRegistry.set('lp1', instantiate(onePole, 'lp1'))
    allocateOutputSlots(s, 'lp1', onePole)
    s.graphOutputs.push({ instance: 'lp1', output: 'out' })

    const planA = compileSessionSlotted(s)
    const rt = new Runtime(64)
    rt.loadPlan(JSON.stringify(planA))

    const idxA = rt.slotIndex('lp1.out')
    rt.setSlot(idxA, 7.7)
    expect(rt.getSlot(idxA)).toBe(7.7)

    // Reload the same plan — slot 'lp1.out' still exists and value should
    // survive even if the index were to change.
    rt.loadPlan(JSON.stringify(planA))
    const idxB = rt.slotIndex('lp1.out')
    expect(idxB).toBeGreaterThanOrEqual(0)
    expect(rt.getSlot(idxB)).toBe(7.7)
    rt.dispose()
  })
})
