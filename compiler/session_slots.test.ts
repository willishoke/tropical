/**
 * Smoke tests for the slot-model session helpers (M2).
 *
 * Verifies that allocateOutputSlots and allocateParamSlot populate the
 * registries correctly without disturbing the legacy session state. These
 * are the data-shape additions; the actual wiring through add_instance
 * lands in M3.
 */
import { describe, expect, test } from 'bun:test'
import {
  makeSession, allocateOutputSlots, allocateParamSlot, expandPortToSlots,
} from './session.ts'
import { loadStdlib } from './program.ts'
import { Float, Int, Bool } from './ir/port_type.ts'
import { ArrayType } from './ir/port_type.ts'

describe('expandPortToSlots', () => {
  test('scalar port → 1 slot, original name', () => {
    const r = expandPortToSlots('osc.out', Float)
    expect(r).toEqual({ names: ['osc.out'], types: ['float'] })
  })

  test('scalar bool port preserves type', () => {
    const r = expandPortToSlots('inst.gate', Bool)
    expect(r).toEqual({ names: ['inst.gate'], types: ['bool'] })
  })

  test('1-D array port → N indexed slots', () => {
    const r = expandPortToSlots('voice.bands', ArrayType('float', [4]))
    expect(r.names).toEqual([
      'voice.bands[0]', 'voice.bands[1]',
      'voice.bands[2]', 'voice.bands[3]',
    ])
    expect(r.types).toEqual(['float', 'float', 'float', 'float'])
  })

  test('2-D array port → product(shape) slots', () => {
    const r = expandPortToSlots('m', ArrayType('int', [2, 3]))
    expect(r.names.length).toBe(6)
    expect(r.types).toEqual(['int', 'int', 'int', 'int', 'int', 'int'])
  })
})

describe('session slot allocation', () => {
  test('makeSession starts with empty slot state', () => {
    const s = makeSession()
    expect(s.slotCount).toBe(0)
    expect(s.outputSlotRegistry.size).toBe(0)
    expect(s.paramSlotRegistry.size).toBe(0)
    expect(s.outputPortMeta.size).toBe(0)
    expect(s.inputExprs.size).toBe(0)
  })

  test('allocateOutputSlots assigns one slot per scalar output of OnePole', () => {
    const s = makeSession()
    loadStdlib(s)
    const onePole = s.typeRegistry.get('OnePole')!
    allocateOutputSlots(s, 'lp1', onePole)
    // OnePole has one output 'out' (scalar float)
    expect(s.outputSlotRegistry.get('lp1.out')).toBe(0)
    expect(s.slotCount).toBe(1)
    const meta = s.outputPortMeta.get('lp1.out')!
    expect(meta.scalarSlotNames).toEqual(['lp1.out'])
    expect(meta.scalarTypes).toEqual(['float'])
  })

  test('allocateOutputSlots is idempotent', () => {
    const s = makeSession()
    loadStdlib(s)
    const onePole = s.typeRegistry.get('OnePole')!
    allocateOutputSlots(s, 'lp1', onePole)
    allocateOutputSlots(s, 'lp1', onePole)  // second call: no-op
    expect(s.slotCount).toBe(1)
  })

  test('allocateOutputSlots assigns distinct slots to distinct instances', () => {
    const s = makeSession()
    loadStdlib(s)
    const onePole = s.typeRegistry.get('OnePole')!
    allocateOutputSlots(s, 'lp1', onePole)
    allocateOutputSlots(s, 'lp2', onePole)
    expect(s.outputSlotRegistry.get('lp1.out')).toBe(0)
    expect(s.outputSlotRegistry.get('lp2.out')).toBe(1)
    expect(s.slotCount).toBe(2)
  })

  test('allocateParamSlot returns unique indices and is idempotent by name', () => {
    const s = makeSession()
    const a = allocateParamSlot(s, 'cutoff')
    const b = allocateParamSlot(s, 'gain')
    const a2 = allocateParamSlot(s, 'cutoff')  // idempotent
    expect(a).toBe(0)
    expect(b).toBe(1)
    expect(a2).toBe(0)
    expect(s.slotCount).toBe(2)
  })

  test('output and param allocations share the slotCount space', () => {
    const s = makeSession()
    loadStdlib(s)
    const onePole = s.typeRegistry.get('OnePole')!
    allocateOutputSlots(s, 'lp1', onePole)            // slot 0
    const p = allocateParamSlot(s, 'cutoff')          // slot 1
    allocateOutputSlots(s, 'lp2', onePole)            // slot 2
    expect(s.outputSlotRegistry.get('lp1.out')).toBe(0)
    expect(p).toBe(1)
    expect(s.outputSlotRegistry.get('lp2.out')).toBe(2)
    expect(s.slotCount).toBe(3)
  })

  test('legacy state untouched by slot allocation', () => {
    const s = makeSession()
    loadStdlib(s)
    const onePole = s.typeRegistry.get('OnePole')!
    allocateOutputSlots(s, 'lp1', onePole)
    expect(s.inputExprNodes.size).toBe(0)
    expect(s.paramRegistry.size).toBe(0)
    expect(s.triggerRegistry.size).toBe(0)
  })
})
