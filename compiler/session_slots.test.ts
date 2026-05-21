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
import { loadStdlib, loadProgramAsSession } from './program.ts'
import { Float, Int, Bool } from './ir/port_type.ts'
import { ArrayType } from './ir/port_type.ts'
import { slotKey, instanceName } from './ir/branded_names.js'

const sk = (i: string, n: string) => slotKey(instanceName(i), n)

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
    expect(s.outputSlotRegistry.get(sk("lp1", "out"))).toBe(0)
    expect(s.slotCount).toBe(1)
    const meta = s.outputPortMeta.get(sk("lp1", "out"))!
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
    expect(s.outputSlotRegistry.get(sk("lp1", "out"))).toBe(0)
    expect(s.outputSlotRegistry.get(sk("lp2", "out"))).toBe(1)
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
    allocateOutputSlots(s, 'lp1', onePole)            // slot 0 (out)
    const p = allocateParamSlot(s, 'cutoff')          // slot 1
    allocateOutputSlots(s, 'lp2', onePole)            // slot 2 (out)
    expect(s.outputSlotRegistry.get(sk("lp1", "out"))).toBe(0)
    expect(p).toBe(1)
    expect(s.outputSlotRegistry.get(sk("lp2", "out"))).toBe(2)
    expect(s.slotCount).toBe(3)
  })

  test('legacy state untouched by slot allocation', () => {
    const s = makeSession()
    loadStdlib(s)
    const onePole = s.typeRegistry.get('OnePole')!
    allocateOutputSlots(s, 'lp1', onePole)
    expect(s.inputExprNodes.size).toBe(0)
    expect(s.paramRegistry.size).toBe(0)
  })

  test('output ports without explicit PortType fall back to one scalar slot', () => {
    // Some stdlib outputs have no explicit PortType in the post-strata IR
    // (e.g. unannotated outputs from Delay's specialized form).
    // allocateOutputSlots must fall back to a single scalar-float slot
    // rather than throwing. The wire_dac.test.ts "generic stdlib types"
    // case exercises this path end-to-end via the MCP add_instance handler;
    // this test directly synthesizes a Compiled with one undefined output
    // PortType to pin the fallback behavior here.
    const s = makeSession()
    loadStdlib(s)
    // Construct a Compiled-shaped object with a single output whose port
    // type is undefined. We borrow the fields off OnePole and overwrite
    // the output decl's `type` to undefined.
    const onePole = s.typeRegistry.get('OnePole')!
    const fakePortless = {
      ...onePole,
      prog: {
        ...onePole.prog,
        ports: {
          ...onePole.prog.ports,
          outputs: [{ ...onePole.prog.ports.outputs[0], type: undefined }],
        },
      },
      slotsCache: undefined,
    }
    allocateOutputSlots(s, 'fp1', fakePortless as any)
    expect(s.outputSlotRegistry.get(sk(`fp1`, onePole.prog.ports.outputs[0].name))).toBe(0)
    expect(s.slotCount).toBe(1)
  })
})

describe('M3: applyParamSpecs allocates param slots transitively', () => {
  test('loadProgramAsSession with declared params populates paramSlotRegistry', () => {
    const s = makeSession()
    loadStdlib(s)
    // Minimal program declaring two top-level control params: one smoothed,
    // one trigger. The body is a paramDecl form — applyParamSpecs sees the
    // params via mergeParamSources(prog, topLevel) and registers them.
    const prog = {
      name: 'TestPatch',
      ports: { inputs: [], outputs: [] },
      body: {
        op: 'block',
        decls: [
          { op: 'paramDecl', name: 'cutoff', type: 'param',   value: 1000, time_const: 0.005 },
          { op: 'paramDecl', name: 'fire',   type: 'trigger' },
        ],
        assigns: [],
      },
    }
    const topLevel = {}
    loadProgramAsSession(prog as any, topLevel as any, s)
    // Both names land in the unified paramRegistry (legacy `type: trigger`
    // becomes a non-smoothed param after the trigger refactor).
    expect(s.paramRegistry.has('cutoff')).toBe(true)
    expect(s.paramRegistry.has('fire')).toBe(true)
    // ALSO populated by the slot model (M3 wire-up via allocateParamSlot)
    expect(s.paramSlotRegistry.get('cutoff')).toBeDefined()
    expect(s.paramSlotRegistry.get('fire')).toBeDefined()
    expect(s.paramSlotRegistry.get('cutoff')).not.toBe(s.paramSlotRegistry.get('fire'))
  })

  test('loadProgramAsSession resets slot registries between loads', () => {
    const s = makeSession()
    loadStdlib(s)
    const onePole = s.typeRegistry.get('OnePole')!
    allocateOutputSlots(s, 'lp1', onePole)            // slotCount = 1 (out)
    allocateParamSlot(s, 'cutoff')                     // slotCount = 2
    expect(s.slotCount).toBe(2)
    // Loading any program should reset the slot state cleanly.
    loadProgramAsSession(
      { name: 'Empty', ports: { inputs: [], outputs: [] }, body: { op: 'block', decls: [], assigns: [] } } as any,
      {} as any,
      s,
    )
    expect(s.slotCount).toBe(0)
    expect(s.outputSlotRegistry.size).toBe(0)
    expect(s.paramSlotRegistry.size).toBe(0)
    expect(s.outputPortMeta.size).toBe(0)
    expect(s.inputExprs.size).toBe(0)
  })
})
