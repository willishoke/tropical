/**
 * n_write_slot_expansion.test.ts — array-output emit shape.
 *
 * Verifies that the array-I/O refactor changed the per-port writeback
 * shape as designed:
 *   - scalar/alias output port  → 1 WriteSlot (into a module slot)
 *   - array output port of shape [N] → 0 WriteSlots, N SetElement
 *     instructions writing the per-element temps into the port's
 *     session-array slot.
 *
 * The old shape (N WriteSlots into N scalar slots per array port)
 * was an artifact of decomposing arrays into scalar bundles at the
 * slot layer. First-class array slots make that decomposition
 * unnecessary; the test now witnesses the new shape.
 */

import { describe, test, expect } from 'bun:test'
import { makeSession } from '../session.js'
import { loadProgramAsType } from '../program.js'
import { compileSession } from './compile_session.js'
import { slotKey, instanceName } from './branded_names.js'

const sk = (i: string, n: string) => slotKey(instanceName(i), n)

describe('array-output writeback shape', () => {
  test('an instance with array_out: float[3] emits 3 SetElements + 1 WriteSlot', () => {
    const session = makeSession(64)

    // Register a type with BOTH an array-typed output (for SetElement
    // expansion) and a scalar output (for the DAC wire). Array-typed
    // graphOutputs are still out of scope for DAC stitching; using a
    // scalar port for the audio output keeps the test focused on the
    // structural assertion.
    loadProgramAsType({
      op: 'program',
      name: 'ArrayOut',
      ports: {
        outputs: [
          { name: 'arr',    type: { element: 'float', shape: [3] } },
          { name: 'single', type: 'float' },
        ],
      },
      body: { op: 'block', assigns: [
        { op: 'outputAssign', name: 'arr',    expr: [10, 20, 30] },
        { op: 'outputAssign', name: 'single', expr: 0.5 },
      ]},
    } as Parameters<typeof loadProgramAsType>[0], session)

    const { loadJSON } = require('../session.js') as typeof import('../session.js')
    loadJSON({
      schema: 'tropical_program_2',
      name: 'patch',
      body: { op: 'block', decls: [
        { op: 'instanceDecl', name: 'a', program: 'ArrayOut', inputs: {} },
      ]},
      audio_outputs: [{ instance: 'a', output: 'single' }],
    } as Parameters<typeof loadJSON>[0], session)

    // Array outputs no longer decompose into scalar slot names. The
    // port's WirePortMeta carries a session-array slot index instead.
    const aMeta = session.outputPortMeta.get(sk("a", "arr"))
    expect(aMeta).toBeDefined()
    expect(aMeta!.scalarSlotNames).toEqual([])
    expect(aMeta!.arraySlot).toBeDefined()
    expect(aMeta!.arraySize).toBe(3)

    const plan = compileSession(session)
    expect(plan.instance_functions.length).toBe(1)
    const instFn = plan.instance_functions[0]

    // Scalar output port: one WriteSlot per scalar element (= 1 here).
    const writeSlots = instFn.instructions.filter(i => i.tag === 'WriteSlot')
    expect(writeSlots.length).toBe(1)

    // Array output port: one SetElement per element of the declared
    // shape (= 3 here). The emit-side packing of the literal
    // `[10,20,30]` produces its own SetElements too (via Pack
    // unboxing), so the count is at least 3; assert lower-bound so
    // the test isn't brittle to incidental Pack-emit shape changes.
    const setElements = instFn.instructions.filter(i => i.tag === 'SetElement')
    expect(setElements.length).toBeGreaterThanOrEqual(3)

    // The array slot allocated for `arr` should appear in the FlatPlan's
    // array_slot_sizes (at the session-absolute index recorded on the
    // port meta).
    expect(plan.array_slot_count).toBeGreaterThanOrEqual(1)
    expect(plan.array_slot_sizes[aMeta!.arraySlot!]).toBe(3)
  })

  test('scalar output port still emits 1 WriteSlot', () => {
    const session = makeSession(64)

    loadProgramAsType({
      op: 'program',
      name: 'ScalarOut',
      ports: { outputs: [{ name: 's', type: 'float' }] },
      body: { op: 'block', assigns: [
        { op: 'outputAssign', name: 's', expr: 5 },
      ]},
    } as Parameters<typeof loadProgramAsType>[0], session)

    const { loadJSON } = require('../session.js') as typeof import('../session.js')
    loadJSON({
      schema: 'tropical_program_2',
      name: 'patch',
      body: { op: 'block', decls: [
        { op: 'instanceDecl', name: 'a', program: 'ScalarOut', inputs: {} },
      ]},
      audio_outputs: [{ instance: 'a', output: 's' }],
    } as Parameters<typeof loadJSON>[0], session)

    const plan = compileSession(session)
    const writeSlots = plan.instance_functions[0].instructions.filter(i => i.tag === 'WriteSlot')
    // 1 for the scalar output (alive comes from scheduler preamble).
    expect(writeSlots.length).toBe(1)
  })
})
