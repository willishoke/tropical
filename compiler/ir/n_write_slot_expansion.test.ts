/**
 * n_write_slot_expansion.test.ts — Phase 3 structural witness.
 *
 * Verifies that array-typed instance OUTPUT ports compile to N
 * WriteSlots (one per scalar element) rather than 1 WriteSlot per port.
 *
 * The actual end-to-end stdlib tests (Clock equivalence, BubbleCloud
 * round-trip) are still blocked on Phase 5's array-input handling.
 * This test isolates the output-side fix.
 */

import { describe, test, expect } from 'bun:test'
import { makeSession } from '../session.js'
import { loadProgramAsType } from '../program.js'
import { compileSession } from './compile_session.js'
import { slotKey, instanceName } from './branded_names.js'

const sk = (i: string, n: string) => slotKey(instanceName(i), n)

describe('Phase 3 — N-WriteSlot expansion for array output ports', () => {
  test('an instance with array_out: float[3] emits 3 WriteSlots', () => {
    const session = makeSession(64)

    // Register a type with BOTH an array-typed output (for WriteSlot
    // expansion) and a scalar output (for the DAC wire). DAC stitch
    // for array-typed graphOutputs is out of M11 Phase 3 scope.
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

    // Confirm slot allocation expanded to 3 for the array port.
    const aMeta = session.outputPortMeta.get(sk("a", "arr"))
    expect(aMeta).toBeDefined()
    expect(aMeta!.scalarSlotNames).toEqual(['a.arr[0]', 'a.arr[1]', 'a.arr[2]'])

    // Compile and inspect.
    const plan = compileSession(session)
    expect(plan.instance_functions.length).toBe(1)
    const instFn = plan.instance_functions[0]

    // Count WriteSlot instructions in the instance body. Expected:
    //   3 for arr[0], arr[1], arr[2]
    // + 1 for single
    // = 4
    const writeSlots = instFn.instructions.filter(i => i.tag === 'WriteSlot')
    expect(writeSlots.length).toBe(4)
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
