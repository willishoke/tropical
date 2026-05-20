// trigger_dead_code.test.ts
//
// Forensic regression test for the trigger-removal refactor. Pins down
// the current behavior: the compiler never emits `SmoothParam` or
// `TriggerParam` instructions. Param/trigger refs resolve to `slot`
// operand reads via translateNode (compile_session_slotted_helpers.ts),
// not via dedicated param ops.
//
// The C++ JIT still understands `OpTag::SmoothParam` /
// `OpTag::TriggerParam` and the C API still exposes
// `tropical_param_new_trigger`, but nothing in the compilation pipeline
// produces an instruction that exercises them. This test makes that
// invariant explicit so Phase 3's deletion of those ops doesn't
// silently regress.
//
// This file is deleted in Phase 3 once the dead instructions are gone.

import { describe, test, expect } from 'bun:test'
import { makeSession, loadJSON } from './session.js'
import { loadStdlib } from './program.js'
import { compileSession } from './ir/compile_session.js'
import type { FlatInstr } from './flat_plan.js'

function* allInstructions(plan: ReturnType<typeof compileSession>): Generator<FlatInstr> {
  for (const instr of plan.scheduler_function.preamble) yield instr
  for (const instr of plan.scheduler_function.state_evolution) yield instr
  for (const instr of plan.scheduler_function.postamble) yield instr
  for (const inst of plan.instance_functions) {
    for (const instr of inst.instructions) yield instr
  }
}

describe('trigger dead-code', () => {
  test('compileSession emits no SmoothParam or TriggerParam instructions', () => {
    const session = makeSession(44100)
    loadStdlib(session)

    // Use a stdlib module with a trigger input wired through a derived
    // rising-edge signal — the canonical trigger pattern in stdlib.
    loadJSON({
      schema: 'tropical_program_2',
      name: 'test',
      body: { op: 'block', decls: [
        { op: 'instanceDecl', name: 'sh', program: 'SampleHold', inputs: {
          trigger: {
            op: 'select',
            args: [{ op: 'eq', args: [{ op: 'sampleIndex' }, 100] }, 1, 0],
          },
          input: { op: 'mul', args: [{ op: 'sampleIndex' }, 0.01] },
        }},
      ]},
      audio_outputs: [{ instance: 'sh', output: 'value' }],
    }, session)

    const plan = compileSession(session)

    const tagsSeen = new Set<string>()
    for (const instr of allInstructions(plan)) {
      tagsSeen.add(instr.tag)
    }

    expect(tagsSeen.has('SmoothParam')).toBe(false)
    expect(tagsSeen.has('TriggerParam')).toBe(false)
  })

  test('param refs resolve to slot operands, not param operands', () => {
    // A direct {op:'param', name} ref should compile to a slot read,
    // not to an OperandKind::Param operand on any arithmetic instruction.
    // We verify by allocating a param slot and using it in a wire.
    const session = makeSession(44100)
    loadStdlib(session)

    loadJSON({
      schema: 'tropical_program_2',
      name: 'test',
      params: [{ name: 'gate', type: 'param', value: 0 }],
      body: { op: 'block', decls: [
        { op: 'instanceDecl', name: 'sh', program: 'SampleHold', inputs: {
          trigger: { op: 'param', name: 'gate' },
          input: 1.0,
        }},
      ]},
      audio_outputs: [{ instance: 'sh', output: 'value' }],
    }, session)

    const plan = compileSession(session)

    let sawParamOperand = false
    for (const instr of allInstructions(plan)) {
      for (const arg of instr.args ?? []) {
        if (arg && typeof arg === 'object' && arg.kind === 'param') {
          sawParamOperand = true
        }
      }
    }
    expect(sawParamOperand).toBe(false)
  })
})
