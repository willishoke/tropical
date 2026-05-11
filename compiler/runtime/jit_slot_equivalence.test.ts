/**
 * jit_slot_equivalence.test.ts — M7 equivalence gate for slot-mode plans.
 *
 * Two complementary checks:
 *
 * 1. **Slot-mode preserves audio.** Under M4's "metadata only" scoping,
 *    slot-mode plans must produce sample-exact identical audio to the
 *    legacy plan. This test runs a representative patch through both
 *    paths and asserts byte-equal output.
 *
 * 2. **Slot operands match legacy operands.** A hand-crafted plan that
 *    uses slot operands for a value should produce the same audio as
 *    the equivalent plan that uses const/state_reg operands. Verifies
 *    the JIT's M6 slot codegen is semantically correct, not just
 *    structurally valid.
 *
 * Note: extending interpret_resolved to mirror slot operands is deferred
 * until M8 introduces slot operands at the IR level. Today, the IR
 * (ResolvedProgram) is unchanged by slot mode — only the FlatPlan
 * acquires slot fields. The interpreter operates on IR and is unaffected.
 */
import { describe, expect, test } from 'bun:test'
import { Runtime } from './runtime.ts'
import { makeSession, allocateOutputSlots } from '../session.ts'
import { loadStdlib } from '../program.ts'
import { instantiate } from '../program_types.ts'
import { compileSessionLegacy } from '../ir/compile_session.ts'
import { compileSessionSlotted } from '../ir/compile_session_slotted.ts'

describe('M7: JIT slot-mode preserves audio output', () => {
  test('SinOsc → dac patch: slot-mode and legacy plans produce identical audio', () => {
    const s = makeSession()
    loadStdlib(s)
    const sinOsc = s.typeRegistry.get('SinOsc')!
    s.instanceRegistry.set('osc1', instantiate(sinOsc, 'osc1'))
    allocateOutputSlots(s, 'osc1', sinOsc)
    s.graphOutputs.push({ instance: 'osc1', output: 'sine' })
    s.inputExprNodes.set('osc1:freq', 440)

    const legacyPlan  = compileSessionLegacy(s)
    const slottedPlan = compileSessionSlotted(s)

    // Run both through the JIT and collect sample-by-sample outputs
    const N_FRAMES = 4
    const collect = (plan: object) => {
      const rt = new Runtime(64)
      rt.loadPlan(JSON.stringify(plan))
      const samples: number[] = []
      for (let f = 0; f < N_FRAMES; f++) {
        rt.process()
        for (const v of rt.outputBuffer) samples.push(v)
      }
      rt.dispose()
      return samples
    }
    const legacyAudio = collect(legacyPlan)
    const slottedAudio = collect(slottedPlan)

    expect(legacyAudio.length).toBe(slottedAudio.length)
    expect(legacyAudio.length).toBe(64 * N_FRAMES)
    // Byte-exact: slot mode is metadata-only at M4, so no perturbation
    // of the audio path is acceptable.
    for (let i = 0; i < legacyAudio.length; i++) {
      expect(slottedAudio[i]).toBe(legacyAudio[i])
    }
  })

  test('multi-instance chain: slot mode preserves audio', () => {
    const s = makeSession()
    loadStdlib(s)
    const sinOsc  = s.typeRegistry.get('SinOsc')!
    const onePole = s.typeRegistry.get('OnePole')!
    s.instanceRegistry.set('osc',  instantiate(sinOsc, 'osc'))
    s.instanceRegistry.set('lp',   instantiate(onePole, 'lp'))
    allocateOutputSlots(s, 'osc', sinOsc)
    allocateOutputSlots(s, 'lp', onePole)
    s.inputExprNodes.set('osc:freq', 220)
    s.inputExprNodes.set('lp:input', { op: 'ref', instance: 'osc', output: 'sine' })
    s.inputExprNodes.set('lp:g',     0.1)
    s.graphOutputs.push({ instance: 'lp', output: 'out' })

    const legacyPlan  = compileSessionLegacy(s)
    const slottedPlan = compileSessionSlotted(s)

    const collect = (plan: object) => {
      const rt = new Runtime(64)
      rt.loadPlan(JSON.stringify(plan))
      const samples: number[] = []
      for (let f = 0; f < 2; f++) {
        rt.process()
        for (const v of rt.outputBuffer) samples.push(v)
      }
      rt.dispose()
      return samples
    }
    const legacyAudio = collect(legacyPlan)
    const slottedAudio = collect(slottedPlan)

    expect(legacyAudio.length).toBe(slottedAudio.length)
    for (let i = 0; i < legacyAudio.length; i++) {
      expect(slottedAudio[i]).toBe(legacyAudio[i])
    }
  })
})

describe('M7: hand-crafted slot operands match equivalent legacy formulations', () => {
  test('slot read produces same value as direct const', () => {
    // Plan A (slot path): slot 0 holds 0.3, kernel reads via slot operand.
    const planSlot = {
      schema: 'tropical_plan_4',
      config: { sampleRate: 44100 },
      state_init: [],
      register_names: [],
      register_types: [],
      array_slot_names: [],
      outputs: [0],
      register_count: 1,
      array_slot_count: 0,
      array_slot_sizes: [],
      output_targets: [0],
      register_targets: [],
      instructions: [
        {
          tag: 'Add',
          dst: 0,
          args: [
            { kind: 'slot', index: 0, scalar_type: 'float' },
            { kind: 'const', val: 0, scalar_type: 'float' },
          ],
          loop_count: 1,
          strides: [],
          result_type: 'float',
        },
      ],
      slot_count: 1,
      slot_names: ['x'],
      slot_defaults: [0.3],
    }

    // Plan B (legacy): same value as a const.
    const planConst = {
      schema: 'tropical_plan_4',
      config: { sampleRate: 44100 },
      state_init: [],
      register_names: [],
      register_types: [],
      array_slot_names: [],
      outputs: [0],
      register_count: 1,
      array_slot_count: 0,
      array_slot_sizes: [],
      output_targets: [0],
      register_targets: [],
      instructions: [
        {
          tag: 'Add',
          dst: 0,
          args: [
            { kind: 'const', val: 0.3, scalar_type: 'float' },
            { kind: 'const', val: 0, scalar_type: 'float' },
          ],
          loop_count: 1,
          strides: [],
          result_type: 'float',
        },
      ],
    }

    const collect = (plan: object) => {
      const rt = new Runtime(32)
      rt.loadPlan(JSON.stringify(plan))
      rt.process()
      const out = Array.from(rt.outputBuffer)
      rt.dispose()
      return out
    }

    const slotAudio  = collect(planSlot)
    const constAudio = collect(planConst)
    expect(slotAudio).toEqual(constAudio)
  })

  test('WriteSlot then read in same kernel: equivalent to direct compute', () => {
    // Plan A: WriteSlot puts a const in slot 0, Add reads slot 0 + const.
    const planRoundTrip = {
      schema: 'tropical_plan_4',
      config: { sampleRate: 44100 },
      state_init: [],
      register_names: [],
      register_types: [],
      array_slot_names: [],
      outputs: [0],
      register_count: 1,
      array_slot_count: 0,
      array_slot_sizes: [],
      output_targets: [0],
      register_targets: [],
      instructions: [
        {
          tag: 'WriteSlot',
          dst: 0,
          args: [{ kind: 'const', val: 0.7, scalar_type: 'float' }],
          loop_count: 1,
          strides: [],
          result_type: 'float',
        },
        {
          tag: 'Mul',
          dst: 0,
          args: [
            { kind: 'slot', index: 0, scalar_type: 'float' },
            { kind: 'const', val: 2, scalar_type: 'float' },
          ],
          loop_count: 1,
          strides: [],
          result_type: 'float',
        },
      ],
      slot_count: 1,
      slot_names: ['scratch'],
      slot_defaults: [0],
    }

    // Plan B: direct multiply with no slot round-trip.
    const planDirect = {
      schema: 'tropical_plan_4',
      config: { sampleRate: 44100 },
      state_init: [],
      register_names: [],
      register_types: [],
      array_slot_names: [],
      outputs: [0],
      register_count: 1,
      array_slot_count: 0,
      array_slot_sizes: [],
      output_targets: [0],
      register_targets: [],
      instructions: [
        {
          tag: 'Mul',
          dst: 0,
          args: [
            { kind: 'const', val: 0.7, scalar_type: 'float' },
            { kind: 'const', val: 2, scalar_type: 'float' },
          ],
          loop_count: 1,
          strides: [],
          result_type: 'float',
        },
      ],
    }

    const collect = (plan: object) => {
      const rt = new Runtime(32)
      rt.loadPlan(JSON.stringify(plan))
      rt.process()
      const out = Array.from(rt.outputBuffer)
      rt.dispose()
      return out
    }

    expect(collect(planRoundTrip)).toEqual(collect(planDirect))
  })
})
