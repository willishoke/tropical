/**
 * jit_slots.test.ts — M6 end-to-end test: hand-crafted slot-mode plan
 * exercises the JIT's new code paths (slot operand reads, WriteSlot
 * instruction, slots arg in kernel signature).
 *
 * Constructs a minimal plan that:
 *   - Allocates 1 slot (slot 0)
 *   - The kernel WriteSlot's a constant value into slot 0
 *   - Reads slot 0 back via the slot operand and writes it to a temp
 *   - The temp drives the audio output
 *
 * If the JIT correctly emits slot loads/stores, the audio output equals
 * the constant we wrote. Without the M6 JIT changes, the kernel would
 * either crash (signature mismatch with FlatRuntime) or produce silence.
 */
import { describe, expect, test } from 'bun:test'
import { Runtime } from './runtime.ts'

describe('M6: JIT slot operand + WriteSlot', () => {
  test('hand-crafted slot-mode plan: WriteSlot then slot read', () => {
    // Hand-craft a tropical_plan_4 with slot fields. The kernel:
    //   instr 0: WriteSlot(slot=0, args=[Const(0.42)])  — write 0.42 to slot 0
    //   instr 1: Add temp[0] = slot[0] + 0              — read slot 0
    // outputs: [0] → output_targets[0] = temp 0 → audio buffer
    const plan = {
      schema: 'tropical_plan_4',
      config: { sampleRate: 44100 },
      state_init: [],
      register_names: [],
      register_types: [],
      array_slot_names: [],
      outputs: [0],
      register_count: 1,                   // 1 temp register
      array_slot_count: 0,
      array_slot_sizes: [],
      output_targets: [0],                 // output[0] = temp[0]
      register_targets: [],                // no state registers
      instructions: [
        // WriteSlot(slot=0, args=[Const(0.42)])
        {
          tag: 'WriteSlot',
          dst: 0,                          // slot index
          args: [{ kind: 'const', val: 0.42, scalar_type: 'float' }],
          loop_count: 1,
          strides: [],
          result_type: 'float',
        },
        // Add temp[0] = slot[0] + 0
        {
          tag: 'Add',
          dst: 0,                          // temp index
          args: [
            { kind: 'slot', index: 0, scalar_type: 'float' },
            { kind: 'const', val: 0, scalar_type: 'float' },
          ],
          loop_count: 1,
          strides: [],
          result_type: 'float',
        },
      ],
      // Slot model fields
      slot_count: 1,
      slot_names: ['my_slot'],
      slot_defaults: [0],
    }

    const rt = new Runtime(64)
    rt.loadPlan(JSON.stringify(plan))
    rt.process()
    const out = rt.outputBuffer
    // The kernel applies a 1/20 mix-bus gain on output. So a temp value
    // of 0.42 → output of 0.42/20 = 0.021 per sample. The slot path is
    // verified end-to-end if every sample equals this fixed value.
    expect(out.length).toBe(64)
    expect(out[0]).toBeCloseTo(0.42 / 20, 10)
    expect(out[63]).toBeCloseTo(0.42 / 20, 10)
    rt.dispose()
  })

  test('control-plane slot write feeds JIT slot operand', () => {
    // Same shape as above but no WriteSlot — the kernel ONLY reads from
    // slot 0. The control plane writes the value via set_slot before
    // process(). Verifies the JIT slot read picks up writes from set_slot.
    const plan = {
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
      slot_names: ['external'],
      slot_defaults: [0],
    }

    const rt = new Runtime(64)
    rt.loadPlan(JSON.stringify(plan))
    rt.process()
    expect(rt.outputBuffer[0]).toBe(0)        // default value (still 0/20 = 0)

    rt.setSlot(0, 0.75)
    rt.process()
    expect(rt.outputBuffer[0]).toBeCloseTo(0.75 / 20, 10)  // /20 mix-bus gain
    rt.dispose()
  })

  test('slot defaults survive into kernel reads on first sample', () => {
    // Verifies slot_defaults seeds slot 0 to 1.5 and the kernel reads it
    // on the very first process() call (no set_slot beforehand).
    const plan = {
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
      slot_names: ['preset'],
      slot_defaults: [1.5],
    }
    const rt = new Runtime(64)
    rt.loadPlan(JSON.stringify(plan))
    rt.process()
    expect(rt.outputBuffer[0]).toBeCloseTo(1.5 / 20, 10)
    rt.dispose()
  })
})
