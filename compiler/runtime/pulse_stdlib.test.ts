/**
 * pulse_stdlib.test.ts — M9b: verify stdlib Pulse module emits a
 * one-sample pulse on rising edges of its input signal.
 *
 * Audio test: drive Pulse with a slot whose value the control plane
 * sets, observe the output. The Pulse module is the user-facing way to
 * get fire-once semantics under the slot model — replaces the legacy
 * kernel-side TriggerParam consume.
 */
import { describe, expect, test } from 'bun:test'
import { Runtime } from './runtime.ts'
import { makeSession, allocateOutputSlots, allocateParamSlot } from '../session.ts'
import { loadStdlib } from '../program.ts'
import { instantiate } from '../program_types.ts'
import { compileSessionSlotted } from '../ir/compile_session_slotted.ts'

describe('M9b: Pulse stdlib emits rising-edge pulses', () => {
  test('Pulse fires exactly once per low-to-high transition', () => {
    // Set up: source slot 'fire' drives Pulse; Pulse output → dac.
    // Set fire=1.0, run a buffer, observe Pulse out spikes briefly.
    const s = makeSession(8)
    loadStdlib(s)
    const Pulse = s.typeRegistry.get('Pulse')!
    expect(Pulse).toBeDefined()
    s.instanceRegistry.set('p', instantiate(Pulse, 'p'))
    allocateOutputSlots(s, 'p', Pulse)
    // Use a session-level "param" as the fire source. Per M3,
    // applyParamSpecs would allocate a slot — we mimic it here.
    // Need to set up paramRegistry so slot_defaults can be read; M9b
    // uses allocateParamSlot + paramRegistry lookup.
    s.paramRegistry.set('fire', { value: 0 } as any)
    allocateParamSlot(s, 'fire')
    s.inputExprNodes.set('p:signal', { op: 'param', name: 'fire' })
    s.graphOutputs.push({ instance: 'p', output: 'out' })

    process.env.TROPICAL_SLOT_OPS = '1'
    try {
      const plan = compileSessionSlotted(s)
      const rt = new Runtime(8)
      rt.loadPlan(JSON.stringify(plan))

      // Buffer 1: fire is 0.0 (default). Pulse out should be 0.
      rt.process()
      const buf1 = Array.from(rt.outputBuffer)
      for (const v of buf1) expect(v).toBe(0)

      // Fire: write 1.0 to the slot. On the next process, Pulse sees
      // signal=1 with prev=0 → fires for one sample. Then prev becomes
      // 1, so subsequent samples in the same buffer don't fire again.
      const fireIdx = rt.slotIndex('param:fire')
      expect(fireIdx).toBeGreaterThanOrEqual(0)
      rt.setSlot(fireIdx, 1.0)
      rt.process()
      const buf2 = Array.from(rt.outputBuffer)

      // First sample of buf2 should be the rising-edge pulse;
      // subsequent samples should be 0 (signal stays at 1.0, prev=1.0
      // after first sample so no more edge).
      // Note: due to /20 mix-bus gain, a bool 1 reads as 1/20 = 0.05 in output.
      expect(buf2[0]).toBeCloseTo(1 / 20, 10)
      for (let i = 1; i < buf2.length; i++) {
        expect(buf2[i]).toBe(0)
      }

      // Buffer 3 with fire still 1.0: no more edge, all zeros.
      rt.process()
      const buf3 = Array.from(rt.outputBuffer)
      for (const v of buf3) expect(v).toBe(0)

      // Reset and re-fire: write 0 then 1 → another rising edge.
      rt.setSlot(fireIdx, 0)
      rt.process()
      const buf4 = Array.from(rt.outputBuffer)
      for (const v of buf4) expect(v).toBe(0)  // staying low

      rt.setSlot(fireIdx, 1)
      rt.process()
      const buf5 = Array.from(rt.outputBuffer)
      expect(buf5[0]).toBeCloseTo(1 / 20, 10)  // new rising edge

      rt.dispose()
    } finally {
      delete process.env.TROPICAL_SLOT_OPS
    }
  })
})
