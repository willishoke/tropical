/**
 * m9a_per_instance_equiv.test.ts — M9a equivalence gate.
 *
 * Asserts that the per-instance slot-mode compile path
 * (`compileSessionSlotted` under `TROPICAL_SLOT_OPS=1`) produces
 * byte-exact identical audio to `compileSessionLegacy` on the M9a-
 * supported patch shapes: single instance with constants, single
 * instance wired to dac, multi-instance ref chains.
 *
 * Out of scope (subsequent sub-milestones throw with clear errors
 * and are tested separately): fan-in, arbitrary input expressions,
 * params/triggers in input expressions, arrays, sums.
 */
import { describe, expect, test, beforeEach, afterEach } from 'bun:test'
import { Runtime } from './runtime.ts'
import { makeSession, allocateOutputSlots } from '../session.ts'
import { loadStdlib } from '../program.ts'
import { instantiate } from '../program_types.ts'
import { compileSessionLegacy } from '../ir/compile_session.ts'
import { compileSessionSlotted } from '../ir/compile_session_slotted.ts'

// Enable per-instance path for the duration of this file's tests
let prevEnv: string | undefined
beforeEach(() => {
  prevEnv = process.env.TROPICAL_SLOT_OPS
  process.env.TROPICAL_SLOT_OPS = '1'
})
afterEach(() => {
  if (prevEnv === undefined) delete process.env.TROPICAL_SLOT_OPS
  else process.env.TROPICAL_SLOT_OPS = prevEnv
})

function runAudio(plan: object, frames = 2, bufferLen = 64): number[] {
  const rt = new Runtime(bufferLen)
  rt.loadPlan(JSON.stringify(plan))
  const samples: number[] = []
  for (let f = 0; f < frames; f++) {
    rt.process()
    for (const v of rt.outputBuffer) samples.push(v)
  }
  rt.dispose()
  return samples
}

describe('M9a: per-instance slot-mode equivalence', () => {
  test('single SinOsc → dac (const freq input): byte-equal to legacy', () => {
    const s = makeSession()
    loadStdlib(s)
    const sinOsc = s.typeRegistry.get('SinOsc')!
    s.instanceRegistry.set('osc', instantiate(sinOsc, 'osc'))
    allocateOutputSlots(s, 'osc', sinOsc)
    s.inputExprNodes.set('osc:freq', 440)
    s.graphOutputs.push({ instance: 'osc', output: 'sine' })

    const legacyPlan  = compileSessionLegacy(s)
    const slottedPlan = compileSessionSlotted(s)
    const legacyAudio  = runAudio(legacyPlan)
    const slottedAudio = runAudio(slottedPlan)

    expect(slottedAudio.length).toBe(legacyAudio.length)
    for (let i = 0; i < legacyAudio.length; i++) {
      expect(slottedAudio[i]).toBe(legacyAudio[i])
    }
  })

  test('SinOsc → OnePole → dac chain: byte-equal to legacy', () => {
    const s = makeSession()
    loadStdlib(s)
    const sinOsc  = s.typeRegistry.get('SinOsc')!
    const onePole = s.typeRegistry.get('OnePole')!
    s.instanceRegistry.set('osc', instantiate(sinOsc, 'osc'))
    s.instanceRegistry.set('lp',  instantiate(onePole, 'lp'))
    allocateOutputSlots(s, 'osc', sinOsc)
    allocateOutputSlots(s, 'lp',  onePole)
    s.inputExprNodes.set('osc:freq', 220)
    s.inputExprNodes.set('lp:input', { op: 'ref', instance: 'osc', output: 'sine' })
    s.inputExprNodes.set('lp:g',     0.1)
    s.graphOutputs.push({ instance: 'lp', output: 'out' })

    const legacyPlan  = compileSessionLegacy(s)
    const slottedPlan = compileSessionSlotted(s)
    const legacyAudio  = runAudio(legacyPlan)
    const slottedAudio = runAudio(slottedPlan)

    expect(slottedAudio.length).toBe(legacyAudio.length)
    for (let i = 0; i < legacyAudio.length; i++) {
      expect(slottedAudio[i]).toBe(legacyAudio[i])
    }
  })

  test('plan has slot operands and WriteSlot instructions (not legacy form)', () => {
    // Smoke test that the per-instance path is actually emitting slot
    // operands — if a regression accidentally fell through to the
    // metadata-only path, the instruction stream would have no slot
    // operands at all.
    const s = makeSession()
    loadStdlib(s)
    const sinOsc = s.typeRegistry.get('SinOsc')!
    s.instanceRegistry.set('osc', instantiate(sinOsc, 'osc'))
    allocateOutputSlots(s, 'osc', sinOsc)
    s.inputExprNodes.set('osc:freq', 440)
    s.graphOutputs.push({ instance: 'osc', output: 'sine' })

    const plan = compileSessionSlotted(s)
    const hasSlotOperand = plan.instructions.some(instr =>
      instr.args.some(op => op.kind === 'slot'))
    const hasWriteSlot = plan.instructions.some(instr => instr.tag === 'WriteSlot')
    expect(hasSlotOperand).toBe(true)
    expect(hasWriteSlot).toBe(true)
  })
})

describe('M9a: roadmap sanity', () => {
  test('nested instance call in input expression (Sin(x: ...)) still throws (M9d)', () => {
    // Sin is a stdlib type — wiring `Sin(x: ref)` into an input is a
    // nested instance call that M9d would handle. M9c covers arithmetic
    // / comparison / unary / ternary; nested instance calls are a
    // separate, harder case.
    const s = makeSession()
    loadStdlib(s)
    const onePole = s.typeRegistry.get('OnePole')!
    s.instanceRegistry.set('lp', instantiate(onePole, 'lp'))
    allocateOutputSlots(s, 'lp', onePole)
    // {op:'call', callee:..., args:...} is the nested-call shape;
    // not in BINARY/UNARY/TERNARY tag tables.
    s.inputExprNodes.set('lp:input', { op: 'call', callee: 'Sin', args: { x: 1.0 } })
    s.inputExprNodes.set('lp:g', 0.1)
    s.graphOutputs.push({ instance: 'lp', output: 'out' })
    expect(() => compileSessionSlotted(s)).toThrow(/M9d|nested instance|not.*supported/i)
  })
})
