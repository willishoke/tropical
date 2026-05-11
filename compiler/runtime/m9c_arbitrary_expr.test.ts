/**
 * m9c_arbitrary_expr.test.ts — M9c equivalence gate.
 *
 * Arbitrary input expressions (binary arithmetic, comparison, unary,
 * ternary) emitted as preamble instructions before the consuming
 * instance's body. The result feeds the instance via a `reg` operand.
 *
 * Out of scope (M9d): nested instance calls in input expressions
 * (Sin(x:...) etc.), array ops, fan-in via combine.
 */
import { describe, expect, test, beforeEach, afterEach } from 'bun:test'
import { Runtime } from './runtime.ts'
import { makeSession, allocateOutputSlots } from '../session.ts'
import { loadStdlib } from '../program.ts'
import { instantiate } from '../program_types.ts'
import { compileSessionLegacy } from '../ir/compile_session.ts'
import { compileSessionSlotted } from '../ir/compile_session_slotted.ts'

let prevEnv: string | undefined
beforeEach(() => {
  prevEnv = process.env.TROPICAL_SLOT_OPS
  process.env.TROPICAL_SLOT_OPS = '1'
})
afterEach(() => {
  if (prevEnv === undefined) delete process.env.TROPICAL_SLOT_OPS
  else process.env.TROPICAL_SLOT_OPS = prevEnv
})

function audio(plan: object, frames = 2, bufferLen = 64): number[] {
  const rt = new Runtime(bufferLen)
  rt.loadPlan(JSON.stringify(plan))
  const out: number[] = []
  for (let f = 0; f < frames; f++) {
    rt.process()
    for (const v of rt.outputBuffer) out.push(v)
  }
  rt.dispose()
  return out
}

describe('M9c: arbitrary input expressions', () => {
  test('mul of ref by const (osc.out * 0.5): byte-equal to legacy', () => {
    const s = makeSession()
    loadStdlib(s)
    const sinOsc  = s.typeRegistry.get('SinOsc')!
    const onePole = s.typeRegistry.get('OnePole')!
    s.instanceRegistry.set('osc', instantiate(sinOsc, 'osc'))
    s.instanceRegistry.set('lp',  instantiate(onePole, 'lp'))
    allocateOutputSlots(s, 'osc', sinOsc)
    allocateOutputSlots(s, 'lp',  onePole)
    s.inputExprNodes.set('osc:freq', 220)
    s.inputExprNodes.set('lp:input', {
      op: 'mul',
      args: [{ op: 'ref', instance: 'osc', output: 'sine' }, 0.5],
    })
    s.inputExprNodes.set('lp:g', 0.1)
    s.graphOutputs.push({ instance: 'lp', output: 'out' })

    const legacy  = compileSessionLegacy(s)
    const slotted = compileSessionSlotted(s)
    const ll = audio(legacy)
    const ss = audio(slotted)
    expect(ss.length).toBe(ll.length)
    for (let i = 0; i < ll.length; i++) expect(ss[i]).toBe(ll[i])
  })

  test('binary add of two refs (osc1.out + osc2.out): byte-equal to legacy', () => {
    const s = makeSession()
    loadStdlib(s)
    const sinOsc  = s.typeRegistry.get('SinOsc')!
    const onePole = s.typeRegistry.get('OnePole')!
    s.instanceRegistry.set('osc1', instantiate(sinOsc, 'osc1'))
    s.instanceRegistry.set('osc2', instantiate(sinOsc, 'osc2'))
    s.instanceRegistry.set('lp',   instantiate(onePole, 'lp'))
    allocateOutputSlots(s, 'osc1', sinOsc)
    allocateOutputSlots(s, 'osc2', sinOsc)
    allocateOutputSlots(s, 'lp',   onePole)
    s.inputExprNodes.set('osc1:freq', 220)
    s.inputExprNodes.set('osc2:freq', 330)
    s.inputExprNodes.set('lp:input', {
      op: 'add',
      args: [
        { op: 'ref', instance: 'osc1', output: 'sine' },
        { op: 'ref', instance: 'osc2', output: 'sine' },
      ],
    })
    s.inputExprNodes.set('lp:g', 0.1)
    s.graphOutputs.push({ instance: 'lp', output: 'out' })

    const ll = audio(compileSessionLegacy(s))
    const ss = audio(compileSessionSlotted(s))
    expect(ss.length).toBe(ll.length)
    for (let i = 0; i < ll.length; i++) expect(ss[i]).toBe(ll[i])
  })

  test('nested arithmetic ((a + b) * 0.5): byte-equal to legacy', () => {
    const s = makeSession()
    loadStdlib(s)
    const sinOsc  = s.typeRegistry.get('SinOsc')!
    const onePole = s.typeRegistry.get('OnePole')!
    s.instanceRegistry.set('a', instantiate(sinOsc, 'a'))
    s.instanceRegistry.set('b', instantiate(sinOsc, 'b'))
    s.instanceRegistry.set('lp', instantiate(onePole, 'lp'))
    allocateOutputSlots(s, 'a', sinOsc)
    allocateOutputSlots(s, 'b', sinOsc)
    allocateOutputSlots(s, 'lp', onePole)
    s.inputExprNodes.set('a:freq', 200)
    s.inputExprNodes.set('b:freq', 300)
    s.inputExprNodes.set('lp:input', {
      op: 'mul',
      args: [
        {
          op: 'add',
          args: [
            { op: 'ref', instance: 'a', output: 'sine' },
            { op: 'ref', instance: 'b', output: 'sine' },
          ],
        },
        0.5,
      ],
    })
    s.inputExprNodes.set('lp:g', 0.1)
    s.graphOutputs.push({ instance: 'lp', output: 'out' })

    const ll = audio(compileSessionLegacy(s))
    const ss = audio(compileSessionSlotted(s))
    for (let i = 0; i < ll.length; i++) expect(ss[i]).toBe(ll[i])
  })

  test('select(cond, then, else) input expression: byte-equal to legacy', () => {
    const s = makeSession()
    loadStdlib(s)
    const sinOsc  = s.typeRegistry.get('SinOsc')!
    const onePole = s.typeRegistry.get('OnePole')!
    s.instanceRegistry.set('osc', instantiate(sinOsc, 'osc'))
    s.instanceRegistry.set('lp',  instantiate(onePole, 'lp'))
    allocateOutputSlots(s, 'osc', sinOsc)
    allocateOutputSlots(s, 'lp',  onePole)
    s.inputExprNodes.set('osc:freq', 220)
    s.inputExprNodes.set('lp:input', {
      op: 'select',
      args: [
        { op: 'gt', args: [{ op: 'ref', instance: 'osc', output: 'sine' }, 0] },
        { op: 'ref', instance: 'osc', output: 'sine' },
        0,
      ],
    })
    s.inputExprNodes.set('lp:g', 0.1)
    s.graphOutputs.push({ instance: 'lp', output: 'out' })

    const ll = audio(compileSessionLegacy(s))
    const ss = audio(compileSessionSlotted(s))
    for (let i = 0; i < ll.length; i++) expect(ss[i]).toBe(ll[i])
  })

  test('clamp(x, lo, hi) input expression: byte-equal to legacy', () => {
    const s = makeSession()
    loadStdlib(s)
    const sinOsc  = s.typeRegistry.get('SinOsc')!
    const onePole = s.typeRegistry.get('OnePole')!
    s.instanceRegistry.set('osc', instantiate(sinOsc, 'osc'))
    s.instanceRegistry.set('lp',  instantiate(onePole, 'lp'))
    allocateOutputSlots(s, 'osc', sinOsc)
    allocateOutputSlots(s, 'lp',  onePole)
    s.inputExprNodes.set('osc:freq', 220)
    s.inputExprNodes.set('lp:input', {
      op: 'clamp',
      args: [{ op: 'ref', instance: 'osc', output: 'sine' }, -0.3, 0.3],
    })
    s.inputExprNodes.set('lp:g', 0.1)
    s.graphOutputs.push({ instance: 'lp', output: 'out' })

    const ll = audio(compileSessionLegacy(s))
    const ss = audio(compileSessionSlotted(s))
    for (let i = 0; i < ll.length; i++) expect(ss[i]).toBe(ll[i])
  })
})
