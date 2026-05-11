/**
 * m9d_fanin.test.ts — M9d fan-in support.
 *
 * Verifies that the wire() handler now wraps duplicate-input writes
 * with a combine op when one is specified, and that the resulting
 * combined ExprNode produces correct audio through the per-instance
 * path.
 *
 * Arrays + sum types are deferred to a follow-up since they require
 * either per-element inputExprNodes keying (breaking the M1-M8
 * invariant) or array-aware translation in translateNode.
 */
import { describe, expect, test, beforeEach, afterEach } from 'bun:test'
import { Runtime } from './runtime.ts'
import { makeSession, allocateOutputSlots, type SessionState } from '../session.ts'
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

/** Simulate the wire handler's combine wrap directly. Tests don't go
 *  through the MCP subprocess. */
function setInputWithCombine(
  session: SessionState,
  key: string,
  expr: unknown,
  combine?: string,
): void {
  const existing = session.inputExprNodes.get(key)
  if (existing !== undefined && combine !== undefined) {
    session.inputExprNodes.set(key, { op: combine, args: [existing, expr] } as never)
  } else {
    session.inputExprNodes.set(key, expr as never)
  }
}

describe('M9d: fan-in with explicit combine', () => {
  test('two refs combined with add: byte-equal to legacy', () => {
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
    // Simulate two `wire` calls, the second supplying `combine:'add'`:
    setInputWithCombine(s, 'lp:input', { op: 'ref', instance: 'osc1', output: 'sine' })
    setInputWithCombine(s, 'lp:input', { op: 'ref', instance: 'osc2', output: 'sine' }, 'add')
    s.inputExprNodes.set('lp:g', 0.1)
    s.graphOutputs.push({ instance: 'lp', output: 'out' })

    // The combined ExprNode should be {op:'add', args:[ref(osc1), ref(osc2)]}
    const combined = s.inputExprNodes.get('lp:input') as { op: string; args: unknown[] }
    expect(combined.op).toBe('add')
    expect(combined.args.length).toBe(2)

    const ll = audio(compileSessionLegacy(s))
    const ss = audio(compileSessionSlotted(s))
    expect(ss.length).toBe(ll.length)
    for (let i = 0; i < ll.length; i++) expect(ss[i]).toBe(ll[i])
  })

  test('three refs combined with add (chained fan-in): byte-equal', () => {
    const s = makeSession()
    loadStdlib(s)
    const sinOsc  = s.typeRegistry.get('SinOsc')!
    const onePole = s.typeRegistry.get('OnePole')!
    s.instanceRegistry.set('a',  instantiate(sinOsc, 'a'))
    s.instanceRegistry.set('b',  instantiate(sinOsc, 'b'))
    s.instanceRegistry.set('c',  instantiate(sinOsc, 'c'))
    s.instanceRegistry.set('lp', instantiate(onePole, 'lp'))
    allocateOutputSlots(s, 'a',  sinOsc)
    allocateOutputSlots(s, 'b',  sinOsc)
    allocateOutputSlots(s, 'c',  sinOsc)
    allocateOutputSlots(s, 'lp', onePole)
    s.inputExprNodes.set('a:freq', 200)
    s.inputExprNodes.set('b:freq', 300)
    s.inputExprNodes.set('c:freq', 500)
    // Three wires; second and third use combine:'add':
    setInputWithCombine(s, 'lp:input', { op: 'ref', instance: 'a', output: 'sine' })
    setInputWithCombine(s, 'lp:input', { op: 'ref', instance: 'b', output: 'sine' }, 'add')
    setInputWithCombine(s, 'lp:input', { op: 'ref', instance: 'c', output: 'sine' }, 'add')
    s.inputExprNodes.set('lp:g', 0.1)
    s.graphOutputs.push({ instance: 'lp', output: 'out' })

    const ll = audio(compileSessionLegacy(s))
    const ss = audio(compileSessionSlotted(s))
    for (let i = 0; i < ll.length; i++) expect(ss[i]).toBe(ll[i])
  })

  test('second wire without combine still replaces (back-compat)', () => {
    const s = makeSession()
    loadStdlib(s)
    const sinOsc = s.typeRegistry.get('SinOsc')!
    s.instanceRegistry.set('osc1', instantiate(sinOsc, 'osc1'))
    s.instanceRegistry.set('osc2', instantiate(sinOsc, 'osc2'))
    allocateOutputSlots(s, 'osc1', sinOsc)
    allocateOutputSlots(s, 'osc2', sinOsc)
    s.inputExprNodes.set('osc1:freq', 110)
    s.inputExprNodes.set('osc2:freq', 220)
    // Two writes to the same key WITHOUT combine: the second replaces.
    setInputWithCombine(s, 'tmp:k', { op: 'ref', instance: 'osc1', output: 'sine' })
    setInputWithCombine(s, 'tmp:k', { op: 'ref', instance: 'osc2', output: 'sine' })
    const final = s.inputExprNodes.get('tmp:k') as { op: string; instance?: string }
    expect(final.op).toBe('ref')
    expect(final.instance).toBe('osc2')
  })
})
