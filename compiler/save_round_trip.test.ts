/**
 * save_round_trip.test.ts — `saveProgramFromSession` round-trips a
 * compiled feedback patch.
 *
 * Regression: every MCP wire is unit-delay-wrapped, and a compile
 * rewrites the ring's wires into `sessionSlot` reads (the delayed source
 * moves into `delaySlotRegistry`). `save` serialized those raw
 * `sessionSlot` indices, so the emitted patch was unloadable — the index
 * means nothing in a fresh session. `save` now reverses the hoisting
 * (`reconstructWireDelays`) and emits the authored `delay(<source>)`
 * form, which reloads and re-hoists cleanly.
 */

import { describe, test, expect } from 'bun:test'
import { makeSession, resolveProgramType, instantiate, setWireExpr, loadJSON, v2NodeToFile } from './session.js'
import { loadStdlib, saveProgramFromSession } from './program.js'
import { compileSession } from './ir/compile_session.js'
import { portRef, instanceName, portName } from './ir/branded_names.js'

function feedbackRing() {
  const s = makeSession(64)
  loadStdlib(s)
  for (const n of ['lfo1', 'lfo2', 'lfo3']) {
    const { type } = resolveProgramType(s, 'SinOsc', undefined, undefined)
    s.instanceRegistry.set(n, instantiate(type, n, { baseTypeName: 'SinOsc' }))
  }
  const fm = (base: number, from: string) =>
    ({ op: 'add', args: [base, { op: 'ref', instance: from, output: 'sine' }] } as const)
  setWireExpr(s, portRef(instanceName('lfo1'), portName('freq')), fm(0.3, 'lfo3'))
  setWireExpr(s, portRef(instanceName('lfo2'), portName('freq')), fm(0.5, 'lfo1'))
  setWireExpr(s, portRef(instanceName('lfo3'), portName('freq')), fm(0.7, 'lfo2'))
  s.graphOutputs.push({ instance: 'lfo1', output: 'sine' })
  return s
}

describe('save round-trips a compiled feedback patch', () => {
  test('emits delay() wires, not sessionSlot indices, and reloads', () => {
    const s = feedbackRing()
    compileSession(s) // hoists the ring's delays → sessionSlot in inputExprNodes
    expect([...s.inputExprNodes.values()].some(v => (v as { op?: string }).op === 'sessionSlot')).toBe(true)

    const { node, topLevel } = saveProgramFromSession(s)
    const file = v2NodeToFile(node as never, topLevel)
    const json = JSON.stringify(file)
    // No raw hoisted-slot ops leak into the saved patch.
    expect(json).not.toContain('sessionSlot')
    expect(json).toContain('"delay"')

    // Reload into a fresh session and recompile — the reconstructed
    // delays must break the ring's cycle again (no CycleViolation).
    const s2 = makeSession(64)
    loadStdlib(s2)
    loadJSON(file as { schema: string; [k: string]: unknown }, s2)
    expect([...s2.instanceRegistry.keys()].sort()).toEqual(['lfo1', 'lfo2', 'lfo3'])
    expect(() => compileSession(s2)).not.toThrow()
  })
})
