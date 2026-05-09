/**
 * compile_session.test.ts — structural soundness for `compileSession`.
 *
 * Covers single-instance and two-instance ref-wiring shapes; asserts the
 * produced FlatPlan validates as `tropical_plan_4` with non-degenerate
 * counts. The audio-equivalence vs. legacy `flattenSession` gate
 * (`compile_session_equiv.test.ts`) carried the cutover; this file is
 * the lightweight per-PR sanity for the materialization shape.
 */

import { describe, test, expect } from 'bun:test'
import { makeSession, resolveProgramType, instantiate, outputNames } from '../session.js'
import { loadStdlib } from '../program.js'
import { compileSession } from './compile_session.js'

function singleInstanceSession(typeName: string) {
  const session = makeSession()
  loadStdlib(session)
  const { type } = resolveProgramType(session, typeName, undefined, undefined)
  const inst = instantiate(type, 'inst', { baseTypeName: typeName, typeArgs: new Map() })
  session.instanceRegistry.set('inst', inst)
  for (const outName of outputNames(inst)) {
    session.graphOutputs.push({ instance: 'inst', output: outName })
  }
  return session
}

describe('compileSession — single-instance sessions', () => {
  for (const typeName of [
    'Sin', 'Cos', 'Exp', 'Log', 'Tanh',
    'OnePole', 'SoftClip', 'BitCrusher', 'CrossFade',
    'NoiseLFSR', 'AllpassDelay', 'CombDelay',
    'VCA', 'Clock',
  ] as const) {
    test(`emits well-formed tropical_plan_4: ${typeName}`, () => {
      const session = singleInstanceSession(typeName)
      const plan = compileSession(session)

      expect(plan.schema).toBe('tropical_plan_4')
      expect(plan.outputs.length).toBeGreaterThan(0)
      expect(plan.outputs.length).toBe(session.graphOutputs.length)
      expect(plan.instructions.length).toBeGreaterThan(0)
      expect(plan.output_targets.length).toBe(plan.outputs.length)
      expect(plan.register_targets.length).toBe(plan.register_names.length)
      expect(plan.array_slot_sizes.length).toBe(plan.array_slot_count)
    })
  }
})

describe('compileSession — two-instance refs', () => {
  test('VCA driven by Sin output via ref wiring', () => {
    const session = makeSession()
    loadStdlib(session)
    const sin = resolveProgramType(session, 'Sin', undefined, undefined).type
    const vca = resolveProgramType(session, 'VCA', undefined, undefined).type

    const sinInst = instantiate(sin, 'osc', { baseTypeName: 'Sin' })
    const vcaInst = instantiate(vca, 'amp', { baseTypeName: 'VCA' })
    session.instanceRegistry.set('osc', sinInst)
    session.instanceRegistry.set('amp', vcaInst)

    // Wire osc.out → amp.audio (ref node).
    session.inputExprNodes.set('amp:audio', { op: 'ref', instance: 'osc', output: 'out' })
    session.inputExprNodes.set('amp:cv', 0.5)

    session.graphOutputs.push({ instance: 'amp', output: 'out' })

    const plan = compileSession(session)
    expect(plan.outputs.length).toBe(1)
    expect(plan.instructions.length).toBeGreaterThan(0)
  })
})
