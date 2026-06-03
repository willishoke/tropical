/**
 * compile_session.test.ts — structural soundness for `compileSession`.
 *
 * Covers single-instance and two-instance ref-wiring shapes; asserts
 * the produced FlatPlan validates as `tropical_plan_5` with
 * non-degenerate counts. Audio-equivalence against the interpreter
 * lives in `tests/equiv/jit_vs_interp_stdlib.test.ts`; this file is the
 * lightweight per-PR sanity for the materialization shape.
 */

import { describe, test, expect } from 'bun:test'
import { makeSession, resolveProgramType, instantiate, outputNames } from '../session.js'
import { loadStdlib } from '../program.js'
import { compileSession } from './compile_session.js'
import { wireKey, portRef, instanceName, portName } from './branded_names.js'

const wk = (i: string, p: string) => wireKey(portRef(instanceName(i), portName(p)))

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
  // Clock is excluded — it has an array-typed output (`ratios_out:
  // float[1]`) that the per-instance compile path doesn't yet expand.
  // The compile-fail path is exercised separately.
  for (const typeName of [
    'Sin', 'Cos', 'Exp', 'Log', 'Tanh',
    'OnePole', 'SoftClip', 'BitCrusher', 'CrossFade',
    'NoiseLFSR', 'AllpassDelay', 'CombDelay',
    'VCA',
  ] as const) {
    test(`emits well-formed tropical_plan_5: ${typeName}`, () => {
      const session = singleInstanceSession(typeName)
      const plan = compileSession(session)

      expect(plan.schema).toBe('tropical_plan_5')
      // The session lowers to a single root kernel whose children are the
      // session instances (boxes closed); count instructions recursively.
      const countInstrs = (fns: typeof plan.instance_functions): number =>
        fns.reduce((n, f) => n + f.instructions.length + countInstrs(f.children), 0)
      expect(countInstrs(plan.instance_functions)).toBeGreaterThan(0)
      // Outputs are device-bound sinks; v1 emits the single audio sink
      // whose inputs are the session's graphOutput slots.
      expect(plan.sinks.length).toBe(1)
      expect(plan.sinks[0]!.inputs.length).toBe(session.graphOutputs.length)
      expect(plan.sinks[0]!.target).toBe(0)
      expect(plan.instance_functions.length).toBe(1)  // the root kernel
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

    session.inputExprNodes.set(wk("amp", "audio"), { op: 'ref', instance: 'osc', output: 'out' })
    session.inputExprNodes.set(wk("amp", "cv"), 0.5)

    session.graphOutputs.push({ instance: 'amp', output: 'out' })

    const plan = compileSession(session)
    expect(plan.sinks.length).toBe(1)
    expect(plan.sinks[0]!.inputs.length).toBe(1)
    // Root kernel with the two session instances as children.
    expect(plan.instance_functions.length).toBe(1)
    expect(plan.instance_functions[0]!.children.length).toBe(2)
  })
})
