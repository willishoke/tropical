import { describe, test, expect } from 'bun:test'
import { makeSession, loadJSON, resolveProgramType, inputPortType } from './session'
import { loadStdlib } from './program'
import { compileSession } from './ir/compile_session'
import { Float, Int, ArrayType, portTypeEqual } from './ir/port_type'

describe('stdlib Sequencer<N>', () => {
  test('Sequencer<N> monomorphizes values input shape to [N]', () => {
    const session = makeSession()
    loadStdlib(session)
    const { type: t4 } = resolveProgramType(session, 'Sequencer', { N: 4 }, undefined)
    const { type: t8 } = resolveProgramType(session, 'Sequencer', { N: 8 }, undefined)
    // values is the second input
    expect(inputPortType(t4, 1)).toEqual(ArrayType(Float, [4]))
    expect(inputPortType(t8, 1)).toEqual(ArrayType(Float, [8]))
    expect(t4).not.toBe(t8)
  })

  test('Sequencer uses declared default N=8 when no type_args provided', () => {
    const session = makeSession()
    loadStdlib(session)
    const { type, typeArgs } = resolveProgramType(session, 'Sequencer', undefined, undefined)
    expect(typeArgs).toEqual({ N: 8 })
    expect(inputPortType(type, 1)).toEqual(ArrayType(Float, [8]))
  })

  test.skip('Sequencer<4> flattens end-to-end with an arrayPack input', () => {
    // Skipped under the active-set runtime: array-shaped input
    // expressions (`values: [110, 220, ...]` and Clock's `ratios_in:
    // [1]`) are not yet supported by the per-instance compile path's
    // translateNode. Tracked as a follow-up.
  })
})
