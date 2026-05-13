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

  test.skip('Sequencer<4> compiles end-to-end with an arrayPack input', () => {
    // Phase 5 of M11 wire-lift handles the WIRE SIDE: the array literal
    // [110, 220, ...] is lifted to a __wire_N program whose body is
    // `out = arrayPack(...)`. But Sequencer's BODY does
    // `index(InputRef(values), step)` — array-typed instance INPUTS
    // still aren't materialized as array_reg operands by the per-
    // instance compile path. emit_resolved throws "non-array operand"
    // on the body-side index. Body-side array-input materialization
    // is a separate follow-up to Phase 5.
  })
})
