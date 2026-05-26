import { describe, test, expect } from 'bun:test'
import { makeSession, loadJSON, resolveProgramType, inputPortType, instantiate, allocateOutputSlots } from './session'
import { loadStdlib } from './program'
import { compileSession } from './ir/compile_session'
import { Float, Int, ArrayType, portTypeEqual } from './ir/port_type'
import { wireKey, portRef, instanceName as toInstanceName, portName as toPortName } from './ir/branded_names'

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

  test('Sequencer<4> compiles end-to-end with an arrayPack input', () => {
    const session = makeSession(256, { inlineNested: false })
    loadStdlib(session)
    const { type } = resolveProgramType(session, 'Sequencer', { N: 4 }, undefined)
    const inst = instantiate(type, 'seq', { baseTypeName: 'Sequencer', typeArgs: { N: 4 } })
    session.instanceRegistry.set(inst.name, inst)
    allocateOutputSlots(session, toInstanceName(inst.name), type)

    // Wire `seq.values = [110, 220, 440, 880]`. liftWiresToInstances
    // converts the array literal to a __wire_N session instance with
    // an array output; allocateInputSlots then aliases seq.values'
    // input array slot to __wire_N's output array slot (no fresh
    // session slot, no per-sample copy — the consumer reads the
    // producer's slot directly).
    session.inputExprNodes.set(
      wireKey(portRef(toInstanceName('seq'), toPortName('values'))),
      [110, 220, 440, 880],
    )
    session.inputExprNodes.set(
      wireKey(portRef(toInstanceName('seq'), toPortName('clock'))),
      0.0,
    )
    session.graphOutputs.push({ instance: 'seq', output: 'value' })

    const plan = compileSession(session)
    expect(plan.schema).toBe('tropical_plan_5')
    // The lifted __wire_N program contributes 1 array slot (size 4)
    // for the values literal; alias semantics mean seq.values shares
    // that same slot, so total array_slot_count = 1.
    expect(plan.array_slot_count).toBeGreaterThanOrEqual(1)
    expect(plan.array_slot_sizes[0]).toBe(4)
    session.graph.dispose()
  })
})
