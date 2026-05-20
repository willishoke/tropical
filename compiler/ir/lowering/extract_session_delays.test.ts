/**
 * extract_session_delays.test.ts — unit-test the pre-emit pass that
 * hoists unit-delay wires into session-level module slots.
 */

import { describe, test, expect } from 'bun:test'
import { makeSession, setWireExpr } from '../../session.js'
import { extractSessionDelays } from './extract_session_delays.js'
import { portRef, instanceName, portName, wireKey } from '../branded_names.js'

const wk = (i: string, p: string) => wireKey(portRef(instanceName(i), portName(p)))

/** Local sugar: build a PortRef from raw strings — keeps test
 *  expectations readable without bypassing the branded constructors. */
const pr = (i: string, p: string) => portRef(instanceName(i), portName(p))

describe('extractSessionDelays', () => {
  test('hoists a top-level delay wire to a sessionSlot read', () => {
    const session = makeSession()
    const baseSlotCount = session.slotCount
    setWireExpr(session, pr('a', 'in'), {
      op: 'ref', instance: 'b', output: 'out',
    })

    extractSessionDelays(session)

    expect(session.delaySlotRegistry).toHaveLength(1)
    const entry = session.delaySlotRegistry[0]
    expect(entry.slotIdx).toBe(baseSlotCount)
    expect(entry.slotName).toBe('__autodelay:a:in')
    expect(entry.init).toBe(0)
    expect(entry.scalarType).toBe('float')
    expect(entry.sourceExpr).toEqual({
      op: 'ref', instance: 'b', output: 'out',
    })

    // Wire is rewritten to a sessionSlot read pointing at the new slot.
    expect(session.inputExprNodes.get(wk("a", "in"))).toEqual({
      op: 'sessionSlot', index: baseSlotCount,
    })
    expect(session.slotCount).toBe(baseSlotCount + 1)
  })

  test('honors explicit init and id on the delay wrap', () => {
    const session = makeSession()
    setWireExpr(session, pr('osc', 'phase'),
      { op: 'ref', instance: 'src', output: 'y' },
      { init: 0.5, id: 'feedback-tap' },
    )

    extractSessionDelays(session)

    expect(session.delaySlotRegistry).toHaveLength(1)
    const entry = session.delaySlotRegistry[0]
    expect(entry.init).toBe(0.5)
    expect(entry.slotName).toBe('feedback-tap')
  })

  test('leaves non-delay wires alone (legacy patches loaded via loadJSON)', () => {
    const session = makeSession()
    // Direct write, bypassing setWireExpr — simulates a legacy
    // patch ingest path.
    session.inputExprNodes.set(wk("a", "gain"), { op: 'param', name: 'gain' })

    const before = session.inputExprNodes.get(wk("a", "gain"))
    extractSessionDelays(session)
    const after = session.inputExprNodes.get(wk("a", "gain"))

    expect(after).toEqual(before!)
    expect(session.delaySlotRegistry).toHaveLength(0)
  })

  test('idempotent: re-running finds no delays to extract', () => {
    const session = makeSession()
    setWireExpr(session, pr('a', 'in'), {
      op: 'ref', instance: 'b', output: 'out',
    })
    extractSessionDelays(session)

    const registryAfterFirst = [...session.delaySlotRegistry]
    const slotCountAfterFirst = session.slotCount

    extractSessionDelays(session)

    expect(session.delaySlotRegistry).toEqual(registryAfterFirst)
    expect(session.slotCount).toBe(slotCountAfterFirst)
  })

  test('handles multiple delays in allocation order', () => {
    const session = makeSession()
    const baseSlot = session.slotCount
    setWireExpr(session, pr('a', 'in'), { op: 'ref', instance: 'b', output: 'out' })
    setWireExpr(session, pr('b', 'in'), { op: 'ref', instance: 'a', output: 'out' })

    extractSessionDelays(session)

    expect(session.delaySlotRegistry).toHaveLength(2)
    expect(session.delaySlotRegistry[0].slotIdx).toBe(baseSlot)
    expect(session.delaySlotRegistry[1].slotIdx).toBe(baseSlot + 1)
    expect(session.slotCount).toBe(baseSlot + 2)
  })
})
