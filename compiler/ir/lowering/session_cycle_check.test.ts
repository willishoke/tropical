/**
 * session_cycle_check.test.ts — exercise the defensive cycle
 * assertion that fires at `compileSession`'s entry.
 */

import { describe, test, expect } from 'bun:test'
import { makeSession, setWireExpr, type ExprNode } from '../../session.js'
import { extractSessionDelays } from './extract_session_delays.js'
import { assertSessionAcyclic, SessionCycleViolation } from './session_cycle_check.js'
import { portRef, instanceName, portName } from '../branded_names.js'

const pr = (i: string, p: string) => portRef(instanceName(i), portName(p))

function makeInstance(session: ReturnType<typeof makeSession>, name: string): void {
  // We don't need a real Compiled here — assertSessionAcyclic only
  // reads inputExprNodes + instanceRegistry keys. Cast to satisfy the
  // type. The check is structural; instance contents are irrelevant.
  session.instanceRegistry.set(name, {} as never)
}

describe('assertSessionAcyclic', () => {
  test('accepts a graph with no inter-instance edges', () => {
    const session = makeSession()
    makeInstance(session, 'a')
    makeInstance(session, 'b')
    expect(() => assertSessionAcyclic(session)).not.toThrow()
  })

  test('accepts an MCP-built cyclic session (auto-delays extracted)', () => {
    const session = makeSession()
    makeInstance(session, 'a')
    makeInstance(session, 'b')
    setWireExpr(session, pr('a', 'in'), { op: 'ref', instance: 'b', output: 'out' })
    setWireExpr(session, pr('b', 'in'), { op: 'ref', instance: 'a', output: 'out' })
    extractSessionDelays(session)
    expect(() => assertSessionAcyclic(session)).not.toThrow()
  })

  test('rejects a bypass-constructed cycle (2-member SCC)', () => {
    const session = makeSession()
    makeInstance(session, 'a')
    makeInstance(session, 'b')
    // Bypass setWireExpr — write directly, simulating a programmatic
    // session that doesn't honor the auto-delay convention.
    session.inputExprNodes.set('a:in', { op: 'ref', instance: 'b', output: 'out' })
    session.inputExprNodes.set('b:in', { op: 'ref', instance: 'a', output: 'out' })
    expect(() => assertSessionAcyclic(session)).toThrow(SessionCycleViolation)
  })

  test('rejects a self-cycle (single instance with self-edge)', () => {
    const session = makeSession()
    makeInstance(session, 'a')
    session.inputExprNodes.set('a:in', { op: 'ref', instance: 'a', output: 'out' })
    expect(() => assertSessionAcyclic(session)).toThrow(SessionCycleViolation)
  })

  test('cycle through nested arithmetic still detected', () => {
    const session = makeSession()
    makeInstance(session, 'a')
    makeInstance(session, 'b')
    const expr: ExprNode = {
      op: 'add',
      args: [
        { op: 'ref', instance: 'b', output: 'out' },
        1,
      ],
    }
    session.inputExprNodes.set('a:in', expr)
    session.inputExprNodes.set('b:in', { op: 'ref', instance: 'a', output: 'out' })
    expect(() => assertSessionAcyclic(session)).toThrow(SessionCycleViolation)
  })

  test('error names the cycle members', () => {
    const session = makeSession()
    makeInstance(session, 'a')
    makeInstance(session, 'b')
    session.inputExprNodes.set('a:in', { op: 'ref', instance: 'b', output: 'out' })
    session.inputExprNodes.set('b:in', { op: 'ref', instance: 'a', output: 'out' })
    try {
      assertSessionAcyclic(session)
      throw new Error('expected SessionCycleViolation')
    } catch (e) {
      expect(e).toBeInstanceOf(SessionCycleViolation)
      const msg = (e as Error).message
      expect(msg).toContain('a')
      expect(msg).toContain('b')
    }
  })
})
