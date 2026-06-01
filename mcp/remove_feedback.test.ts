/**
 * remove_feedback.test.ts — removing an instance that participates in a
 * feedback ring must not throw, and must not leave dangling wiring.
 *
 * Regression: every MCP wire is unit-delay-wrapped, and after a compile
 * `extractSessionDelays` rewrites the ring's wires into `sessionSlot`
 * reads (the delayed source moves into `delaySlotRegistry`).
 * `remove_instance` then (a) crashed because `exprDependencies` →
 * `mapChildren` threw on the unhandled `sessionSlot` op, and (b) even
 * past that, could not see that a `sessionSlot` wire still referenced
 * the removed instance, so it left a wire whose registry source pointed
 * at a now-missing instance — the post-removal recompile then threw.
 *
 * Requires libtropical.dylib (the engine's session owns a runtime).
 */

import { describe, test, expect } from 'bun:test'
import { handleTool, session } from './engine.js'
import type { ExprNode } from '../compiler/expr.js'

function call(name: string, args: Record<string, unknown>) {
  const res = handleTool(name, args) as { content: { text: string }[]; isError?: boolean }
  return JSON.parse(res.content[0].text) as { status: 'ok' | 'error'; data?: unknown; error?: unknown }
}

const ref = (instance: string): ExprNode => ({ op: 'ref', instance, output: 'sine' })
const fm  = (base: number, from: string): ExprNode => ({ op: 'add', args: [base, ref(from)] })

describe('remove_instance on a feedback ring', () => {
  test('does not throw and clears wiring that fed off the removed instance', () => {
    // 3-LFO feedback ring: lfo1 ← lfo3 ← lfo2 ← lfo1. Unique names so
    // the engine's shared session stays clean across the suite.
    for (const n of ['rfb1', 'rfb2', 'rfb3']) {
      expect(call('add_instance', { program: 'SinOsc', instance_name: n }).status).toBe('ok')
    }
    // Close the ring (MCP auto-delay breaks the cycle) and tap one to dac.
    const wired = call('wire', { set: [
      { instance: 'rfb1', input: 'freq', expr: fm(0.3, 'rfb3') },
      { instance: 'rfb2', input: 'freq', expr: fm(0.5, 'rfb1') },
      { instance: 'rfb3', input: 'freq', expr: fm(0.7, 'rfb2') },
      { instance: 'dac',  input: 'out',  expr: ref('rfb1') },
    ]})
    expect(wired.status).toBe('ok') // compiled → ring wires became sessionSlot

    // Remove an instance in the ring. Must succeed (no throw), and the
    // recompile inside it must succeed too.
    const removed = call('remove_instance', { instance_name: 'rfb3' })
    expect(removed.status).toBe('ok')

    // No surviving wire may reference the removed instance — including
    // through a hoisted sessionSlot (rfb1.freq fed off rfb3).
    const wiring = call('list_wiring', {})
    expect(wiring.status).toBe('ok')
    const pretty = JSON.stringify(wiring.data)
    expect(pretty).not.toContain('rfb3')

    // And the removed instance's own wires are gone.
    expect(session.instanceRegistry.has('rfb3')).toBe(false)
    expect([...session.inputExprNodes.keys()].some(k => k.startsWith('rfb3:'))).toBe(false)

    // Inspection tools work on the survivors (would have thrown on the
    // sessionSlot op before the fix).
    expect(call('get_info', { instance_name: 'rfb2' }).status).toBe('ok')

    // Clean up so other tests in this shared-session suite aren't affected.
    call('remove_instance', { instance_name: 'rfb1' })
    call('remove_instance', { instance_name: 'rfb2' })
  })
})
