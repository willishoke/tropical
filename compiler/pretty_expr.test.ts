/**
 * pretty_expr.test.ts — `prettyExpr` renders the hoisted unit-delay
 * slots that `extractSessionDelays` writes into wires during compile.
 *
 * Regression: inspecting a session AFTER a compile (e.g. `list_wiring`
 * / `get_info` over MCP on a patch with a feedback ring) threw
 * `prettyExpr: unhandled op 'sessionSlot'`, because every MCP wire is
 * unit-delay-wrapped and the compile rewrites those `delay()` nodes to
 * `sessionSlot` / `sessionArraySlot` reads. `prettyExpr` now resolves
 * the slot back to its registered source when given the registry.
 */

import { describe, test, expect } from 'bun:test'
import { makeSession, resolveProgramType, instantiate, setWireExpr, prettyExpr } from './session.js'
import { loadStdlib } from './program.js'
import { compileSession } from './ir/compile_session.js'
import { portRef, instanceName, portName, wireKey } from './ir/branded_names.js'

const wk = (i: string, p: string) => wireKey(portRef(instanceName(i), portName(p)))

describe('prettyExpr on post-compile sessionSlot wires', () => {
  function feedbackRingSession() {
    const s = makeSession(64)
    loadStdlib(s)
    for (const n of ['lfo1', 'lfo2']) {
      const { type } = resolveProgramType(s, 'SinOsc', undefined, undefined)
      s.instanceRegistry.set(n, instantiate(type, n, { baseTypeName: 'SinOsc' }))
    }
    // Two-LFO feedback ring; MCP wires are unit-delay-wrapped, so the
    // cycle is broken automatically and the compile hoists the delays
    // into sessionSlot reads.
    setWireExpr(s, portRef(instanceName('lfo1'), portName('freq')),
      { op: 'add', args: [0.3, { op: 'ref', instance: 'lfo2', output: 'sine' }] })
    setWireExpr(s, portRef(instanceName('lfo2'), portName('freq')),
      { op: 'ref', instance: 'lfo1', output: 'sine' })
    s.graphOutputs.push({ instance: 'lfo1', output: 'sine' })
    compileSession(s) // mutates inputExprNodes: delay() → sessionSlot
    return s
  }

  test('resolves the slot source to delay(<source>) with the registry', () => {
    const s = feedbackRingSession()
    const rendered = [...s.inputExprNodes.values()]
      .map(v => prettyExpr(v, s.instanceRegistry, s.delaySlotRegistry))
      .sort()
    // The feedback wires render as delays of their resolved sources,
    // not opaque slot indices — and nothing throws.
    expect(rendered).toEqual([
      'delay((0.3 + lfo2.sine), 0)',
      'delay(lfo1.sine, 0)',
    ])
  })

  test('does not throw, and degrades to slot index, without the registry', () => {
    const s = feedbackRingSession()
    for (const v of s.inputExprNodes.values()) {
      // No registry: must not throw; renders a slot placeholder.
      const out = prettyExpr(v, s.instanceRegistry)
      expect(out).toMatch(/^delay\(slot:\d+\)$/)
    }
  })

  test('the wire keys are intact', () => {
    const s = feedbackRingSession()
    expect(new Set(s.inputExprNodes.keys())).toEqual(new Set([wk('lfo1', 'freq'), wk('lfo2', 'freq')]))
  })
})
