/**
 * session_to_parsed.test.ts — the session→root lowering.
 *
 * `compileSessionSlottedRoot` builds the synthetic root by serializing the
 * session to a `ParsedProgram` (`sessionToParsedProgram`) and running it
 * through the shared `elaborate` front door, with the instances' already-
 * resolved types supplied via the elaborator's `ExternalProgramResolver`
 * hook (LINK). These tests pin the root's structure directly, plus two
 * whole-path properties (plan-byte determinism, full example-patch corpus).
 *
 * The sample-for-sample audio equivalence against the per-instance oracle
 * lives in `tests/equiv/root_vs_flat.test.ts`.
 */

import { describe, test, expect } from 'bun:test'
import { readFileSync, readdirSync } from 'node:fs'
import { join } from 'node:path'
import {
  makeSession, resolveProgramType, instantiate, setWireExpr, loadJSON,
  type SessionState,
} from '../session.js'
import { loadStdlib } from '../program.js'
import { compileSession } from './compile_session.js'
import {
  wireKey, portRef, instanceName as toInstanceName, portName as toPortName,
} from './branded_names.js'
import { liftWiresToInstances } from './lift_wires.js'
import { extractSessionDelays } from './lowering/extract_session_delays.js'
import { elaborate } from './elaborator.js'
import { sessionToParsedProgram, sessionTypeResolver } from './session_to_parsed.js'
import { CycleViolation } from './elaboration_diagnostics.js'
import type { ResolvedProgram } from './nodes.js'

/** Build the synthetic root the production path builds (the body of
 *  `compile_session_slotted.ts:buildSessionRoot`). */
function buildRoot(session: SessionState): ResolvedProgram {
  return elaborate(sessionToParsedProgram(session), sessionTypeResolver(session))
}

function addInst(session: SessionState, name: string, typeName: string) {
  const { type } = resolveProgramType(session, typeName, undefined, undefined)
  session.instanceRegistry.set(name, instantiate(type, name, { baseTypeName: typeName }))
}

/** osc→amp→dac, with the two pre-passes that precede root construction in
 *  the real compile path (array-literal wire lift, then delay extraction). */
function makeOscAmpSession(): SessionState {
  const session = makeSession()
  loadStdlib(session)
  addInst(session, 'osc', 'BlepSaw')
  addInst(session, 'amp', 'VCA')
  setWireExpr(session, portRef(toInstanceName('amp'), toPortName('audio')),
    { op: 'ref', instance: 'osc', output: 'saw' })
  setWireExpr(session, portRef(toInstanceName('amp'), toPortName('cv')), 0.5)
  // dac stays on session.graphOutputs (NOT via setWireExpr, which would
  // auto-wrap it in a spurious unit delay).
  session.graphOutputs.push({ instance: 'amp', output: 'out' })
  liftWiresToInstances(session)
  extractSessionDelays(session)
  return session
}

describe('session → root ResolvedProgram (via elaborate)', () => {
  test('produces a valid root program; boxes stay closed (no inline)', () => {
    const session = makeOscAmpSession()
    const root = buildRoot(session)

    expect(root.name).toBe('__session__')
    expect(root.op).toBe('program')
    // Every top-level instance survives as an InstanceDecl (LINK, not FLATTEN).
    expect(root.instances.length).toBe(session.instanceRegistry.size)
    expect(root.instances.map(i => i.name).sort())
      .toEqual([...session.instanceRegistry.keys()].sort())
    // DAC stays on session.graphOutputs — the root carries no ports/assigns.
    expect(root.ports.outputs.length).toBe(0)
    expect(root.body.assigns.length).toBe(0)
    expect(root.ports.inputs.length).toBe(0)
  })

  test('scalar session delays become root RegDecls (naming-transparent)', () => {
    const session = makeOscAmpSession()
    const scalarDelays = session.delaySlotRegistry.filter(e => !e.isArray)
    expect(scalarDelays.length).toBeGreaterThan(0)

    const root = buildRoot(session)
    expect(root.regs.length).toBe(scalarDelays.length)
    // Naming transparency: each delay reg carries the registry's stable
    // slotName (the hot-swap state key), NOT a root-prefixed name — what
    // keeps state-transfer-by-name byte-identical to the flat path.
    expect(root.regs.map(r => r.name).sort())
      .toEqual(scalarDelays.map(e => e.slotName).sort())
    for (const r of root.regs) expect(r.update).toBeDefined()
  })

  test('wires land on InstanceDecl.inputs', () => {
    const root = buildRoot(makeOscAmpSession())
    const amp = root.instances.find(i => i.name === 'amp')!
    expect(amp.inputs.length).toBe(2)
  })

  test('array session delays become array-typed root RegDecls', () => {
    const session = makeOscAmpSession()
    // Inject an array delay entry directly (as `extractSessionDelays` would
    // for a `delay()` over an array-shaped source).
    session.delaySlotRegistry.push({
      slotName: '__arr_delay_test',
      sourceExpr: { op: 'ref', instance: 'osc', output: 'saw' },
      init: 0, scalarType: 'float', isArray: true, arraySlot: 0, arraySize: 8,
    })
    const root = buildRoot(session)
    const arrReg = root.regs.find(r => r.name === '__arr_delay_test')
    expect(arrReg).toBeDefined()
    expect(Array.isArray(arrReg!.init)).toBe(true)
    expect((arrReg!.init as unknown[]).length).toBe(8)
    expect(arrReg!.update).toBeDefined()
  })

  test('throws CycleViolation on an undelayed inter-instance cycle', () => {
    const session = makeSession()
    loadStdlib(session)
    addInst(session, 'a', 'VCA')
    addInst(session, 'b', 'VCA')
    // Combinational cycle a.audio←b.out, b.audio←a.out written directly
    // (bypassing setWireExpr's auto-delay), so no delay breaks the loop.
    // The elaborator's strict cycle policy rejects it — the same
    // `CycleViolation` the deleted materializer raised by hand.
    session.inputExprNodes.set(
      wireKey(portRef(toInstanceName('a'), toPortName('audio'))),
      { op: 'ref', instance: 'b', output: 'out' })
    session.inputExprNodes.set(
      wireKey(portRef(toInstanceName('b'), toPortName('audio'))),
      { op: 'ref', instance: 'a', output: 'out' })
    expect(() => buildRoot(session)).toThrow(CycleViolation)
  })
})

// ── Whole-path properties ────────────────────────────────────────────

describe('compileSession plan bytes are deterministic across re-elaboration', () => {
  // Re-elaboration is free relative to the JIT, which keys its kernel cache
  // on the serialized plan bytes (OrcJitEngine.cpp). Determinism is the
  // invariant that keeps that cache warm no matter how often we re-elaborate.
  const cases: Array<[string, () => SessionState]> = [
    ['oscAmp', () => makeOscAmpSession()],
    ['SinOsc', () => { const s = makeSession(); loadStdlib(s); addInst(s, 'o', 'SinOsc'); s.graphOutputs.push({ instance: 'o', output: 'sine' }); return s }],
  ]
  for (const [label, build] of cases) {
    test(label, () => {
      expect(JSON.stringify(compileSession(build())))
        .toBe(JSON.stringify(compileSession(build())))
    })
  }
})

describe('every example patch compiles through the root path + is deterministic', () => {
  const PATCHES_DIR = join(import.meta.dir, '..', '..', 'patches')
  const files = readdirSync(PATCHES_DIR).filter(f => f.endsWith('.json'))
  test('corpus is non-empty', () => { expect(files.length).toBeGreaterThan(0) })
  for (const file of files) {
    test(file, () => {
      const load = (): SessionState => {
        const s = makeSession()
        loadStdlib(s)
        loadJSON(JSON.parse(readFileSync(join(PATCHES_DIR, file), 'utf8')), s)
        return s
      }
      const a = JSON.stringify(compileSession(load()))
      const b = JSON.stringify(compileSession(load()))
      expect(a).toBe(b)               // deterministic
      expect(a.length).toBeGreaterThan(0)
    })
  }
})
