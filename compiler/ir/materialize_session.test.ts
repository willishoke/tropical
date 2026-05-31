import { describe, test, expect } from 'bun:test'
import { makeSession, resolveProgramType, instantiate, setWireExpr } from '../session.js'
import { loadStdlib } from '../program.js'
import { wireKey, portRef, instanceName as toInstanceName, portName as toPortName } from './branded_names.js'
import { liftWiresToInstances } from './lift_wires.js'
import { extractSessionDelays } from './lowering/extract_session_delays.js'
import { materializeSessionToResolvedIR } from './materialize_session.js'
import { CycleViolation } from './elaboration_diagnostics.js'

/** Instantiate a stdlib type into the session's instanceRegistry. */
function addInst(session: ReturnType<typeof makeSession>, name: string, typeName: string) {
  const { type } = resolveProgramType(session, typeName, undefined, undefined)
  const inst = instantiate(type, name, { baseTypeName: typeName })
  session.instanceRegistry.set(name, inst)
}

/** Build a minimal osc→amp→dac session and run the two pre-passes that
 *  precede materialization in the real compile path (array-literal wire
 *  lift, then session-delay extraction). Returns the prepared session. */
function makeOscAmpSession() {
  const session = makeSession()
  loadStdlib(session)
  addInst(session, 'osc', 'BlepSaw')
  addInst(session, 'amp', 'VCA')
  setWireExpr(
    session,
    portRef(toInstanceName('amp'), toPortName('audio')),
    { op: 'ref', instance: 'osc', output: 'saw' },
  )
  setWireExpr(session, portRef(toInstanceName('amp'), toPortName('cv')), 0.5)
  // dac — special-cased into session.graphOutputs (NOT via setWireExpr,
  // which would auto-wrap it in a spurious unit delay).
  session.graphOutputs.push({ instance: 'amp', output: 'out' })
  liftWiresToInstances(session)
  extractSessionDelays(session)
  return session
}

describe('materializeSessionToResolvedIR', () => {
  test('produces a valid root program (no strata, no flatten)', () => {
    const session = makeOscAmpSession()
    const root = materializeSessionToResolvedIR(session)

    // mkProgram validated the programRegistry covers every instance
    // typeKey (it throws otherwise) — reaching here proves that.
    expect(root.name).toBe('__session__')
    expect(root.op).toBe('program')

    // Boxes stay closed: every top-level instance survives as an
    // InstanceDecl (NOT inlined away). One per session instance.
    expect(root.instances.length).toBe(session.instanceRegistry.size)
    const instNames = root.instances.map(i => i.name).sort()
    expect(instNames).toEqual([...session.instanceRegistry.keys()].sort())

    // The root carries NO output ports — DAC stays on session.graphOutputs.
    expect(root.ports.outputs.length).toBe(0)
    expect(root.body.assigns.length).toBe(0)
    expect(root.ports.inputs.length).toBe(0)
  })

  test('scalar session delays become root RegDecls (naming-transparent)', () => {
    const session = makeOscAmpSession()
    const scalarDelays = session.delaySlotRegistry.filter(e => !e.isArray)
    // Two non-dac wires (amp.audio, amp.cv) each auto-delayed by setWireExpr.
    expect(scalarDelays.length).toBeGreaterThan(0)

    const root = materializeSessionToResolvedIR(session)

    // One root RegDecl per scalar delay entry.
    expect(root.regs.length).toBe(scalarDelays.length)
    // Naming transparency: each delay reg carries the registry's stable
    // slotName (the hot-swap state key), NOT a root-prefixed name. This
    // is what keeps state-transfer-by-name byte-identical to the flat path.
    const regNames = root.regs.map(r => r.name).sort()
    const slotNames = scalarDelays.map(e => e.slotName).sort()
    expect(regNames).toEqual(slotNames)
    // Every delay reg has an update populated (it's a unit delay).
    for (const r of root.regs) expect(r.update).toBeDefined()
  })

  test('wires land on InstanceDecl.inputs', () => {
    const session = makeOscAmpSession()
    const root = materializeSessionToResolvedIR(session)
    const amp = root.instances.find(i => i.name === 'amp')!
    // amp has two wired inputs (audio, cv).
    expect(amp.inputs.length).toBe(2)
  })

  test('array session delays become array RegDecls', () => {
    const session = makeOscAmpSession()
    // Inject an array delay entry directly (as `extractSessionDelays`
    // would for a `delay()` over an array-shaped source). It should
    // materialize to an array-typed root RegDecl whose init is a
    // size-long literal array — the backing store `compileResolved`
    // turns into an array slot.
    session.delaySlotRegistry.push({
      slotName: '__arr_delay_test',
      sourceExpr: { op: 'ref', instance: 'osc', output: 'saw' },
      init: 0,
      scalarType: 'float',
      isArray: true,
      arraySlot: 0,
      arraySize: 8,
    })
    const root = materializeSessionToResolvedIR(session)
    const arrReg = root.regs.find(r => r.name === '__arr_delay_test')
    expect(arrReg).toBeDefined()
    // Array-typed init: a length-8 literal array (the scalar delays
    // carry plain-number inits).
    expect(Array.isArray(arrReg!.init)).toBe(true)
    expect((arrReg!.init as unknown[]).length).toBe(8)
    expect(arrReg!.update).toBeDefined()
  })

  test('throws CycleViolation on an undelayed inter-instance cycle', () => {
    const session = makeSession()
    loadStdlib(session)
    addInst(session, 'a', 'VCA')
    addInst(session, 'b', 'VCA')
    // Hand-build a combinational cycle a.audio←b.out, b.audio←a.out by
    // writing inputExprNodes directly (bypassing setWireExpr's auto-delay),
    // so no delay breaks the loop.
    session.inputExprNodes.set(
      wireKey(portRef(toInstanceName('a'), toPortName('audio'))),
      { op: 'ref', instance: 'b', output: 'out' },
    )
    session.inputExprNodes.set(
      wireKey(portRef(toInstanceName('b'), toPortName('audio'))),
      { op: 'ref', instance: 'a', output: 'out' },
    )
    expect(() => materializeSessionToResolvedIR(session)).toThrow(CycleViolation)
  })
})
