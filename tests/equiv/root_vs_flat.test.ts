/**
 * root_vs_flat.test.ts — Option A equivalence gate.
 *
 * The session compiler has two lowerings that must agree sample-for-
 * sample on the audio output:
 *
 *   • flat (default) — `compileSessionSlottedPerInstance`: each
 *     top-level instance compiles to its own `InstanceFunction`; a
 *     scheduler runs them in topo order, then a `state_evolution`
 *     phase writes one slot per extracted per-wire unit delay, then
 *     the DAC-stitch postamble.
 *
 *   • root (`rootProgram:true`) — `compileSessionSlottedRoot`: the
 *     whole session materializes into one synthetic root
 *     `ResolvedProgram` (instances → `InstanceDecl` children, per-wire
 *     unit delays → root `RegDecl`s) lowered through the SAME
 *     `partitionKernel` path the per-program fractal lowering uses.
 *     The scheduler is reduced to the DAC postamble; the delays are
 *     root RegDecl writebacks.
 *
 * Both feed the JIT through the identical `tropical_plan_5` schema.
 * Any divergence is a materializer, naming-transparency, or delay-
 * ordering bug. The flat path is the oracle.
 *
 * Requires libtropical.dylib (build with `make build` first).
 */

import { describe, test, expect } from 'bun:test'
import { makeSession, loadJSON, resolveProgramType, instantiate, inputNames, outputNames, setWireExpr } from '../../compiler/session.js'
import type { ExprNode } from '../../compiler/expr.js'
import { loadStdlib } from '../../compiler/program.js'
import { applyFlatPlan } from '../../compiler/apply_plan.js'
import { wireKey, portRef, instanceName, portName } from '../../compiler/ir/branded_names.js'

const wk = (i: string, p: string) => wireKey(portRef(instanceName(i), portName(p)))

const BUFFER_LENGTH = 256
const N_BUFFERS     = 4
const TOTAL_SAMPLES = BUFFER_LENGTH * N_BUFFERS
// Both paths run identical per-instruction LLVM codegen; the only
// difference is how the cross-instance/delay plumbing is expressed
// (scheduler state_evolution slots vs. root RegDecl writebacks). LLVM
// folds the equivalent store/load round-trips → byte-equal output.
const TOLERANCE     = 1e-12

function pulseEvery(n: number): ExprNode {
  return { op: 'lt', args: [{ op: 'mod', args: [{ op: 'sampleIndex' }, n] }, 1] }
}

const DEFAULT_INPUTS: Record<string, ExprNode> = {
  freq: 220, x: 0.5, y: 0.5, audio: 0.5, input: 0.5, cv: 0.5,
  cutoff: 1000, q: 0.5, drive: 1.0, mix: 0.5, a: 0.3, b: 0.7,
  coeff: 0.4, feedback: 0.4, lfo_speed: 0.2, decay: 0.99,
  rate: 5, g: 0.1, resonance: 0.5,
  trigger: pulseEvery(64), clock: pulseEvery(32),
}

const STDLIB_TARGETS: Array<[string, Record<string, number>?]> = [
  ['SinOsc'], ['Sin'], ['Cos'], ['Tanh'], ['Exp'], ['Log'], ['Pow'],
  ['OnePole'], ['BlepSaw'], ['SoftClip'], ['VCA'], ['CrossFade'],
  ['SVF'], ['LadderFilter'], ['Phaser'], ['Phaser16'],
  ['Bubble'], ['BubbleCloud'],
  ['AllpassDelay'], ['CombDelay'],
  ['Delay', { N: 1024 }],
]

function setupInstance(
  typeName: string,
  typeArgs: Record<string, number> | undefined,
) {
  const session = makeSession(BUFFER_LENGTH)
  loadStdlib(session)
  const { type, typeArgs: resolved } = resolveProgramType(session, typeName, typeArgs, undefined)
  const inst = instantiate(type, 'inst', { baseTypeName: typeName, typeArgs: resolved })
  session.instanceRegistry.set('inst', inst)
  for (const pn of inputNames(inst)) {
    if (pn in DEFAULT_INPUTS) {
      session.inputExprNodes.set(wk('inst', pn), DEFAULT_INPUTS[pn])
    }
  }
  session.graphOutputs.push({ instance: 'inst', output: outputNames(inst)[0] })
  return session
}

/** osc(BlepSaw) → amp(VCA) → dac, wired via `setWireExpr` so every
 *  cross-instance wire is auto-delayed → a delaySlotRegistry entry →
 *  (root path) a root RegDecl. This is the case the per-instance
 *  state_evolution phase exists for; the root differential proves the
 *  RegDecl-writeback replacement is byte-identical. */
function setupOscAmp() {
  const session = makeSession(BUFFER_LENGTH)
  loadStdlib(session)
  for (const [name, ty] of [['osc', 'BlepSaw'], ['amp', 'VCA']] as const) {
    const { type } = resolveProgramType(session, ty, undefined, undefined)
    session.instanceRegistry.set(name, instantiate(type, name, { baseTypeName: ty }))
  }
  // Wire freq explicitly. An UNWIRED typed-bound input (`freq: freq =
  // 440`) is handled differently by the two paths — the flat
  // top-level builder only honors plain-number defaults (→ 0 here),
  // while the root path's child default uses `rawInputDefaults` (→
  // 440). That divergence is a pre-existing default-policy quirk
  // orthogonal to Option A; every realistic session (and every other
  // equiv test) wires its inputs, so we do too.
  session.inputExprNodes.set(wk('osc', 'freq'), 220)
  setWireExpr(session, portRef(instanceName('amp'), portName('audio')),
    { op: 'ref', instance: 'osc', output: 'saw' })
  setWireExpr(session, portRef(instanceName('amp'), portName('cv')), 0.5)
  session.graphOutputs.push({ instance: 'amp', output: 'out' })
  return session
}

function captureOutput(
  session: ReturnType<typeof setupInstance>,
  rootProgram: boolean,
): Float64Array {
  applyFlatPlan(session, session.runtime, { rootProgram })
  session.graph.primeJit()
  const out = new Float64Array(TOTAL_SAMPLES)
  for (let f = 0; f < N_BUFFERS; f++) {
    session.runtime.process()
    const buf = session.runtime.outputBuffer
    out.set(buf.subarray(0, BUFFER_LENGTH), f * BUFFER_LENGTH)
  }
  return out
}

function assertEqual(label: string, flat: Float64Array, root: Float64Array) {
  let maxAbsDiff = 0
  let firstDiffIdx = -1
  for (let i = 0; i < TOTAL_SAMPLES; i++) {
    const f = flat[i]
    const r = root[i]
    if (Number.isNaN(f) && Number.isNaN(r)) continue
    const d = Math.abs(f - r)
    if (d > maxAbsDiff) maxAbsDiff = d
    if (d > TOLERANCE && firstDiffIdx < 0) firstDiffIdx = i
  }
  if (maxAbsDiff > TOLERANCE) {
    const i = firstDiffIdx
    throw new Error(
      `${label}: flat/root diverged at sample ${i} ` +
      `(flat=${flat[i]}, root=${root[i]}, maxAbsDiff=${maxAbsDiff})`,
    )
  }
  expect(maxAbsDiff).toBeLessThanOrEqual(TOLERANCE)
}

describe('root-vs-flat stdlib equivalence (single instance)', () => {
  for (const [typeName, typeArgs] of STDLIB_TARGETS) {
    test(`${typeName}${typeArgs ? `<${JSON.stringify(typeArgs)}>` : ''}`, () => {
      // Independent sessions per path so state inits match (no
      // hot-swap state transfer crossing the axis).
      const flat = captureOutput(setupInstance(typeName, typeArgs), false)
      const root = captureOutput(setupInstance(typeName, typeArgs), true)
      assertEqual(typeName, flat, root)
    })
  }
})

describe('root-vs-flat multi-instance (auto-delayed wires)', () => {
  test('BlepSaw → VCA → dac', () => {
    const flat = captureOutput(setupOscAmp(), false)
    const root = captureOutput(setupOscAmp(), true)
    assertEqual('osc→amp', flat, root)
  })
})

describe('root-vs-flat array-port wiring', () => {
  // A Sequencer's `values: float[N]` array input wired with an array
  // literal. `liftWiresToInstances` lifts the literal to an anonymous
  // producer instance whose array output is aliased into the
  // sequencer's input slot — a same-sample (un-delayed) array wire, so
  // the root path must emit the producer before the consumer and copy
  // the array into the child's session-array slot.
  test('Sequencer with array-literal values', () => {
    function setup() {
      const session = makeSession(BUFFER_LENGTH)
      loadStdlib(session)
      const { type, typeArgs } = resolveProgramType(session, 'Sequencer', { N: 8 }, undefined)
      const inst = instantiate(type, 'seq', { baseTypeName: 'Sequencer', typeArgs })
      session.instanceRegistry.set('seq', inst)
      session.inputExprNodes.set(wk('seq', 'clock'), pulseEvery(64))
      session.inputExprNodes.set(wk('seq', 'values'),
        { op: 'array', items: [110, 138.59, 164.81, 220, 261.63, 329.63, 220, 164.81] })
      session.graphOutputs.push({ instance: 'seq', output: outputNames(inst)[0] })
      return session
    }
    const flat = captureOutput(setup(), false)
    const root = captureOutput(setup(), true)
    assertEqual('Sequencer[values]', flat, root)
  })
})

describe('root-vs-flat array session delay', () => {
  // An array-shaped `delay()`: the wire `sum.a = delay([s, s+10, s+20])`
  // carries a time-varying array literal. `liftWiresToInstances` lifts
  // the literal to an anonymous producer instance; `extractSessionDelays`
  // hoists the surrounding `delay()` into an `ioArraySlot` (an `isArray`
  // registry entry). On the per-instance path that's a `state_evolution`
  // elementwise array `Add`; on the root path it becomes an array
  // `RegDecl` whose elementwise-copy writeback must reproduce the SAME
  // one-sample, per-element latency. The time-varying source makes any
  // latency or element-permutation bug visible.
  const arrDelayProgram = {
    schema: 'tropical_program_2', name: 'arr_delay_test',
    body: { op: 'block', decls: [
      { op: 'programDecl', name: 'ArrSum', program: {
        op: 'program', name: 'ArrSum',
        ports: { inputs: [{ name: 'a', type: { element: 'float', shape: [3] } }], outputs: ['out'] },
        body: { op: 'block', decls: [], assigns: [
          { op: 'outputAssign', name: 'out', expr: { op: 'add', args: [
            { op: 'index', args: [{ op: 'input', name: 'a' }, 0] },
            { op: 'mul', args: [{ op: 'index', args: [{ op: 'input', name: 'a' }, 1] }, 100] } ] } } ] } } },
      { op: 'instanceDecl', name: 'sum', program: 'ArrSum', inputs: { a: { op: 'delay', args: [
        { op: 'array', items: [
          { op: 'sampleIndex' },
          { op: 'add', args: [{ op: 'sampleIndex' }, 10] },
          { op: 'add', args: [{ op: 'sampleIndex' }, 20] } ] } ] } } } ],
      assigns: [] },
    audio_outputs: [{ instance: 'sum', output: 'out' }],
  }
  test('delay([s, s+10, s+20]) → ArrSum', () => {
    function setup() {
      const session = makeSession(BUFFER_LENGTH)
      loadStdlib(session)
      loadJSON(arrDelayProgram as Parameters<typeof loadJSON>[0], session)
      return session
    }
    const flat = captureOutput(setup(), false)
    const root = captureOutput(setup(), true)
    assertEqual('array-delay', flat, root)
  })
})

describe('root-vs-flat polyphony', () => {
  test('8x SinOsc voices', () => {
    function setupPolyphony() {
      const session = makeSession(BUFFER_LENGTH)
      loadStdlib(session)
      const { type, typeArgs } = resolveProgramType(session, 'SinOsc', undefined, undefined)
      for (let i = 0; i < 8; i++) {
        const name = `osc${i}`
        const inst = instantiate(type, name, { baseTypeName: 'SinOsc', typeArgs })
        session.instanceRegistry.set(name, inst)
        session.inputExprNodes.set(wk(name, 'freq'), 110 + 22 * i)
        session.graphOutputs.push({ instance: name, output: outputNames(inst)[0] })
      }
      return session
    }
    const flat = captureOutput(setupPolyphony(), false)
    const root = captureOutput(setupPolyphony(), true)
    assertEqual('8x SinOsc', flat, root)
  })
})
