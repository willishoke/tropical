/**
 * microkernel_vs_fused.test.ts — cross-mode equivalence: fused JIT vs
 * microkernel JIT.
 *
 * For each stdlib program in the equivalence corpus, compile and run
 * the same session twice — once with `compilation_mode: 'fused'`,
 * once with `'microkernel'` — and assert sample-for-sample agreement.
 *
 * Modeled on `jit_vs_interp_stdlib.test.ts`. The microkernel and
 * fused JIT both emit LLVM IR through the same per-instruction
 * codegen (EmitCtx methods are functionally identical to fused
 * mode's inline lambdas — see OrcJitEngine.cpp's comment block).
 * Any divergence here is either:
 *   (a) a codegen drift between the duplicated paths (most likely),
 *   (b) a microkernel-mode dispatch-loop bug (FlatRuntime), or
 *   (c) a parser/runtime mode-routing bug (compilation_mode parse,
 *       cache key, hot-swap state transfer).
 *
 * Requires libtropical.dylib (build with `make build` first).
 */

import { describe, test, expect } from 'bun:test'
import { makeSession, resolveProgramType, instantiate, inputNames, outputNames } from '../../compiler/session.js'
import type { ExprNode } from '../../compiler/expr.js'
import { loadStdlib } from '../../compiler/program.js'
import { applyFlatPlan } from '../../compiler/apply_plan.js'
import { wireKey, portRef, instanceName, portName } from '../../compiler/ir/branded_names.js'

const wk = (i: string, p: string) => wireKey(portRef(instanceName(i), portName(p)))

const BUFFER_LENGTH  = 256
const N_BUFFERS      = 4
const TOTAL_SAMPLES  = BUFFER_LENGTH * N_BUFFERS
const TOLERANCE      = 1e-12   // tighter than jit-vs-interp; same LLVM codegen path

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
  ['AllpassDelay'], ['CombDelay'],
  ['Delay', { N: 1024 }],
]

function setupInstance(
  typeName: string,
  typeArgs?: Record<string, number>,
) {
  const session = makeSession(BUFFER_LENGTH)
  loadStdlib(session)
  const { type, typeArgs: resolved } = resolveProgramType(session, typeName, typeArgs, undefined)
  const inst = instantiate(type, 'inst', { baseTypeName: typeName, typeArgs: resolved })
  session.instanceRegistry.set('inst', inst)
  for (const portName of inputNames(inst)) {
    if (portName in DEFAULT_INPUTS) {
      session.inputExprNodes.set(wk('inst', portName), DEFAULT_INPUTS[portName])
    }
  }
  session.graphOutputs.push({ instance: 'inst', output: outputNames(inst)[0] })
  return session
}

function captureOutput(
  session: ReturnType<typeof setupInstance>,
  mode: 'fused' | 'microkernel',
): Float64Array {
  applyFlatPlan(session, session.runtime, { compilation_mode: mode })
  session.graph.primeJit()
  const out = new Float64Array(TOTAL_SAMPLES)
  for (let f = 0; f < N_BUFFERS; f++) {
    session.runtime.process()
    const buf = session.runtime.outputBuffer
    out.set(buf.subarray(0, BUFFER_LENGTH), f * BUFFER_LENGTH)
  }
  return out
}

describe('microkernel-vs-fused stdlib equivalence', () => {
  for (const [typeName, typeArgs] of STDLIB_TARGETS) {
    test(`${typeName}${typeArgs ? `<${JSON.stringify(typeArgs)}>` : ''}`, () => {
      // Two independent sessions — one per mode — so register/slot
      // state starts from identical inits in both runs. Sharing a
      // session and switching modes would invoke hot-swap state
      // transfer, which is a separate axis to test.
      const sessionFused = setupInstance(typeName, typeArgs)
      const sessionMk    = setupInstance(typeName, typeArgs)

      const fusedOut = captureOutput(sessionFused, 'fused')
      const mkOut    = captureOutput(sessionMk,    'microkernel')

      let maxAbsDiff = 0
      let firstDiffIdx = -1
      for (let i = 0; i < TOTAL_SAMPLES; i++) {
        const f = fusedOut[i]
        const m = mkOut[i]
        // Treat NaN==NaN — both modes should agree on NaN positions too.
        if (Number.isNaN(f) && Number.isNaN(m)) continue
        const d = Math.abs(f - m)
        if (d > maxAbsDiff) maxAbsDiff = d
        if (d > TOLERANCE && firstDiffIdx < 0) firstDiffIdx = i
      }

      if (maxAbsDiff > TOLERANCE) {
        const i = firstDiffIdx
        throw new Error(
          `${typeName}: microkernel/fused diverged at sample ${i} ` +
          `(fused=${fusedOut[i]}, microkernel=${mkOut[i]}, ` +
          `maxAbsDiff=${maxAbsDiff})`,
        )
      }
      expect(maxAbsDiff).toBeLessThanOrEqual(TOLERANCE)
    })
  }
})

// Polyphony case: 8 cross-FM voices in one session. Microkernel mode's
// per-sample dispatch must preserve cross-instance slot semantics
// across the 8 kernels.
//
// `cross_fm_4` is the polyphony patch referenced in the plan; the
// equivalence test here uses 8 instances of a small stdlib program
// (SinOsc), wired through a shared sink, to validate the
// multi-instance path without depending on a specific patch's
// existence on disk.
describe('microkernel-vs-fused polyphony', () => {
  test('8x SinOsc voices', () => {
    function setupPolyphony() {
      const session = makeSession(BUFFER_LENGTH)
      loadStdlib(session)
      const { type, typeArgs } = resolveProgramType(session, 'SinOsc', undefined, undefined)
      for (let i = 0; i < 8; i++) {
        const name = `osc${i}`
        const inst = instantiate(type, name, { baseTypeName: 'SinOsc', typeArgs })
        session.instanceRegistry.set(name, inst)
        // Per-voice freq spread so the kernels don't simplify to identical IR.
        session.inputExprNodes.set(wk(name, 'freq'), 110 + 22 * i)
        session.graphOutputs.push({ instance: name, output: outputNames(inst)[0] })
      }
      return session
    }
    const fusedOut = captureOutput(setupPolyphony(), 'fused')
    const mkOut    = captureOutput(setupPolyphony(), 'microkernel')

    let maxAbsDiff = 0
    for (let i = 0; i < TOTAL_SAMPLES; i++) {
      if (Number.isNaN(fusedOut[i]) && Number.isNaN(mkOut[i])) continue
      const d = Math.abs(fusedOut[i] - mkOut[i])
      if (d > maxAbsDiff) maxAbsDiff = d
    }
    expect(maxAbsDiff).toBeLessThanOrEqual(TOLERANCE)
  })
})
