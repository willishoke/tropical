/**
 * nested_vs_inlined.test.ts — cross-mode equivalence: the legacy
 * flat-IR compile path (`inlineNested:true`) vs the M11 fractal
 * slot-based compile path (`inlineNested:false`).
 *
 * Each session compiles its programs through one path; the test
 * builds the same session shape twice (once per mode) and asserts
 * sample-for-sample agreement on the audio output.
 *
 * This is the correctness gate for the input-slot refactor. The
 * flat path is the oracle (it's been correct since the strata
 * pipeline landed); the slot path must produce identical output.
 *
 * Inline path: `inlineInstances` strata pass splats every nested
 * `InstanceDecl` into its parent's body via expression substitution.
 * By the time `partition_recursive` sees the IR, no nesting remains.
 *
 * Slot path: `inlineInstances` is skipped. `partition_recursive`
 * allocates per-port INPUT slots for each child, emits per-child
 * `WriteSlot` blocks stored on each child's `pre_input_instructions`,
 * and the child's `InputRef`s lower to `Slot` reads via the
 * `inputSlotOverride` map. The boundary is the slot, not the
 * substituted expression. Per-child placement (vs hoisting into a
 * single parent-wide pre-children block) preserves sibling-to-sibling
 * NestedOut dependencies — Pow's `exp = Exp(x: y * log_x.out)` only
 * works because `log_x` has already run by the time `exp`'s pre-input
 * wires evaluate.
 *
 * If the two modes diverge, that's either:
 *   (a) a bug in `partition_recursive`'s slot-wiring step
 *   (b) a bug in `emit_resolved`'s InputRef → Slot lowering
 *   (c) an engine-side ordering bug (`pre_input_instructions` not
 *       emitted before each child)
 *   (d) a temp / slot index shift mismatch in `remapInstancePlan`
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
// Tight tolerance: both modes go through the same LLVM codegen for
// the per-instruction work; the only difference is whether wire
// expressions get inlined into the child body (flat path) or
// evaluated in the parent and crossed via a slot (slot path). LLVM's
// CSE + GVN should fold the slot store-load round-trip in fused mode,
// so we expect byte-equal results.
const TOLERANCE      = 1e-12

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

// Same stdlib corpus as microkernel_vs_fused — covers oscillators,
// filters, transcendentals, delays. The honest depth-2 cases
// (Phaser16 with 16 _allpassStage children; OnePole with 2 Tanh;
// Pow with Log + Exp; etc.) are where the slot path actually does
// work; the flat cases (Sin, Cos, Tanh) reduce to no-op trees on
// both sides.
const STDLIB_TARGETS: Array<[string, Record<string, number>?]> = [
  ['SinOsc'], ['Sin'], ['Cos'], ['Tanh'], ['Exp'], ['Log'], ['Pow'],
  ['OnePole'], ['BlepSaw'], ['SoftClip'], ['VCA'], ['CrossFade'],
  ['SVF'], ['LadderFilter'], ['Phaser'], ['Phaser16'],
  // Depth-3 cases unlocked by the bubble-fix-via-levels work:
  // Bubble contains sum-typed state regs (TriggerRamp's state,
  // EnvExpDecay's state) that previously failed to compile in
  // nested mode because the slot path arrived at them un-strata-
  // processed. The topological registry build (Phase 3) +
  // sumLower's pure construction (Phase 2) close the gap.
  ['Bubble'], ['BubbleCloud'],
  ['AllpassDelay'], ['CombDelay'],
  ['Delay', { N: 1024 }],
]

function setupInstance(
  typeName: string,
  typeArgs: Record<string, number> | undefined,
  inlineNested: boolean,
) {
  const session = makeSession(BUFFER_LENGTH, { inlineNested })
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

function captureOutput(session: ReturnType<typeof setupInstance>): Float64Array {
  applyFlatPlan(session, session.runtime)
  session.graph.primeJit()
  const out = new Float64Array(TOTAL_SAMPLES)
  for (let f = 0; f < N_BUFFERS; f++) {
    session.runtime.process()
    const buf = session.runtime.outputBuffer
    out.set(buf.subarray(0, BUFFER_LENGTH), f * BUFFER_LENGTH)
  }
  return out
}

describe('nested-vs-inlined stdlib equivalence', () => {
  for (const [typeName, typeArgs] of STDLIB_TARGETS) {
    test(`${typeName}${typeArgs ? `<${JSON.stringify(typeArgs)}>` : ''}`, () => {
      // Two independent sessions — one per mode — so register/slot
      // state starts from identical inits in both runs. Sharing a
      // session and switching modes would invoke hot-swap state
      // transfer, which is a separate axis to test.
      const flatOut   = captureOutput(setupInstance(typeName, typeArgs, /* inlineNested */ true))
      const nestedOut = captureOutput(setupInstance(typeName, typeArgs, /* inlineNested */ false))

      let maxAbsDiff = 0
      let firstDiffIdx = -1
      for (let i = 0; i < TOTAL_SAMPLES; i++) {
        const f = flatOut[i]
        const n = nestedOut[i]
        if (Number.isNaN(f) && Number.isNaN(n)) continue
        const d = Math.abs(f - n)
        if (d > maxAbsDiff) maxAbsDiff = d
        if (d > TOLERANCE && firstDiffIdx < 0) firstDiffIdx = i
      }

      if (maxAbsDiff > TOLERANCE) {
        const i = firstDiffIdx
        throw new Error(
          `${typeName}: flat/nested diverged at sample ${i} ` +
          `(flat=${flatOut[i]}, nested=${nestedOut[i]}, ` +
          `maxAbsDiff=${maxAbsDiff})`,
        )
      }
      expect(maxAbsDiff).toBeLessThanOrEqual(TOLERANCE)
    })
  }
})

// Polyphony case: 8 voices of SinOsc with freq spread. Each voice is
// a top-level session instance with a nested Sin child. The slot
// path must wire each voice's freq → its own Sin's x slot
// independently — no cross-voice slot collisions.
describe('nested-vs-inlined polyphony', () => {
  test('8x SinOsc voices', () => {
    function setupPolyphony(inlineNested: boolean) {
      const session = makeSession(BUFFER_LENGTH, { inlineNested })
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
    const flatOut   = captureOutput(setupPolyphony(true))
    const nestedOut = captureOutput(setupPolyphony(false))

    let maxAbsDiff = 0
    for (let i = 0; i < TOTAL_SAMPLES; i++) {
      if (Number.isNaN(flatOut[i]) && Number.isNaN(nestedOut[i])) continue
      const d = Math.abs(flatOut[i] - nestedOut[i])
      if (d > maxAbsDiff) maxAbsDiff = d
    }
    expect(maxAbsDiff).toBeLessThanOrEqual(TOLERANCE)
  })
})
