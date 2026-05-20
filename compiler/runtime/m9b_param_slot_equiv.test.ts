/**
 * m9b_param_slot_equiv.test.ts — M9b equivalence gate.
 *
 * Verifies that param and trigger refs in input expressions compile to
 * slot operands under the per-instance path and produce correct audio.
 * Today's MCP set_param writes to both the legacy ControlParam and the
 * paramSlotRegistry's slot (the latter via the M5 set_slot bridge,
 * coming up implicitly through wire reads); the per-instance path
 * reads from slots, so the audio behavior should mirror legacy.
 *
 * Scope note: this tests the *compilation* — that a {op:'param'} or
 * {op:'triggerParamExpr'} ExprNode reaching the per-instance path
 * resolves to a slot operand. Full byte-equivalence with the legacy
 * SmoothParam/TriggerParam path is a *behavioral* statement (the
 * legacy path applies kernel-side smoothing; the slot path is raw
 * reads). Where the two diverge by design, we assert the slot path
 * produces the unsmoothed-but-correct value.
 */
import { describe, expect, test, beforeEach, afterEach } from 'bun:test'
import { Runtime } from './runtime.ts'
import { makeSession, allocateOutputSlots, allocateParamSlot } from '../session.ts'
import { loadStdlib } from '../program.ts'
import { instantiate } from '../program_types.ts'
import { compileSessionSlotted } from '../ir/compile_session_slotted.ts'
import { toWirePlan } from '../flat_plan.ts'
import { Param, Trigger } from './param.ts'
import { wireKey, portRef, instanceName, portName } from '../ir/branded_names.js'

const wk = (i: string, p: string) => wireKey(portRef(instanceName(i), portName(p)))

let prevEnv: string | undefined
beforeEach(() => {
  prevEnv = process.env.TROPICAL_SLOT_OPS
  process.env.TROPICAL_SLOT_OPS = '1'
})
afterEach(() => {
  if (prevEnv === undefined) delete process.env.TROPICAL_SLOT_OPS
  else process.env.TROPICAL_SLOT_OPS = prevEnv
})

describe('M9b: params as slot reads in per-instance path', () => {
  test('paramRef compiles to slot operand pointing at paramSlotRegistry', () => {
    const s = makeSession()
    loadStdlib(s)
    const sinOsc = s.typeRegistry.get('SinOsc')!
    s.instanceRegistry.set('osc', instantiate(sinOsc, 'osc'))
    allocateOutputSlots(s, 'osc', sinOsc)
    // Declare a 'cutoff' param at the session level and wire it
    s.paramRegistry.set('cutoff', new Param(880, 0))   // unsmoothed (time_const=0)
    const paramSlotIdx = allocateParamSlot(s, 'cutoff')
    s.inputExprNodes.set(wk("osc", "freq"), { op: 'param', name: 'cutoff' })
    s.graphOutputs.push({ instance: 'osc', output: 'sine' })

    const plan = compileSessionSlotted(s)

    // Find the SinOsc body's freq read and verify it's a slot operand
    // pointing at the param's slot index. Search across all instance
    // function bodies + scheduler preamble + postamble.
    const allInstrs = [
      ...plan.scheduler_function.preamble,
      ...plan.scheduler_function.postamble,
      ...plan.instance_functions.flatMap(i => i.instructions),
    ]
    const slotReadsAtParamIdx = allInstrs.flatMap(instr => instr.args)
      .filter(op => op.kind === 'slot' && op.index === paramSlotIdx)
    expect(slotReadsAtParamIdx.length).toBeGreaterThan(0)
  })

  test('paramRef + set_slot drives kernel audio: matches direct const', () => {
    // Set the param slot to a frequency, run the kernel, and verify the
    // output matches what a literal-wired version produces.
    const FREQ = 660
    // Variant A: param-wired
    const sParam = makeSession(64)
    loadStdlib(sParam)
    const sin = sParam.typeRegistry.get('SinOsc')!
    sParam.instanceRegistry.set('osc', instantiate(sin, 'osc'))
    allocateOutputSlots(sParam, 'osc', sin)
    sParam.paramRegistry.set('freq', new Param(FREQ, 0))
    const slotIdx = allocateParamSlot(sParam, 'freq')
    sParam.inputExprNodes.set(wk("osc", "freq"), { op: 'param', name: 'freq' })
    sParam.graphOutputs.push({ instance: 'osc', output: 'sine' })
    const planParam = compileSessionSlotted(sParam)

    // Variant B: literal freq
    const sLit = makeSession(64)
    loadStdlib(sLit)
    const sin2 = sLit.typeRegistry.get('SinOsc')!
    sLit.instanceRegistry.set('osc', instantiate(sin2, 'osc'))
    allocateOutputSlots(sLit, 'osc', sin2)
    sLit.inputExprNodes.set(wk("osc", "freq"), FREQ)
    sLit.graphOutputs.push({ instance: 'osc', output: 'sine' })
    const planLit = compileSessionSlotted(sLit)

    const rtParam = new Runtime(64)
    rtParam.loadPlan(JSON.stringify(toWirePlan(planParam)))
    // Defaults seeded the slot to FREQ already (via slot_defaults from
    // paramRegistry.value), so no set_slot needed for first buffer.
    rtParam.process()
    const paramAudio = Array.from(rtParam.outputBuffer)

    const rtLit = new Runtime(64)
    rtLit.loadPlan(JSON.stringify(toWirePlan(planLit)))
    rtLit.process()
    const litAudio = Array.from(rtLit.outputBuffer)

    expect(paramAudio.length).toBe(litAudio.length)
    for (let i = 0; i < paramAudio.length; i++) {
      expect(paramAudio[i]).toBe(litAudio[i])
    }
    rtParam.dispose()
    rtLit.dispose()
  })

  test('set_slot mid-run: kernel picks up new param value next buffer', () => {
    const s = makeSession(64)
    loadStdlib(s)
    const sin = s.typeRegistry.get('SinOsc')!
    s.instanceRegistry.set('osc', instantiate(sin, 'osc'))
    allocateOutputSlots(s, 'osc', sin)
    s.paramRegistry.set('f', new Param(110, 0))
    allocateParamSlot(s, 'f')
    s.inputExprNodes.set(wk("osc", "freq"), { op: 'param', name: 'f' })
    s.graphOutputs.push({ instance: 'osc', output: 'sine' })

    const plan = compileSessionSlotted(s)
    const rt = new Runtime(64)
    rt.loadPlan(JSON.stringify(toWirePlan(plan)))

    rt.process()
    const audio_110 = Array.from(rt.outputBuffer)

    // Mid-run param change
    const idx = rt.slotIndex('param:f')
    expect(idx).toBeGreaterThanOrEqual(0)
    rt.setSlot(idx, 880)
    rt.process()
    const audio_880 = Array.from(rt.outputBuffer)

    // Different frequencies → different waveforms. They shouldn't be equal.
    let anyDiff = false
    for (let i = 0; i < audio_110.length; i++) {
      if (audio_110[i] !== audio_880[i]) { anyDiff = true; break }
    }
    expect(anyDiff).toBe(true)
    rt.dispose()
  })
})

describe('M9b: triggers as slot reads', () => {
  test('triggerParamExpr compiles to slot operand', () => {
    const s = makeSession()
    loadStdlib(s)
    const sin = s.typeRegistry.get('SinOsc')!
    s.instanceRegistry.set('osc', instantiate(sin, 'osc'))
    allocateOutputSlots(s, 'osc', sin)
    s.triggerRegistry.set('go', new Trigger())
    const trigSlot = allocateParamSlot(s, 'go')
    // Wire trigger into freq — odd but exercises the compile path
    s.inputExprNodes.set(wk("osc", "freq"), { op: 'triggerParamExpr', name: 'go' })
    s.graphOutputs.push({ instance: 'osc', output: 'sine' })

    const plan = compileSessionSlotted(s)
    const allInstrs = [
      ...plan.scheduler_function.preamble,
      ...plan.scheduler_function.postamble,
      ...plan.instance_functions.flatMap(i => i.instructions),
    ]
    const slotReads = allInstrs.flatMap(instr => instr.args)
      .filter(op => op.kind === 'slot' && op.index === trigSlot)
    expect(slotReads.length).toBeGreaterThan(0)
  })
})
