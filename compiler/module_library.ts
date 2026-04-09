/**
 * Preset module types. Port of tropical/module_library.py.
 */

import {
  SignalExpr, ExprCoercible,
  add, sub, mul, div, mod, pow_, neg, floorDiv,
  lt, lte, gt, gte,
  abs_, sin, log, tanh,
  bitAnd, bitXor, rshift,
  clamp, select, arrayPack, arraySet, matmul,
  sampleRate, sampleIndex,
} from './expr.js'
import {
  defineModule, definePureFunction,
  delay, SymbolMap, ModuleType, PureFunction,
} from './module.js'

const TWO_PI = 6.283185307179586

/** Narrow a single-output call result to SignalExpr. */
function one(r: SignalExpr | SignalExpr[]): SignalExpr {
  return Array.isArray(r) ? r[0] : r
}

// ─── Private sub-modules ──────────────────────────────────────────────────────

const _wrap01: PureFunction = definePureFunction(['x'], ['value'], (inp) => {
  const x = inp.get('x')
  return { value: mod(add(mod(x, 1.0), 1.0), 1.0) }
})

const _polyBlep: PureFunction = definePureFunction(['t', 'dt'], ['value'], (inp) => {
  const t = inp.get('t')
  const dt = inp.get('dt')
  const leftT = div(t, dt)
  const rightT = div(sub(t, 1.0), dt)
  const left = sub(sub(add(leftT, leftT), mul(leftT, leftT)), 1.0)
  const right = add(add(mul(rightT, rightT), mul(2.0, rightT)), 1.0)
  const valid = mul(gt(dt, 0.0), lt(dt, 1.0))
  const leftMask = lt(t, dt)
  const rightMask = mul(gte(t, dt), gt(t, sub(1.0, dt)))
  return { value: mul(valid, add(mul(leftMask, left), mul(rightMask, right))) }
})

const _allpassStage: ModuleType = defineModule(
  'AllpassStage', ['x', 'a'], ['y'], { x_prev: 0.0, y_prev: 0.0 },
  (inp, reg) => {
    const a = inp.get('a')
    const x = inp.get('x')
    const output = add(add(mul(neg(a), x), reg.get('x_prev')), mul(a, reg.get('y_prev')))
    return {
      outputs: { y: output },
      nextRegs: { x_prev: x, y_prev: output },
    }
  },
)

const _polyBlamp: PureFunction = definePureFunction(['t', 'dt'], ['value'], (inp) => {
  const t = inp.get('t')
  const dt = inp.get('dt')
  const valid = mul(gt(dt, 0.0), lt(dt, 1.0))

  const u = div(t, dt)
  const u2 = mul(u, u)
  const u3 = mul(u2, u)
  const left = sub(sub(u2, div(u3, 3.0)), u)

  const v = div(sub(1.0, t), dt)
  const v2 = mul(v, v)
  const v3 = mul(v2, v)
  const right = neg(sub(sub(v2, div(v3, 3.0)), v))

  const leftMask  = lt(t, dt)
  const rightMask = lt(sub(1.0, t), dt)
  return { value: mul(valid, add(mul(leftMask, left), mul(rightMask, right))) }
})

// ─── VCO ──────────────────────────────────────────────────────────────────────

export function vco(name = 'VCO'): ModuleType {
  return defineModule(
    name,
    [{ name: 'freq', type: 'float' }, { name: 'fm', type: 'float' }, { name: 'fm_index', type: 'float' }],
    [{ name: 'saw', type: 'float' }, { name: 'tri', type: 'float' }, { name: 'sin', type: 'float' }, { name: 'sqr', type: 'float' }],
    { phase: 0.0, tri_state: 0.0 },
    (inp, reg) => {
      const fmRatio = pow_(2.0, div(mul(inp.get('fm_index'), inp.get('fm')), 5.0))
      const freq    = mul(inp.get('freq'), fmRatio)
      const sr      = sampleRate()
      const dt      = clamp(div(abs_(freq), sr), 0.0, 0.5)
      const phase   = one(_wrap01.call(add(reg.get('phase'), div(freq, sr))))

      const saw0 = sub(mul(2.0, phase), 1.0)
      const saw  = sub(saw0, one(_polyBlep.call(phase, dt)))

      const sqr0 = sub(1.0, mul(2.0, gte(phase, 0.5)))
      const sqr1 = add(sqr0, one(_polyBlep.call(phase, dt)))
      const sqr  = sub(sqr1, one(_polyBlep.call(one(_wrap01.call(add(phase, 0.5))), dt)))

      const triState = clamp(add(reg.get('tri_state'), mul(mul(sqr, dt), 4.0)), -1.0, 1.0)
      const sine     = sin(mul(TWO_PI, phase))

      return {
        outputs: {
          saw: mul(5.0, saw),
          tri: mul(5.0, triState),
          sin: mul(5.0, sine),
          sqr: mul(5.0, sqr),
        },
        nextRegs: { phase, tri_state: triState },
      }
    },
    44100.0,
    { freq: 100.0, fm: 0.0, fm_index: 5.0 },
  )
}

// ─── Phaser ───────────────────────────────────────────────────────────────────

function _phaser(stageCount: number, name: string): ModuleType {
  return defineModule(
    name,
    ['input', 'feedback', 'lfo_speed'],
    ['output', 'lfo'],
    { fb: 0.0 },
    (inp, reg) => {
      const lfo = sin(mul(TWO_PI, div(mul(sampleIndex(), inp.get('lfo_speed')), sampleRate())))
      const a = add(0.6, mul(0.35, lfo))
      let stageInput: ExprCoercible = add(inp.get('input'), mul(inp.get('feedback'), reg.get('fb')))
      for (let i = 0; i < stageCount; i++) {
        stageInput = one(_allpassStage.call(stageInput, a))
      }
      return {
        outputs: {
          output: add(mul(0.5, inp.get('input')), mul(0.5, stageInput)),
          lfo,
        },
        nextRegs: { fb: stageInput as SignalExpr },
      }
    },
    44100.0,
    { input: 0.0, feedback: 0.4, lfo_speed: 0.2 },
  )
}

export function phaser(name = 'Phaser'): ModuleType    { return _phaser(4,  name) }
export function phaser16(name = 'Phaser16'): ModuleType { return _phaser(16, name) }

// ─── Clock ────────────────────────────────────────────────────────────────────

export function clock(name = 'Clock'): ModuleType {
  return defineModule(
    name,
    ['freq', { name: 'ratios_in', type: 'float[1]' }],
    ['output', { name: 'ratios_out', type: 'float[1]' }],
    {},
    (inp) => {
      const sr = sampleRate()
      const basePhase = one(_wrap01.call(div(mul(sampleIndex(), inp.get('freq')), sr)))
      const output    = mul(lt(basePhase, 0.5), 1.0)

      const ratioPhase = one(_wrap01.call(div(mul(mul(sampleIndex(), inp.get('freq')), inp.get('ratios_in')), sr)))
      const ratiosOut  = mul(lt(ratioPhase, 0.5), 1.0)

      return {
        outputs: { output, ratios_out: ratiosOut },
        nextRegs: {},
      }
    },
    44100.0,
    { freq: 1.0, ratios_in: [1.0] },
  )
}

// ─── Delay line ───────────────────────────────────────────────────────────────

export function delayLine(delayLen: number, name = 'Delay'): ModuleType {
  return defineModule(
    name,
    ['x'],
    ['y'],
    { buf: new Array(delayLen).fill(0.0) },
    (inp, reg) => {
      const buf      = reg.get('buf')
      const writeIdx = mod(sampleIndex(), delayLen)
      const y        = buf.at(writeIdx)
      const newBuf   = arraySet(buf, writeIdx, inp.get('x'))
      return {
        outputs: { y },
        nextRegs: { buf: newBuf },
      }
    },
    44100.0,
    { x: 0.0 },
    { breaksCycles: true },
  )
}

// ─── Comb filter (private) ────────────────────────────────────────────────────

function _defineCombFilter(delayLen: number, name: string): ModuleType {
  return defineModule(
    name,
    ['x', 'decay', 'damp'],
    ['y'],
    { lpf: 0.0, buf: new Array(delayLen).fill(0.0) },
    (inp, reg) => {
      const x        = inp.get('x')
      const decay    = clamp(inp.get('decay'), 0.0, 0.98)
      const damp     = clamp(inp.get('damp'),  0.0, 0.99)
      const lpfPrev  = reg.get('lpf')
      const buf      = reg.get('buf')
      const writeIdx = mod(sampleIndex(), delayLen)
      const state    = buf.at(writeIdx)
      const newBuf   = arraySet(buf, writeIdx, add(x, mul(decay, lpfPrev)))
      const lpfOut   = add(mul(sub(1.0, damp), state), mul(damp, lpfPrev))
      return {
        outputs: { y: state },
        nextRegs: { lpf: lpfOut, buf: newBuf },
      }
    },
    44100.0,
    { x: 0.0, decay: 0.84, damp: 0.4 },
  )
}

// ─── Reverb ───────────────────────────────────────────────────────────────────

export function reverb(name = 'Reverb'): ModuleType {
  const COMB_DELAYS = [1557, 1617, 1491, 1422]
  const combFilters = COMB_DELAYS.map((d, i) => _defineCombFilter(d, `${name}_Comb${i}`))
  const AP_IN  = [0.70, 0.65]
  const AP_OUT = [0.60, 0.55, 0.50, 0.45]

  return defineModule(
    name,
    ['input', 'mix', 'decay', 'damp'],
    ['output'],
    {},
    (inp) => {
      const x     = inp.get('input')
      const mix   = clamp(inp.get('mix'),   0.0, 1.0)
      const decay = clamp(inp.get('decay'), 0.0, 0.99)
      const damp  = clamp(inp.get('damp'),  0.0, 1.0)

      let diff: ExprCoercible = x
      for (const a of AP_IN) {
        diff = one(_allpassStage.call(diff, a))
      }

      let wet: ExprCoercible = 0.0
      for (const cf of combFilters) {
        wet = add(wet, one(cf.call(diff, decay, damp)))
      }
      wet = mul(wet as SignalExpr, 0.25)

      for (const a of AP_OUT) {
        wet = one(_allpassStage.call(wet, a))
      }

      const output = add(mul(sub(1.0, mix), x), mul(mix, wet as SignalExpr))
      return {
        outputs: { output },
        nextRegs: {},
      }
    },
    44100.0,
    { input: 0.0, mix: 0.35, decay: 0.84, damp: 0.4 },
  )
}

// ─── AD Envelope ──────────────────────────────────────────────────────────────

export function adEnvelope(name = 'ADEnvelope'): ModuleType {
  return defineModule(
    name,
    ['gate', 'attack', 'decay'],
    ['env'],
    { stage: 0.0, phase: 0.0, startLevel: 0.0 },
    (inp, reg) => {
      const sr      = sampleRate()
      const attack  = clamp(inp.get('attack'), 1e-4, 10.0)
      const decayT  = clamp(inp.get('decay'),  1e-4, 10.0)
      const gate    = inp.get('gate')

      const prevGate = delay(gate, 0.0)
      const trig     = mul(gt(gate, 0.5), lte(prevGate, 0.5))

      const stage      = reg.get('stage')
      const phase      = reg.get('phase')
      const startLevel = reg.get('startLevel')

      const inAttack = mul(gt(stage, 0.5), lt(stage, 1.5))
      const inDecay  = gt(stage, 1.5)

      const dtA = div(1.0, mul(attack, sr))
      const dtD = div(1.0, mul(decayT, sr))
      const dt  = add(mul(inAttack, dtA), mul(inDecay, dtD))

      const newPhase = add(phase, dt)
      const aToD     = mul(inAttack, gte(newPhase, 1.0))
      const dDone    = mul(inDecay,  gte(newPhase, 1.0))

      const notTrig = sub(1.0, trig)
      const nextStage = add(
        mul(trig, 1.0),
        mul(notTrig, add(
          add(mul(mul(inAttack, sub(1.0, aToD)), 1.0), mul(aToD, 2.0)),
          mul(mul(inDecay, sub(1.0, dDone)), 2.0),
        )),
      )
      const nextPhase = mul(
        notTrig,
        add(
          mul(aToD, clamp(sub(newPhase, 1.0), 0.0, 1.0)),
          mul(mul(sub(1.0, aToD), sub(1.0, dDone)), clamp(newPhase, 0.0, 1.0)),
        ),
      )

      // Attack ramps from startLevel (captured at retrigger) up to 1.0,
      // so retriggering mid-cycle never jumps discontinuously to zero.
      const attackSlope = sub(1.0, startLevel)
      const rawEnv = add(
        mul(inAttack, add(startLevel, mul(phase, attackSlope))),
        mul(inDecay,  sub(1.0, phase)),
      )

      // Capture envelope level at the retrigger moment for the next attack.
      // Uses rawEnv (blamp correction is negligibly small at capture time).
      const nextStartLevel = add(mul(trig, rawEnv), mul(notTrig, startLevel))

      const blampStart    = one(_polyBlamp.call(phase,           dtA))
      const blampAtkEnd   = one(_polyBlamp.call(sub(1.0, phase), dtA))
      const blampDecStart = one(_polyBlamp.call(phase,           dtD))
      const blampEnd      = one(_polyBlamp.call(sub(1.0, phase), dtD))

      let env: ExprCoercible = rawEnv
      // Scale attack BLAMP by attackSlope: actual per-sample rise is dtA * attackSlope
      env = sub(env, mul(mul(mul(inAttack, dtA), attackSlope), blampStart))
      env = add(env, mul(mul(mul(inAttack, dtA), attackSlope), blampAtkEnd))
      env = add(env, mul(mul(inDecay,  dtD), blampDecStart))
      env = sub(env, mul(mul(inDecay,  dtD), blampEnd))

      return {
        outputs: { env: clamp(env as SignalExpr, 0.0, 1.0) },
        nextRegs: { stage: nextStage, phase: nextPhase, startLevel: nextStartLevel },
      }
    },
    44100.0,
    { gate: 0.0, attack: 0.01, decay: 0.3 },
  )
}

// ─── ADSR Envelope ───────────────────────────────────────────────────────────

export function adsrEnvelope(name = 'ADSREnvelope'): ModuleType {
  return defineModule(
    name,
    ['gate', 'attack', 'decay', 'sustain', 'release'],
    ['env'],
    { stage: 0.0, phase: 0.0, release_level: 0.0 },
    (inp, reg) => {
      const sr       = sampleRate()
      const attack   = clamp(inp.get('attack'),  1e-4, 10.0)
      const decayT   = clamp(inp.get('decay'),   1e-4, 10.0)
      const sustain  = clamp(inp.get('sustain'),  0.0,  1.0)
      const releaseT = clamp(inp.get('release'), 1e-4, 10.0)
      const gate     = inp.get('gate')

      const prevGate = delay(gate, 0.0)
      const trig     = mul(gt(gate, 0.5), lte(prevGate, 0.5))
      const gateOff  = mul(lte(gate, 0.5), gt(prevGate, 0.5))

      const stage = reg.get('stage')
      const phase = reg.get('phase')
      const relLvl = reg.get('release_level')

      // Branchless stage detection
      const inAttack  = mul(gt(stage, 0.5), lt(stage, 1.5))
      const inDecay   = mul(gt(stage, 1.5), lt(stage, 2.5))
      const inSustain = mul(gt(stage, 2.5), lt(stage, 3.5))
      const inRelease = gt(stage, 3.5)
      const inADS     = mul(gt(stage, 0.5), lt(stage, 3.5))

      // Phase increments per stage
      const dtA = div(1.0, mul(attack,   sr))
      const dtD = div(1.0, mul(decayT,   sr))
      const dtR = div(1.0, mul(releaseT, sr))
      const dt  = add(add(mul(inAttack, dtA), mul(inDecay, dtD)), mul(inRelease, dtR))

      const newPhase = add(phase, dt)
      const atkDone  = mul(inAttack,  gte(newPhase, 1.0))
      const decDone  = mul(inDecay,   gte(newPhase, 1.0))
      const relDone  = mul(inRelease, gte(newPhase, 1.0))

      // Current envelope value (before transitions) for release capture
      const currentEnv = add(
        add(mul(inAttack, phase),
            mul(inDecay, sub(1.0, mul(phase, sub(1.0, sustain))))),
        add(mul(inSustain, sustain),
            mul(inRelease, mul(relLvl, sub(1.0, phase)))),
      )

      // Gate-off: capture current env and enter release
      const enterRelease = mul(gateOff, inADS)

      const notTrig = sub(1.0, trig)
      const notGateOff = sub(1.0, enterRelease)

      // Next stage logic (trig overrides gate-off overrides normal transitions)
      const normalNext = add(
        add(
          mul(mul(inAttack, sub(1.0, atkDone)), 1.0),  // stay in attack
          mul(atkDone, 2.0),                            // attack → decay
        ),
        add(
          add(
            mul(mul(inDecay, sub(1.0, decDone)), 2.0),  // stay in decay
            mul(decDone, 3.0),                           // decay → sustain
          ),
          add(
            mul(inSustain, 3.0),                         // stay in sustain
            mul(mul(inRelease, sub(1.0, relDone)), 4.0), // stay in release (idle if done = 0)
          ),
        ),
      )

      const afterGateOff = add(
        mul(enterRelease, 4.0),
        mul(notGateOff, normalNext),
      )

      const nextStage = add(
        mul(trig, 1.0),
        mul(notTrig, afterGateOff),
      )

      // Next phase
      const normalPhase = add(
        mul(atkDone, clamp(sub(newPhase, 1.0), 0.0, 1.0)),
        mul(
          sub(1.0, add(atkDone, add(decDone, relDone))),
          clamp(newPhase, 0.0, 1.0),
        ),
      )

      const afterGateOffPhase = add(
        mul(enterRelease, 0.0),
        mul(notGateOff, normalPhase),
      )

      const nextPhase = mul(notTrig, afterGateOffPhase)

      // Next release_level: capture on gate-off, keep otherwise
      const nextRelLvl = add(
        mul(enterRelease, currentEnv),
        mul(sub(1.0, enterRelease), relLvl),
      )

      // Compute output envelope with polyBLAMP antialiasing
      const rawEnv = currentEnv

      let env: ExprCoercible = rawEnv
      // Attack BLAMP
      env = sub(env, mul(mul(inAttack, dtA), one(_polyBlamp.call(phase, dtA))))
      env = add(env, mul(mul(inAttack, dtA), one(_polyBlamp.call(sub(1.0, phase), dtA))))
      // Decay BLAMP
      const decaySlope = sub(1.0, sustain)
      env = add(env, mul(mul(mul(inDecay, dtD), decaySlope), one(_polyBlamp.call(phase, dtD))))
      env = sub(env, mul(mul(mul(inDecay, dtD), decaySlope), one(_polyBlamp.call(sub(1.0, phase), dtD))))
      // Release BLAMP
      env = add(env, mul(mul(mul(inRelease, dtR), relLvl), one(_polyBlamp.call(phase, dtR))))
      env = sub(env, mul(mul(mul(inRelease, dtR), relLvl), one(_polyBlamp.call(sub(1.0, phase), dtR))))

      return {
        outputs: { env: clamp(env as SignalExpr, 0.0, 1.0) },
        nextRegs: { stage: nextStage, phase: nextPhase, release_level: nextRelLvl },
      }
    },
    44100.0,
    { gate: 0.0, attack: 0.01, decay: 0.1, sustain: 0.7, release: 0.3 },
  )
}

// ─── Compressor ───────────────────────────────────────────────────────────────

export function compressor(name = 'Compressor'): ModuleType {
  const LOG10E      = 0.4342944819309
  const LN10_20_INV = 8.68588963807

  return defineModule(
    name,
    ['input', 'sidechain', 'threshold', 'ratio', 'attack_ms', 'release_ms', 'makeup'],
    ['output', 'gr'],
    { env: 0.0, gr: 0.0 },
    (inp, reg) => {
      const sr        = sampleRate()
      const threshold = inp.get('threshold')
      const ratio     = clamp(inp.get('ratio'),      1.0,  1000.0)
      const attackMs  = clamp(inp.get('attack_ms'),  0.01, 2000.0)
      const releaseMs = clamp(inp.get('release_ms'), 1.0,  10000.0)
      const makeup    = inp.get('makeup')

      const sc      = abs_(inp.get('sidechain'))
      const prevEnv = reg.get('env')

      const atkCoeff = pow_(10.0, div(-LOG10E, clamp(mul(mul(attackMs,  0.001), sr), 1.0, 1e8)))
      const relCoeff = pow_(10.0, div(-LOG10E, clamp(mul(mul(releaseMs, 0.001), sr), 1.0, 1e8)))

      const rising = gt(sc, prevEnv)
      const newEnv = add(
        mul(rising,         add(mul(atkCoeff, prevEnv), mul(sub(1.0, atkCoeff), sc))),
        mul(sub(1.0, rising), add(mul(relCoeff, prevEnv), mul(sub(1.0, relCoeff), sc))),
      )

      const levelDb = mul(LN10_20_INV, log(clamp(newEnv, 1e-9, 1e9)))
      const overDb  = sub(levelDb, threshold)
      const grDb    = mul(mul(gt(overDb, 0.0), overDb), sub(div(1.0, ratio), 1.0))

      const prevGr   = reg.get('gr')
      const grAttack = lt(grDb, prevGr)
      const smoothGr = add(
        mul(grAttack,         add(mul(atkCoeff, prevGr), mul(sub(1.0, atkCoeff), grDb))),
        mul(sub(1.0, grAttack), add(mul(relCoeff, prevGr), mul(sub(1.0, relCoeff), grDb))),
      )

      const gain   = pow_(10.0, div(smoothGr, 20.0))
      const output = mul(mul(inp.get('input'), gain), makeup)

      return {
        outputs: { output, gr: smoothGr },
        nextRegs: { env: newEnv, gr: smoothGr },
      }
    },
    44100.0,
    {
      input: 0.0, sidechain: 0.0,
      threshold: -12.0, ratio: 4.0,
      attack_ms: 10.0, release_ms: 100.0, makeup: 1.0,
    },
  )
}

// ─── Bass Drum ────────────────────────────────────────────────────────────────

export function bassDrum(name = 'BassDrum'): ModuleType {
  const PI = 3.14159265358979

  return defineModule(
    name,
    ['gate', 'freq', 'punch', 'decay', 'tone'],
    ['output'],
    { amp_env: 0.0, pitch_env: 0.0, ic1: 0.0, ic2: 0.0 },
    (inp, reg) => {
      const sr    = sampleRate()
      const freq  = clamp(inp.get('freq'),  20.0, 500.0)
      const punch = clamp(inp.get('punch'),  0.0,   1.0)
      const decayT = clamp(inp.get('decay'), 0.01,   4.0)
      const tone  = clamp(inp.get('tone'),  0.5,   50.0)

      const gate     = inp.get('gate')
      const prevGate = delay(gate, 0.0)
      const trig     = mul(gt(gate, 0.5), lte(prevGate, 0.5))
      const notTrig  = sub(1.0, trig)

      const ampEnv   = reg.get('amp_env')
      const ampCoeff = pow_(10.0, div(-0.4342944819, clamp(mul(decayT, sr), 1.0, 1e8)))
      const newAmp   = add(mul(trig, 1.0), mul(mul(notTrig, ampCoeff), ampEnv))

      const pitchEnv   = reg.get('pitch_env')
      const pitchCoeff = pow_(10.0, div(-0.4342944819, clamp(mul(0.040, sr), 1.0, 1e8)))
      const newPitch   = add(mul(trig, 1.0), mul(mul(notTrig, pitchCoeff), pitchEnv))

      const fc = mul(freq, add(1.0, mul(mul(punch, 4.0), newPitch)))
      const g  = clamp(div(mul(PI, fc), sr), 0.0, 0.98)
      const R  = div(0.5, tone)

      const ic1 = add(mul(trig, 0.0), mul(notTrig, reg.get('ic1')))
      const ic2 = add(mul(trig, 2.0), mul(notTrig, reg.get('ic2')))

      const denom = add(add(1.0, mul(mul(2.0, R), g)), mul(g, g))
      const v3    = div(sub(sub(0.0, ic2), mul(add(mul(2.0, R), g), ic1)), denom)
      const v1    = mul(g, v3)
      const yBp   = add(v1, ic1)
      const v2    = mul(g, yBp)
      const yLp   = add(v2, ic2)

      const newIc1 = add(yBp, v1)
      const newIc2 = add(yLp, v2)
      const output = mul(mul(newAmp, clamp(mul(yLp, 0.5), -1.0, 1.0)), 5.0)

      return {
        outputs: { output },
        nextRegs: { amp_env: newAmp, pitch_env: newPitch, ic1: newIc1, ic2: newIc2 },
      }
    },
    44100.0,
    { gate: 0.0, freq: 60.0, punch: 0.5, decay: 0.35, tone: 8.0 },
  )
}

// ─── Topological Waveguide ────────────────────────────────────────────────────

export function topoWaveguide(nx = 4, ny = 4, name = 'TopoWaveguide'): ModuleType {
  nx = Math.max(1, Math.trunc(nx))
  ny = Math.max(1, Math.trunc(ny))
  const nodeCount = nx * ny
  const centerX   = 0.5 * (nx - 1)
  const centerY   = 0.5 * (ny - 1)
  const maxRadius = Math.max(1.0, Math.sqrt(centerX * centerX + centerY * centerY))
  const MODAL_RATIOS = [1.0, 2.76, 5.4, 8.93, 13.3, 18.64, 24.97, 32.31]

  // JS % can be negative for negative operands — use proper modulo
  const posmod = (a: number, b: number) => ((a % b) + b) % b
  const gidx   = (i: number, j: number) => posmod(i, nx) * ny + posmod(j, ny)

  // Adjacency list (toroidal)
  const adjacency: number[][] = Array.from({ length: nodeCount }, () => [])
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      const n = gidx(i, j)
      for (const [di, dj] of [[-1,0],[1,0],[0,-1],[0,1]] as [number,number][]) {
        const v = gidx(i + di, j + dj)
        if (v !== n && !adjacency[n].includes(v)) adjacency[n].push(v)
      }
    }
  }

  // Pre-compute per-node constants (plain JS)
  const fcValues: number[]           = []
  const stiffnessValues: number[]    = []
  const phaseOffsetValues: number[]  = []
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      const nodeId = gidx(i, j)
      const ratio  = MODAL_RATIOS[nodeId % MODAL_RATIOS.length]
      const dx     = i - centerX, dy = j - centerY
      const radial = Math.sqrt(dx * dx + dy * dy) / maxRadius
      const skew   = ((posmod(nodeId * 17, 9)) - 4) / 4.0
      let   fc     = 180.0 * ratio * (1.0 + 0.16 * radial) * (1.0 + 0.035 * skew)
      fc = Math.max(120.0, Math.min(6400.0, fc))
      let stiffness = 1.0 + 0.08 * radial + 0.025 * (posmod(nodeId * 11, 7) - 3)
      stiffness = Math.max(0.82, Math.min(1.22, stiffness))
      fcValues.push(fc)
      stiffnessValues.push(stiffness)
      phaseOffsetValues.push(TWO_PI * (posmod(nodeId * 19, 23) / 23.0))
    }
  }

  return defineModule(
    name,
    ['input', 'fc', 'g', 'decay', 'brightness'],
    ['out'],
    {
      amp:   new Array(nodeCount).fill(0.0),
      phase: new Array(nodeCount).fill(0.0),
    },
    (inp, reg) => {
      const g          = clamp(inp.get('g'),          0.0,    0.2)
      const decay      = clamp(inp.get('decay'),      0.95,   0.999995)
      const brightness = clamp(inp.get('brightness'), 0.0,    1.0)
      const ampPrev    = reg.get('amp')
      const phasePrev  = reg.get('phase')

      const outputs:  SignalExpr[] = []
      const nextAmp:  SignalExpr[] = []
      const nextPhase: SignalExpr[] = []

      for (let i = 0; i < nodeCount; i++) {
        // Sum neighbour amplitudes
        let neighborEnergy: ExprCoercible = 0.0
        for (const j of adjacency[i]) {
          neighborEnergy = add(neighborEnergy, ampPrev.at(j))
        }
        const neighborAvg = div(neighborEnergy, Math.max(1, adjacency[i].length))

        const fc           = clamp(inp.get('fc').at(i), 120.0, 8000.0)
        const normalizedFc = clamp(div(fc, 8000.0), 0.0, 1.0)
        const decayFactor  = mul(decay, sub(1.0, mul(add(0.008, mul(0.055, sub(1.0, brightness))), normalizedFc)))
        const localDecay   = clamp(decayFactor, 0.94, 0.99997)

        const strike   = mul(inp.get('input').at(i), add(0.18, mul(0.82, brightness)))
        const coupling = mul(mul(mul(0.018, g), neighborAvg), sub(1.0, mul(0.5, normalizedFc)))
        const amp      = add(add(mul(localDecay, ampPrev.at(i)), strike), coupling)

        const phaseInc = add(phasePrev.at(i), div(mul(fc, stiffnessValues[i]), sampleRate()))
        const phase    = one(_wrap01.call(phaseInc))

        const fundamental = sin(add(mul(TWO_PI, phase), phaseOffsetValues[i]))
        const overtone    = mul(mul(fundamental, fundamental), fundamental)
        const y = mul(amp, add(
          mul(sub(0.9,  mul(0.26, brightness)), fundamental),
          mul(add(0.06, mul(0.24, brightness)), overtone),
        ))

        outputs.push(y)
        nextAmp.push(amp)
        nextPhase.push(phase)
      }

      return {
        outputs: { out: arrayPack(outputs) },
        nextRegs: { amp: arrayPack(nextAmp), phase: arrayPack(nextPhase) },
      }
    },
    44100.0,
    {
      input:      new Array(nodeCount).fill(0.0),
      fc:         fcValues,
      g:          0.035,
      decay:      0.9997,
      brightness: 0.88,
    },
  )
}

// ─── VCA ──────────────────────────────────────────────────────────────────────

export function vca(name = 'VCA'): ModuleType {
  return defineModule(
    name,
    [{ name: 'audio', type: 'float' }, { name: 'cv', type: 'float' }],
    [{ name: 'out', type: 'float' }],
    {},
    (inp) => ({
      outputs: { out: mul(inp.get('audio'), inp.get('cv')) },
      nextRegs: {},
    }),
    44100.0,
    { audio: 0.0, cv: 0.0 },
  )
}

// ─── BitCrusher ──────────────────────────────────────────────────────────────

export function bitCrusher(name = 'BitCrusher'): ModuleType {
  return defineModule(
    name,
    ['audio', 'bit_depth', 'sample_rate_hz'],
    ['output'],
    { hold_sample: 0.0, hold_counter: 0.0 },
    (inp, reg) => {
      const sr = sampleRate()

      // Clamp parameters: bit_depth ∈ [1, 24], sample_rate_hz ∈ [1, sr]
      const bd       = clamp(inp.get('bit_depth'), 1.0, 24.0)
      const targetSr = clamp(inp.get('sample_rate_hz'), 1.0, sr)

      // Quantisation levels for bit-depth reduction
      const levels    = pow_(2.0, sub(bd, 1.0))
      const audio     = inp.get('audio')
      const quantized = div(floorDiv(add(mul(audio, levels), 0.5), 1.0), levels)

      // Sample-and-hold: capture a new quantised sample every N input samples
      const samplesPerHold = clamp(floorDiv(sr, targetSr), 1.0, 44100.0)
      const counter        = reg.get('hold_counter')
      const incremented    = add(counter, 1.0)
      const shouldCapture  = gte(incremented, samplesPerHold)

      const nextHold    = select(shouldCapture, quantized, reg.get('hold_sample'))
      const nextCounter = select(shouldCapture, 0.0, incremented)

      return {
        outputs:  { output: nextHold },
        nextRegs: { hold_sample: nextHold, hold_counter: nextCounter },
      }
    },
    44100.0,
    { audio: 0.0, bit_depth: 24.0, sample_rate_hz: 44100.0 },
  )
}

// ─── Ladder Filter ───────────────────────────────────────────────────────────

export function ladderFilter(name = 'LadderFilter'): ModuleType {
  const E = 2.718281828459045

  return defineModule(
    name,
    ['input', 'cutoff', 'resonance', 'drive'],
    ['lp', 'bp', 'hp', 'notch'],
    { s1: 0.0, s2: 0.0, s3: 0.0, s4: 0.0 },
    (inp, reg) => {
      const sr = sampleRate()
      const cutoff = clamp(inp.get('cutoff'), 20.0, 20000.0)
      const reso = mul(clamp(inp.get('resonance'), 0.0, 1.0), 4.0)
      const drive = clamp(inp.get('drive'), 1.0, 10.0)
      const raw = inp.get('input')
      const driven = mul(raw, drive)

      // 2x oversampled one-pole coefficient
      const g = sub(1.0, pow_(E, neg(div(mul(TWO_PI, cutoff), mul(sr, 2.0)))))

      let s1: ExprCoercible = reg.get('s1')
      let s2: ExprCoercible = reg.get('s2')
      let s3: ExprCoercible = reg.get('s3')
      let s4: ExprCoercible = reg.get('s4')

      // 2x oversampled Huovilainen ladder with tanh saturation
      for (let i = 0; i < 2; i++) {
        const fb = mul(reso, tanh(s4))
        const u = sub(driven, fb)
        s1 = add(s1, mul(g, sub(tanh(u),  tanh(s1))))
        s2 = add(s2, mul(g, sub(tanh(s1), tanh(s2))))
        s3 = add(s3, mul(g, sub(tanh(s2), tanh(s3))))
        s4 = add(s4, mul(g, sub(tanh(s3), tanh(s4))))
      }

      const lp = s4 as SignalExpr      // 24dB/oct lowpass
      const bp = s2 as SignalExpr       // 12dB/oct (bandpass character with resonance)
      const hp = sub(raw, lp)           // highpass (input minus lowpass)
      const notch = sub(raw, bp)        // notch (input minus bandpass)

      return {
        outputs: { lp, bp, hp, notch },
        nextRegs: { s1: s1 as SignalExpr, s2: s2 as SignalExpr, s3: s3 as SignalExpr, s4: s4 as SignalExpr },
      }
    },
    44100.0,
    { input: 0.0, cutoff: 1000.0, resonance: 0.5, drive: 1.0 },
  )
}

// ─── Noise LFSR ──────────────────────────────────────────────────────────────

export function noiseLFSR(name = 'NoiseLFSR'): ModuleType {
  return defineModule(
    name,
    ['clock'],
    ['out'],
    { state: { init: 0xACE1, type: 'int' }, value: 0.0 },
    (inp, reg) => {
      const clock = inp.get('clock')
      const prevClock = delay(clock, 0.0)
      const tick = mul(gt(clock, 0.5), lte(prevClock, 0.5))

      const state = reg.get('state')

      // Galois LFSR 16-bit: x^16 + x^14 + x^13 + x^11 + 1
      const lsb = bitAnd(state, 1)
      const shifted = rshift(state, 1)
      const newState = select(lsb, bitXor(shifted, 0xB400), shifted)

      // Normalize to [-1, 1]
      const normalized = sub(div(mul(newState, 2.0), 65535.0), 1.0)

      // Sample-and-hold: advance on rising edge, hold between ticks
      const prevValue = reg.get('value')
      const outValue = select(tick, normalized, prevValue)

      return {
        outputs: { out: outValue },
        nextRegs: {
          state: select(tick, newState, state),
          value: outValue,
        },
      }
    },
    44100.0,
    { clock: 0.0 },
  )
}

// ─── Builtin registry ─────────────────────────────────────────────────────────

/** Canonical type names shipped with the library. */
export const BUILTIN_NAMES = [
  'VCO', 'Phaser', 'Phaser16', 'Clock',
  'Reverb', 'ADEnvelope', 'ADSREnvelope', 'Compressor', 'BassDrum', 'TopoWaveguide',
  'VCA', 'BitCrusher', 'LadderFilter', 'NoiseLFSR',
  'Delay8', 'Delay16', 'Delay512', 'Delay4410', 'Delay44100',
] as const

/** Load all builtin module types into a type registry. */
export function loadBuiltins(typeRegistry: Map<string, ModuleType>): void {
  typeRegistry.set('VCO',           vco())
  typeRegistry.set('Phaser',        phaser())
  typeRegistry.set('Phaser16',      phaser16())
  typeRegistry.set('Clock',         clock())
  typeRegistry.set('Reverb',        reverb())
  typeRegistry.set('ADEnvelope',    adEnvelope())
  typeRegistry.set('ADSREnvelope',  adsrEnvelope())
  typeRegistry.set('Compressor',    compressor())
  typeRegistry.set('BassDrum',      bassDrum())
  typeRegistry.set('TopoWaveguide', topoWaveguide())
  typeRegistry.set('VCA',           vca())
  typeRegistry.set('BitCrusher',    bitCrusher())
  typeRegistry.set('LadderFilter',  ladderFilter())
  typeRegistry.set('NoiseLFSR',    noiseLFSR())
  // Common delay lengths
  typeRegistry.set('Delay8',     delayLine(8,     'Delay8'))
  typeRegistry.set('Delay16',    delayLine(16,    'Delay16'))
  typeRegistry.set('Delay512',   delayLine(512,   'Delay512'))
  typeRegistry.set('Delay4410',  delayLine(4410,  'Delay4410'))
  typeRegistry.set('Delay44100', delayLine(44100, 'Delay44100'))
}
