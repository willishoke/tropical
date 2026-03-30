/**
 * Preset module types. Port of egress/module_library.py.
 */

import {
  SignalExpr, ExprCoercible,
  add, sub, mul, div, mod, pow_, neg,
  lt, lte, gt, gte,
  abs_, sin, log,
  clamp, arrayPack, arraySet, matmul,
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
    ['freq', 'fm', 'fm_index'],
    ['saw', 'tri', 'sin', 'sqr'],
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
    ['freq', 'ratios_in'],
    ['output', 'ratios_out'],
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
    { stage: 0.0, phase: 0.0 },
    (inp, reg) => {
      const sr      = sampleRate()
      const attack  = clamp(inp.get('attack'), 1e-4, 10.0)
      const decayT  = clamp(inp.get('decay'),  1e-4, 10.0)
      const gate    = inp.get('gate')

      const prevGate = delay(gate, 0.0)
      const trig     = mul(gt(gate, 0.5), lte(prevGate, 0.5))

      const stage = reg.get('stage')
      const phase = reg.get('phase')

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

      const rawEnv = add(mul(inAttack, phase), mul(inDecay, sub(1.0, phase)))

      const blampStart    = one(_polyBlamp.call(phase,           dtA))
      const blampAtkEnd   = one(_polyBlamp.call(sub(1.0, phase), dtA))
      const blampDecStart = one(_polyBlamp.call(phase,           dtD))
      const blampEnd      = one(_polyBlamp.call(sub(1.0, phase), dtD))

      let env: ExprCoercible = rawEnv
      env = sub(env, mul(mul(inAttack, dtA), blampStart))
      env = add(env, mul(mul(inAttack, dtA), blampAtkEnd))
      env = add(env, mul(mul(inDecay,  dtD), blampDecStart))
      env = sub(env, mul(mul(inDecay,  dtD), blampEnd))

      return {
        outputs: { env: clamp(env as SignalExpr, 0.0, 1.0) },
        nextRegs: { stage: nextStage, phase: nextPhase },
      }
    },
    44100.0,
    { gate: 0.0, attack: 0.01, decay: 0.3 },
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
    ['audio', 'cv'],
    ['out'],
    {},
    (inp) => ({
      outputs: { out: mul(inp.get('audio'), inp.get('cv')) },
      nextRegs: {},
    }),
    44100.0,
    { audio: 0.0, cv: 0.0 },
  )
}

// ─── Builtin registry ─────────────────────────────────────────────────────────

/** Canonical type names shipped with the library. */
export const BUILTIN_NAMES = [
  'VCO', 'Phaser', 'Phaser16', 'Clock',
  'Reverb', 'ADEnvelope', 'Compressor', 'BassDrum', 'TopoWaveguide',
  'VCA',
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
  typeRegistry.set('Compressor',    compressor())
  typeRegistry.set('BassDrum',      bassDrum())
  typeRegistry.set('TopoWaveguide', topoWaveguide())
  typeRegistry.set('VCA',           vca())
  // Common delay lengths
  typeRegistry.set('Delay8',     delayLine(8,     'Delay8'))
  typeRegistry.set('Delay16',    delayLine(16,    'Delay16'))
  typeRegistry.set('Delay512',   delayLine(512,   'Delay512'))
  typeRegistry.set('Delay4410',  delayLine(4410,  'Delay4410'))
  typeRegistry.set('Delay44100', delayLine(44100, 'Delay44100'))
}
