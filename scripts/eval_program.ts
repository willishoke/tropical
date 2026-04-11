/**
 * eval_patch.ts — Pure-JS interpreter for a tropical_plan_4 flat plan.
 * Compiles a patch through the TS pipeline and runs it sample-by-sample,
 * flagging blowup and reporting where it first occurs.
 *
 * Usage: bun scripts/eval_patch.ts <patch.json> [n_samples] [sample_rate]
 */

import { readFileSync } from 'node:fs'
import { makeSession, loadJSON } from '../compiler/session.js'
import { loadStdlib }           from '../compiler/program.js'
import { flattenSession }         from '../compiler/flatten.js'
import type { FlatPlan }        from '../compiler/flatten.js'
import type { NOperand }        from '../compiler/emit_numeric.js'

// ─── CLI ─────────────────────────────────────────────────────────────────────

const patchPath  = process.argv[2]
const nSamples   = parseInt(process.argv[3] ?? '88200', 10)
const sampleRate = parseFloat(process.argv[4] ?? '44100')
const BLOW_THRESHOLD = 100   // flag output when |val| exceeds this

if (!patchPath) {
  console.error('Usage: bun scripts/eval_patch.ts <patch.json> [n_samples] [sample_rate]')
  process.exit(1)
}

// ─── Compile ─────────────────────────────────────────────────────────────────

const session = makeSession()
loadStdlib(session)
loadJSON(JSON.parse(readFileSync(patchPath, 'utf-8')), session)
const plan: FlatPlan = flattenSession(session)

console.log(`Compiled: ${plan.instructions.length} instrs, ` +
            `${plan.state_init.length} state regs, ` +
            `${plan.array_slot_count} array slots, ` +
            `${plan.output_targets.length} outputs`)
console.log()

// ─── State ───────────────────────────────────────────────────────────────────

const nTemps = plan.register_count
const temps  = new Float64Array(nTemps)

// Persistent state registers
const state  = new Float64Array(plan.state_init.length)
for (let i = 0; i < plan.state_init.length; i++) state[i] = Number(plan.state_init[i])

// Array slots — each is a mutable Float64Array
const arrayData: Float64Array[] = plan.array_slot_sizes.map(sz => new Float64Array(sz))

// ─── Operand resolution ───────────────────────────────────────────────────────

let sampleIdx = 0

function resolveScalar(op: NOperand, loopI = 0, stride = 1): number {
  switch (op.kind) {
    case 'const':     return op.val
    case 'reg':       return temps[op.slot]
    case 'state_reg': return state[op.slot]
    case 'rate':      return sampleRate
    case 'tick':      return sampleIdx
    case 'input':     return 0
    case 'param':     return 0
    case 'array_reg': {
      const arr = arrayData[op.slot]
      const idx = stride === 0 ? 0 : loopI
      return arr[idx % arr.length]
    }
    default: return 0
  }
}

// ─── Op implementations ───────────────────────────────────────────────────────

const warnedOps = new Set<string>()

function applyScalarOp(tag: string, v: number[]): number {
  switch (tag) {
    case 'Add':      return v[0] + v[1]
    case 'Sub':      return v[0] - v[1]
    case 'Mul':      return v[0] * v[1]
    case 'Div':      return v[1] === 0 ? 0 : v[0] / v[1]
    case 'Mod':      return v[1] === 0 ? 0 : v[0] % v[1]
    case 'Pow':      return Math.pow(v[0], v[1])
    case 'FloorDiv': return v[1] === 0 ? 0 : Math.floor(v[0] / v[1])
    case 'Neg':      return -v[0]
    case 'Abs':      return Math.abs(v[0])
    case 'Sin':      return Math.sin(v[0])
    case 'Cos':      return Math.cos(v[0])
    case 'Tanh':     return Math.tanh(v[0])
    case 'Log':      return v[0] <= 0 ? -1e38 : Math.log(v[0])
    case 'Exp':      return Math.exp(v[0])
    case 'Sqrt':     return v[0] < 0 ? 0 : Math.sqrt(v[0])
    case 'Floor':    return Math.floor(v[0])
    case 'Ceil':     return Math.ceil(v[0])
    case 'Round':    return Math.round(v[0])
    case 'Less':     return v[0] <  v[1] ? 1 : 0
    case 'LessEq':   return v[0] <= v[1] ? 1 : 0
    case 'Greater':  return v[0] >  v[1] ? 1 : 0
    case 'GreaterEq':return v[0] >= v[1] ? 1 : 0
    case 'Equal':    return v[0] === v[1] ? 1 : 0
    case 'NotEqual': return v[0] !== v[1] ? 1 : 0
    case 'And':      return (v[0] !== 0 && v[1] !== 0) ? 1 : 0
    case 'Or':       return (v[0] !== 0 || v[1] !== 0) ? 1 : 0
    case 'Not':      return v[0] === 0 ? 1 : 0
    case 'BitAnd':   return (v[0] | 0) & (v[1] | 0)
    case 'BitOr':    return (v[0] | 0) | (v[1] | 0)
    case 'BitXor':   return (v[0] | 0) ^ (v[1] | 0)
    case 'LShift':   return (v[0] | 0) << (v[1] | 0)
    case 'RShift':   return (v[0] | 0) >> (v[1] | 0)
    case 'BitNot':   return ~(v[0] | 0)
    case 'Clamp':    return Math.max(v[1], Math.min(v[2], v[0]))
    case 'Select':   return v[0] !== 0 ? v[1] : v[2]
    // Index: read element from array slot
    // args: [array_reg, index]
    case 'Index': {
      const arr = arrayData[(plan.instructions[currentInstr].args[0] as { kind: 'array_reg'; slot: number }).slot]
      const idx = (v[1] | 0 + arr.length) % arr.length
      return arr[idx < 0 ? idx + arr.length : idx]
    }
    default:
      if (!warnedOps.has(tag)) { warnedOps.add(tag); console.warn(`  [warn] unimplemented op: ${tag}`) }
      return 0
  }
}

// ─── Instruction execution ────────────────────────────────────────────────────

let currentInstr = 0

function execInstrs(): void {
  const N = plan.instructions.length
  for (let i = 0; i < N; i++) {
    currentInstr = i
    const instr = plan.instructions[i]
    const { tag, dst, args, loop_count, strides } = instr

    // Pack: write scalar args into array slot
    if (tag === 'Pack') {
      const arr = arrayData[dst]
      for (let k = 0; k < args.length; k++)
        arr[k] = resolveScalar(args[k])
      continue
    }

    // SetElement: functional array update — copy src, set one index, write to dst array slot
    if (tag === 'SetElement') {
      // args: [src_array_reg, index, value]
      const srcSlot = (args[0] as { kind: 'array_reg'; slot: number }).slot
      const src  = arrayData[srcSlot]
      const idx  = (resolveScalar(args[1]) | 0 + src.length) % src.length
      const val  = resolveScalar(args[2])
      const dst_arr = arrayData[dst]
      if (dst_arr !== src) dst_arr.set(src)
      dst_arr[idx < 0 ? idx + src.length : idx] = val
      continue
    }

    // Elementwise loop (e.g. map over array)
    if (loop_count > 1) {
      const arr = arrayData[dst]
      for (let k = 0; k < loop_count; k++) {
        const vals = args.map((op, ai) => resolveScalar(op, k, strides[ai] ?? 1))
        arr[k] = applyScalarOp(tag, vals)
      }
      continue
    }

    // Scalar instruction
    const vals = args.map(op => resolveScalar(op))
    temps[dst] = applyScalarOp(tag, vals)
  }
}

// ─── Run ─────────────────────────────────────────────────────────────────────

const nOut = plan.output_targets.length
const peakOut = new Float64Array(nOut)

// First blowup: [sampleIdx, outputIndex, value]
let blowSample = -1
let blowOutIdx = -1
let blowVal    = 0

const REPORT_EVERY = Math.max(1, Math.floor(nSamples / 20))

for (sampleIdx = 0; sampleIdx < nSamples; sampleIdx++) {
  temps.fill(0)
  execInstrs()

  // Read outputs and track peaks / blowup
  for (let o = 0; o < nOut; o++) {
    const v = temps[plan.output_targets[o]]
    const av = Math.abs(v)
    if (av > peakOut[o]) peakOut[o] = av
    if (blowSample === -1 && (!isFinite(v) || av > BLOW_THRESHOLD)) {
      blowSample = sampleIdx
      blowOutIdx = o
      blowVal    = v
    }
  }

  // Periodic report
  if (sampleIdx % REPORT_EVERY === 0) {
    const snap = Array.from({ length: nOut }, (_, o) => {
      const v = temps[plan.output_targets[o]]
      return Math.abs(v) > BLOW_THRESHOLD ? `!!!${v.toExponential(2)}` : v.toFixed(3)
    }).join('  ')
    console.log(`s=${String(sampleIdx).padStart(6)}  [${snap}]`)
  }

  // Advance state registers
  for (let r = 0; r < plan.register_targets.length; r++) {
    const src = plan.register_targets[r]
    if (src >= 0) state[r] = temps[src]
  }

  if (blowSample !== -1 && sampleIdx > blowSample + 512) break
}

// ─── Report ──────────────────────────────────────────────────────────────────

console.log()

if (blowSample === -1) {
  console.log(`✓  No blowup (threshold ±${BLOW_THRESHOLD}) in ${nSamples} samples.`)
} else {
  const sec = (blowSample / sampleRate).toFixed(3)
  console.log(`✗  Blowup at sample ${blowSample} (${sec}s) — output[${blowOutIdx}] = ${blowVal}`)
  console.log()

  // Show which state regs are already extreme at the blowup point
  console.log('  State register snapshot (non-zero / extreme):')
  for (let r = 0; r < state.length; r++) {
    const v = state[r]
    const name = plan.register_names[r] ?? `state[${r}]`
    if (!isFinite(v) || Math.abs(v) > 1.5)
      console.log(`    ${name.padEnd(40)} = ${v}  ← !!!`)
    else if (Math.abs(v) > 0.001)
      console.log(`    ${name.padEnd(40)} = ${v.toFixed(6)}`)
  }
}

console.log()
console.log('Peak output levels:')
for (let o = 0; o < nOut; o++) {
  const flag = peakOut[o] > BLOW_THRESHOLD ? '  ← BLOWN' : peakOut[o] > 1.0 ? '  ← clipping' : ''
  console.log(`  output[${String(o).padStart(2)}]: peak = ${peakOut[o].toFixed(4)}${flag}`)
}
