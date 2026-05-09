/**
 * interpret_resolved.ts — pure-TS interpreter that walks `ResolvedExpr`.
 *
 * The independent-oracle half of `jit_interp_equiv`: the JIT consumes
 * `tropical_plan_4` (an instruction stream emitted from the resolved
 * IR), this interpreter consumes the same `ResolvedProgram` directly.
 * Both run the same strata pipeline + materialization through
 * `materializeSessionToResolvedIR`, so the cross-check tests
 * end-to-end semantic agreement of the IR against the emitter.
 *
 * Phase D D3-b: replaces the retired `interpret.ts` (which walked
 * Expr produced by the legacy `flatten.ts:flattenExpressions`).
 *
 * No FFI, no C++ dependency. Pure TS, fully deterministic.
 */

import type {
  ResolvedExpr, ResolvedExprOp, ResolvedProgram,
  RegDecl, DelayDecl, InputDecl, ParamDecl,
} from './ir/nodes.js'
import type { SessionState } from './session.js'
import { materializeSessionToResolvedIR } from './ir/materialize_session.js'

// ─────────────────────────────────────────────────────────────
// Environment
// ─────────────────────────────────────────────────────────────

type Value = number | boolean | number[]

interface InterpretEnv {
  sampleRate: number
  sampleIndex: number
  /** Decl-identity-keyed state. Reads see the *current* sample's state;
   *  next-sample state is computed into shadow maps and atomically
   *  swapped at sample end. */
  regs:    Map<RegDecl,    Value>
  delays:  Map<DelayDecl,  Value>
  /** Top-level synthetic program has no inputs (audio inputs are a host
   *  concern, not in the model today), but lifted-from-instance inner
   *  inputRefs have already been substituted by `inlineInstances`, so
   *  this only fires on the rare case of a literal inputRef surviving
   *  to the top level — which would be a strata bug, not a runtime
   *  case. Keep the slot for completeness. */
  inputs:  Map<InputDecl,  Value>
  params:  Map<ParamDecl,  number>
}

// ─────────────────────────────────────────────────────────────
// Helpers (mirrors interpret.ts coercions)
// ─────────────────────────────────────────────────────────────

function toNum(v: Value): number {
  if (typeof v === 'boolean') return v ? 1 : 0
  if (Array.isArray(v))       return v[0] ?? 0
  return v
}

function toInt(v: Value): number  { return Math.trunc(toNum(v)) }
function toBool(v: Value): boolean {
  if (typeof v === 'boolean') return v
  if (Array.isArray(v))       return (v[0] ?? 0) !== 0
  return v !== 0
}

/** Apply a binary fn elementwise, broadcasting scalar args. */
function binOp(a: Value, b: Value, fn: (x: number, y: number) => number | boolean): Value {
  const aArr = Array.isArray(a)
  const bArr = Array.isArray(b)
  if (!aArr && !bArr) return fn(toNum(a), toNum(b))
  if (aArr && bArr) {
    const aa = a as number[], bb = b as number[]
    const len = Math.max(aa.length, bb.length)
    const out = new Array<number>(len)
    for (let i = 0; i < len; i++) out[i] = toNum(fn(aa[i % aa.length], bb[i % bb.length]))
    return out
  }
  if (aArr) {
    const aa = a as number[], bv = toNum(b)
    return aa.map(x => toNum(fn(x, bv)))
  }
  const bb = b as number[], av = toNum(a)
  return bb.map(y => toNum(fn(av, y)))
}

function unOp(a: Value, fn: (x: number) => number | boolean): Value {
  if (Array.isArray(a)) return (a as number[]).map(x => toNum(fn(x)))
  return fn(toNum(a))
}

// ─────────────────────────────────────────────────────────────
// Core evaluator: walk ResolvedExpr against the env
// ─────────────────────────────────────────────────────────────

function evalExpr(node: ResolvedExpr, env: InterpretEnv): Value {
  if (typeof node === 'number')  return node
  if (typeof node === 'boolean') return node
  if (Array.isArray(node)) {
    return (node as ResolvedExpr[]).map(n => toNum(evalExpr(n, env)))
  }
  return evalOpNode(node as ResolvedExprOp, env)
}

function evalOpNode(node: ResolvedExprOp, env: InterpretEnv): Value {
  const recur = (e: ResolvedExpr): Value => evalExpr(e, env)

  switch (node.op) {
    // ── References ─────────────────────────────────────────
    case 'inputRef':
      return env.inputs.get(node.decl) ?? 0
    case 'regRef':
      return env.regs.get(node.decl) ?? regInitValue(node.decl)
    case 'delayRef':
      return env.delays.get(node.decl) ?? delayInitValue(node.decl)
    case 'paramRef':
      return env.params.get(node.decl) ?? node.decl.value ?? 0
    case 'typeParamRef':
      throw new Error(`interpret: typeParamRef '${node.decl.name}' should have been substituted by specialize`)
    case 'bindingRef':
      throw new Error(`interpret: bindingRef '${node.decl.name}' should have been substituted by array_lower`)
    case 'nestedOut':
      throw new Error(`interpret: nestedOut '${node.instance.name}.${node.output.name}' should have been inlined`)

    // ── Sentinels ──────────────────────────────────────────
    case 'sampleRate':  return env.sampleRate
    case 'sampleIndex': return env.sampleIndex

    // ── Binary arithmetic ──────────────────────────────────
    case 'add':       return binOp(recur(node.args[0]), recur(node.args[1]), (a, b) => a + b)
    case 'sub':       return binOp(recur(node.args[0]), recur(node.args[1]), (a, b) => a - b)
    case 'mul':       return binOp(recur(node.args[0]), recur(node.args[1]), (a, b) => a * b)
    case 'div':       return binOp(recur(node.args[0]), recur(node.args[1]), (a, b) => b !== 0 ? a / b : 0)
    case 'mod':       return binOp(recur(node.args[0]), recur(node.args[1]), (a, b) => b !== 0 ? a % b : 0)
    case 'ldexp':     return binOp(recur(node.args[0]), recur(node.args[1]), (a, b) => a * Math.pow(2, Math.trunc(b)))
    case 'floorDiv':  return binOp(recur(node.args[0]), recur(node.args[1]), (a, b) => b !== 0 ? Math.floor(a / b) : 0)

    // ── Binary comparison ──────────────────────────────────
    case 'lt':  return binOp(recur(node.args[0]), recur(node.args[1]), (a, b) => a < b)
    case 'lte': return binOp(recur(node.args[0]), recur(node.args[1]), (a, b) => a <= b)
    case 'gt':  return binOp(recur(node.args[0]), recur(node.args[1]), (a, b) => a > b)
    case 'gte': return binOp(recur(node.args[0]), recur(node.args[1]), (a, b) => a >= b)
    case 'eq':  return binOp(recur(node.args[0]), recur(node.args[1]), (a, b) => a === b)
    case 'neq': return binOp(recur(node.args[0]), recur(node.args[1]), (a, b) => a !== b)

    // ── Binary bitwise ─────────────────────────────────────
    case 'bitAnd':  return binOp(recur(node.args[0]), recur(node.args[1]), (a, b) => toInt(a) & toInt(b))
    case 'bitOr':   return binOp(recur(node.args[0]), recur(node.args[1]), (a, b) => toInt(a) | toInt(b))
    case 'bitXor':  return binOp(recur(node.args[0]), recur(node.args[1]), (a, b) => toInt(a) ^ toInt(b))
    case 'lshift':  return binOp(recur(node.args[0]), recur(node.args[1]), (a, b) => toInt(a) << toInt(b))
    case 'rshift':  return binOp(recur(node.args[0]), recur(node.args[1]), (a, b) => toInt(a) >> toInt(b))

    // ── Binary logical ─────────────────────────────────────
    case 'and': return binOp(recur(node.args[0]), recur(node.args[1]), (a, b) => toBool(a) && toBool(b))
    case 'or':  return binOp(recur(node.args[0]), recur(node.args[1]), (a, b) => toBool(a) || toBool(b))

    // ── Unary math ─────────────────────────────────────────
    case 'neg':           return unOp(recur(node.args[0]), x => -x)
    case 'abs':           return unOp(recur(node.args[0]), x => Math.abs(x))
    case 'sqrt':          return unOp(recur(node.args[0]), x => Math.sqrt(x))
    case 'floor':         return unOp(recur(node.args[0]), x => Math.floor(x))
    case 'ceil':          return unOp(recur(node.args[0]), x => Math.ceil(x))
    case 'round':         return unOp(recur(node.args[0]), x => Math.round(x))
    case 'floatExponent': return unOp(recur(node.args[0]), x => {
      if (x === 0 || !isFinite(x)) return 0
      return Math.floor(Math.log2(Math.abs(x)))
    })
    case 'toInt':   return unOp(recur(node.args[0]), x => Math.trunc(x))
    case 'toBool':  return unOp(recur(node.args[0]), x => x !== 0)
    case 'toFloat': return unOp(recur(node.args[0]), x => x)

    // ── Unary logical/bitwise ──────────────────────────────
    case 'not':    return unOp(recur(node.args[0]), x => !toBool(x))
    case 'bitNot': return unOp(recur(node.args[0]), x => ~toInt(x))

    // ── Ternary ────────────────────────────────────────────
    case 'select': {
      const cond = recur(node.args[0])
      const then_ = recur(node.args[1])
      const else_ = recur(node.args[2])
      if (Array.isArray(cond)) {
        const c = cond as number[]
        const n = c.length
        const out = new Array<number>(n)
        for (let i = 0; i < n; i++) {
          out[i] = toBool(c[i])
            ? toNum(Array.isArray(then_) ? (then_ as number[])[i] ?? 0 : then_)
            : toNum(Array.isArray(else_) ? (else_ as number[])[i] ?? 0 : else_)
        }
        return out
      }
      return toBool(cond) ? then_ : else_
    }
    case 'clamp': {
      const v = recur(node.args[0])
      const lo = recur(node.args[1])
      const hi = recur(node.args[2])
      if (Array.isArray(v)) {
        return (v as number[]).map((x, i) => {
          const l = toNum(Array.isArray(lo) ? (lo as number[])[i] ?? 0 : lo)
          const h = toNum(Array.isArray(hi) ? (hi as number[])[i] ?? 0 : hi)
          return Math.min(Math.max(x, l), h)
        })
      }
      return Math.min(Math.max(toNum(v), toNum(lo)), toNum(hi))
    }

    // ── Index ──────────────────────────────────────────────
    case 'index': {
      const arr = recur(node.args[0])
      const idx = toInt(recur(node.args[1]))
      if (Array.isArray(arr)) return (arr as number[])[idx] ?? 0
      return arr   // scalar-indexed-as-array semantics: return the scalar
    }

    // ── Array ops (post-arrayLower these should be unrolled, but the
    //    elaborator may leave a literal `zeros{count}` as a reg init — eval
    //    it as an array literal here). ───────────────────────────────────
    case 'zeros': {
      const c = recur(node.count)
      const n = typeof c === 'number' ? c : toNum(c)
      return new Array<number>(n).fill(0)
    }
    case 'arraySet': {
      const arr = recur(node.args[0])
      const idx = toInt(recur(node.args[1]))
      const val = toNum(recur(node.args[2]))
      if (Array.isArray(arr)) {
        const copy = [...(arr as number[])]
        if (idx >= 0 && idx < copy.length) copy[idx] = val
        return copy
      }
      return val
    }

    // ── Combinators / let / ADT — should all be lowered out before
    //    interpret runs. Treat reaching one as a strata bug. ──────────
    case 'fold':
    case 'scan':
    case 'generate':
    case 'iterate':
    case 'chain':
    case 'map2':
    case 'zipWith':
    case 'let':
    case 'tag':
    case 'match':
      throw new Error(`interpret: '${node.op}' should have been lowered (strata bug?)`)
  }
}

function regInitValue(d: RegDecl): Value {
  const init = d.init
  if (typeof init === 'number')  return init
  if (typeof init === 'boolean') return init
  if (Array.isArray(init))       return init as number[]
  // {op:'zeros', count:N} — eval it
  if (typeof init === 'object' && init !== null && (init as { op?: string }).op === 'zeros') {
    const count = (init as { count: ResolvedExpr }).count
    if (typeof count !== 'number') throw new Error('interpret: regDecl init zeros count must be a literal int')
    return new Array<number>(count).fill(0)
  }
  return 0
}

function delayInitValue(d: DelayDecl): Value {
  const init = d.init
  if (typeof init === 'number')  return init
  if (typeof init === 'boolean') return init
  return 0
}

// ─────────────────────────────────────────────────────────────
// Sample runner
// ─────────────────────────────────────────────────────────────

/**
 * Interpret a session for `nSamples`, returning the audio output buffer.
 * Mirrors `FlatRuntime` execution order:
 *   1. Evaluate output expressions (audio outputs only)
 *   2. Evaluate register / delay update expressions
 *   3. Atomic writeback
 *   4. Mix and scale (sum / 20.0, matching the engine's gain compensation)
 *   5. Advance sampleIndex
 *
 * Audio is summed across `dac.out` outputs and divided by 20.0 — same
 * fade-compensation factor `FlatRuntime` applies on the C++ side.
 */
export function interpretSession(
  session: SessionState,
  nSamples: number,
  params?: ReadonlyMap<string, number>,
): Float64Array {
  const prog = materializeSessionToResolvedIR(session)

  // Build initial state from decl init values.
  const env: InterpretEnv = {
    sampleRate: 44100,
    sampleIndex: 0,
    regs:    new Map(),
    delays:  new Map(),
    inputs:  new Map(),
    params:  new Map(),
  }
  for (const d of prog.body.decls) {
    if (d.op === 'regDecl')   env.regs.set(d, regInitValue(d))
    else if (d.op === 'delayDecl') env.delays.set(d, delayInitValue(d))
    else if (d.op === 'paramDecl') {
      const v = params?.get(d.name) ?? d.value ?? 0
      env.params.set(d, v)
    }
  }

  // Find audio outputs: the OutputDecls in `ports.outputs` whose names
  // are `${instance}.${output}` (synthesized by `compile_session.ts`
  // step 3). Their outputAssigns drive the audio mix.
  const outputAssignByDecl = new Map<typeof prog.ports.outputs[number], ResolvedExpr>()
  for (const a of prog.body.assigns) {
    if (a.op !== 'outputAssign') continue
    if (!('op' in a.target)) continue
    if (a.target.op !== 'outputDecl') continue
    outputAssignByDecl.set(a.target, a.expr)
  }
  const outputExprs: ResolvedExpr[] = prog.ports.outputs.map(out => {
    const expr = outputAssignByDecl.get(out)
    if (expr === undefined) throw new Error(`interpret: output '${out.name}' has no outputAssign`)
    return expr
  })

  // Reg/delay updates: map decl → update expression. Register updates
  // come from nextUpdate assigns (or null for hold-current); delay updates
  // come from either nextUpdate (override) or the decl's own `update` field.
  const regUpdateByDecl   = new Map<RegDecl, ResolvedExpr | null>()
  const delayUpdateByDecl = new Map<DelayDecl, ResolvedExpr>()
  for (const a of prog.body.assigns) {
    if (a.op !== 'nextUpdate') continue
    if (a.target.op === 'regDecl')   regUpdateByDecl.set(a.target, a.expr)
    if (a.target.op === 'delayDecl') delayUpdateByDecl.set(a.target, a.expr)
  }
  for (const d of prog.body.decls) {
    if (d.op === 'regDecl' && !regUpdateByDecl.has(d)) regUpdateByDecl.set(d, null)
    if (d.op === 'delayDecl' && !delayUpdateByDecl.has(d)) delayUpdateByDecl.set(d, d.update)
  }

  const output = new Float64Array(nSamples)

  for (let s = 0; s < nSamples; s++) {
    env.sampleIndex = s

    // 1. Evaluate output expressions for the audio mix.
    let mixed = 0
    for (const expr of outputExprs) {
      const v = evalExpr(expr, env)
      mixed += toNum(v)
    }

    // 2. Evaluate register/delay updates against the current state.
    const newRegs   = new Map<RegDecl, Value>()
    const newDelays = new Map<DelayDecl, Value>()
    for (const [d, update] of regUpdateByDecl) {
      if (update === null) newRegs.set(d, env.regs.get(d) ?? regInitValue(d))
      else newRegs.set(d, evalExpr(update, env))
    }
    for (const [d, update] of delayUpdateByDecl) {
      newDelays.set(d, evalExpr(update, env))
    }

    // 3. Atomic writeback.
    for (const [d, v] of newRegs)   env.regs.set(d, v)
    for (const [d, v] of newDelays) env.delays.set(d, v)

    // 4. Mix + scale (matches the C++ runtime's /20 gain compensation).
    output[s] = mixed / 20.0
  }

  return output
}
