/**
 * inline_instances.test.ts — Phase C5 unit tests.
 *
 * Each test starts from a parsed surface-syntax fixture, elaborates to
 * `ResolvedProgram`, runs `inlineInstances`, and asserts on the
 * structural outcome: instance decls dropped, register/delay decls
 * lifted with renamed prefixes, NestedOut refs replaced.
 *
 * The fixtures cover:
 *   - one instance, no nesting
 *   - two instances composed (A's output wires into B's input)
 *   - nested generic Delay<N> resolution
 *   - output-port reference replacement
 *   - sub-instances (instance whose program contains another instance)
 *   - lifted reg/delay name discipline (`${instance}_${innerName}`)
 */

import { describe, test, expect } from 'bun:test'
import { parseProgram } from '../parse/declarations.js'
import { elaborate } from './elaborator.js'
import { inlineInstances } from './inline_instances.js'
import type { ResolvedProgram, BodyDecl, RegDecl } from './nodes.js'

function elab(src: string): ResolvedProgram {
  return elaborate(parseProgram(src))
}

/** Walk a resolved object graph (cycle-safe) and return every `op`
 *  present in the targets set. Used to assert that a pass eliminated
 *  certain ops (e.g. inline drops every `nestedOut`).
 *
 *  programDecl bodies are *passive type bindings* — they record the
 *  inner program template so other decls can reference it, but the
 *  runtime never evaluates them. Inline doesn't rewrite programDecl
 *  bodies, so this walker skips into them by default. Tests asserting
 *  on the outer's evaluated state should not see those interiors. */
function findOps(root: unknown, targets: string[]): string[] {
  const set = new Set(targets)
  const out: string[] = []
  const seen = new WeakSet<object>()
  const walk = (v: unknown): void => {
    if (v === null || typeof v !== 'object') return
    if (seen.has(v as object)) return
    seen.add(v as object)
    if (Array.isArray(v)) { v.forEach(walk); return }
    const o = v as Record<string, unknown>
    if (typeof o.op === 'string' && set.has(o.op)) out.push(o.op)
    // Skip programDecl bodies (passive type bindings; not evaluated).
    if (o.op === 'programDecl') return
    for (const k of Object.keys(o)) walk(o[k])
  }
  walk(root)
  return out
}

function regDecls(p: ResolvedProgram): RegDecl[] {
  return p.body.decls.filter((d): d is RegDecl => d.op === 'regDecl')
}

/** Post-Phase-0a: former DelayDecls are RegDecls with update populated. */
function delayLikeRegs(p: ResolvedProgram): RegDecl[] {
  return p.body.decls.filter(
    (d): d is RegDecl => d.op === 'regDecl' && d.update !== undefined,
  )
}

function declNames(decls: BodyDecl[]): string[] {
  return decls.map(d => `${d.op}:${d.name}`)
}

describe('inlineInstances — basic shapes', () => {
  test('passthrough: program with no instances returns input by reference', () => {
    const p = elab(`
      program X(a: float) -> (out: float) {
        reg s: float = 0
        out = s + a
        next s = a
      }
    `)
    expect(inlineInstances(p)).toBe(p)
  })

  test('one instance, no nesting: NestedOut replaced, instance dropped', () => {
    const p = elab(`
      program X(a: float) -> (out: float) {
        program Inner(x: float) -> (y: float) { y = x + 1 }
        inst = Inner(x: a)
        out = inst.y
      }
    `)
    const out = inlineInstances(p)
    // No InstanceDecl, no NestedOut.
    expect(findOps(out, ['instanceDecl'])).toEqual([])
    expect(findOps(out, ['nestedOut'])).toEqual([])
    // The output expression is now (a + 1) — the inner's body with
    // input `x` substituted by the wired `a`.
    expect(out.body.assigns).toHaveLength(1)
    const assign = out.body.assigns[0]
    if (assign.op !== 'outputAssign') throw new Error('expected outputAssign')
    const expr = assign.expr
    expect(typeof expr === 'object' && (expr as { op: string }).op).toBe('add')
  })

  test('two instances composed: A.out wired into B.in', () => {
    const p = elab(`
      program X(a: float) -> (out: float) {
        program A(x: float) -> (y: float) { y = x * 2 }
        program B(z: float) -> (w: float) { w = z + 10 }
        i1 = A(x: a)
        i2 = B(z: i1.y)
        out = i2.w
      }
    `)
    const out = inlineInstances(p)
    expect(findOps(out, ['instanceDecl'])).toEqual([])
    expect(findOps(out, ['nestedOut'])).toEqual([])
    // The composition (a * 2) + 10 should be reachable as the output
    // expression. We just verify shape: a top-level `add`.
    expect(out.body.assigns).toHaveLength(1)
    const assign = out.body.assigns[0]
    if (assign.op !== 'outputAssign') throw new Error('expected outputAssign')
    const expr = assign.expr
    expect(typeof expr === 'object' && (expr as { op: string }).op).toBe('add')
  })

  test('inner with reg: lifted reg renamed with instance prefix', () => {
    const p = elab(`
      program X(a: float) -> (out: float) {
        program Inner(x: float) -> (y: float) {
          reg s: float = 0
          y = s + x
          next s = x
        }
        inst = Inner(x: a)
        out = inst.y
      }
    `)
    const out = inlineInstances(p)
    expect(findOps(out, ['instanceDecl'])).toEqual([])
    // The inner's `s` is now lifted with prefix `inst_`.
    const regs = regDecls(out)
    const liftedS = regs.find(r => r.name === 'inst_s')
    expect(liftedS).toBeDefined()
    // Post-Phase-0a: the next-update for the lifted reg travels on the
    // decl's `update` field (no separate NextUpdate body-assign).
    expect(liftedS!.update).toBeDefined()
  })

  test('multiple instances of the same inner program: names disambiguate', () => {
    const p = elab(`
      program X(a: float, b: float) -> (out: float) {
        program Inner(x: float) -> (y: float) {
          reg s: float = 0
          y = s + x
          next s = x
        }
        i1 = Inner(x: a)
        i2 = Inner(x: b)
        out = i1.y + i2.y
      }
    `)
    const out = inlineInstances(p)
    expect(findOps(out, ['instanceDecl'])).toEqual([])
    const regs = regDecls(out)
    expect(regs.map(r => r.name).sort()).toEqual(['i1_s', 'i2_s'])
  })

  test('inner with delay: lifted delay-form reg renamed with instance prefix', () => {
    // Post-Phase-0a: `delay d = x init 0` is a RegDecl with update set.
    const p = elab(`
      program X(a: float) -> (out: float) {
        program Inner(x: float) -> (y: float) {
          delay d: float = x init 0
          y = d
        }
        inst = Inner(x: a)
        out = inst.y
      }
    `)
    const out = inlineInstances(p)
    expect(findOps(out, ['instanceDecl'])).toEqual([])
    const liftedDelays = delayLikeRegs(out)
    expect(liftedDelays.map(d => d.name)).toContain('inst_d')
  })

  test('nested instance: inner contains another instance', () => {
    const p = elab(`
      program X(a: float) -> (out: float) {
        program Leaf(x: float) -> (y: float) { y = x + 1 }
        program Mid(p: float) -> (q: float) {
          inner = Leaf(x: p)
          q = inner.y * 2
        }
        outer = Mid(p: a)
        out = outer.q
      }
    `)
    const out = inlineInstances(p)
    expect(findOps(out, ['instanceDecl'])).toEqual([])
    expect(findOps(out, ['nestedOut'])).toEqual([])
    // Output expression: (a + 1) * 2 — top is `mul`.
    const assign = out.body.assigns[0]
    if (assign.op !== 'outputAssign') throw new Error('expected outputAssign')
    expect((assign.expr as { op: string }).op).toBe('mul')
  })

  test('nested generic: outer instantiates FixedDelay<N=8>', () => {
    // A simple generic delay that allocates an array reg of size N.
    // After inlining + specialization, the lifted reg's init is a
    // zeros-of-size-8 expression.
    const p = elab(`
      program X(a: float) -> (out: float) {
        program FixedDelay<N: int = 4>(x: float) -> (y: float) {
          reg buf: float = zeros(N)
          y = buf[0]
          next buf = arraySet(buf, 0, x)
        }
        d = FixedDelay<N=8>(x: a)
        out = d.y
      }
    `)
    const out = inlineInstances(p)
    expect(findOps(out, ['instanceDecl'])).toEqual([])
    // The reg `d_buf` is lifted and its init is zeros{8}.
    const regs = regDecls(out)
    const buf = regs.find(r => r.name === 'd_buf')
    expect(buf).toBeDefined()
    if (!buf) throw new Error('unreachable')
    // After specialization, the typeParam `N` resolves to literal 8.
    const init = buf.init
    if (typeof init !== 'object' || Array.isArray(init)) throw new Error('expected zeros init')
    expect(init.op).toBe('zeros')
    if (init.op === 'zeros') expect(init.count).toBe(8)
  })

  test('lifted decls follow surviving outer decls in body order', () => {
    const p = elab(`
      program X(a: float) -> (out: float) {
        program Inner(x: float) -> (y: float) {
          reg t: float = 0
          y = t + x
          next t = x
        }
        reg own: float = 0
        inst = Inner(x: a)
        out = own + inst.y
        next own = a
      }
    `)
    const out = inlineInstances(p)
    expect(findOps(out, ['instanceDecl'])).toEqual([])
    // Outer's own reg comes first, then the lifted inner's reg.
    const names = declNames(out.body.decls).filter(s => s.startsWith('regDecl:'))
    expect(names).toEqual(['regDecl:own', 'regDecl:inst_t'])
  })
})

describe('inlineInstances — multi-instance + nested-generic interactions', () => {
  test('two instances of the same generic with different type-args', () => {
    const p = elab(`
      program X(a: float) -> (out: float) {
        program Buffer<N: int = 4>(x: float) -> (y: float) {
          reg buf: float = zeros(N)
          y = buf[0]
          next buf = arraySet(buf, 0, x)
        }
        d4 = Buffer<N=4>(x: a)
        d8 = Buffer<N=8>(x: a)
        out = d4.y + d8.y
      }
    `)
    const out = inlineInstances(p)
    expect(findOps(out, ['instanceDecl'])).toEqual([])
    const regs = regDecls(out)
    const d4buf = regs.find(r => r.name === 'd4_buf')
    const d8buf = regs.find(r => r.name === 'd8_buf')
    expect(d4buf).toBeDefined(); expect(d8buf).toBeDefined()
    if (d4buf && typeof d4buf.init === 'object' && !Array.isArray(d4buf.init) && d4buf.init.op === 'zeros') {
      expect(d4buf.init.count).toBe(4)
    } else { throw new Error('d4buf init not zeros{4}') }
    if (d8buf && typeof d8buf.init === 'object' && !Array.isArray(d8buf.init) && d8buf.init.op === 'zeros') {
      expect(d8buf.init.count).toBe(8)
    } else { throw new Error('d8buf init not zeros{8}') }
  })

  test('chained instances composed via NestedOut do not infinite-loop', () => {
    // The nested-out substitution must distinguish (instance, output)
    // pairs by InstanceDecl, not OutputDecl. Two instances of the
    // same program share OutputDecl objects, so substitution keyed
    // only by OutputDecl forms a self-cycle and recurses forever.
    const p = elab(`
      program X(input: float) -> (output: float) {
        program Stage(x: float) -> (y: float) {
          reg s: float = 0
          y = s + x
          next s = x
        }
        a = Stage(x: input)
        b = Stage(x: a.y)
        c = Stage(x: b.y)
        output = c.y
      }
    `)
    const out = inlineInstances(p)
    expect(findOps(out, ['instanceDecl'])).toEqual([])
    expect(findOps(out, ['nestedOut'])).toEqual([])
    // Three distinct lifted regs.
    expect(regDecls(out).map(r => r.name).sort()).toEqual(['a_s', 'b_s', 'c_s'])
  })
})

describe('inlineInstances — error and edge cases', () => {
  test('input default is used when the outer doesn\'t wire the port', () => {
    const p = elab(`
      program X(a: float) -> (out: float) {
        program Inner(x: float = 5, y: float) -> (z: float) { z = x + y }
        inst = Inner(y: a)
        out = inst.z
      }
    `)
    const out = inlineInstances(p)
    expect(findOps(out, ['instanceDecl'])).toEqual([])
    // The output expression should now be (5 + a).
    const assign = out.body.assigns[0]
    if (assign.op !== 'outputAssign') throw new Error('expected outputAssign')
    expect((assign.expr as { op: string }).op).toBe('add')
  })
})
