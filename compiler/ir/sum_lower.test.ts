/**
 * sum_lower.test.ts — coverage for the Phase C4 sum-type decomposer.
 *
 * Properties tested:
 *   1. A program with no sums returns its input by identity.
 *   2. A simple two-variant nullary enum (Toggle) lowers `match` to
 *      a chained `select` over the tag-slot read.
 *   3. A sum with a payload variant (EnvExpDecay-style) decomposes
 *      the sum-typed delay into N+1 scalar slots and rewrites
 *      payload-bearing match arms to read the per-variant field
 *      slots.
 */

import { describe, test, expect } from 'bun:test'
import { parseProgram } from '../parse/declarations.js'
import { elaborate } from './elaborator.js'
import { sumLower } from './sum_lower.js'
import type {
  ResolvedProgram, ResolvedExpr, ResolvedExprOp,
  DelayDecl, OutputAssign,
} from './nodes.js'

function elab(src: string): ResolvedProgram {
  return elaborate(parseProgram(src))
}

/** Walk a resolved-program graph (cycle-safe via WeakSet) and return
 *  every `op` value matching the targets. Walks the program body's
 *  decls and assigns; for each decl, visits its expression-shaped
 *  fields (init / update / inputs.value). */
function findOps(prog: ResolvedProgram, targets: string[]): string[] {
  const set = new Set(targets)
  const out: string[] = []
  const seen = new WeakSet<object>()
  const visitExpr = (e: ResolvedExpr): void => {
    if (e === null || typeof e !== 'object') return
    if (seen.has(e as object)) return
    seen.add(e as object)
    if (Array.isArray(e)) { e.forEach(visitExpr); return }
    if (set.has(e.op)) out.push(e.op)
    walkChildren(e, visitExpr)
  }
  for (const d of prog.body.decls) {
    if (d.op === 'regDecl') visitExpr(d.init)
    else if (d.op === 'delayDecl') { visitExpr(d.init); visitExpr(d.update) }
    else if (d.op === 'instanceDecl') for (const i of d.inputs) visitExpr(i.value)
  }
  for (const a of prog.body.assigns) visitExpr(a.expr)
  return out
}

// ─────────────────────────────────────────────────────────────
// 1. Identity on programs without sums
// ─────────────────────────────────────────────────────────────

describe('sumLower — no-op when no sums are used', () => {
  test('a sum-free program returns input by identity', () => {
    const p = elab('program X(a: float) -> (out: float) { out = a + 1 }')
    expect(sumLower(p)).toBe(p)
  })

  test('a program declaring (but not using) a sum type returns input by identity', () => {
    const p = elab(`
      program X(a: float) -> (out: float) {
        enum E { A, B }
        out = a
      }
    `)
    expect(sumLower(p)).toBe(p)
  })
})

// ─────────────────────────────────────────────────────────────
// 2. Two-variant nullary enum: Toggle
// ─────────────────────────────────────────────────────────────

describe('sumLower — nullary-only enum', () => {
  test('Toggle (Off|On): sum-typed delay → 1 tag slot; match → select chain', () => {
    const p = elab(`
      program Toggle() -> (value: float) {
        enum St { Off, On }
        delay state: St = match state { Off => On { }, On => Off { } } init Off { }
        value = match state { Off => 0.0, On => 1.0 }
      }
    `)
    const out = sumLower(p)
    // No `tag` / `match` ops remain anywhere in the body.
    expect(findOps(out, ['tag', 'match'])).toEqual([])
    // The sum-typed delay decomposes into a single tag-slot delay
    // (no payload variants → just the discriminator).
    expect(out.body.decls.length).toBe(1)
    const d = out.body.decls[0]
    expect(d.op).toBe('delayDecl')
    if (d.op === 'delayDecl') {
      expect(d.name).toBe('state#tag')
      // init = variant index of `Off` = 0.
      expect(d.init).toBe(0)
    }
    // The output assign now reads the tag slot. The shape is a single
    // top-level `select` whose condition is `eq(<tag-read>, 0)` and
    // whose branches are the lowered arm bodies.
    const a = out.body.assigns[0] as OutputAssign
    expect(typeof a.expr).toBe('object')
    if (typeof a.expr === 'object' && a.expr !== null && !Array.isArray(a.expr)) {
      expect(a.expr.op).toBe('select')
    }
  })
})

// ─────────────────────────────────────────────────────────────
// 3. Payload variant: EnvExpDecay-style (Idle | Decaying(level))
// ─────────────────────────────────────────────────────────────

describe('sumLower — payload variant', () => {
  test('decomposes Idle|Decaying(level) into 2 scalar slots and rewrites bindings', () => {
    const p = elab(`
      program EnvLite(trigger: float = 0, decay: float = 0.999) -> (env: float) {
        enum Env { Idle, Decaying(level: float) }
        delay state: Env = match state {
          Idle => select(trigger > 0.5, Decaying { level: 1 }, Idle { }),
          Decaying { level: level } => select(trigger > 0.5, Decaying { level: 1 }, Decaying { level: level * decay })
        } init Idle { }
        env = match state { Idle => 0, Decaying { level: level } => level }
      }
    `)
    const out = sumLower(p)
    // Two slots: `state#tag` (int discriminator) + `state#Decaying__level` (float payload).
    expect(out.body.decls.length).toBe(2)
    const names = out.body.decls.map(d => (d as DelayDecl).name)
    expect(names).toEqual(['state#tag', 'state#Decaying__level'])
    // No tag / match remain anywhere.
    expect(findOps(out, ['tag', 'match'])).toEqual([])
    // The output expression's payload binding (`level`) was rewritten
    // to a DelayRef reading state#Decaying__level. We assert the
    // structural shape: the output is a select chain whose `then`
    // branch (Decaying arm) eventually contains a delayRef whose decl
    // is the state#Decaying__level slot.
    const a = out.body.assigns[0] as OutputAssign
    const refs = collectDelayRefNames(a.expr)
    expect(refs).toContain('state#Decaying__level')
    expect(refs).toContain('state#tag')
  })
})

/** Collect delay-decl names referenced by `delayRef` nodes in the
 *  given expression. Does NOT recurse through `delayRef.decl` (the
 *  decl carries its own update expression which would re-enter the
 *  walker indefinitely). */
function collectDelayRefNames(expr: ResolvedExpr): string[] {
  const out: string[] = []
  const walk = (e: ResolvedExpr): void => {
    if (typeof e !== 'object' || e === null) return
    if (Array.isArray(e)) { e.forEach(walk); return }
    if (e.op === 'delayRef') { out.push(e.decl.name); return }
    // Recurse into structured children only — skip back-pointer fields.
    walkChildren(e, walk)
  }
  walk(expr)
  return out
}

function walkChildren(node: ResolvedExprOp, k: (e: ResolvedExpr) => void): void {
  switch (node.op) {
    case 'add': case 'sub': case 'mul': case 'div': case 'mod':
    case 'lt': case 'lte': case 'gt': case 'gte': case 'eq': case 'neq':
    case 'and': case 'or':
    case 'bitAnd': case 'bitOr': case 'bitXor': case 'lshift': case 'rshift':
    case 'floorDiv': case 'ldexp':
    case 'neg': case 'not': case 'bitNot':
    case 'sqrt': case 'abs': case 'floor': case 'ceil': case 'round':
    case 'floatExponent': case 'toInt': case 'toBool': case 'toFloat':
    case 'clamp': case 'select': case 'index': case 'arraySet':
      node.args.forEach(k); return
    case 'zeros': k(node.count); return
    case 'fold': case 'scan': k(node.over); k(node.init); k(node.body); return
    case 'generate': k(node.count); k(node.body); return
    case 'iterate': case 'chain': k(node.count); k(node.init); k(node.body); return
    case 'map2': k(node.over); k(node.body); return
    case 'zipWith': k(node.a); k(node.b); k(node.body); return
    case 'let': for (const b of node.binders) k(b.value); k(node.in); return
    case 'tag': for (const p of node.payload) k(p.value); return
    case 'match': k(node.scrutinee); for (const arm of node.arms) k(arm.body); return
    case 'inputRef': case 'regRef': case 'delayRef': case 'paramRef':
    case 'typeParamRef': case 'bindingRef': case 'nestedOut':
    case 'sampleRate': case 'sampleIndex':
      return
  }
}

// (Removed in C9: stdlib EnvExpDecay/TriggerRamp byte-equal dual-run
// gate. With the legacy `loadProgramDef` deleted, the comparison has
// no dual to compare against; the strata pipeline alone is the source
// of truth, exercised end-to-end by tests/equiv/jit_vs_interp and apply_plan.)
