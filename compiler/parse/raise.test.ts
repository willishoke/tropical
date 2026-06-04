/**
 * raise.test.ts — Phase D D6 invariant assertions.
 *
 * Two properties the category-theorist + maintainer reviews of Phase D
 * called out:
 *
 *  1. **Zero scope analysis**: `raiseProgram` produces only NameRef-
 *     bearing references. No resolved-IR ref op (`inputRef`, `regRef`,
 *     `delayRef`, `paramRef`, `typeParamRef`, `bindingRef`, `nestedOut`
 *     with non-NameRef components) appears anywhere in the output. The
 *     type system enforces this today via `RaiseOutput = ParsedProgramNode`,
 *     but the runtime walk catches anyone reaching past types via casts.
 *
 *  2. **Drift detection (round-trip)**: every stdlib program parses, is
 *     re-saved as legacy `tropical_program_2` JSON via
 *     `saveProgramFromSession` (modulo surface-specific forms),
 *     then raised back. The raise output structurally matches the
 *     direct parser output — proves `raise.ts`'s op-coverage stays in
 *     sync with the parser's.
 *
 * Today (D6) the invariant test is the load-bearing one; the drift
 * round-trip needs a few-step setup (parse → resolve → ProgramNode →
 * raise) and lives behind a focused fixture corpus rather than the full
 * stdlib so failures point cleanly at op-coverage drift.
 */

import { describe, test, expect } from 'bun:test'
import { raiseProgram } from './raise.js'
import type { ProgramNode as LegacyProgramNode } from '../program.js'

/** Resolved-IR-only op tags that must NEVER appear in raise output. The
 *  parser output uses `nameRef` for all of these positions. */
const RESOLVED_REF_OPS: ReadonlySet<string> = new Set([
  'inputRef', 'regRef', 'delayRef', 'paramRef',
  'typeParamRef', 'bindingRef',
])

function walk(node: unknown, visit: (n: unknown, path: string) => void, path = '$'): void {
  visit(node, path)
  if (node === null || typeof node !== 'object') return
  if (Array.isArray(node)) {
    node.forEach((item, i) => walk(item, visit, `${path}[${i}]`))
    return
  }
  for (const [k, v] of Object.entries(node)) walk(v, visit, `${path}.${k}`)
}

function assertNoResolvedRefs(node: unknown, ctx: string): void {
  walk(node, (n, path) => {
    if (n === null || typeof n !== 'object' || Array.isArray(n)) return
    const op = (n as { op?: unknown }).op
    if (typeof op !== 'string') return
    if (RESOLVED_REF_OPS.has(op)) {
      throw new Error(
        `${ctx}: forbidden resolved-IR op '${op}' at ${path}. ` +
        `raise.ts must emit only nameRef-bearing references; resolution belongs to the elaborator.`,
      )
    }
  })
}

describe('raise — invariant: zero scope analysis (no resolved-IR refs)', () => {
  test('minimal program with input/reg/delay/output passes', () => {
    const legacy: LegacyProgramNode = {
      op: 'program',
      name: 'P',
      ports: { inputs: ['x'], outputs: ['out'] },
      body: {
        op: 'block',
        decls: [
          { op: 'regDecl', name: 'r', init: 0 },
          { op: 'delayDecl', name: 'd', update: { op: 'reg', name: 'r' }, init: 0 },
        ],
        assigns: [
          { op: 'outputAssign', name: 'out', expr: { op: 'add', args: [{ op: 'input', name: 'x' }, { op: 'reg', name: 'r' }] } },
          { op: 'nextUpdate', target: { kind: 'reg', name: 'r' }, expr: { op: 'mul', args: [{ op: 'reg', name: 'r' }, 0.5] } },
        ],
      },
    }
    const raised = raiseProgram(legacy)
    assertNoResolvedRefs(raised, 'minimal')
  })

  test('program with nested instance outputs (nestedOut + ref) emits NameRefs only', () => {
    const legacy: LegacyProgramNode = {
      op: 'program',
      name: 'P',
      ports: { outputs: ['out'] },
      body: {
        op: 'block',
        decls: [
          { op: 'instanceDecl', name: 'osc', program: 'Sin', inputs: { x: 440 } },
        ],
        assigns: [
          { op: 'outputAssign', name: 'out', expr: { op: 'nestedOut', ref: 'osc', output: 'out' } },
        ],
      },
    }
    const raised = raiseProgram(legacy)
    assertNoResolvedRefs(raised, 'nestedOut-bearing')
  })

  test('match + tag forms emit NameRefs only', () => {
    const legacy: LegacyProgramNode = {
      op: 'program',
      name: 'P',
      ports: { outputs: ['out'] },
      body: {
        op: 'block',
        decls: [],
        assigns: [
          {
            op: 'outputAssign',
            name: 'out',
            expr: {
              op: 'match',
              type: 'Maybe',
              scrutinee: { op: 'tag', type: 'Maybe', variant: 'Some', payload: { value: 1 } },
              arms: {
                Some: { bind: 'v', body: { op: 'binding', name: 'v' } },
                None: { body: 0 },
              },
            },
          },
        ],
      },
    }
    const raised = raiseProgram(legacy)
    assertNoResolvedRefs(raised, 'match/tag')
  })
})
