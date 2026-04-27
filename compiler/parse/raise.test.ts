/**
 * raise.test.ts — coverage for the legacy → parsed ProgramNode raiser.
 *
 * The strongest assertion is round-trip identity through `lower`:
 * `lower(raise(legacy))` should deep-equal the original `legacy` for any
 * legacy ProgramNode whose construction the lowerer can re-produce. Where
 * the lowerer cannot (e.g. the `{zeros: <N>}` sugar in Delay.trop's reg
 * init, which has no parser-shape inverse), we assert structurally on the
 * raise output.
 */

import { describe, test, expect } from 'bun:test'
import { readFileSync } from 'fs'
import { join, dirname } from 'path'
import { fileURLToPath } from 'url'
import { raiseProgram } from './raise.js'
import { lowerProgram } from './lower.js'
import { extractMarkdown } from './markdown.js'
import { parseProgram } from './declarations.js'
import { nameRef, type ProgramNode as ParsedProgramNode } from './nodes.js'
import type { ProgramNode as LegacyProgramNode } from '../program.js'

const __dirname = dirname(fileURLToPath(import.meta.url))

/** Load a stdlib program from its .trop source as a legacy ProgramNode
 *  (i.e., the same shape `loadProgramAsType` consumes). The legacy node
 *  is produced by the production parse+lower pipeline; that's the only
 *  source-of-truth for stdlib programs after Phase B8e removed JSON.
 *  The round-trip assertion `lower(raise(legacy)) === legacy` still
 *  verifies that raise is the categorical inverse of lower. */
function loadLegacy(filename: string): LegacyProgramNode {
  const text = readFileSync(join(__dirname, '../../stdlib', filename), 'utf-8')
  const ext = extractMarkdown(text)
  if (ext.blocks.length !== 1) {
    throw new Error(`${filename}: expected exactly 1 tropical block, got ${ext.blocks.length}`)
  }
  return lowerProgram(parseProgram(ext.blocks[0].source))
}

/** Round-trip assertion: lower(raise(legacy)) deep-equals legacy. */
function assertRoundTrip(legacy: LegacyProgramNode): void {
  const raised = raiseProgram(legacy)
  const lowered = lowerProgram(raised)
  expect(lowered).toEqual(legacy)
}

// ─────────────────────────────────────────────────────────────
// 1. Empty program
// ─────────────────────────────────────────────────────────────

describe('raise — empty program', () => {
  test('round-trips through lower', () => {
    const legacy: LegacyProgramNode = {
      op: 'program',
      name: 'X',
      body: { op: 'block', decls: [], assigns: [] },
    }
    assertRoundTrip(legacy)
  })
})

// ─────────────────────────────────────────────────────────────
// 2. OnePole — full stdlib JSON round-trip
// ─────────────────────────────────────────────────────────────

describe('raise — stdlib round-trips', () => {
  test('OnePole.trop', () => {
    assertRoundTrip(loadLegacy('OnePole.trop'))
  })

  test('AllpassDelay.trop (let-binding)', () => {
    assertRoundTrip(loadLegacy('AllpassDelay.trop'))
  })

  test('EnvExpDecay.trop (sum types, tag, match, payload, bind)', () => {
    // The on-disk file decorates `delayDecl` with a `type: "Env"` annotation
    // that the parser AST does not carry, so we strip it before round-trip.
    const legacy = loadLegacy('EnvExpDecay.trop')
    for (const decl of legacy.body?.decls ?? []) {
      const obj = decl as { op?: string; type?: unknown }
      if (obj.op === 'delayDecl' && obj.type !== undefined) delete obj.type
    }
    assertRoundTrip(legacy)
  })

  test('Sin.trop (fold over array of literals)', () => {
    assertRoundTrip(loadLegacy('Sin.trop'))
  })

  test('Exp.trop (fold + clamp + ldexp)', () => {
    assertRoundTrip(loadLegacy('Exp.trop'))
  })

  test('LadderFilter.trop (deeply-nested nestedOut chain)', () => {
    assertRoundTrip(loadLegacy('LadderFilter.trop'))
  })

  test('Phaser.trop (nested programDecl)', () => {
    assertRoundTrip(loadLegacy('Phaser.trop'))
  })
})

// ─────────────────────────────────────────────────────────────
// 3. Delay — has the {zeros: <N>} sugar; lowerer has no inverse,
//    so assert structurally on the raise output.
// ─────────────────────────────────────────────────────────────

describe('raise — Delay (zeros sugar)', () => {
  test('regDecl.init {zeros: {typeParam: N}} raises to call(zeros, [nameRef(N)])', () => {
    const legacy = loadLegacy('Delay.trop')
    const raised = raiseProgram(legacy)
    const reg = raised.body.decls[0]
    if (reg.op !== 'regDecl') throw new Error('expected regDecl')
    expect(reg.init).toEqual({
      op: 'call',
      callee: nameRef('zeros'),
      args: [nameRef('N')],
    })
  })

  test('bare-name output port "y" preserved as a string', () => {
    const legacy = loadLegacy('Delay.trop')
    const raised = raiseProgram(legacy)
    expect(raised.ports?.outputs).toEqual(['y'])
  })

  test('typeParam ref in expression position raises to nameRef', () => {
    const legacy = loadLegacy('Delay.trop')
    const raised = raiseProgram(legacy)
    // outputAssign('y') = index(nameRef('buf'), mod(call(sampleIndex), nameRef('N')))
    const outAssign = raised.body.assigns[0]
    if (outAssign.op !== 'outputAssign') throw new Error('expected outputAssign')
    if (typeof outAssign.expr !== 'object' || outAssign.expr === null || Array.isArray(outAssign.expr)) {
      throw new Error('expected object expr')
    }
    const idx = outAssign.expr
    if (idx.op !== 'index') throw new Error('expected index')
    const modNode = idx.args[1]
    if (typeof modNode !== 'object' || modNode === null || Array.isArray(modNode)) {
      throw new Error('expected mod object')
    }
    if (modNode.op !== 'mod') throw new Error('expected mod')
    expect(modNode.args[1]).toEqual(nameRef('N'))
  })
})

// ─────────────────────────────────────────────────────────────
// 4. Builtin call recognition (sampleIndex, clamp, etc.)
// ─────────────────────────────────────────────────────────────

describe('raise — builtin call recognition', () => {
  test('{op: "sampleIndex"} raises to call(nameRef("sampleIndex"), [])', () => {
    const legacy: LegacyProgramNode = {
      op: 'program',
      name: 'P',
      ports: { outputs: [{ name: 'y' }] },
      body: {
        op: 'block',
        decls: [],
        assigns: [
          { op: 'outputAssign', name: 'y', expr: { op: 'sampleIndex' } } as never,
        ],
      },
    }
    const raised = raiseProgram(legacy)
    const a = raised.body.assigns[0]
    if (a.op !== 'outputAssign') throw new Error('expected outputAssign')
    expect(a.expr).toEqual({ op: 'call', callee: nameRef('sampleIndex'), args: [] })
    // And it round-trips.
    assertRoundTrip(legacy)
  })

  test('clamp(x, 0, 1) raises to call(nameRef("clamp"), [...])', () => {
    const legacy: LegacyProgramNode = {
      op: 'program',
      name: 'P',
      ports: { inputs: [{ name: 'x' }], outputs: [{ name: 'y' }] },
      body: {
        op: 'block',
        decls: [],
        assigns: [
          {
            op: 'outputAssign',
            name: 'y',
            expr: {
              op: 'clamp',
              args: [{ op: 'input', name: 'x' }, 0, 1],
            },
          } as never,
        ],
      },
    }
    assertRoundTrip(legacy)
  })
})

// ─────────────────────────────────────────────────────────────
// 5. Unknown op → throw
// ─────────────────────────────────────────────────────────────

describe('raise — error cases', () => {
  test('unknown expression op throws', () => {
    const legacy: LegacyProgramNode = {
      op: 'program',
      name: 'P',
      ports: { outputs: [{ name: 'y' }] },
      body: {
        op: 'block',
        decls: [],
        assigns: [
          { op: 'outputAssign', name: 'y', expr: { op: 'mysteryOp' } } as never,
        ],
      },
    }
    expect(() => raiseProgram(legacy)).toThrow(/unknown expression op 'mysteryOp'/)
  })
})

// ─────────────────────────────────────────────────────────────
// 6. Sanity check: raise output is the lower test's hand-authored shape
// ─────────────────────────────────────────────────────────────

describe('raise — parser-shape sanity', () => {
  test('raise(OnePole legacy) produces NameRefs at every reference site', () => {
    const raised = raiseProgram(loadLegacy('OnePole.trop'))
    // The same hand-authored shape from lower.test.ts; if raise produces it,
    // raise and the lowerer agree on the parsed shape.
    const expected: ParsedProgramNode = {
      op: 'program',
      name: 'OnePole',
      ports: {
        inputs: [
          { name: 'input', type: nameRef('signal'), default: 0 },
          { name: 'g',     type: nameRef('float'),  default: 0.1 },
        ],
        outputs: [{ name: 'out', type: nameRef('signal') }],
      },
      body: {
        op: 'block',
        decls: [
          { op: 'regDecl', name: 's', init: 0 },
          {
            op: 'instanceDecl',
            name: 'tanh_in',
            program: nameRef('Tanh'),
            inputs: [{ port: nameRef('x'), value: nameRef('input') }],
          },
          {
            op: 'instanceDecl',
            name: 'tanh_s',
            program: nameRef('Tanh'),
            inputs: [{ port: nameRef('x'), value: nameRef('s') }],
          },
        ],
        assigns: [
          {
            op: 'outputAssign',
            name: 'out',
            expr: {
              op: 'add',
              args: [
                nameRef('s'),
                {
                  op: 'mul',
                  args: [
                    nameRef('g'),
                    {
                      op: 'sub',
                      args: [
                        { op: 'nestedOut', ref: nameRef('tanh_in'), output: nameRef('out') },
                        { op: 'nestedOut', ref: nameRef('tanh_s'),  output: nameRef('out') },
                      ],
                    },
                  ],
                },
              ],
            },
          },
          {
            op: 'nextUpdate',
            target: { kind: 'reg', name: 's' },
            expr: {
              op: 'add',
              args: [
                nameRef('s'),
                {
                  op: 'mul',
                  args: [
                    nameRef('g'),
                    {
                      op: 'sub',
                      args: [
                        { op: 'nestedOut', ref: nameRef('tanh_in'), output: nameRef('out') },
                        { op: 'nestedOut', ref: nameRef('tanh_s'),  output: nameRef('out') },
                      ],
                    },
                  ],
                },
              ],
            },
          },
        ],
      },
    }
    expect(raised).toEqual(expected)
  })
})
