/**
 * lower.test.ts — coverage for the parsed → legacy ProgramNode lowerer.
 *
 * The lowerer's job is scope-driven NameRef disambiguation plus a few
 * structural rewrites (instance inputs/type_args list → object form,
 * tag/match get sum-type names, builtins call → op forms). These tests
 * construct ParsedProgram values by hand and check the legacy output
 * shape.
 */

import { describe, test, expect } from 'bun:test'
import { readFileSync } from 'fs'
import { join, dirname } from 'path'
import { fileURLToPath } from 'url'
import { lowerProgram } from './lower.js'
import { nameRef, type ProgramNode as ParsedProgramNode } from './nodes.js'

const __dirname = dirname(fileURLToPath(import.meta.url))

// ─────────────────────────────────────────────────────────────
// 1. Empty program
// ─────────────────────────────────────────────────────────────

describe('lower — empty program', () => {
  test('no ports, no body', () => {
    const parsed: ParsedProgramNode = {
      op: 'program',
      name: 'X',
      body: { op: 'block', decls: [], assigns: [] },
    }
    const out = lowerProgram(parsed)
    expect(out).toEqual({
      op: 'program',
      name: 'X',
      body: { op: 'block', decls: [], assigns: [] },
    })
  })
})

// ─────────────────────────────────────────────────────────────
// 2. OnePole — full structural deep-equal against stdlib JSON
// ─────────────────────────────────────────────────────────────

describe('lower — OnePole matches stdlib JSON', () => {
  test('OnePole.json round-trip', () => {
    // Hand-author the parsed-shape equivalent of stdlib/OnePole.json. Every
    // reference site uses NameRef; the lowerer must produce the same legacy
    // shape on the disk.
    const parsed: ParsedProgramNode = {
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

    const lowered = lowerProgram(parsed)
    const onDisk = JSON.parse(readFileSync(join(__dirname, '../../stdlib/OnePole.json'), 'utf-8'))

    // The on-disk JSON is a ProgramFile (no top-level `op`, has `schema`);
    // the lowerer emits a ProgramNode. Convert the file shape to the
    // ProgramNode shape: drop `schema`, add `op: 'program'`, drop the
    // `value: null` placeholder in the body block (legacy carry-over).
    const { schema: _schema, ...rest } = onDisk
    const expected = { op: 'program' as const, ...rest }
    if (expected.body && expected.body.value === null) {
      delete expected.body.value
    }

    expect(lowered).toEqual(expected)
  })
})

// ─────────────────────────────────────────────────────────────
// 3. Type-param resolution in expression position
// ─────────────────────────────────────────────────────────────

describe('lower — type params', () => {
  test('NameRef in expr position resolves to typeParam op when in type_params scope', () => {
    const parsed: ParsedProgramNode = {
      op: 'program',
      name: 'P',
      type_params: { N: { type: 'int', default: 16 } },
      ports: {
        inputs: [{ name: 'x' }],
        outputs: [{ name: 'y' }],
      },
      body: {
        op: 'block',
        decls: [],
        assigns: [
          {
            op: 'outputAssign',
            name: 'y',
            // x % N — N must lower to {op:'typeParam', name:'N'}
            expr: { op: 'mod', args: [nameRef('x'), nameRef('N')] },
          },
        ],
      },
    }
    const out = lowerProgram(parsed)
    expect(out.body.assigns?.[0]).toEqual({
      op: 'outputAssign',
      name: 'y',
      expr: {
        op: 'mod',
        args: [
          { op: 'input', name: 'x' },
          { op: 'typeParam', name: 'N' },
        ],
      },
    })
  })
})

// ─────────────────────────────────────────────────────────────
// 4. ADTs — tag, match, payload object form, single-bind preserved
// ─────────────────────────────────────────────────────────────

describe('lower — ADTs (EnvExpDecay-shaped)', () => {
  test('tag.type / match.type filled in; payload + arms become object form', () => {
    const parsed: ParsedProgramNode = {
      op: 'program',
      name: 'EnvExpDecay',
      ports: {
        inputs: [{ name: 'trigger' }, { name: 'decay' }],
        outputs: [{ name: 'env' }],
        type_defs: [
          {
            kind: 'sum',
            name: 'Env',
            variants: [
              { name: 'Idle', payload: [] },
              { name: 'Decaying', payload: [{ name: 'level', scalar_type: 'float' }] },
            ],
          },
        ],
      },
      body: {
        op: 'block',
        decls: [
          {
            op: 'delayDecl',
            name: 'state',
            update: { op: 'tag', variant: nameRef('Idle') },
            init: { op: 'tag', variant: nameRef('Idle') },
          },
        ],
        assigns: [
          {
            op: 'outputAssign',
            name: 'env',
            expr: {
              op: 'match',
              scrutinee: nameRef('state'),
              arms: [
                { variant: nameRef('Idle'), body: 0 },
                {
                  variant: nameRef('Decaying'),
                  bind: 'level',
                  body: { op: 'binding', name: 'level' },
                },
              ],
            },
          },
        ],
      },
    }

    const out = lowerProgram(parsed)
    const envAssign = out.body.assigns?.[0] as { op: string; expr: { op: string; type: string; arms: Record<string, { bind?: string; body: unknown }> } }

    expect(envAssign.expr.op).toBe('match')
    expect(envAssign.expr.type).toBe('Env')
    expect(envAssign.expr.arms).toEqual({
      Idle:     { body: 0 },
      Decaying: { bind: 'level', body: { op: 'binding', name: 'level' } },
    })

    // Tag in the delay update should also gain `type: 'Env'`.
    const stateDecl = out.body.decls?.[0] as { op: string; update: { op: string; type: string; variant: string } }
    expect(stateDecl.update).toEqual({ op: 'tag', type: 'Env', variant: 'Idle' })
  })

  test('tag with payload object form', () => {
    const parsed: ParsedProgramNode = {
      op: 'program',
      name: 'P',
      ports: {
        inputs: [{ name: 'level' }],
        outputs: [{ name: 'out' }],
        type_defs: [
          {
            kind: 'sum',
            name: 'Env',
            variants: [
              { name: 'Decaying', payload: [{ name: 'level', scalar_type: 'float' }] },
            ],
          },
        ],
      },
      body: {
        op: 'block',
        decls: [],
        assigns: [
          {
            op: 'outputAssign',
            name: 'out',
            expr: {
              op: 'tag',
              variant: nameRef('Decaying'),
              payload: [
                { field: nameRef('level'), value: nameRef('level') },
              ],
            },
          },
        ],
      },
    }
    const out = lowerProgram(parsed)
    const expr = (out.body.assigns?.[0] as { expr: unknown }).expr
    expect(expr).toEqual({
      op: 'tag',
      type: 'Env',
      variant: 'Decaying',
      // The value `level` resolves to input(level) since `level` is an
      // input in this program. The field name `level` is identity.
      payload: { level: { op: 'input', name: 'level' } },
    })
  })
})

// ─────────────────────────────────────────────────────────────
// 5. NestedOut
// ─────────────────────────────────────────────────────────────

describe('lower — nestedOut', () => {
  test('osc.out becomes {ref: "osc", output: "out"}', () => {
    const parsed: ParsedProgramNode = {
      op: 'program',
      name: 'P',
      ports: {
        inputs: [],
        outputs: [{ name: 'out' }],
      },
      body: {
        op: 'block',
        decls: [
          {
            op: 'instanceDecl',
            name: 'osc',
            program: nameRef('Sin'),
          },
        ],
        assigns: [
          {
            op: 'outputAssign',
            name: 'out',
            expr: { op: 'nestedOut', ref: nameRef('osc'), output: nameRef('out') },
          },
        ],
      },
    }
    const out = lowerProgram(parsed)
    expect((out.body.assigns?.[0] as { expr: unknown }).expr).toEqual({
      op: 'nestedOut',
      ref: 'osc',
      output: 'out',
    })
  })
})

// ─────────────────────────────────────────────────────────────
// 6. Builtin call recognition
// ─────────────────────────────────────────────────────────────

describe('lower — builtin calls', () => {
  test('clamp(x, 0, 1) becomes {op: "clamp", args: [...]}', () => {
    const parsed: ParsedProgramNode = {
      op: 'program',
      name: 'P',
      ports: {
        inputs: [{ name: 'x' }],
        outputs: [{ name: 'y' }],
      },
      body: {
        op: 'block',
        decls: [],
        assigns: [
          {
            op: 'outputAssign',
            name: 'y',
            expr: {
              op: 'call',
              callee: nameRef('clamp'),
              args: [nameRef('x'), 0, 1],
            },
          },
        ],
      },
    }
    const out = lowerProgram(parsed)
    expect((out.body.assigns?.[0] as { expr: unknown }).expr).toEqual({
      op: 'clamp',
      args: [{ op: 'input', name: 'x' }, 0, 1],
    })
  })

  test('sampleIndex() becomes {op: "sampleIndex"}', () => {
    const parsed: ParsedProgramNode = {
      op: 'program',
      name: 'P',
      ports: {
        inputs: [],
        outputs: [{ name: 'y' }],
      },
      body: {
        op: 'block',
        decls: [],
        assigns: [
          {
            op: 'outputAssign',
            name: 'y',
            expr: { op: 'call', callee: nameRef('sampleIndex'), args: [] },
          },
        ],
      },
    }
    const out = lowerProgram(parsed)
    expect((out.body.assigns?.[0] as { expr: unknown }).expr).toEqual({ op: 'sampleIndex' })
  })

  test('unrecognized function call throws', () => {
    const parsed: ParsedProgramNode = {
      op: 'program',
      name: 'P',
      ports: {
        inputs: [],
        outputs: [{ name: 'y' }],
      },
      body: {
        op: 'block',
        decls: [],
        assigns: [
          {
            op: 'outputAssign',
            name: 'y',
            expr: { op: 'call', callee: nameRef('unknownFn'), args: [1] },
          },
        ],
      },
    }
    expect(() => lowerProgram(parsed)).toThrow(/unknownFn.*not a recognized builtin/)
  })
})

// ─────────────────────────────────────────────────────────────
// 7. Combinator binders
// ─────────────────────────────────────────────────────────────

describe('lower — combinator binders', () => {
  test("fold body's reference to acc lowers to {op: 'binding', name: 'acc'}", () => {
    const parsed: ParsedProgramNode = {
      op: 'program',
      name: 'P',
      ports: {
        inputs: [],
        outputs: [{ name: 'y' }],
      },
      body: {
        op: 'block',
        decls: [],
        assigns: [
          {
            op: 'outputAssign',
            name: 'y',
            expr: {
              op: 'fold',
              over: [1, 2, 3],
              init: 0,
              acc_var: 'acc',
              elem_var: 'e',
              // body refers to `acc` and `e` — both must lower to bindings,
              // not inputs/regs/etc.
              body: {
                op: 'add',
                args: [
                  { op: 'binding', name: 'acc' },
                  { op: 'binding', name: 'e' },
                ],
              },
            },
          },
        ],
      },
    }
    const out = lowerProgram(parsed)
    expect((out.body.assigns?.[0] as { expr: unknown }).expr).toEqual({
      op: 'fold',
      over: [1, 2, 3],
      init: 0,
      acc_var: 'acc',
      elem_var: 'e',
      body: {
        op: 'add',
        args: [
          { op: 'binding', name: 'acc' },
          { op: 'binding', name: 'e' },
        ],
      },
    })
  })

  test('NameRef inside fold body resolves to binding when name matches a binder', () => {
    // If a parser ever emits a bare NameRef instead of BindingNode for a
    // binder-introduced name, the lowerer's binder-stack check should
    // still produce a binding.
    const parsed: ParsedProgramNode = {
      op: 'program',
      name: 'P',
      ports: {
        inputs: [],
        outputs: [{ name: 'y' }],
      },
      body: {
        op: 'block',
        decls: [],
        assigns: [
          {
            op: 'outputAssign',
            name: 'y',
            expr: {
              op: 'fold',
              over: [1, 2],
              init: 0,
              acc_var: 'acc',
              elem_var: 'e',
              body: { op: 'add', args: [nameRef('acc'), nameRef('e')] },
            },
          },
        ],
      },
    }
    const out = lowerProgram(parsed)
    const fold = (out.body.assigns?.[0] as { expr: { body: unknown } }).expr.body
    expect(fold).toEqual({
      op: 'add',
      args: [
        { op: 'binding', name: 'acc' },
        { op: 'binding', name: 'e' },
      ],
    })
  })
})

// ─────────────────────────────────────────────────────────────
// 8. Unresolved name
// ─────────────────────────────────────────────────────────────

describe('lower — unresolved name', () => {
  test('reference to a name not in any scope throws', () => {
    const parsed: ParsedProgramNode = {
      op: 'program',
      name: 'P',
      ports: {
        inputs: [],
        outputs: [{ name: 'y' }],
      },
      body: {
        op: 'block',
        decls: [],
        assigns: [
          { op: 'outputAssign', name: 'y', expr: nameRef('whoIsThis') },
        ],
      },
    }
    expect(() => lowerProgram(parsed)).toThrow(/unresolved reference 'whoIsThis'/)
  })
})

// ─────────────────────────────────────────────────────────────
// 9. InstanceDecl shape rewrites
// ─────────────────────────────────────────────────────────────

describe('lower — instanceDecl shape', () => {
  test('inputs array becomes object; type_args array becomes object', () => {
    const parsed: ParsedProgramNode = {
      op: 'program',
      name: 'P',
      ports: {
        inputs: [{ name: 'sig' }],
        outputs: [{ name: 'out' }],
      },
      body: {
        op: 'block',
        decls: [
          {
            op: 'instanceDecl',
            name: 'd',
            program: nameRef('Delay'),
            type_args: [{ param: nameRef('N'), value: 1024 }],
            inputs: [{ port: nameRef('x'), value: nameRef('sig') }],
          },
        ],
        assigns: [
          {
            op: 'outputAssign',
            name: 'out',
            expr: { op: 'nestedOut', ref: nameRef('d'), output: nameRef('y') },
          },
        ],
      },
    }
    const out = lowerProgram(parsed)
    expect(out.body.decls?.[0]).toEqual({
      op: 'instanceDecl',
      name: 'd',
      program: 'Delay',
      type_args: { N: 1024 },
      inputs: { x: { op: 'input', name: 'sig' } },
    })
  })
})
