/**
 * statements.test.ts — body-statement parser coverage (Phase B3).
 */

import { describe, test, expect } from 'bun:test'
import { parseBody, type BlockNode } from './statements.js'
import { ParseError } from './expressions.js'
import { nameRef } from './nodes.js'

describe('body — empty and minimal', () => {
  test('empty body', () => {
    expect(parseBody('{}')).toEqual({ op: 'block', decls: [], assigns: [] })
  })

  test('body with whitespace and comments', () => {
    expect(parseBody('{ /* nothing */ \n  // also nothing\n }'))
      .toEqual({ op: 'block', decls: [], assigns: [] })
  })

  test('trailing input after body rejected', () => {
    expect(() => parseBody('{} extra')).toThrow(/unexpected trailing/)
  })
})

describe('body — regDecl', () => {
  test('reg without type', () => {
    const b = parseBody('{ reg s = 0 }')
    expect(b.decls).toEqual([{ op: 'regDecl', name: 's', init: 0 }])
  })

  test('reg with type', () => {
    const b = parseBody('{ reg s: float = 0 }')
    expect(b.decls).toEqual([{ op: 'regDecl', name: 's', init: 0, type: nameRef('float') }])
  })

  test('reg with expression init', () => {
    const b = parseBody('{ reg state: float = 1 + 2 }')
    expect(b.decls).toHaveLength(1)
    const d = b.decls[0] as { op: string; name: string; type: { name: string }; init: { op: string } }
    expect(d.op).toBe('regDecl')
    expect(d.name).toBe('state')
    expect(d.type.name).toBe('float')
    expect(d.init.op).toBe('add')
  })

  test('reg without `=` rejected', () => {
    expect(() => parseBody('{ reg s 0 }')).toThrow(ParseError)
  })
})

describe('body — delayDecl', () => {
  test('delay with simple update and init', () => {
    const b = parseBody('{ delay z = x init 0 }')
    expect(b.decls).toEqual([
      { op: 'delayDecl', name: 'z', update: { op: 'nameRef', name: 'x' }, init: 0 },
    ])
  })

  test('delay with compound update expression', () => {
    const b = parseBody('{ delay z = a + b * c init -1 }')
    expect(b.decls).toHaveLength(1)
    const d = b.decls[0] as { op: string; name: string; init: number; update: { op: string } }
    expect(d.op).toBe('delayDecl')
    expect(d.name).toBe('z')
    expect(d.init).toBe(-1)
    expect(d.update.op).toBe('add')
  })

  test('delay missing `init` keyword rejected', () => {
    expect(() => parseBody('{ delay z = x 0 }')).toThrow(/expected 'init'/)
  })

  test('delay missing init value rejected', () => {
    expect(() => parseBody('{ delay z = x init }')).toThrow(ParseError)
  })
})

describe('body — paramDecl', () => {
  test('smoothed param with default', () => {
    const b = parseBody('{ param cutoff: smoothed = 1000 }')
    expect(b.decls).toEqual([
      { op: 'paramDecl', name: 'cutoff', type: 'param', value: 1000 },
    ])
  })

  test('smoothed param without default', () => {
    const b = parseBody('{ param freq: smoothed }')
    expect(b.decls).toEqual([
      { op: 'paramDecl', name: 'freq', type: 'param' },
    ])
  })

  test('trigger param', () => {
    const b = parseBody('{ param fire: trigger }')
    expect(b.decls).toEqual([
      { op: 'paramDecl', name: 'fire', type: 'trigger' },
    ])
  })

  test('trigger param with default rejected', () => {
    expect(() => parseBody('{ param fire: trigger = 1 }')).toThrow(/cannot have a default/)
  })

  test('unknown param kind rejected', () => {
    expect(() => parseBody('{ param x: knob }')).toThrow(/'smoothed' or 'trigger'/)
  })

  test('param default must be a number literal', () => {
    expect(() => parseBody('{ param x: smoothed = a + b }')).toThrow(/number literal/)
  })
})

describe('body — nextUpdate', () => {
  test('register update', () => {
    const b = parseBody('{ next state = state + 1 }')
    expect(b.assigns).toEqual([{
      op: 'nextUpdate',
      target: { kind: 'reg', name: 'state' },
      expr: {
        op: 'add',
        args: [{ op: 'nameRef', name: 'state' }, 1],
      },
    }])
  })

  test('next without `=` rejected', () => {
    expect(() => parseBody('{ next x x }')).toThrow(ParseError)
  })

  test('next on undeclared name parses cleanly (semantic check is elaborator-level)', () => {
    // The parser does not check that `next foo = ...` references a
    // previously-declared `reg foo`; that's a scope-resolution concern
    // for the elaborator. Pin the current behaviour so future scope
    // tightening is intentional.
    const b = parseBody('{ next undeclared = 0 }')
    expect(b.assigns).toEqual([{
      op: 'nextUpdate',
      target: { kind: 'reg', name: 'undeclared' },
      expr: 0,
    }])
  })
})

describe('body — outputAssign', () => {
  test('simple output assign', () => {
    const b = parseBody('{ out = sig }')
    expect(b.assigns).toEqual([{
      op: 'outputAssign', name: 'out', expr: { op: 'nameRef', name: 'sig' },
    }])
  })

  test('output assign with expression rhs', () => {
    const b = parseBody('{ y = a * x + b }')
    expect(b.assigns).toHaveLength(1)
    const a = b.assigns[0] as { op: string; name: string; expr: { op: string } }
    expect(a.op).toBe('outputAssign')
    expect(a.name).toBe('y')
    expect(a.expr.op).toBe('add')
  })

  test('dac.out wire', () => {
    const b = parseBody('{ dac.out = osc.sin }')
    expect(b.assigns).toEqual([{
      op: 'outputAssign',
      name: 'dac.out',
      expr: { op: 'nestedOut', ref: nameRef('osc'), output: nameRef('sin') },
    }])
  })

  test('dac with wrong port rejected', () => {
    expect(() => parseBody('{ dac.left = sig }')).toThrow(/only one output port/)
  })
})

describe('body — instanceDecl', () => {
  test('instance with positional-keyword args', () => {
    const b = parseBody('{ osc = SinOsc(freq: 440) }')
    expect(b.decls).toEqual([{
      op: 'instanceDecl',
      name: 'osc',
      program: nameRef('SinOsc'),
      inputs: [{ port: nameRef('freq'), value: 440 }],
    }])
  })

  test('instance with multiple inputs', () => {
    const b = parseBody('{ filt = OnePole(cutoff: 1000, x: osc.sin) }')
    const d = b.decls[0] as {
      op: string; name: string; program: { name: string };
      inputs: Array<{ port: { name: string }; value: unknown }>
    }
    expect(d.op).toBe('instanceDecl')
    expect(d.name).toBe('filt')
    expect(d.program).toEqual(nameRef('OnePole'))
    const ports = d.inputs.map(i => i.port.name).sort()
    expect(ports).toEqual(['cutoff', 'x'])
    const cutoff = d.inputs.find(i => i.port.name === 'cutoff')!
    const x = d.inputs.find(i => i.port.name === 'x')!
    expect(cutoff.value).toBe(1000)
    expect(x.value).toEqual({ op: 'nestedOut', ref: nameRef('osc'), output: nameRef('sin') })
  })

  test('instance with type args', () => {
    const b = parseBody('{ seq = Sequencer<N=4>(clock: trig) }')
    expect(b.decls).toEqual([{
      op: 'instanceDecl',
      name: 'seq',
      program: nameRef('Sequencer'),
      type_args: [{ param: nameRef('N'), value: 4 }],
      inputs: [{ port: nameRef('clock'), value: { op: 'nameRef', name: 'trig' } }],
    }])
  })

  test('instance with multiple type args', () => {
    const b = parseBody('{ d = Delay<N=4, M=8>()  }')
    const d = b.decls[0] as { type_args: Array<{ param: { name: string }; value: number }> }
    const byName = Object.fromEntries(d.type_args.map(a => [a.param.name, a.value]))
    expect(byName).toEqual({ N: 4, M: 8 })
  })

  test('instance with no inputs', () => {
    const b = parseBody('{ d = Delay<N=44100>() }')
    const dd = b.decls[0] as { op: string; inputs?: unknown }
    expect(dd.op).toBe('instanceDecl')
    expect(dd.inputs).toBeUndefined()
  })

  test('non-integer type-arg value rejected', () => {
    expect(() => parseBody('{ d = Delay<N=4.5>() }')).toThrow(/integer/)
  })

  test('duplicate input port rejected', () => {
    expect(() => parseBody('{ x = OnePole(cutoff: 1, cutoff: 2) }')).toThrow(/duplicate instance input/)
  })

  test('duplicate type-arg name rejected', () => {
    expect(() => parseBody('{ x = Sequencer<N=4, N=8>() }')).toThrow(/duplicate type-arg/)
  })

  test('lowercase RHS is treated as outputAssign, not instanceDecl', () => {
    // `out = sinosc(freq: 440)` would be ambiguous if we matched on the
    // call shape alone. Capitalization disambiguates.
    const b = parseBody('{ out = sinosc(440) }')
    expect(b.assigns).toHaveLength(1)
    expect(b.decls).toHaveLength(0)
  })
})

describe('body — multiple statements', () => {
  test('mixed decls and assigns', () => {
    const b = parseBody(`{
      reg s: float = 0
      osc = SinOsc(freq: 440)
      out = osc.sin
      next s = s + 1
      dac.out = osc.sin
    }`)
    expect(b.decls).toHaveLength(2)
    expect(b.assigns).toHaveLength(3)
    expect((b.decls[0] as { op: string }).op).toBe('regDecl')
    expect((b.decls[1] as { op: string }).op).toBe('instanceDecl')
    expect((b.assigns[0] as { op: string; name: string }).name).toBe('out')
    expect((b.assigns[1] as { op: string }).op).toBe('nextUpdate')
    expect((b.assigns[2] as { op: string; name: string }).name).toBe('dac.out')
  })

  test('semicolon separators are accepted but optional', () => {
    const a = parseBody('{ reg s = 0; out = s; next s = s + 1; }')
    const b = parseBody('{ reg s = 0  out = s  next s = s + 1 }')
    expect(a).toEqual(b)
  })

  test('source order of decls preserved', () => {
    const b = parseBody(`{
      reg a = 1
      reg b = 2
      reg c = 3
    }`)
    expect((b.decls as Array<{ name: string }>).map(d => d.name)).toEqual(['a', 'b', 'c'])
  })
})

describe('body — realistic stdlib-ish patterns', () => {
  test('VCA-shaped program (rough)', () => {
    const b = parseBody(`{
      out = audio * cv
    }`)
    expect(b.assigns).toEqual([{
      op: 'outputAssign',
      name: 'out',
      expr: {
        op: 'mul',
        args: [{ op: 'nameRef', name: 'audio' }, { op: 'nameRef', name: 'cv' }],
      },
    }])
  })

  test('OnePole-shaped program (rough)', () => {
    const b = parseBody(`{
      reg s: float = 0
      out = s
      next s = x * (1 - g) + s * g
    }`)
    expect(b.decls).toHaveLength(1)
    expect(b.assigns).toHaveLength(2)
    expect((b.decls[0] as { name: string }).name).toBe('s')
    expect((b.assigns[1] as { op: string }).op).toBe('nextUpdate')
  })

  test('LadderFilter-shaped program (instance composition)', () => {
    const b = parseBody(`{
      delay z = x init 0
      lp1 = OnePole(x: x - 4 * z, cutoff: cutoff)
      lp2 = OnePole(x: lp1.out, cutoff: cutoff)
      out = lp2.out
    }`)
    expect(b.decls).toHaveLength(3)  // 1 delay + 2 instances
    expect(b.assigns).toHaveLength(1)
    expect((b.decls[0] as { op: string }).op).toBe('delayDecl')
    expect((b.decls[1] as { op: string; name: string }).name).toBe('lp1')
    expect((b.decls[2] as { op: string; name: string }).name).toBe('lp2')
  })

  test('patch with dac.out wiring', () => {
    const b = parseBody(`{
      osc = SinOsc(freq: 220)
      filt = OnePole(x: osc.sin, cutoff: 1000)
      dac.out = filt.out
    }`)
    expect(b.decls).toHaveLength(2)
    expect(b.assigns).toHaveLength(1)
    expect((b.assigns[0] as { name: string }).name).toBe('dac.out')
  })
})

describe('body — error cases', () => {
  test('missing closing brace', () => {
    expect(() => parseBody('{ out = sig')).toThrow(ParseError)
  })

  test('unknown leading token', () => {
    expect(() => parseBody('{ 42 }')).toThrow(/expected body item/)
  })

  test('error positions reflect token line/col', () => {
    let err: ParseError | undefined
    try { parseBody('{\n  reg\n}') } catch (e) { err = e as ParseError }
    expect(err).toBeInstanceOf(ParseError)
    // `reg` is followed by `}` (after newline), so reg-name expected at the
    // `}` token (line 3, col 1) — the message carries the position.
    expect(err?.message).toMatch(/3:1/)
  })
})
