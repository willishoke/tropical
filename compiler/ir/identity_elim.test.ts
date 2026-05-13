import { describe, test, expect } from 'bun:test'
import type {
  ResolvedProgram, InstanceDecl, InputDecl, OutputDecl,
  ResolvedExpr, NestedOut, BinaryOp, RegDecl,
} from './nodes.js'
import { identityElim } from './identity_elim.js'

// ─── Fixture helpers ──────────────────────────────────────────────────────

/** Build a trivial identity program: takes one input `in_p`, has one
 *  output `out`, body assigns `out = inputRef(in_p)`. */
function identityProgram(name = 'IdProg'): ResolvedProgram {
  const inDecl: InputDecl  = { op: 'inputDecl',  name: 'in_p' }
  const outDecl: OutputDecl = { op: 'outputDecl', name: 'out' }
  return {
    op: 'program',
    name,
    typeParams: [],
    ports: { inputs: [inDecl], outputs: [outDecl], typeDefs: [] },
    body: {
      op: 'block',
      decls: [],
      assigns: [{
        op: 'outputAssign',
        target: outDecl,
        expr: { op: 'inputRef', decl: inDecl },
      }],
    },
  }
}

/** Build a non-identity program: body assigns `out = inputRef(p) + 1`. */
function nonIdentityProgram(name = 'NonIdProg'): ResolvedProgram {
  const inDecl: InputDecl  = { op: 'inputDecl',  name: 'in_p' }
  const outDecl: OutputDecl = { op: 'outputDecl', name: 'out' }
  return {
    op: 'program',
    name,
    typeParams: [],
    ports: { inputs: [inDecl], outputs: [outDecl], typeDefs: [] },
    body: {
      op: 'block',
      decls: [],
      assigns: [{
        op: 'outputAssign',
        target: outDecl,
        expr: { op: 'add', args: [{ op: 'inputRef', decl: inDecl }, 1] },
      }],
    },
  }
}

/** Build an InstanceDecl wrapping a program and wiring one input. */
function makeInstance(
  prog: ResolvedProgram,
  instName: string,
  wireValue: ResolvedExpr,
): InstanceDecl {
  return {
    op: 'instanceDecl',
    name: instName,
    type: prog,
    typeArgs: [],
    inputs: [{ port: prog.ports.inputs[0], value: wireValue }],
  }
}

// ─── No-op cases ──────────────────────────────────────────────────────────

describe('identityElim — no-op cases', () => {
  test('empty program returns same shape', () => {
    const prog: ResolvedProgram = {
      op: 'program', name: 'empty', typeParams: [],
      ports: { inputs: [], outputs: [], typeDefs: [] },
      body: { op: 'block', decls: [], assigns: [] },
    }
    const after = identityElim(prog)
    expect(after.body.decls).toEqual([])
    expect(after.body.assigns).toEqual([])
  })

  test('returns input identity when no identity decls present', () => {
    const inDecl: InputDecl  = { op: 'inputDecl',  name: 'x' }
    const outDecl: OutputDecl = { op: 'outputDecl', name: 'y' }
    const prog: ResolvedProgram = {
      op: 'program', name: 'p', typeParams: [],
      ports: { inputs: [inDecl], outputs: [outDecl], typeDefs: [] },
      body: {
        op: 'block', decls: [],
        assigns: [{
          op: 'outputAssign',
          target: outDecl,
          expr: { op: 'inputRef', decl: inDecl },
        }],
      },
    }
    // Even though THIS program body looks like an identity, the pass
    // only eliminates *InstanceDecls* whose body is identity — it
    // doesn't change the top-level program's I/O behavior.
    const after = identityElim(prog)
    expect(after).toBe(prog)   // identity preserved when no work
  })

  test('non-identity InstanceDecl is preserved', () => {
    const idProg = nonIdentityProgram()
    const inst = makeInstance(idProg, 'i1', 42)
    const outerOut: OutputDecl = { op: 'outputDecl', name: 'out' }
    const prog: ResolvedProgram = {
      op: 'program', name: 'outer', typeParams: [],
      ports: { inputs: [], outputs: [outerOut], typeDefs: [] },
      body: {
        op: 'block',
        decls: [inst],
        assigns: [{
          op: 'outputAssign',
          target: outerOut,
          expr: { op: 'nestedOut', instance: inst, output: idProg.ports.outputs[0] },
        }],
      },
    }
    const after = identityElim(prog)
    expect(after.body.decls.length).toBe(1)   // not eliminated
  })
})

// ─── Elimination ──────────────────────────────────────────────────────────

describe('identityElim — elimination', () => {
  test('single identity InstanceDecl eliminated; NestedOut rewritten', () => {
    const idProg = identityProgram()
    const inst = makeInstance(idProg, 'i1', 42)
    const outerOut: OutputDecl = { op: 'outputDecl', name: 'out' }
    const prog: ResolvedProgram = {
      op: 'program', name: 'outer', typeParams: [],
      ports: { inputs: [], outputs: [outerOut], typeDefs: [] },
      body: {
        op: 'block',
        decls: [inst],
        assigns: [{
          op: 'outputAssign',
          target: outerOut,
          expr: { op: 'nestedOut', instance: inst, output: idProg.ports.outputs[0] },
        }],
      },
    }
    const after = identityElim(prog)
    // The InstanceDecl is gone.
    expect(after.body.decls).toEqual([])
    // The outputAssign now reads the constant 42 directly.
    expect(after.body.assigns.length).toBe(1)
    const oa = after.body.assigns[0]
    expect(oa.op).toBe('outputAssign')
    expect((oa as { expr: ResolvedExpr }).expr).toBe(42)
  })

  test('rewrite is recursive — chained identities collapse', () => {
    // Build: outerOut <- inst2.out  where  inst2.in_p = nestedOut(inst1.out)  where  inst1.in_p = 7
    const idProg = identityProgram()
    const inst1 = makeInstance(idProg, 'i1', 7)
    const inst2 = makeInstance(
      idProg,
      'i2',
      { op: 'nestedOut', instance: inst1, output: idProg.ports.outputs[0] },
    )
    const outerOut: OutputDecl = { op: 'outputDecl', name: 'out' }
    const prog: ResolvedProgram = {
      op: 'program', name: 'outer', typeParams: [],
      ports: { inputs: [], outputs: [outerOut], typeDefs: [] },
      body: {
        op: 'block',
        decls: [inst1, inst2],
        assigns: [{
          op: 'outputAssign',
          target: outerOut,
          expr: { op: 'nestedOut', instance: inst2, output: idProg.ports.outputs[0] },
        }],
      },
    }
    const after = identityElim(prog)
    expect(after.body.decls).toEqual([])
    expect((after.body.assigns[0] as { expr: ResolvedExpr }).expr).toBe(7)
  })

  test('NestedOut deep inside a binary expression is rewritten', () => {
    const idProg = identityProgram()
    const inst = makeInstance(idProg, 'i1', 3.14)
    const outerOut: OutputDecl = { op: 'outputDecl', name: 'out' }
    const nestedRef: NestedOut = {
      op: 'nestedOut', instance: inst, output: idProg.ports.outputs[0],
    }
    const prog: ResolvedProgram = {
      op: 'program', name: 'outer', typeParams: [],
      ports: { inputs: [], outputs: [outerOut], typeDefs: [] },
      body: {
        op: 'block',
        decls: [inst],
        assigns: [{
          op: 'outputAssign',
          target: outerOut,
          expr: { op: 'mul', args: [nestedRef, 2] },
        }],
      },
    }
    const after = identityElim(prog)
    expect(after.body.decls).toEqual([])
    const expr = (after.body.assigns[0] as { expr: ResolvedExpr }).expr as BinaryOp
    expect(expr.op).toBe('mul')
    expect(expr.args[0]).toBe(3.14)
    expect(expr.args[1]).toBe(2)
  })

  test('identity at one level + non-identity sibling: only identity eliminated', () => {
    const idProg = identityProgram()
    const nonIdProg = nonIdentityProgram()
    const idInst = makeInstance(idProg, 'id1', 5)
    const nonIdInst = makeInstance(nonIdProg, 'non1', 10)
    const outerOut: OutputDecl = { op: 'outputDecl', name: 'out' }
    const prog: ResolvedProgram = {
      op: 'program', name: 'outer', typeParams: [],
      ports: { inputs: [], outputs: [outerOut], typeDefs: [] },
      body: {
        op: 'block',
        decls: [idInst, nonIdInst],
        assigns: [{
          op: 'outputAssign',
          target: outerOut,
          expr: {
            op: 'add',
            args: [
              { op: 'nestedOut', instance: idInst,    output: idProg.ports.outputs[0] },
              { op: 'nestedOut', instance: nonIdInst, output: nonIdProg.ports.outputs[0] },
            ],
          },
        }],
      },
    }
    const after = identityElim(prog)
    // Identity gone, non-identity remains. The pass clones the program
    // before mutating, so surviving InstanceDecls are fresh objects
    // (different identity from the input `nonIdInst`) but they're
    // shared with the rest of the cloned program — NestedOut refs in
    // expressions point at the same surviving object.
    expect(after.body.decls.length).toBe(1)
    expect(after.body.decls[0].op).toBe('instanceDecl')
    const survivor = after.body.decls[0] as InstanceDecl
    expect(survivor.name).toBe('non1')
    const expr = (after.body.assigns[0] as { expr: ResolvedExpr }).expr as BinaryOp
    expect(expr.args[0]).toBe(5)
    expect((expr.args[1] as NestedOut).op).toBe('nestedOut')
    // The rhs NestedOut MUST reference the surviving instance (not the
    // dangling original). This is the consistency invariant the
    // clone-then-mutate approach preserves.
    expect((expr.args[1] as NestedOut).instance).toBe(survivor)
  })

  test('substitution into RegDecl init / update fields', () => {
    const idProg = identityProgram()
    const inst = makeInstance(idProg, 'i1', 100)
    const reg: RegDecl = {
      op: 'regDecl',
      name: 'r',
      init: { op: 'nestedOut', instance: inst, output: idProg.ports.outputs[0] },
    }
    const outerOut: OutputDecl = { op: 'outputDecl', name: 'out' }
    const prog: ResolvedProgram = {
      op: 'program', name: 'outer', typeParams: [],
      ports: { inputs: [], outputs: [outerOut], typeDefs: [] },
      body: {
        op: 'block',
        decls: [inst, reg],
        assigns: [
          {
            op: 'outputAssign',
            target: outerOut,
            expr: { op: 'regRef', decl: reg },
          },
          {
            op: 'nextUpdate',
            target: reg,
            expr: { op: 'nestedOut', instance: inst, output: idProg.ports.outputs[0] },
          },
        ],
      },
    }
    const after = identityElim(prog)
    expect(after.body.decls.length).toBe(1)   // only the RegDecl
    expect(after.body.decls[0].op).toBe('regDecl')
    const newReg = after.body.decls[0] as RegDecl
    expect(newReg.init).toBe(100)              // substituted from nestedOut → 100
    const nextAssign = after.body.assigns.find(a => a.op === 'nextUpdate')!
    expect((nextAssign as { expr: ResolvedExpr }).expr).toBe(100)
  })
})

// ─── Properties ───────────────────────────────────────────────────────────

describe('identityElim — properties', () => {
  test('idempotent: applying twice gives the same result as once', () => {
    const idProg = identityProgram()
    const inst = makeInstance(idProg, 'i1', 99)
    const outerOut: OutputDecl = { op: 'outputDecl', name: 'out' }
    const prog: ResolvedProgram = {
      op: 'program', name: 'outer', typeParams: [],
      ports: { inputs: [], outputs: [outerOut], typeDefs: [] },
      body: {
        op: 'block',
        decls: [inst],
        assigns: [{
          op: 'outputAssign',
          target: outerOut,
          expr: { op: 'nestedOut', instance: inst, output: idProg.ports.outputs[0] },
        }],
      },
    }
    const once = identityElim(prog)
    const twice = identityElim(once)
    expect(JSON.stringify(twice)).toEqual(JSON.stringify(once))
  })

  test('does not mutate input program', () => {
    const idProg = identityProgram()
    const inst = makeInstance(idProg, 'i1', 99)
    const outerOut: OutputDecl = { op: 'outputDecl', name: 'out' }
    const prog: ResolvedProgram = {
      op: 'program', name: 'outer', typeParams: [],
      ports: { inputs: [], outputs: [outerOut], typeDefs: [] },
      body: {
        op: 'block',
        decls: [inst],
        assigns: [{
          op: 'outputAssign',
          target: outerOut,
          expr: { op: 'nestedOut', instance: inst, output: idProg.ports.outputs[0] },
        }],
      },
    }
    const snapshot = JSON.stringify(prog)
    identityElim(prog)
    expect(JSON.stringify(prog)).toBe(snapshot)
  })
})
