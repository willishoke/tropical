/**
 * acyclic.test.ts — coverage for the strataPipeline-boundary
 * acyclicity check.
 *
 * Tested properties:
 *   1. A program with no instances is trivially acyclic.
 *   2. An acyclic DAG of instances passes assertion.
 *   3. A two-instance cycle is detected and reported.
 *   4. A self-loop SCC is detected.
 *   5. AcyclicityViolation carries the SCC structure.
 *
 * Today this check sits inside strataPipeline immediately after
 * traceCycles, where it's vacuously true for any input that reached
 * the cycle-breaker. Phase 3 moves traceCycles out of the compiler,
 * at which point this check becomes the load-bearing contract that
 * callers honor.
 */

import { describe, test, expect } from 'bun:test'
import { parseProgram } from '../parse/declarations.js'
import { elaborate } from './elaborator.js'
import { assertAcyclic, findInstanceCycles, AcyclicityViolation } from './acyclic.js'
import type { ResolvedProgram, InstanceDecl, OutputDecl, InputDecl } from './nodes.js'

function elab(src: string): ResolvedProgram {
  return elaborate(parseProgram(src))
}

describe('assertAcyclic / findInstanceCycles', () => {
  test('a program with no instances is trivially acyclic', () => {
    const p = elab('program X(a: float) -> (out: float) { out = a + 1 }')
    expect(findInstanceCycles(p)).toEqual([])
    expect(() => assertAcyclic(p)).not.toThrow()
  })

  test('an acyclic two-instance DAG passes', () => {
    const p = elab(`
      program Top(x: float) -> (out: float) {
        program Inner(in_: float) -> (out_: float) { out_ = in_ + 1 }
        a = Inner(in_: x)
        b = Inner(in_: a.out_)
        out = b.out_
      }
    `)
    expect(findInstanceCycles(p)).toEqual([])
    expect(() => assertAcyclic(p)).not.toThrow()
  })

  test('a two-instance cycle is detected and thrown', () => {
    // We construct the cyclic program by hand because the parser-level
    // path normally runs through traceCycles before reaching strata,
    // and we want to test the assertion in isolation.
    const innerIn: InputDecl  = { op: 'inputDecl', name: 'in_' }
    const innerOut: OutputDecl = { op: 'outputDecl', name: 'out_' }
    const innerProg: ResolvedProgram = {
      op: 'program', name: 'Inner', typeParams: [],
      ports: { inputs: [innerIn], outputs: [innerOut], typeDefs: [] },
      body: {
        op: 'block',
        decls: [],
        assigns: [{ op: 'outputAssign', target: innerOut, expr: { op: 'inputRef', decl: innerIn } }],
      },
    }
    // Build two instances of Inner that reference each other.
    const a: InstanceDecl = {
      op: 'instanceDecl', name: 'a', type: innerProg,
      typeArgs: [], inputs: [],
    }
    const b: InstanceDecl = {
      op: 'instanceDecl', name: 'b', type: innerProg,
      typeArgs: [], inputs: [],
    }
    // Now wire them cyclically: a.in_ = b.out_, b.in_ = a.out_.
    a.inputs.push({ port: innerIn, value: { op: 'nestedOut', instance: b, output: innerOut } })
    b.inputs.push({ port: innerIn, value: { op: 'nestedOut', instance: a, output: innerOut } })
    const cyclic: ResolvedProgram = {
      op: 'program', name: 'Cyclic', typeParams: [],
      ports: { inputs: [], outputs: [], typeDefs: [] },
      body: { op: 'block', decls: [a, b], assigns: [] },
    }
    const cycles = findInstanceCycles(cyclic)
    expect(cycles.length).toBe(1)
    expect(new Set(cycles[0].map(i => i.name))).toEqual(new Set(['a', 'b']))
    expect(() => assertAcyclic(cyclic)).toThrow(AcyclicityViolation)
  })

  test('a self-loop SCC (single instance with self-edge) is detected', () => {
    const innerIn: InputDecl  = { op: 'inputDecl', name: 'in_' }
    const innerOut: OutputDecl = { op: 'outputDecl', name: 'out_' }
    const innerProg: ResolvedProgram = {
      op: 'program', name: 'Inner', typeParams: [],
      ports: { inputs: [innerIn], outputs: [innerOut], typeDefs: [] },
      body: {
        op: 'block',
        decls: [],
        assigns: [{ op: 'outputAssign', target: innerOut, expr: { op: 'inputRef', decl: innerIn } }],
      },
    }
    const self: InstanceDecl = {
      op: 'instanceDecl', name: 'self', type: innerProg,
      typeArgs: [], inputs: [],
    }
    self.inputs.push({ port: innerIn, value: { op: 'nestedOut', instance: self, output: innerOut } })
    const cyclic: ResolvedProgram = {
      op: 'program', name: 'SelfLoop', typeParams: [],
      ports: { inputs: [], outputs: [], typeDefs: [] },
      body: { op: 'block', decls: [self], assigns: [] },
    }
    const cycles = findInstanceCycles(cyclic)
    expect(cycles.length).toBe(1)
    expect(cycles[0].map(i => i.name)).toEqual(['self'])
  })

  test('AcyclicityViolation carries the detected SCCs as structured data', () => {
    const innerIn: InputDecl  = { op: 'inputDecl', name: 'in_' }
    const innerOut: OutputDecl = { op: 'outputDecl', name: 'out_' }
    const innerProg: ResolvedProgram = {
      op: 'program', name: 'Inner', typeParams: [],
      ports: { inputs: [innerIn], outputs: [innerOut], typeDefs: [] },
      body: {
        op: 'block',
        decls: [],
        assigns: [{ op: 'outputAssign', target: innerOut, expr: { op: 'inputRef', decl: innerIn } }],
      },
    }
    const a: InstanceDecl = { op: 'instanceDecl', name: 'a', type: innerProg, typeArgs: [], inputs: [] }
    const b: InstanceDecl = { op: 'instanceDecl', name: 'b', type: innerProg, typeArgs: [], inputs: [] }
    a.inputs.push({ port: innerIn, value: { op: 'nestedOut', instance: b, output: innerOut } })
    b.inputs.push({ port: innerIn, value: { op: 'nestedOut', instance: a, output: innerOut } })
    const cyclic: ResolvedProgram = {
      op: 'program', name: 'Cyclic', typeParams: [],
      ports: { inputs: [], outputs: [], typeDefs: [] },
      body: { op: 'block', decls: [a, b], assigns: [] },
    }
    try {
      assertAcyclic(cyclic)
      throw new Error('expected throw')
    } catch (e) {
      expect(e).toBeInstanceOf(AcyclicityViolation)
      const violation = e as AcyclicityViolation
      expect(violation.sccs.length).toBe(1)
      expect(new Set(violation.sccs[0].map(i => i.name))).toEqual(new Set(['a', 'b']))
      expect(violation.message).toContain('unbroken inter-instance cycle')
    }
  })
})
