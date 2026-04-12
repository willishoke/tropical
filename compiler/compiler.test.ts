/**
 * compiler.test.ts — Tests for graph utilities (dependency extraction, topo sort, SCC).
 */

import { describe, test, expect } from 'bun:test'
import {
  portTypeFromString,
  exprDependencies,
  buildDependencyGraph,
  topologicalSort,
  tarjanSCC,
  extractInstanceInfo,
  CompilerError,
  type InstanceInfo,
} from './compiler'
import {
  Float, Int, Bool, Unit, StructType, ArrayType,
  portTypeEqual,
} from './term'
import type { ExprNode } from './session'

// ─────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────

/** Build a simple InstanceInfo for testing. */
function modInfo(
  name: string,
  opts: {
    typeName?: string
    inputs?: string[]
    outputs?: string[]
    registers?: string[]
    inputTypes?: string[]
    outputTypes?: string[]
    registerTypes?: string[]
  } = {},
): InstanceInfo {
  const inputs = opts.inputs ?? ['in']
  const outputs = opts.outputs ?? ['out']
  const registers = opts.registers ?? []
  return {
    name,
    typeName: opts.typeName ?? name,
    inputNames: inputs,
    outputNames: outputs,
    registerNames: registers,
    inputTypes: (opts.inputTypes ?? inputs.map(() => 'float')).map(s => portTypeFromString(s)),
    outputTypes: (opts.outputTypes ?? outputs.map(() => 'float')).map(s => portTypeFromString(s)),
    registerTypes: (opts.registerTypes ?? registers.map(() => 'float')).map(s => portTypeFromString(s)),
  }
}

// ─────────────────────────────────────────────────────────────
// portTypeFromString
// ─────────────────────────────────────────────────────────────

describe('portTypeFromString', () => {
  test('scalar types', () => {
    expect(portTypeEqual(portTypeFromString('float'), Float)).toBe(true)
    expect(portTypeEqual(portTypeFromString('int'), Int)).toBe(true)
    expect(portTypeEqual(portTypeFromString('bool'), Bool)).toBe(true)
  })

  test('unit type', () => {
    expect(portTypeEqual(portTypeFromString('unit'), Unit)).toBe(true)
  })

  test('undefined defaults to float', () => {
    expect(portTypeEqual(portTypeFromString(undefined), Float)).toBe(true)
  })

  test('unknown name becomes struct type', () => {
    const t = portTypeFromString('MyStruct')
    expect(t.tag).toBe('struct')
    expect(portTypeEqual(t, StructType('MyStruct'))).toBe(true)
  })

  test('array type with shape', () => {
    const t = portTypeFromString('float[4]')
    expect(portTypeEqual(t, ArrayType(Float, [4]))).toBe(true)
  })

  test('array type multi-dimensional', () => {
    const t = portTypeFromString('float[4,4]')
    expect(portTypeEqual(t, ArrayType(Float, [4, 4]))).toBe(true)
  })

  test('array type with int element', () => {
    const t = portTypeFromString('int[8]')
    expect(portTypeEqual(t, ArrayType(Int, [8]))).toBe(true)
  })

  test('legacy array string', () => {
    const t = portTypeFromString('array')
    expect(t.tag).toBe('array')
  })

  test('legacy matrix string', () => {
    const t = portTypeFromString('matrix')
    expect(t.tag).toBe('array')
    if (t.tag === 'array') {
      expect(t.shape.length).toBe(2)
    }
  })
})

// ─────────────────────────────────────────────────────────────
// exprDependencies
// ─────────────────────────────────────────────────────────────

describe('exprDependencies', () => {
  test('literal has no deps', () => {
    expect(exprDependencies(440).size).toBe(0)
    expect(exprDependencies(true).size).toBe(0)
  })

  test('ref extracts module name', () => {
    const expr: ExprNode = { op: 'ref', instance: 'VCO1', output: 'sin' }
    const deps = exprDependencies(expr)
    expect(deps.size).toBe(1)
    expect(deps.has('VCO1')).toBe(true)
  })

  test('binary op with nested refs', () => {
    const expr: ExprNode = {
      op: 'mul',
      args: [
        { op: 'ref', instance: 'VCO1', output: 'sin' },
        { op: 'ref', instance: 'LFO1', output: 'out' },
      ],
    }
    const deps = exprDependencies(expr)
    expect(deps.size).toBe(2)
    expect(deps.has('VCO1')).toBe(true)
    expect(deps.has('LFO1')).toBe(true)
  })

  test('mixed refs and constants', () => {
    const expr: ExprNode = {
      op: 'mul',
      args: [
        { op: 'ref', instance: 'VCO1', output: 'sin' },
        0.5,
      ],
    }
    const deps = exprDependencies(expr)
    expect(deps.size).toBe(1)
    expect(deps.has('VCO1')).toBe(true)
  })

  test('array expression', () => {
    const expr: ExprNode = [
      { op: 'ref', instance: 'A', output: 'x' },
      { op: 'ref', instance: 'B', output: 'y' },
    ]
    const deps = exprDependencies(expr)
    expect(deps.size).toBe(2)
  })

  test('non-ref ops with no args', () => {
    expect(exprDependencies({ op: 'sample_rate' }).size).toBe(0)
    expect(exprDependencies({ op: 'param', name: 'freq' }).size).toBe(0)
  })
})

// ─────────────────────────────────────────────────────────────
// buildDependencyGraph
// ─────────────────────────────────────────────────────────────

describe('buildDependencyGraph', () => {
  test('simple chain A → B', () => {
    const exprs = new Map<string, ExprNode>([
      ['A:freq', 440],
      ['B:in', { op: 'ref', instance: 'A', output: 'out' }],
    ])
    const graph = buildDependencyGraph(['A', 'B'], exprs)
    expect(graph.get('A')!.size).toBe(0)
    expect(graph.get('B')!.has('A')).toBe(true)
  })

  test('diamond: A → B, A → C, B → D, C → D', () => {
    const exprs = new Map<string, ExprNode>([
      ['B:in', { op: 'ref', instance: 'A', output: 'out' }],
      ['C:in', { op: 'ref', instance: 'A', output: 'out' }],
      ['D:x', { op: 'ref', instance: 'B', output: 'out' }],
      ['D:y', { op: 'ref', instance: 'C', output: 'out' }],
    ])
    const graph = buildDependencyGraph(['A', 'B', 'C', 'D'], exprs)
    expect(graph.get('A')!.size).toBe(0)
    expect(graph.get('B')!.has('A')).toBe(true)
    expect(graph.get('C')!.has('A')).toBe(true)
    expect(graph.get('D')!.has('B')).toBe(true)
    expect(graph.get('D')!.has('C')).toBe(true)
  })

  test('ignores self-references', () => {
    const exprs = new Map<string, ExprNode>([
      ['A:in', { op: 'ref', instance: 'A', output: 'out' }],
    ])
    const graph = buildDependencyGraph(['A'], exprs)
    expect(graph.get('A')!.size).toBe(0)
  })

  test('ignores refs to unknown modules', () => {
    const exprs = new Map<string, ExprNode>([
      ['A:in', { op: 'ref', instance: 'UNKNOWN', output: 'out' }],
    ])
    const graph = buildDependencyGraph(['A'], exprs)
    expect(graph.get('A')!.size).toBe(0)
  })
})

// ─────────────────────────────────────────────────────────────
// topologicalSort
// ─────────────────────────────────────────────────────────────

describe('topologicalSort', () => {
  test('empty graph', () => {
    const result = topologicalSort(new Map())
    expect(result.order).toEqual([])
    expect(result.levels).toEqual([])
    expect(result.complete).toBe(true)
  })

  test('single node', () => {
    const deps = new Map([['A', new Set<string>()]])
    const result = topologicalSort(deps)
    expect(result.order).toEqual(['A'])
    expect(result.levels).toEqual([['A']])
    expect(result.complete).toBe(true)
  })

  test('chain: A → B → C', () => {
    const deps = new Map([
      ['A', new Set<string>()],
      ['B', new Set(['A'])],
      ['C', new Set(['B'])],
    ])
    const result = topologicalSort(deps)
    expect(result.order).toEqual(['A', 'B', 'C'])
    expect(result.levels).toEqual([['A'], ['B'], ['C']])
    expect(result.complete).toBe(true)
  })

  test('parallel: A, B independent', () => {
    const deps = new Map([
      ['A', new Set<string>()],
      ['B', new Set<string>()],
    ])
    const result = topologicalSort(deps)
    expect(result.levels).toEqual([['A', 'B']])
    expect(result.complete).toBe(true)
  })

  test('diamond: A → B,C → D', () => {
    const deps = new Map([
      ['A', new Set<string>()],
      ['B', new Set(['A'])],
      ['C', new Set(['A'])],
      ['D', new Set(['B', 'C'])],
    ])
    const result = topologicalSort(deps)
    expect(result.levels[0]).toEqual(['A'])
    expect(result.levels[1]).toEqual(['B', 'C'])
    expect(result.levels[2]).toEqual(['D'])
    expect(result.complete).toBe(true)
  })

  test('incomplete when cycle exists', () => {
    const deps = new Map([
      ['A', new Set(['B'])],
      ['B', new Set(['A'])],
    ])
    const result = topologicalSort(deps)
    expect(result.complete).toBe(false)
    expect(result.order.length).toBe(0)
  })
})

// ─────────────────────────────────────────────────────────────
// tarjanSCC
// ─────────────────────────────────────────────────────────────

describe('tarjanSCC', () => {
  test('no cycles: each node is its own SCC', () => {
    const deps = new Map([
      ['A', new Set<string>()],
      ['B', new Set(['A'])],
    ])
    const sccs = tarjanSCC(deps)
    expect(sccs.every(s => s.length === 1)).toBe(true)
  })

  test('simple cycle: A ↔ B', () => {
    const deps = new Map([
      ['A', new Set(['B'])],
      ['B', new Set(['A'])],
    ])
    const sccs = tarjanSCC(deps)
    const cycles = sccs.filter(s => s.length > 1)
    expect(cycles.length).toBe(1)
    expect(cycles[0].sort()).toEqual(['A', 'B'])
  })

  test('triangle cycle: A → B → C → A', () => {
    const deps = new Map([
      ['A', new Set(['C'])],
      ['B', new Set(['A'])],
      ['C', new Set(['B'])],
    ])
    const sccs = tarjanSCC(deps)
    const cycles = sccs.filter(s => s.length > 1)
    expect(cycles.length).toBe(1)
    expect(cycles[0].sort()).toEqual(['A', 'B', 'C'])
  })

  test('mixed: cycle + acyclic', () => {
    const deps = new Map([
      ['A', new Set(['B'])],
      ['B', new Set(['A'])],
      ['C', new Set(['A'])],  // C depends on A but not in cycle
    ])
    const sccs = tarjanSCC(deps)
    const cycles = sccs.filter(s => s.length > 1)
    expect(cycles.length).toBe(1)
    expect(cycles[0].sort()).toEqual(['A', 'B'])
    const singletons = sccs.filter(s => s.length === 1).map(s => s[0])
    expect(singletons).toContain('C')
  })
})

// ─────────────────────────────────────────────────────────────
// extractInstanceInfo
// ─────────────────────────────────────────────────────────────

describe('extractInstanceInfo', () => {
  test('converts port type strings', () => {
    const def = {
      typeName: 'Test',
      inputNames: ['a', 'b'],
      outputNames: ['out'],
      registerNames: ['state'],
      inputPortTypes: ['float', 'bool'] as (string | undefined)[],
      outputPortTypes: ['int'] as (string | undefined)[],
      registerPortTypes: [undefined] as (string | undefined)[],
    }
    const info = extractInstanceInfo('Test1', def)
    expect(info.name).toBe('Test1')
    expect(portTypeEqual(info.inputTypes[0], Float)).toBe(true)
    expect(portTypeEqual(info.inputTypes[1], Bool)).toBe(true)
    expect(portTypeEqual(info.outputTypes[0], Int)).toBe(true)
    expect(portTypeEqual(info.registerTypes[0], Float)).toBe(true) // undefined → Float
  })
})
