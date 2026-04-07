/**
 * compiler.test.ts — Tests for the patch → Term compiler.
 *
 * Tests the pure functions (dependency extraction, topo sort, SCC, module→Term)
 * and integration tests that compile small hand-built patches to Terms.
 */

import { describe, test, expect } from 'bun:test'
import {
  portTypeFromString,
  exprDependencies,
  buildDependencyGraph,
  topologicalSort,
  tarjanSCC,
  moduleToTerm,
  compilePatch,
  extractModuleInfo,
  CompilerError,
  type ModuleInfo,
  type CompilerInput,
} from './compiler'
import {
  Float, Int, Bool, Unit, ArrayType,
  product, portTypeEqual, portTypeToString,
} from './term'
import { inferType } from './type_check'
import type { ExprNode } from './patch'

// ─────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────

/** Build a simple ModuleInfo for testing. */
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
): ModuleInfo {
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

  test('unknown name without registry defaults to float', () => {
    const t = portTypeFromString('MyStruct')
    expect(portTypeEqual(t, Float)).toBe(true)
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
    const expr: ExprNode = { op: 'ref', module: 'VCO1', output: 'sin' }
    const deps = exprDependencies(expr)
    expect(deps.size).toBe(1)
    expect(deps.has('VCO1')).toBe(true)
  })

  test('binary op with nested refs', () => {
    const expr: ExprNode = {
      op: 'mul',
      args: [
        { op: 'ref', module: 'VCO1', output: 'sin' },
        { op: 'ref', module: 'LFO1', output: 'out' },
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
        { op: 'ref', module: 'VCO1', output: 'sin' },
        0.5,
      ],
    }
    const deps = exprDependencies(expr)
    expect(deps.size).toBe(1)
    expect(deps.has('VCO1')).toBe(true)
  })

  test('array expression', () => {
    const expr: ExprNode = [
      { op: 'ref', module: 'A', output: 'x' },
      { op: 'ref', module: 'B', output: 'y' },
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
      ['B:in', { op: 'ref', module: 'A', output: 'out' }],
    ])
    const graph = buildDependencyGraph(['A', 'B'], exprs)
    expect(graph.get('A')!.size).toBe(0)
    expect(graph.get('B')!.has('A')).toBe(true)
  })

  test('diamond: A → B, A → C, B → D, C → D', () => {
    const exprs = new Map<string, ExprNode>([
      ['B:in', { op: 'ref', module: 'A', output: 'out' }],
      ['C:in', { op: 'ref', module: 'A', output: 'out' }],
      ['D:x', { op: 'ref', module: 'B', output: 'out' }],
      ['D:y', { op: 'ref', module: 'C', output: 'out' }],
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
      ['A:in', { op: 'ref', module: 'A', output: 'out' }],
    ])
    const graph = buildDependencyGraph(['A'], exprs)
    expect(graph.get('A')!.size).toBe(0)
  })

  test('ignores refs to unknown modules', () => {
    const exprs = new Map<string, ExprNode>([
      ['A:in', { op: 'ref', module: 'UNKNOWN', output: 'out' }],
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
// moduleToTerm
// ─────────────────────────────────────────────────────────────

describe('moduleToTerm', () => {
  test('stateless module: morphism inputs → outputs', () => {
    const info = modInfo('Gain', { inputs: ['in'], outputs: ['out'] })
    const term = moduleToTerm(info)
    expect(term.tag).toBe('morphism')
    const t = inferType(term)
    expect(portTypeEqual(t.dom, Float)).toBe(true)
    expect(portTypeEqual(t.cod, Float)).toBe(true)
  })

  test('multi-output stateless module', () => {
    const info = modInfo('VCO', {
      inputs: ['freq'],
      outputs: ['saw', 'tri', 'sin', 'sqr'],
    })
    const term = moduleToTerm(info)
    const t = inferType(term)
    expect(portTypeEqual(t.dom, Float)).toBe(true)
    expect(portTypeEqual(t.cod, product([Float, Float, Float, Float]))).toBe(true)
  })

  test('stateful module: trace wraps morphism', () => {
    const info = modInfo('Counter', {
      inputs: ['gate'],
      outputs: ['count'],
      registers: ['state'],
      inputTypes: ['bool'],
      outputTypes: ['int'],
      registerTypes: ['int'],
    })
    const term = moduleToTerm(info)
    expect(term.tag).toBe('trace')
    const t = inferType(term)
    // trace peels off state: (Bool⊗Int → Int⊗Int) becomes Bool → Int
    expect(portTypeEqual(t.dom, Bool)).toBe(true)
    expect(portTypeEqual(t.cod, Int)).toBe(true)
  })

  test('multi-register stateful module', () => {
    const info = modInfo('VCO', {
      inputs: ['freq', 'fm', 'fm_index'],
      outputs: ['saw', 'tri', 'sin', 'sqr'],
      registers: ['phase', 'tri_state'],
    })
    const term = moduleToTerm(info)
    expect(term.tag).toBe('trace')
    const t = inferType(term)
    // domain: freq ⊗ fm ⊗ fm_index = Float ⊗ Float ⊗ Float
    expect(portTypeEqual(t.dom, product([Float, Float, Float]))).toBe(true)
    // codomain: saw ⊗ tri ⊗ sin ⊗ sqr = Float ⊗ Float ⊗ Float ⊗ Float
    expect(portTypeEqual(t.cod, product([Float, Float, Float, Float]))).toBe(true)
  })
})

// ─────────────────────────────────────────────────────────────
// extractModuleInfo
// ─────────────────────────────────────────────────────────────

describe('extractModuleInfo', () => {
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
    const info = extractModuleInfo('Test1', def)
    expect(info.name).toBe('Test1')
    expect(portTypeEqual(info.inputTypes[0], Float)).toBe(true)
    expect(portTypeEqual(info.inputTypes[1], Bool)).toBe(true)
    expect(portTypeEqual(info.outputTypes[0], Int)).toBe(true)
    expect(portTypeEqual(info.registerTypes[0], Float)).toBe(true) // undefined → Float
  })
})

// ─────────────────────────────────────────────────────────────
// compilePatch — integration tests
// ─────────────────────────────────────────────────────────────

describe('compilePatch', () => {
  test('empty patch', () => {
    const result = compilePatch({
      modules: new Map(),
      inputExprNodes: new Map(),
      graphOutputs: [],
    })
    expect(result.term.tag).toBe('id')
    expect(result.levels).toEqual([])
  })

  test('single stateless module, constant input', () => {
    const input: CompilerInput = {
      modules: new Map([
        ['Gain1', modInfo('Gain1', { inputs: ['in'], outputs: ['out'] })],
      ]),
      inputExprNodes: new Map([['Gain1:in', 1.0]]),
      graphOutputs: [{ module: 'Gain1', output: 'out' }],
    }
    const result = compilePatch(input)
    // Should type-check without error
    const t = inferType(result.term)
    // Unit → Float (wiring generates constant, module processes, output projects)
    expect(portTypeEqual(t.dom, Unit)).toBe(true)
    expect(portTypeEqual(t.cod, Float)).toBe(true)
    expect(result.levels).toEqual([['Gain1']])
  })

  test('two-module chain: A → B', () => {
    const input: CompilerInput = {
      modules: new Map([
        ['A', modInfo('A', { inputs: ['freq'], outputs: ['out'] })],
        ['B', modInfo('B', { inputs: ['in'], outputs: ['out'] })],
      ]),
      inputExprNodes: new Map([
        ['A:freq', 440],
        ['B:in', { op: 'ref', module: 'A', output: 'out' }],
      ]),
      graphOutputs: [{ module: 'B', output: 'out' }],
    }
    const result = compilePatch(input)
    const t = inferType(result.term)
    expect(portTypeEqual(t.dom, Unit)).toBe(true)
    expect(portTypeEqual(t.cod, Float)).toBe(true)
    expect(result.levels).toEqual([['A'], ['B']])
  })

  test('parallel modules at level 0', () => {
    const input: CompilerInput = {
      modules: new Map([
        ['A', modInfo('A', { inputs: ['freq'], outputs: ['out'] })],
        ['B', modInfo('B', { inputs: ['freq'], outputs: ['out'] })],
      ]),
      inputExprNodes: new Map([
        ['A:freq', 440],
        ['B:freq', 880],
      ]),
      graphOutputs: [
        { module: 'A', output: 'out' },
        { module: 'B', output: 'out' },
      ],
    }
    const result = compilePatch(input)
    const t = inferType(result.term)
    expect(portTypeEqual(t.dom, Unit)).toBe(true)
    // Two Float outputs
    expect(portTypeEqual(t.cod, product([Float, Float]))).toBe(true)
    expect(result.levels).toEqual([['A', 'B']])
  })

  test('diamond: A → B,C → D', () => {
    const input: CompilerInput = {
      modules: new Map([
        ['A', modInfo('A', { inputs: ['freq'], outputs: ['out'] })],
        ['B', modInfo('B', { inputs: ['in'], outputs: ['out'] })],
        ['C', modInfo('C', { inputs: ['in'], outputs: ['out'] })],
        ['D', modInfo('D', { inputs: ['x', 'y'], outputs: ['out'] })],
      ]),
      inputExprNodes: new Map<string, ExprNode>([
        ['A:freq', 100],
        ['B:in', { op: 'ref', module: 'A', output: 'out' }],
        ['C:in', { op: 'ref', module: 'A', output: 'out' }],
        ['D:x', { op: 'ref', module: 'B', output: 'out' }],
        ['D:y', { op: 'ref', module: 'C', output: 'out' }],
      ]),
      graphOutputs: [{ module: 'D', output: 'out' }],
    }
    const result = compilePatch(input)
    const t = inferType(result.term)
    expect(portTypeEqual(t.dom, Unit)).toBe(true)
    expect(portTypeEqual(t.cod, Float)).toBe(true)
    expect(result.levels[0]).toEqual(['A'])
    expect(result.levels[1]).toEqual(['B', 'C'])
    expect(result.levels[2]).toEqual(['D'])
  })

  test('stateful module compiles with trace', () => {
    const input: CompilerInput = {
      modules: new Map([
        ['Osc1', modInfo('Osc1', {
          inputs: ['freq'],
          outputs: ['out'],
          registers: ['phase'],
        })],
      ]),
      inputExprNodes: new Map([['Osc1:freq', 440]]),
      graphOutputs: [{ module: 'Osc1', output: 'out' }],
    }
    const result = compilePatch(input)
    const t = inferType(result.term)
    expect(portTypeEqual(t.dom, Unit)).toBe(true)
    expect(portTypeEqual(t.cod, Float)).toBe(true)
  })

  test('expression wiring (mul, add)', () => {
    const input: CompilerInput = {
      modules: new Map([
        ['A', modInfo('A', { inputs: ['freq'], outputs: ['out'] })],
        ['B', modInfo('B', { inputs: ['in'], outputs: ['out'] })],
      ]),
      inputExprNodes: new Map<string, ExprNode>([
        ['A:freq', 440],
        ['B:in', { op: 'mul', args: [
          { op: 'ref', module: 'A', output: 'out' },
          0.5,
        ]}],
      ]),
      graphOutputs: [{ module: 'B', output: 'out' }],
    }
    const result = compilePatch(input)
    const t = inferType(result.term)
    expect(portTypeEqual(t.dom, Unit)).toBe(true)
    expect(portTypeEqual(t.cod, Float)).toBe(true)
  })

  test('cycle detection throws', () => {
    const input: CompilerInput = {
      modules: new Map([
        ['A', modInfo('A', { inputs: ['in'], outputs: ['out'] })],
        ['B', modInfo('B', { inputs: ['in'], outputs: ['out'] })],
      ]),
      inputExprNodes: new Map<string, ExprNode>([
        ['A:in', { op: 'ref', module: 'B', output: 'out' }],
        ['B:in', { op: 'ref', module: 'A', output: 'out' }],
      ]),
      graphOutputs: [],
    }
    expect(() => compilePatch(input)).toThrow(CompilerError)
    expect(() => compilePatch(input)).toThrow(/Feedback cycles/)
  })

  test('no graph outputs — term still type-checks', () => {
    const input: CompilerInput = {
      modules: new Map([
        ['A', modInfo('A', { inputs: ['freq'], outputs: ['out'] })],
      ]),
      inputExprNodes: new Map([['A:freq', 440]]),
      graphOutputs: [],
    }
    const result = compilePatch(input)
    // No output projection, just the module pipeline
    inferType(result.term)
  })

  test('multi-output module feeding two consumers', () => {
    const input: CompilerInput = {
      modules: new Map([
        ['VCO', modInfo('VCO', {
          inputs: ['freq'],
          outputs: ['saw', 'sin'],
        })],
        ['FiltA', modInfo('FiltA', { inputs: ['in'], outputs: ['out'] })],
        ['FiltB', modInfo('FiltB', { inputs: ['in'], outputs: ['out'] })],
      ]),
      inputExprNodes: new Map<string, ExprNode>([
        ['VCO:freq', 440],
        ['FiltA:in', { op: 'ref', module: 'VCO', output: 'saw' }],
        ['FiltB:in', { op: 'ref', module: 'VCO', output: 'sin' }],
      ]),
      graphOutputs: [
        { module: 'FiltA', output: 'out' },
        { module: 'FiltB', output: 'out' },
      ],
    }
    const result = compilePatch(input)
    const t = inferType(result.term)
    expect(portTypeEqual(t.dom, Unit)).toBe(true)
    expect(portTypeEqual(t.cod, product([Float, Float]))).toBe(true)
  })

  test('mixed types (int, bool) propagate correctly', () => {
    const input: CompilerInput = {
      modules: new Map([
        ['Gate', modInfo('Gate', {
          inputs: ['thresh'],
          outputs: ['out'],
          inputTypes: ['float'],
          outputTypes: ['bool'],
        })],
        ['Counter', modInfo('Counter', {
          inputs: ['gate'],
          outputs: ['count'],
          registers: ['state'],
          inputTypes: ['bool'],
          outputTypes: ['int'],
          registerTypes: ['int'],
        })],
      ]),
      inputExprNodes: new Map<string, ExprNode>([
        ['Gate:thresh', 0.5],
        ['Counter:gate', { op: 'ref', module: 'Gate', output: 'out' }],
      ]),
      graphOutputs: [{ module: 'Counter', output: 'count' }],
    }
    const result = compilePatch(input)
    const t = inferType(result.term)
    expect(portTypeEqual(t.dom, Unit)).toBe(true)
    expect(portTypeEqual(t.cod, Int)).toBe(true)
  })
})
