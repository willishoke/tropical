/**
 * allocate_input_slots.test.ts — unit coverage for the input-slot
 * allocator added in Phase 2 of the input-slot refactor.
 *
 * Pure-TS: builds a fake `Compiled` directly (no FFI / no
 * libtropical.dylib needed) so the test can run in any environment.
 * Pins the contract: `allocateInputSlots` mirrors
 * `allocateOutputSlots` (slot indices, port-meta, idempotency,
 * unified slotCount accounting) and shares the unified slot
 * namespace with outputs / params / delays without collision.
 */

import { describe, test, expect } from 'bun:test'
import {
  makeSession,
  allocateInputSlots,
  allocateOutputSlots,
} from './session.js'
import { instanceName, slotKey } from './ir/branded_names.js'
import { makeCompiled } from './program_types.js'
import type { ResolvedProgram, InputDecl, OutputDecl } from './ir/nodes.js'

function fakeCompiled(name: string, inputs: string[], outputs: string[]) {
  const inputDecls: InputDecl[] = inputs.map(n => ({
    op: 'inputDecl', name: n, type: { kind: 'scalar', scalar: 'float' },
  }))
  const outputDecls: OutputDecl[] = outputs.map(n => ({
    op: 'outputDecl', name: n, type: { kind: 'scalar', scalar: 'float' },
  }))
  const prog: ResolvedProgram = {
    op: 'program', name, typeParams: [],
    ports: { inputs: inputDecls, outputs: outputDecls, typeDefs: [] },
    body: { op: 'block', decls: [], assigns: [] },
  }
  return makeCompiled(prog, { displayName: name })
}

describe('allocateInputSlots', () => {
  test('allocates one slot per scalar input port', () => {
    const session = makeSession()
    const compiled = fakeCompiled('Foo', ['a', 'b', 'c'], ['out'])
    allocateInputSlots(session, instanceName('foo'), compiled)
    expect(session.slotCount).toBe(3)
    expect(session.inputSlotRegistry.get(slotKey(instanceName('foo'), 'a'))).toBe(0)
    expect(session.inputSlotRegistry.get(slotKey(instanceName('foo'), 'b'))).toBe(1)
    expect(session.inputSlotRegistry.get(slotKey(instanceName('foo'), 'c'))).toBe(2)
  })

  test('idempotent: second call is a no-op', () => {
    const session = makeSession()
    const compiled = fakeCompiled('Foo', ['a'], ['out'])
    allocateInputSlots(session, instanceName('foo'), compiled)
    const countAfterFirst = session.slotCount
    allocateInputSlots(session, instanceName('foo'), compiled)
    expect(session.slotCount).toBe(countAfterFirst)
  })

  test('shares unified slotCount with allocateOutputSlots; no collisions', () => {
    const session = makeSession()
    const compiled = fakeCompiled('Foo', ['a', 'b'], ['out0', 'out1'])
    // Outputs first
    allocateOutputSlots(session, instanceName('foo'), compiled)
    expect(session.slotCount).toBe(2)
    // Inputs allocated AFTER outputs use the next free indices
    allocateInputSlots(session, instanceName('foo'), compiled)
    expect(session.slotCount).toBe(4)
    expect(session.outputSlotRegistry.get(slotKey(instanceName('foo'), 'out0'))).toBe(0)
    expect(session.outputSlotRegistry.get(slotKey(instanceName('foo'), 'out1'))).toBe(1)
    expect(session.inputSlotRegistry.get(slotKey(instanceName('foo'), 'a'))).toBe(2)
    expect(session.inputSlotRegistry.get(slotKey(instanceName('foo'), 'b'))).toBe(3)
  })

  test('input and output slot maps are disjoint keyspaces — same port-name allowed on both', () => {
    // A port can legally be `x` on both an input and an output of
    // different instances (or even the same instance). The two maps
    // must be queried separately; the key namespace is per-map, not
    // shared.
    const session = makeSession()
    const compiled = fakeCompiled('Foo', ['x'], ['x'])
    allocateInputSlots(session, instanceName('foo'), compiled)
    allocateOutputSlots(session, instanceName('foo'), compiled)
    const key = slotKey(instanceName('foo'), 'x')
    expect(session.inputSlotRegistry.has(key)).toBe(true)
    expect(session.outputSlotRegistry.has(key)).toBe(true)
    expect(session.inputSlotRegistry.get(key)).not.toBe(session.outputSlotRegistry.get(key))
  })

  test('records full WirePortMeta — names + types + portType', () => {
    const session = makeSession()
    const compiled = fakeCompiled('Foo', ['freq'], ['out'])
    allocateInputSlots(session, instanceName('foo'), compiled)
    const meta = session.inputPortMeta.get(slotKey(instanceName('foo'), 'freq'))
    expect(meta).toBeDefined()
    expect(meta!.scalarSlotNames.length).toBe(1)
    expect(meta!.scalarTypes).toEqual(['float'])
    expect(meta!.portType.kind).toBe('scalar')
  })
})
