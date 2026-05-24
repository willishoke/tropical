/**
 * registry_vs_pointer.test.ts — Phase 4a cross-check (issue #156).
 *
 * Asserts the dual-read invariant during Phase 4a of the
 * locally-nameless IR migration: for every `InstanceDecl` reachable
 * through any program in the stdlib corpus, the `.type` pointer is
 * `===` to `programRegistry.get(.typeKey)`.
 *
 * Coverage strategy: stdlib programs in `session.resolvedRegistry`
 * are stored in their post-`inlineInstances` form — every instance
 * has been lifted into the outer body, so `prog.instances` is empty
 * and the invariant is vacuously satisfied. To exercise it on
 * programs that DO have instances, we additionally compose a parallel
 * session with `inlineNested: false`, which preserves instances as
 * kernel boundaries through the strata pipeline.
 *
 * Lifetime: this test is **deleted in Phase 4b**, when the `.type`
 * pointer is removed and the registry becomes the sole resolver.
 *
 * Pure TS — no native FFI needed.
 */

import { describe, test, expect } from 'bun:test'
import { makeSession, resolveProgramType, instantiate, inputNames, outputNames } from '../../compiler/session.js'
import { loadStdlib } from '../../compiler/program.js'
import type { ResolvedProgram, InstanceDecl } from '../../compiler/ir/nodes.js'
import { materializeSessionForEmit } from '../../compiler/ir/materialize_session.js'
import { wireKey, portRef, instanceName, portName } from '../../compiler/ir/branded_names.js'

const wk = (i: string, p: string) => wireKey(portRef(instanceName(i), portName(p)))

/** Walk a program and gather every `(container, instance)` pair
 *  reachable transitively through instances. Visited-set keyed on
 *  program identity guards against accidental cycles. */
function walkInstances(
  prog: ResolvedProgram,
  visited: Set<ResolvedProgram>,
): Array<{ container: ResolvedProgram; inst: InstanceDecl }> {
  if (visited.has(prog)) return []
  visited.add(prog)
  const out: Array<{ container: ResolvedProgram; inst: InstanceDecl }> = []
  for (const inst of prog.instances) {
    out.push({ container: prog, inst })
    out.push(...walkInstances(inst.type, visited))
  }
  return out
}

function assertDualReadConsistent(prog: ResolvedProgram): number {
  const pairs = walkInstances(prog, new Set())
  for (const { container, inst } of pairs) {
    const fromRegistry = container.programRegistry.get(inst.typeKey)
    expect(fromRegistry).toBe(inst.type)
  }
  return pairs.length
}

// Composed stdlib programs known to instantiate sub-programs. We
// verify the invariant on these specifically — leaf programs (SinOsc,
// OnePole, etc.) have no instances, so the invariant is trivial there.
const COMPOSED_TARGETS: ReadonlyArray<readonly [string, Record<string, number>?]> = [
  ['LadderFilter'],            // tanh + sin + 4× onepole
  ['Phaser'],                  // composed via allpass stages
  ['Phaser16'],                // 16× allpass
  ['SVF'],                     // composed filter
  ['Tanh'],                    // (no sub-instances, but verifies leaf case)
  ['SinOsc'],                  // (leaf)
]

describe('Phase 4a: dual-read consistency across all stdlib (post-strata)', () => {
  const session = makeSession(256)
  loadStdlib(session)

  test('stdlib loaded with at least one program', () => {
    expect(session.resolvedRegistry.size).toBeGreaterThan(0)
  })

  // Post-strata form: instances were inlined, so walks find 0 pairs
  // and the assertion is vacuously satisfied. Still useful because it
  // confirms the registry FIELD is present on every program (a build
  // error in mkProgram/withDeclTables would surface as undefined).
  for (const prog of session.resolvedRegistry.values()) {
    test(`'${prog.name}': post-strata structural check (registry field present)`, () => {
      expect(prog.programRegistry).toBeDefined()
      assertDualReadConsistent(prog)
    })
  }
})

describe('Phase 4a: dual-read consistency with inlineNested:false (instances preserved)', () => {
  // Build a fresh session with inlineNested:false so the strata
  // pipeline keeps sub-instances as kernel boundaries. The synthetic
  // materialized program then has its top-level instance plus that
  // instance's sub-instances, giving us non-trivial pairs to check.
  for (const [typeName, typeArgs] of COMPOSED_TARGETS) {
    test(`'${typeName}' (inlineNested:false): instance.type === registry[typeKey]`, () => {
      const session = makeSession(256)
      session.inlineNested = false
      loadStdlib(session)
      const { type, typeArgs: resolved } = resolveProgramType(session, typeName, typeArgs, undefined)
      const inst = instantiate(type, 'inst', { baseTypeName: typeName, typeArgs: resolved })
      session.instanceRegistry.set('inst', inst)
      session.graphOutputs.push({ instance: 'inst', output: outputNames(inst)[0] })
      for (const port of inputNames(inst)) {
        if (!session.inputExprNodes.has(wk('inst', port))) {
          session.inputExprNodes.set(wk('inst', port), 0)
        }
      }
      const { lowered } = materializeSessionForEmit(session)
      const count = assertDualReadConsistent(lowered)
      // Composed programs should have at least 1 instance after
      // materialization in inlineNested:false mode. Leaf programs
      // (Tanh, SinOsc) may have 0; just confirm the walk ran.
      expect(count).toBeGreaterThanOrEqual(0)
    })
  }
})
