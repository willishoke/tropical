/**
 * session_as_parsed_program.test.ts — SPIKE differential.
 *
 * Two assertions, answering the two architecture questions from the
 * design discussion:
 *
 *  Q1 (does `materialize_session.ts` duplicate the elaborator?):
 *      For each session in the slice, the ResolvedProgram built by
 *      `materializeSessionToResolvedIR` (the hand-rolled mini-elaborator)
 *      is STRUCTURALLY EQUAL — modulo idx/decl-order, compared by a
 *      name-keyed canonical form — to the one built by
 *      `elaborate(sessionToParsedProgram(s), sessionTypeResolver(s))`
 *      (the shared front door + the existing ExternalProgramResolver
 *      LINK hook). Equality ⇒ the materializer is a redundant elaborator
 *      on this slice.
 *
 *  Q2 (can we re-elaborate per compile and keep the JIT warm?):
 *      `compileSession` is BYTE-DETERMINISTIC across freshly-built
 *      sessions. The JIT cache keys on the serialized plan bytes
 *      (OrcJitEngine.cpp), downstream of elaboration; identical bytes ⇒
 *      warm cache regardless of how many times we re-elaborated. This is
 *      the invariant that makes "re-elaborate freely" safe.
 *
 * Pure TS — no native lib, no audio device. Run: bun test tests/spike
 */

import { describe, test, expect } from 'bun:test'
import {
  makeSession, resolveProgramType, instantiate, outputNames,
} from '../../compiler/session.js'
import { loadStdlib } from '../../compiler/program.js'
import { compileSession } from '../../compiler/ir/compile_session.js'
import { materializeSessionToResolvedIR } from '../../compiler/ir/materialize_session.js'
import { liftWiresToInstances } from '../../compiler/ir/lift_wires.js'
import { extractSessionDelays } from '../../compiler/ir/lowering/extract_session_delays.js'
import { elaborate } from '../../compiler/ir/elaborator.js'
import {
  sessionToParsedProgram, sessionTypeResolver,
} from '../../compiler/ir/session_to_parsed.js'
import { wireKey, portRef, instanceName, portName } from '../../compiler/ir/branded_names.js'
import type { SessionState } from '../../compiler/session.js'
import type { ResolvedProgram, ResolvedExpr, BodyDecl } from '../../compiler/ir/nodes.js'

// ── Name-keyed canonical form of a (root) ResolvedProgram ────────────
//
// Prints the session graph by NAME — instance names, port names, reg
// names, param names — so two independently-built programs with
// different idx assignments / decl orderings compare equal iff they
// encode the same graph.

function exprToStr(e: ResolvedExpr, prog: ResolvedProgram): string {
  if (typeof e === 'number') return `#${e}`
  if (typeof e === 'boolean') return `#${e}`
  if (Array.isArray(e)) return `[${e.map(x => exprToStr(x, prog)).join(',')}]`
  switch (e.op) {
    case 'regRef':    return `reg:${prog.regs[e.idx].name}`
    case 'paramRef':  return `param:${prog.params[e.idx].name}`
    case 'inputRef':  return `in:${prog.ports.inputs[e.idx].name}`
    case 'nestedOut': {
      const inst = prog.instances[e.instance]
      const tgt = prog.programRegistry.get(inst.typeKey)
      const outName = tgt ? tgt.ports.outputs[e.output].name : `#out${e.output}`
      return `${inst.name}.${outName}`
    }
    case 'sampleRate':  return 'sampleRate'
    case 'sampleIndex': return 'sampleIndex'
    default: {
      const anyE = e as { op: string; args?: ResolvedExpr[] }
      if (Array.isArray(anyE.args)) {
        return `${anyE.op}(${anyE.args.map(a => exprToStr(a, prog)).join(',')})`
      }
      return `<${anyE.op}>`
    }
  }
}

function canon(prog: ResolvedProgram): string[] {
  const lines: string[] = []
  const decls: BodyDecl[] = prog.body.decls
  for (const d of decls) {
    if (d.op === 'instanceDecl') {
      const tgt = prog.programRegistry.get(d.typeKey)
      const ins = [...d.inputs]
        .sort((a, b) => a.port - b.port)
        .map(i => {
          const pn = tgt ? tgt.ports.inputs[i.port].name : `#in${i.port}`
          return `${pn}=${exprToStr(i.value, prog)}`
        })
      const ta = [...d.typeArgs]
        .sort((a, b) => a.param - b.param)
        .map(t => `${t.param}=${t.value}`)
      lines.push(`inst ${d.name}:${d.typeKey}<${ta.join(',')}>(${ins.join(';')})`)
    } else if (d.op === 'regDecl') {
      const upd = d.update !== undefined ? exprToStr(d.update, prog) : '-'
      lines.push(`reg ${d.name} init=${exprToStr(d.init, prog)} update=${upd}`)
    } else if (d.op === 'paramDecl') {
      lines.push(`param ${d.name}`)
    }
  }
  return lines.sort()
}

// ── Session builders (the slice) ─────────────────────────────────────

/** Single instance, no wiring. Exercises the instance-decl + external
 *  resolver path (and, for stateful types, the instance's own internal
 *  regs living inside the boxed type). */
function singleInstance(typeName: string): SessionState {
  const s = makeSession()
  loadStdlib(s)
  const { type } = resolveProgramType(s, typeName, undefined, undefined)
  const inst = instantiate(type, 'inst', { baseTypeName: typeName, typeArgs: new Map() })
  s.instanceRegistry.set('inst', inst)
  for (const outName of outputNames(inst)) {
    s.graphOutputs.push({ instance: 'inst', output: outName })
  }
  return s
}

/** Two instances wired the MCP way (setWireExpr auto-delays every wire),
 *  so the wires become hoisted scalar delays → synthetic regs. Exercises
 *  the wiring + per-wire-delay path that `extractSessionDelays` +
 *  `materializeSessionDelaySlots` hand-manage. */
function wiredPair(): SessionState {
  const s = makeSession()
  loadStdlib(s)
  const sin = resolveProgramType(s, 'Sin', undefined, undefined).type
  const vca = resolveProgramType(s, 'VCA', undefined, undefined).type
  s.instanceRegistry.set('osc', instantiate(sin, 'osc', { baseTypeName: 'Sin' }))
  s.instanceRegistry.set('amp', instantiate(vca, 'amp', { baseTypeName: 'VCA' }))
  // MCP-style wires: each is auto-wrapped in a unit delay by setWireExpr.
  // setWireExpr is imported lazily to keep the import list focused.
  const { setWireExpr } = require('../../compiler/session.js') as typeof import('../../compiler/session.js')
  setWireExpr(s, portRef(instanceName('amp'), portName('audio')),
    { op: 'ref', instance: 'osc', output: 'out' } as never, { id: 'w_audio' })
  setWireExpr(s, portRef(instanceName('amp'), portName('cv')),
    0.5 as never, { id: 'w_cv' })
  s.graphOutputs.push({ instance: 'amp', output: 'out' })
  return s
}

/** Run the same pre-emit passes `compile_session` runs before
 *  `materializeSessionToResolvedIR` sees the session, so both builders
 *  consume identical post-extraction state. */
function prepare(s: SessionState): void {
  liftWiresToInstances(s)
  extractSessionDelays(s)
}

const SINGLE_TYPES = [
  'Sin', 'Cos', 'Exp', 'Log', 'Tanh',
  'OnePole', 'SoftClip', 'BitCrusher', 'CrossFade',
  'NoiseLFSR', 'AllpassDelay', 'CombDelay', 'VCA',
] as const

// ── Q1: materializer ≡ elaborator (structural) ───────────────────────

describe('Q1 — materialize_session ≡ elaborate∘sessionToParsedProgram', () => {
  for (const typeName of SINGLE_TYPES) {
    test(`single instance: ${typeName}`, () => {
      const s = singleInstance(typeName)
      prepare(s)

      const viaMaterialize = materializeSessionToResolvedIR(s)
      const viaElaborate = elaborate(sessionToParsedProgram(s), sessionTypeResolver(s))

      expect(canon(viaElaborate)).toEqual(canon(viaMaterialize))
    })
  }

  test('two-instance wired pair (per-wire scalar delays)', () => {
    const s = wiredPair()
    prepare(s)

    const viaMaterialize = materializeSessionToResolvedIR(s)
    const viaElaborate = elaborate(sessionToParsedProgram(s), sessionTypeResolver(s))

    const cm = canon(viaMaterialize)
    const ce = canon(viaElaborate)
    // Print once so a human can eyeball the graph the spike is matching.
    if (process.env.SPIKE_DUMP) console.log('materialize:', cm, '\nelaborate:', ce)

    expect(ce).toEqual(cm)
    // Guard against a vacuous match: the graph must actually contain the
    // two instances and the two hoisted-delay regs.
    expect(cm.some(l => l.startsWith('inst osc:'))).toBe(true)
    expect(cm.some(l => l.startsWith('inst amp:'))).toBe(true)
    expect(cm.filter(l => l.startsWith('reg ')).length).toBe(2)
  })
})

// ── Q2: compileSession is byte-deterministic across re-elaboration ───

describe('Q2 — compileSession plan bytes are stable across fresh re-elaboration', () => {
  const cases: Array<[string, () => SessionState]> = [
    ...SINGLE_TYPES.map(t => [`single:${t}`, () => singleInstance(t)] as [string, () => SessionState]),
    ['wiredPair', wiredPair],
  ]
  for (const [label, build] of cases) {
    test(label, () => {
      // Two independently-built sessions → two independent elaborations
      // (fresh session ⇒ fresh resolveProgramType ⇒ fresh strata). If the
      // serialized plans are byte-identical, the JIT cache (keyed on these
      // bytes) stays warm no matter how often we re-elaborate.
      const a = JSON.stringify(compileSession(build()))
      const b = JSON.stringify(compileSession(build()))
      expect(a).toBe(b)
    })
  }
})
