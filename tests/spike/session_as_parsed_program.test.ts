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
import { readFileSync, readdirSync } from 'node:fs'
import { join } from 'node:path'
import {
  makeSession, resolveProgramType, instantiate, outputNames,
  allocateOutputSlots, allocateParamSlot, setWireExpr, loadJSON,
} from '../../compiler/session.js'
import { loadStdlib } from '../../compiler/program.js'
import { compileSession } from '../../compiler/ir/compile_session.js'
import { compileSessionSlotted } from '../../compiler/ir/compile_session_slotted.js'
import { materializeSessionToResolvedIR } from '../../compiler/ir/materialize_session.js'
import { liftWiresToInstances } from '../../compiler/ir/lift_wires.js'
import { extractSessionDelays } from '../../compiler/ir/lowering/extract_session_delays.js'
import { elaborate } from '../../compiler/ir/elaborator.js'
import {
  sessionToParsedProgram, sessionTypeResolver,
} from '../../compiler/ir/session_to_parsed.js'
import { portRef, instanceName, portName } from '../../compiler/ir/branded_names.js'
import type { FlatPlan } from '../../compiler/flat_plan.js'
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
  setWireExpr(s, portRef(instanceName('amp'), portName('audio')),
    { op: 'ref', instance: 'osc', output: 'out' } as never, { id: 'w_audio' })
  setWireExpr(s, portRef(instanceName('amp'), portName('cv')),
    0.5 as never, { id: 'w_cv' })
  s.graphOutputs.push({ instance: 'amp', output: 'out' })
  return s
}

/** Like `wiredPair`, but the cv wire reads a control-plane param. This is
 *  the case that exercises the PARAM-SLOT sidecar: the wire lowers to a
 *  `param:gain` module-slot read, and the plan's slot metadata must carry
 *  it. If the elaborator-built root produces the same plan bytes here, the
 *  param sidecar provably rejoins by name. */
function wiredPairWithParam(): SessionState {
  const s = makeSession()
  loadStdlib(s)
  const sin = resolveProgramType(s, 'Sin', undefined, undefined).type
  const vca = resolveProgramType(s, 'VCA', undefined, undefined).type
  s.instanceRegistry.set('osc', instantiate(sin, 'osc', { baseTypeName: 'Sin' }))
  s.instanceRegistry.set('amp', instantiate(vca, 'amp', { baseTypeName: 'VCA' }))
  allocateParamSlot(s, 'gain')
  setWireExpr(s, portRef(instanceName('amp'), portName('audio')),
    { op: 'ref', instance: 'osc', output: 'out' } as never, { id: 'w_audio' })
  setWireExpr(s, portRef(instanceName('amp'), portName('cv')),
    { op: 'param', name: 'gain' } as never, { id: 'w_cv' })
  s.graphOutputs.push({ instance: 'amp', output: 'out' })
  return s
}

/** Mirror `compile_session`'s pre-emit passes (lift → eager output-slot
 *  alloc → delay extraction) so both root builders consume identical
 *  post-extraction session state. */
function prepare(s: SessionState): void {
  liftWiresToInstances(s)
  for (const [name, inst] of s.instanceRegistry) {
    if (inst.compiled !== undefined) {
      allocateOutputSlots(s, instanceName(name), inst.compiled)
    }
  }
  extractSessionDelays(s)
}

/** The cutover shape: run the pre-passes, build the root via the shared
 *  elaborator front door (instead of `materializeSessionToResolvedIR`),
 *  and lower it through the SAME backend tail via the `rootOverride` seam.
 *  Everything downstream of the root — slot allocation, DAC stitch,
 *  param-slot threading — is the production code, unchanged. */
function compileViaElaborate(s: SessionState): FlatPlan {
  prepare(s)
  const root = elaborate(sessionToParsedProgram(s), sessionTypeResolver(s))
  return compileSessionSlotted(s, { rootProgram: true, rootOverride: root })
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

// ── Q3: the sidecar rejoins — full plan bytes match through the tail ──
//
// Lower the elaborator-built root through the production backend tail
// (slot alloc + partitionKernel + DAC stitch + param-slot threading) via
// the `rootOverride` seam, and compare the resulting `tropical_plan_5`
// bytes against the materializer path. Byte-identity ⇒ the DAC and
// param-slot sidecars rejoin by name with zero divergence, i.e.
// `materialize_session.ts` is replaceable with no downstream change.

describe('Q3 — elaborator-built root yields identical plan bytes (sidecar rejoins)', () => {
  for (const typeName of SINGLE_TYPES) {
    test(`single instance: ${typeName}`, () => {
      const baseline = JSON.stringify(compileSession(singleInstance(typeName)))
      const viaElab = JSON.stringify(compileViaElaborate(singleInstance(typeName)))
      expect(viaElab).toBe(baseline)
    })
  }

  test('two-instance wired pair (DAC stitch + per-wire delays)', () => {
    const baseline = JSON.stringify(compileSession(wiredPair()))
    const viaElab = JSON.stringify(compileViaElaborate(wiredPair()))
    expect(viaElab).toBe(baseline)
  })

  test('wired pair with control-plane param (param-slot sidecar)', () => {
    const basePlan = compileSession(wiredPairWithParam())
    // Non-vacuity: the param-slot sidecar must actually be present, else
    // byte-identity proves nothing about it.
    expect(basePlan.slot_names).toContain('param:gain')
    const viaElab = JSON.stringify(compileViaElaborate(wiredPairWithParam()))
    expect(viaElab).toBe(JSON.stringify(basePlan))
  })
})

// ── Q4: the live surface — every example patch matches byte-for-byte ──
//
// The audit, as a regression test. Load each `patches/*.json` (the real
// MCP authoring surface — multi-instance, nested generics, lifted
// array-literal wires, hand-written `delay()`s) and assert the
// elaborator-built root yields a byte-identical plan. Across this corpus
// `arrayDelays === 0` (the materializer's `isArray` delay branch is dead
// code on the live surface); the scalar + lifted-wire slice covers 100%
// of what real patches generate.

const PATCHES_DIR = join(import.meta.dir, '..', '..', 'patches')

function loadPatch(file: string): SessionState {
  const s = makeSession()
  loadStdlib(s)
  loadJSON(JSON.parse(readFileSync(join(PATCHES_DIR, file), 'utf8')), s)
  return s
}

describe('Q4 — every example patch: elaborator root ≡ materializer (byte-identical)', () => {
  const files = readdirSync(PATCHES_DIR).filter(f => f.endsWith('.json'))
  // Guard: the corpus must be non-empty, else this whole block is vacuous.
  test('corpus is non-empty', () => { expect(files.length).toBeGreaterThan(0) })
  for (const file of files) {
    test(file, () => {
      const baseline = JSON.stringify(compileSession(loadPatch(file)))
      const viaElab = JSON.stringify(compileViaElaborate(loadPatch(file)))
      expect(viaElab).toBe(baseline)
    })
  }
})

// ── Q2: compileSession is byte-deterministic across re-elaboration ───

describe('Q2 — compileSession plan bytes are stable across fresh re-elaboration', () => {
  const cases: Array<[string, () => SessionState]> = [
    ...SINGLE_TYPES.map(t => [`single:${t}`, () => singleInstance(t)] as [string, () => SessionState]),
    ['wiredPair', wiredPair],
    ['wiredPairWithParam', wiredPairWithParam],
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
