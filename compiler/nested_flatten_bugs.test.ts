/**
 * Regression tests for two flatten.ts bugs discovered during bubble-synth:
 *
 * Bug 1: Outer program's delay_ref inside a nested instance's input wiring
 *        survives into the flat plan as unresolved delay_value(node_id),
 *        because collectNestedRegisterExprs' caller doesn't run
 *        resolveDelayValues after substituteInputs.
 *
 * Bug 2: Wrapping a stdlib program that has internal state (delays and/or
 *        nested-call register updates) in a program_decl produces wrong
 *        output — either zero, or unbounded amplification on a sum of
 *        such wrapped instances.
 */

import { describe, test, expect } from 'bun:test'
import { makeSession, loadJSON } from './session'
import { loadStdlib } from './program'
import { flattenExpressions } from './flatten'
import { interpretSamples } from './interpret'
import type { ProgramFile } from './program'

function countDelayValueLeaks(node: unknown): number {
  if (!node || typeof node !== 'object') return 0
  const obj = node as Record<string, unknown>
  if (obj.op === 'delay_value') return 1
  let n = 0
  for (const v of Object.values(obj)) {
    if (Array.isArray(v)) for (const el of v) n += countDelayValueLeaks(el)
    else if (v && typeof v === 'object') n += countDelayValueLeaks(v)
  }
  return n
}

describe('flatten bug 1 — outer delay_ref inside nested input wiring', () => {
  test('Wrap with a delay_decl piped into inner OnePole input — no delay_value leaks', () => {
    const session = makeSession(44100)
    loadStdlib(session)
    loadJSON({
      schema: 'tropical_program_2',
      name: 't',
      body: { op: 'block', decls: [
        { op: 'program_decl', name: 'Wrap', program: {
          op: 'program', name: 'Wrap',
          ports: {
            inputs: [{ name: 'x', type: 'signal', default: 0 }],
            outputs: [{ name: 'out', type: 'float' }],
          },
          body: { op: 'block', decls: [
            { op: 'delay_decl', name: 'prev_x',
              update: { op: 'input', name: 'x' }, init: 0 },
            { op: 'instance_decl', name: 'op', program: 'OnePole', inputs: {
              // Inner instance's input wiring references Wrap's own delay_ref.
              // This is the exact pattern that was tripping flatten.ts.
              input: {
                op: 'add',
                args: [
                  { op: 'input', name: 'x' },
                  { op: 'delay_ref', id: 'prev_x' },
                ],
              },
              g: 0.1,
            }},
          ], assigns: [
            { op: 'output_assign', name: 'out', expr: { op: 'nested_out', ref: 'op', output: 'out' } },
          ]},
        }},
        { op: 'instance_decl', name: 'w', program: 'Wrap', inputs: {
          x: { op: 'select', args: [{ op: 'eq', args: [{ op: 'sample_index' }, 10] }, 1, 0] },
        }},
      ]},
      audio_outputs: [{ instance: 'w', output: 'out' }],
    } as ProgramFile, session)
    const flat = flattenExpressions(session)
    let leaks = 0
    for (const e of flat.outputExprs) leaks += countDelayValueLeaks(e)
    for (const e of flat.registerExprs) leaks += countDelayValueLeaks(e)
    expect(leaks).toBe(0)
  })
})

describe('flatten bug 2 — wrapping stateful stdlib programs in program_decl', () => {
  test('Wrap(OnePole) matches unwrapped OnePole', () => {
    const impulse = { op: 'select', args: [{ op: 'eq', args: [{ op: 'sample_index' }, 10] }, 1, 0] }

    const wrapped = (() => {
      const session = makeSession(44100)
      loadStdlib(session)
      loadJSON({
        schema: 'tropical_program_2',
        name: 't',
        body: { op: 'block', decls: [
          { op: 'program_decl', name: 'Wrap', program: {
            op: 'program', name: 'Wrap',
            ports: {
              inputs: [{ name: 'x', type: 'signal', default: 0 }],
              outputs: [{ name: 'out', type: 'float' }],
            },
            body: { op: 'block', decls: [
              { op: 'instance_decl', name: 'op', program: 'OnePole', inputs: {
                input: { op: 'input', name: 'x' }, g: 0.1,
              }},
            ], assigns: [
              { op: 'output_assign', name: 'out', expr: { op: 'nested_out', ref: 'op', output: 'out' } },
            ]},
          }},
          { op: 'instance_decl', name: 'w', program: 'Wrap', inputs: { x: impulse } },
        ]},
        audio_outputs: [{ instance: 'w', output: 'out' }],
      } as ProgramFile, session)
      return interpretSamples(flattenExpressions(session), 30)
    })()

    const bare = (() => {
      const session = makeSession(44100)
      loadStdlib(session)
      loadJSON({
        schema: 'tropical_program_2',
        name: 't',
        body: { op: 'block', decls: [
          { op: 'instance_decl', name: 'op', program: 'OnePole', inputs: {
            input: impulse, g: 0.1,
          }},
        ]},
        audio_outputs: [{ instance: 'op', output: 'out' }],
      } as ProgramFile, session)
      return interpretSamples(flattenExpressions(session), 30)
    })()

    for (let i = 0; i < 30; i++) {
      expect(wrapped[i]).toBeCloseTo(bare[i], 10)
    }
  })

  test('Wrap(LadderFilter) matches unwrapped LadderFilter (no delay_value leaks)', () => {
    const makeImpulsePatch = (useWrap: boolean): ProgramFile => {
      const impulse = { op: 'select', args: [{ op: 'eq', args: [{ op: 'sample_index' }, 10] }, 1, 0] }
      if (useWrap) {
        return {
          schema: 'tropical_program_2',
          name: 't',
          body: { op: 'block', decls: [
            { op: 'program_decl', name: 'Wrap', program: {
              op: 'program', name: 'Wrap',
              ports: {
                inputs: [{ name: 'x', type: 'signal', default: 0 }],
                outputs: [{ name: 'out', type: 'float' }],
              },
              body: { op: 'block', decls: [
                { op: 'instance_decl', name: 'lf', program: 'LadderFilter', inputs: {
                  input: { op: 'input', name: 'x' }, cutoff: 800, resonance: 0.5, drive: 1,
                }},
              ], assigns: [
                { op: 'output_assign', name: 'out', expr: { op: 'nested_out', ref: 'lf', output: 'lp' } },
              ]},
            }},
            { op: 'instance_decl', name: 'w', program: 'Wrap', inputs: { x: impulse } },
          ]},
          audio_outputs: [{ instance: 'w', output: 'out' }],
        } as ProgramFile
      }
      return {
        schema: 'tropical_program_2',
        name: 't',
        body: { op: 'block', decls: [
          { op: 'instance_decl', name: 'lf', program: 'LadderFilter', inputs: {
            input: impulse, cutoff: 800, resonance: 0.5, drive: 1,
          }},
        ]},
        audio_outputs: [{ instance: 'lf', output: 'lp' }],
      } as ProgramFile
    }

    const sessionW = makeSession(44100); loadStdlib(sessionW); loadJSON(makeImpulsePatch(true), sessionW)
    const flatW = flattenExpressions(sessionW)
    let leaks = 0
    for (const e of flatW.outputExprs) leaks += countDelayValueLeaks(e)
    for (const e of flatW.registerExprs) leaks += countDelayValueLeaks(e)
    expect(leaks).toBe(0)

    const wrapped = interpretSamples(flatW, 60)

    const sessionB = makeSession(44100); loadStdlib(sessionB); loadJSON(makeImpulsePatch(false), sessionB)
    const bare = interpretSamples(flattenExpressions(sessionB), 60)

    for (let i = 0; i < 60; i++) {
      expect(wrapped[i]).toBeCloseTo(bare[i], 10)
    }
  })

  test('Wrap(Bubble) matches unwrapped Bubble', () => {
    const impulse = { op: 'select', args: [{ op: 'eq', args: [{ op: 'sample_index' }, 100] }, 1, 0] }

    const wrapped = (() => {
      const session = makeSession(44100)
      loadStdlib(session)
      loadJSON({
        schema: 'tropical_program_2',
        name: 't',
        body: { op: 'block', decls: [
          { op: 'program_decl', name: 'Wrap', program: {
            op: 'program', name: 'Wrap',
            ports: {
              inputs: [{ name: 'trigger', type: 'signal', default: 0 }],
              outputs: [{ name: 'out', type: 'float' }],
            },
            body: { op: 'block', decls: [
              { op: 'instance_decl', name: 'b', program: 'Bubble', inputs: {
                trigger: { op: 'input', name: 'trigger' },
                radius: 0.003, decay_scale: 12, amp_scale: 0.3,
              }},
            ], assigns: [
              { op: 'output_assign', name: 'out', expr: { op: 'nested_out', ref: 'b', output: 'out' } },
            ]},
          }},
          { op: 'instance_decl', name: 'w', program: 'Wrap', inputs: { trigger: impulse } },
        ]},
        audio_outputs: [{ instance: 'w', output: 'out' }],
      } as ProgramFile, session)
      return interpretSamples(flattenExpressions(session), 300)
    })()

    const bare = (() => {
      const session = makeSession(44100)
      loadStdlib(session)
      loadJSON({
        schema: 'tropical_program_2',
        name: 't',
        body: { op: 'block', decls: [
          { op: 'instance_decl', name: 'b', program: 'Bubble', inputs: {
            trigger: impulse, radius: 0.003, decay_scale: 12, amp_scale: 0.3,
          }},
        ]},
        audio_outputs: [{ instance: 'b', output: 'out' }],
      } as ProgramFile, session)
      return interpretSamples(flattenExpressions(session), 300)
    })()

    for (let i = 0; i < 300; i++) {
      expect(wrapped[i]).toBeCloseTo(bare[i], 10)
    }
  })
})
