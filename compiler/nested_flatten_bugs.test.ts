/**
 * Regression tests originally written for two flatten.ts bugs in
 * bubble-synth:
 *
 *   Bug 1: Outer program's delay_ref inside a nested instance's input
 *          wiring survived as unresolved delay_value(node_id).
 *   Bug 2: Wrapping a stateful stdlib program in program_decl produced
 *          wrong output (zero or unbounded amplification).
 *
 * Post-D2-cutover, both shapes go through the strata pipeline +
 * `compileSession`. The leak-detection assertion is structurally
 * impossible to fail (inlineInstances + buildSlotMaps would refuse to
 * produce a delayRef whose decl isn't in the body), so we drop it.
 * The audio-equivalence assertions (wrapped == bare) survive — they
 * stay relevant whatever the pipeline.
 */

import { describe, test, expect } from 'bun:test'
import { makeSession, loadJSON } from './session'
import { loadStdlib } from './program'
import { interpretSession } from './interpret_resolved'
import type { ProgramFile } from './program'

describe('flatten regression — wrapping stateful stdlib programs in program_decl', () => {
  test('Wrap(OnePole) matches unwrapped OnePole', () => {
    const impulse = { op: 'select', args: [{ op: 'eq', args: [{ op: 'sampleIndex' }, 10] }, 1, 0] }

    const wrapped = (() => {
      const session = makeSession(44100)
      loadStdlib(session)
      loadJSON({
        schema: 'tropical_program_2',
        name: 't',
        body: { op: 'block', decls: [
          { op: 'programDecl', name: 'Wrap', program: {
            op: 'program', name: 'Wrap',
            ports: {
              inputs: [{ name: 'x', type: 'signal', default: 0 }],
              outputs: [{ name: 'out', type: 'float' }],
            },
            body: { op: 'block', decls: [
              { op: 'instanceDecl', name: 'op', program: 'OnePole', inputs: {
                input: { op: 'input', name: 'x' }, g: 0.1,
              }},
            ], assigns: [
              { op: 'outputAssign', name: 'out', expr: { op: 'nestedOut', ref: 'op', output: 'out' } },
            ]},
          }},
          { op: 'instanceDecl', name: 'w', program: 'Wrap', inputs: { x: impulse } },
        ]},
        audio_outputs: [{ instance: 'w', output: 'out' }],
      } as ProgramFile, session)
      return interpretSession(session, 30)
    })()

    const bare = (() => {
      const session = makeSession(44100)
      loadStdlib(session)
      loadJSON({
        schema: 'tropical_program_2',
        name: 't',
        body: { op: 'block', decls: [
          { op: 'instanceDecl', name: 'op', program: 'OnePole', inputs: {
            input: impulse, g: 0.1,
          }},
        ]},
        audio_outputs: [{ instance: 'op', output: 'out' }],
      } as ProgramFile, session)
      return interpretSession(session, 30)
    })()

    for (let i = 0; i < 30; i++) {
      expect(wrapped[i]).toBeCloseTo(bare[i], 10)
    }
  })

  test('Wrap(LadderFilter) matches unwrapped LadderFilter', () => {
    const makeImpulsePatch = (useWrap: boolean): ProgramFile => {
      const impulse = { op: 'select', args: [{ op: 'eq', args: [{ op: 'sampleIndex' }, 10] }, 1, 0] }
      if (useWrap) {
        return {
          schema: 'tropical_program_2',
          name: 't',
          body: { op: 'block', decls: [
            { op: 'programDecl', name: 'Wrap', program: {
              op: 'program', name: 'Wrap',
              ports: {
                inputs: [{ name: 'x', type: 'signal', default: 0 }],
                outputs: [{ name: 'out', type: 'float' }],
              },
              body: { op: 'block', decls: [
                { op: 'instanceDecl', name: 'lf', program: 'LadderFilter', inputs: {
                  input: { op: 'input', name: 'x' }, cutoff: 800, resonance: 0.5, drive: 1,
                }},
              ], assigns: [
                { op: 'outputAssign', name: 'out', expr: { op: 'nestedOut', ref: 'lf', output: 'lp' } },
              ]},
            }},
            { op: 'instanceDecl', name: 'w', program: 'Wrap', inputs: { x: impulse } },
          ]},
          audio_outputs: [{ instance: 'w', output: 'out' }],
        } as ProgramFile
      }
      return {
        schema: 'tropical_program_2',
        name: 't',
        body: { op: 'block', decls: [
          { op: 'instanceDecl', name: 'lf', program: 'LadderFilter', inputs: {
            input: impulse, cutoff: 800, resonance: 0.5, drive: 1,
          }},
        ]},
        audio_outputs: [{ instance: 'lf', output: 'lp' }],
      } as ProgramFile
    }

    const sessionW = makeSession(44100); loadStdlib(sessionW); loadJSON(makeImpulsePatch(true), sessionW)
    const wrapped = interpretSession(sessionW, 60)

    const sessionB = makeSession(44100); loadStdlib(sessionB); loadJSON(makeImpulsePatch(false), sessionB)
    const bare = interpretSession(sessionB, 60)

    for (let i = 0; i < 60; i++) {
      expect(wrapped[i]).toBeCloseTo(bare[i], 10)
    }
  })

  // ─────────────────────────────────────────────────────────────
  // Phase G — wrap-nesting depth (TDD plan §Phase G)
  // ─────────────────────────────────────────────────────────────

  test('(D) Wrap(Wrap(Wrap(LadderFilter))) matches bare LadderFilter — 60 samples', () => {
    // Three nested levels of program_decl wrapping. The post-strata
    // `_liftedFrom` provenance chain becomes inst3.inst2.inst1.lf.
    // Behavior, not provenance shape, is the gate.
    const impulse = { op: 'select', args: [{ op: 'eq', args: [{ op: 'sampleIndex' }, 10] }, 1, 0] }

    const wrapped = (() => {
      const session = makeSession(44100)
      loadStdlib(session)
      loadJSON({
        schema: 'tropical_program_2',
        name: 't',
        body: { op: 'block', decls: [
          // Innermost: Wrap1 = LadderFilter passthrough
          { op: 'programDecl', name: 'Wrap1', program: {
            op: 'program', name: 'Wrap1',
            ports: {
              inputs: [{ name: 'x', type: 'signal', default: 0 }],
              outputs: [{ name: 'out', type: 'float' }],
            },
            body: { op: 'block', decls: [
              { op: 'instanceDecl', name: 'lf', program: 'LadderFilter', inputs: {
                input: { op: 'input', name: 'x' }, cutoff: 800, resonance: 0.5, drive: 1,
              }},
            ], assigns: [
              { op: 'outputAssign', name: 'out', expr: { op: 'nestedOut', ref: 'lf', output: 'lp' } },
            ]},
          }},
          // Middle: Wrap2 = instance of Wrap1
          { op: 'programDecl', name: 'Wrap2', program: {
            op: 'program', name: 'Wrap2',
            ports: {
              inputs: [{ name: 'x', type: 'signal', default: 0 }],
              outputs: [{ name: 'out', type: 'float' }],
            },
            body: { op: 'block', decls: [
              { op: 'instanceDecl', name: 'inst1', program: 'Wrap1', inputs: {
                x: { op: 'input', name: 'x' },
              }},
            ], assigns: [
              { op: 'outputAssign', name: 'out', expr: { op: 'nestedOut', ref: 'inst1', output: 'out' } },
            ]},
          }},
          // Outermost: Wrap3 = instance of Wrap2
          { op: 'programDecl', name: 'Wrap3', program: {
            op: 'program', name: 'Wrap3',
            ports: {
              inputs: [{ name: 'x', type: 'signal', default: 0 }],
              outputs: [{ name: 'out', type: 'float' }],
            },
            body: { op: 'block', decls: [
              { op: 'instanceDecl', name: 'inst2', program: 'Wrap2', inputs: {
                x: { op: 'input', name: 'x' },
              }},
            ], assigns: [
              { op: 'outputAssign', name: 'out', expr: { op: 'nestedOut', ref: 'inst2', output: 'out' } },
            ]},
          }},
          { op: 'instanceDecl', name: 'inst3', program: 'Wrap3', inputs: { x: impulse } },
        ]},
        audio_outputs: [{ instance: 'inst3', output: 'out' }],
      } as ProgramFile, session)
      return interpretSession(session, 60)
    })()

    const bare = (() => {
      const session = makeSession(44100)
      loadStdlib(session)
      loadJSON({
        schema: 'tropical_program_2',
        name: 't',
        body: { op: 'block', decls: [
          { op: 'instanceDecl', name: 'lf', program: 'LadderFilter', inputs: {
            input: impulse, cutoff: 800, resonance: 0.5, drive: 1,
          }},
        ]},
        audio_outputs: [{ instance: 'lf', output: 'lp' }],
      } as ProgramFile, session)
      return interpretSession(session, 60)
    })()

    for (let i = 0; i < 60; i++) {
      expect(wrapped[i]).toBeCloseTo(bare[i], 10)
    }
  })

  test('(S) generic Wrap<P>: instantiating Wrap<Sin> lowers without throwing', () => {
    // Smoke test only — assert that a generic wrapper around a generic
    // stdlib type can be instantiated. The denotational stretch goal
    // (specialize ∘ inline_instances == inline_instances ∘ specialize)
    // is deferred per the plan.
    const session = makeSession(64)
    loadStdlib(session)
    expect(() => {
      loadJSON({
        schema: 'tropical_program_2',
        name: 't',
        body: { op: 'block', decls: [
          // Generic Wrap<P>: Wrap<P>(x) = P(x).out where P is the
          // type argument. Sin is a leaf with input `x` and output `out`.
          { op: 'programDecl', name: 'WrapGen', program: {
            op: 'program', name: 'WrapGen',
            ports: {
              inputs: [{ name: 'x', type: 'signal', default: 0 }],
              outputs: [{ name: 'out', type: 'float' }],
            },
            body: { op: 'block', decls: [
              // Concrete inner program (the smoke version doesn't use
              // a true generic type parameter — `Sin` is referenced by
              // name. The plan's stretch goal would parameterize this
              // properly; for now this confirms the multi-level wrap
              // around a stdlib type lowers cleanly).
              { op: 'instanceDecl', name: 'p', program: 'Sin', inputs: {
                x: { op: 'input', name: 'x' },
              }},
            ], assigns: [
              { op: 'outputAssign', name: 'out', expr: { op: 'nestedOut', ref: 'p', output: 'out' } },
            ]},
          }},
          { op: 'instanceDecl', name: 'w', program: 'WrapGen',
            inputs: { x: { op: 'sampleIndex' } } },
        ]},
        audio_outputs: [{ instance: 'w', output: 'out' }],
      } as ProgramFile, session)
      // Run a single sample to confirm the lowered IR also evaluates.
      interpretSession(session, 1)
    }).not.toThrow()
  })

  test('Wrap(Bubble) matches unwrapped Bubble', () => {
    const impulse = { op: 'select', args: [{ op: 'eq', args: [{ op: 'sampleIndex' }, 100] }, 1, 0] }

    const wrapped = (() => {
      const session = makeSession(44100)
      loadStdlib(session)
      loadJSON({
        schema: 'tropical_program_2',
        name: 't',
        body: { op: 'block', decls: [
          { op: 'programDecl', name: 'Wrap', program: {
            op: 'program', name: 'Wrap',
            ports: {
              inputs: [{ name: 'trigger', type: 'signal', default: 0 }],
              outputs: [{ name: 'out', type: 'float' }],
            },
            body: { op: 'block', decls: [
              { op: 'instanceDecl', name: 'b', program: 'Bubble', inputs: {
                trigger: { op: 'input', name: 'trigger' },
                radius: 0.003, decay_scale: 12, amp_scale: 0.3,
              }},
            ], assigns: [
              { op: 'outputAssign', name: 'out', expr: { op: 'nestedOut', ref: 'b', output: 'out' } },
            ]},
          }},
          { op: 'instanceDecl', name: 'w', program: 'Wrap', inputs: { trigger: impulse } },
        ]},
        audio_outputs: [{ instance: 'w', output: 'out' }],
      } as ProgramFile, session)
      return interpretSession(session, 300)
    })()

    const bare = (() => {
      const session = makeSession(44100)
      loadStdlib(session)
      loadJSON({
        schema: 'tropical_program_2',
        name: 't',
        body: { op: 'block', decls: [
          { op: 'instanceDecl', name: 'b', program: 'Bubble', inputs: {
            trigger: impulse, radius: 0.003, decay_scale: 12, amp_scale: 0.3,
          }},
        ]},
        audio_outputs: [{ instance: 'b', output: 'out' }],
      } as ProgramFile, session)
      return interpretSession(session, 300)
    })()

    for (let i = 0; i < 300; i++) {
      expect(wrapped[i]).toBeCloseTo(bare[i], 10)
    }
  })
})
