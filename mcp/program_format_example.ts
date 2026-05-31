/**
 * The canonical `tropical_program_2` example shown in the `tropical://program-format`
 * resource (PROGRAM_FORMAT_DOC). It is a real object, not text embedded in markdown,
 * so the doc renders FROM it and a regression test loads THIS SAME object — the doc
 * and the tested patch cannot drift apart.
 *
 * (This exists because the old hand-written example silently rotted into a deprecated
 * `audio_outputs` form *and* a non-existent `sin` op — agents fetch this doc and copy
 * it verbatim, so a broken example is a broken agent.)
 *
 * Demonstrates: an inline `programDecl` with a state register + `nextUpdate`
 * (a naive `Saw` = 2·phase − 1), instances of it and a stdlib filter, inter-instance
 * wiring, and audio output via a body `outputAssign` named `"dac.out"`.
 */
export const PROGRAM_FORMAT_EXAMPLE = {
  schema: 'tropical_program_2',
  name: 'MyPatch',
  body: {
    op: 'block',
    decls: [
      { op: 'programDecl', name: 'Saw', program: { op: 'program', name: 'Saw',
        ports: { inputs: ['freq'], outputs: ['out'] },
        body: { op: 'block',
          decls: [{ op: 'regDecl', name: 'phase', init: 0 }],
          assigns: [
            { op: 'outputAssign', name: 'out',
              expr: { op: 'sub', args: [{ op: 'mul', args: [2, { op: 'reg', name: 'phase' }] }, 1] } },
            { op: 'nextUpdate', target: { kind: 'reg', name: 'phase' },
              expr: { op: 'mod', args: [{ op: 'add', args: [{ op: 'reg', name: 'phase' },
                { op: 'div', args: [{ op: 'input', name: 'freq' }, { op: 'sampleRate' }] }] }, 1] } } ],
          value: null } } },
      { op: 'instanceDecl', name: 'osc', program: 'Saw', inputs: { freq: 440 } },
      { op: 'instanceDecl', name: 'filt', program: 'LadderFilter',
        inputs: { input: { op: 'ref', instance: 'osc', output: 'out' }, cutoff: 2000 } },
    ],
    assigns: [
      { op: 'outputAssign', name: 'dac.out',
        expr: { op: 'ref', instance: 'filt', output: 'lp' } },
    ],
    value: null,
  },
}
