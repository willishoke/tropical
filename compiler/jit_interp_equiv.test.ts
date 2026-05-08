/**
 * jit_interp_equiv.test.ts — Differential test: TS interpreter vs. LLVM JIT.
 *
 * Runs a small gateable-subgraph patch through both the pure-TS interpreter
 * (interpret_resolved.ts / interpretSession) and the native LLVM JIT
 * (apply_plan → Runtime.process), then asserts the output buffers match
 * sample-for-sample.
 *
 * This is the merge gate for the gateable-subgraph feature: if the JIT's
 * conditional-block optimization diverges from the interpreter's reference
 * source_tag semantics, this test will flag it.
 *
 * Requires libtropical.dylib (build with `make build` first).
 */

import { describe, test, expect } from 'bun:test'
import { makeSession, loadJSON, type ExprNode } from './session'
import { loadStdlib as loadBuiltins, loadProgramAsType, type ProgramNode } from './program'
import { applySessionWiring } from './apply_plan'
import { interpretSession } from './interpret_resolved'

// A trivial leaf program: one input, one state register, output = reg; reg updates
// to (reg + input). Exercises both output and register-update wrapping paths.
const ACCUM: ProgramNode = {
  op: 'program',
  name: 'Accum',
  ports: { inputs: [{ name: 'x', default: 0 }], outputs: ['out'] },
  body: { op: 'block',
    decls: [{ op: 'regDecl', name: 'acc', init: 0 }],
    assigns: [
      { op: 'outputAssign', name: 'out', expr: { op: 'reg', name: 'acc' } },
      { op: 'nextUpdate', target: { kind: 'reg', name: 'acc' },
        expr: { op: 'add', args: [{ op: 'reg', name: 'acc' }, { op: 'input', name: 'x' }] } },
    ],
  },
}

function setupGated(gateInput: ExprNode, bufferLength = 32) {
  const session = makeSession(bufferLength)
  loadBuiltins(session)
  loadProgramAsType(ACCUM, session)

  // Patch: one gated Accum. x drives the register; output is the register value,
  // wrapped in source_tag (zero on skip) / state update wrapped with on_skip=reg.
  loadJSON({
    schema: 'tropical_program_2',
    name: 'patch',
    body: { op: 'block', decls: [
      { op: 'instanceDecl', name: 'a1', program: 'Accum',
        inputs: { x: 1.0 }, gateable: true, gate_input: gateInput },
    ]},
    audio_outputs: [{ instance: 'a1', output: 'out' }],
  }, session)

  return session
}

function runJit(session: ReturnType<typeof setupGated>, nFrames = 1): Float64Array {
  applySessionWiring(session)
  session.graph.primeJit()
  const acc: number[] = []
  for (let f = 0; f < nFrames; f++) {
    session.graph.process()
    for (const v of session.graph.outputBuffer) acc.push(v)
  }
  return Float64Array.from(acc)
}

function runInterp(session: ReturnType<typeof setupGated>, nSamples: number): Float64Array {
  // Independent oracle: re-run the same materialization compileSession
  // uses, then walk the resolved IR directly. Matches what the JIT sees,
  // but through a different evaluator.
  return interpretSession(session, nSamples)
}

describe('JIT ↔ interpreter equivalence for gateable subgraphs', () => {
  test('baseline: ungated Accum accumulates correctly in JIT', () => {
    // No source_tag at all — direct Accum instance. Sanity check the test
    // harness itself before asserting anything about gateable semantics.
    const session = makeSession(16)
    loadBuiltins(session)
    loadProgramAsType(ACCUM, session)
    loadJSON({
      schema: 'tropical_program_2',
      name: 'patch',
      body: { op: 'block', decls: [
        { op: 'instanceDecl', name: 'a1', program: 'Accum', inputs: { x: 1.0 } },
      ]},
      audio_outputs: [{ instance: 'a1', output: 'out' }],
    }, session)
    applySessionWiring(session)
    session.graph.primeJit()
    session.graph.process()
    const jit = new Float64Array(session.graph.outputBuffer)
    const interp = interpretSession(session, jit.length)

    for (let i = 0; i < jit.length; i++) {
      expect(jit[i]).toBeCloseTo(interp[i], 10)
    }
    session.graph.dispose()
  })

  test('gate always true: JIT output matches interpreter (output + reg update)', () => {
    const session = setupGated(true, 16)
    const jit = runJit(session, 1)
    const interp = runInterp(session, jit.length)

    expect(jit.length).toBe(interp.length)
    for (let i = 0; i < jit.length; i++) {
      expect(jit[i]).toBeCloseTo(interp[i], 10)
    }

    session.graph.dispose()
  })

  test('gate always false: JIT output is exactly 0 and matches interpreter', () => {
    const session = setupGated(false, 16)
    const jit = runJit(session, 1)
    const interp = runInterp(session, jit.length)

    expect(jit.length).toBe(interp.length)
    for (let i = 0; i < jit.length; i++) {
      // On skip, output should be zero AND the accumulator should hold (stay 0).
      expect(jit[i]).toBe(0)
      expect(interp[i]).toBe(0)
    }

    session.graph.dispose()
  })

  test('gate driven by param flips: JIT matches interpreter sample-for-sample', () => {
    // Use a sample-index-derived gate: (sample_index % 4) < 2. This exercises
    // a gate that changes every two samples, so half the time the group is
    // skipped and the accumulator must hold on those samples.
    const gate: ExprNode = {
      op: 'lt',
      args: [
        { op: 'mod', args: [{ op: 'sampleIndex' }, 4] },
        2,
      ],
    }
    const session = setupGated(gate, 32)
    const jit = runJit(session, 1)
    const interp = runInterp(session, jit.length)

    expect(jit.length).toBe(interp.length)
    for (let i = 0; i < jit.length; i++) {
      expect(jit[i]).toBeCloseTo(interp[i], 10)
    }

    session.graph.dispose()
  })

  // ─── Bob-recommended coverage: delay register, nested calls, gate dependencies ───

  // Like ACCUM but with a delay register alongside the named reg, so both
  // register-update paths (reg_decl and delay_decl) get wrapped with on_skip.
  const ACCUM_WITH_DELAY: ProgramNode = {
    op: 'program',
    name: 'AccumDelay',
    ports: { inputs: [{ name: 'x', default: 0 }], outputs: ['out'] },
    body: { op: 'block',
      decls: [
        { op: 'regDecl', name: 'acc', init: 0 },
        { op: 'delayDecl', name: 'prev_x', update: { op: 'input', name: 'x' }, init: 0 },
      ],
      assigns: [
        // Output is current register value.
        { op: 'outputAssign', name: 'out', expr: { op: 'reg', name: 'acc' } },
        // Accumulator adds the PREVIOUS sample's x (via the delay reg).
        { op: 'nextUpdate', target: { kind: 'reg', name: 'acc' },
          expr: { op: 'add', args: [{ op: 'reg', name: 'acc' }, { op: 'delayRef', id: 'prev_x' }] } },
      ],
    },
  }

  test('gateable instance with both reg and delay registers: JIT matches interpreter', () => {
    const session = makeSession(32)
    loadBuiltins(session)
    loadProgramAsType(ACCUM_WITH_DELAY, session)

    // Dynamic gate: (sample_index % 4) < 2 — alternates live / skip.
    const gate: ExprNode = {
      op: 'lt',
      args: [{ op: 'mod', args: [{ op: 'sampleIndex' }, 4] }, 2],
    }
    loadJSON({
      schema: 'tropical_program_2',
      name: 'patch',
      body: { op: 'block', decls: [
        { op: 'instanceDecl', name: 'a1', program: 'AccumDelay',
          inputs: { x: 1.0 }, gateable: true, gate_input: gate },
      ]},
      audio_outputs: [{ instance: 'a1', output: 'out' }],
    }, session)

    applySessionWiring(session)
    session.graph.primeJit()
    session.graph.process()
    const jit = new Float64Array(session.graph.outputBuffer)
    const interp = interpretSession(session, jit.length)

    for (let i = 0; i < jit.length; i++) {
      expect(jit[i]).toBeCloseTo(interp[i], 10)
    }
    session.graph.dispose()
  })

  test('gateable LadderFilter (4 nested OnePoles): JIT matches interpreter', () => {
    // LadderFilter from stdlib has 4 internal OnePole instances — this
    // exercises the nested-call register wrapping path that simple Accum
    // doesn't hit.
    const session = makeSession(64)
    loadBuiltins(session)

    // Gate alternates every 8 samples so the filter's state is meaningfully
    // held through skip windows and resumes cleanly on live windows.
    const gate: ExprNode = {
      op: 'lt',
      args: [{ op: 'mod', args: [{ op: 'sampleIndex' }, 16] }, 8],
    }
    loadJSON({
      schema: 'tropical_program_2',
      name: 'patch',
      body: { op: 'block', decls: [
        { op: 'instanceDecl', name: 'ladder', program: 'LadderFilter',
          inputs: { input: 0.5, cutoff: 1000, resonance: 0.5 },
          gateable: true, gate_input: gate },
      ]},
      audio_outputs: [{ instance: 'ladder', output: 'lp' }],
    }, session)

    applySessionWiring(session)
    session.graph.primeJit()
    session.graph.process()
    const jit = new Float64Array(session.graph.outputBuffer)
    const interp = interpretSession(session, jit.length)

    for (let i = 0; i < jit.length; i++) {
      expect(jit[i]).toBeCloseTo(interp[i], 8)
    }
    session.graph.dispose()
  })

  // ─────────────────────────────────────────────────────────────
  // Phase H — Param / Trigger handle stitching (TDD plan §Phase H)
  //
  // Both tests are deferred. The live-JIT path for params/triggers has
  // never been end-to-end exercised, and surfacing it requires THREE
  // coordinated changes (it isn't a single focused PR):
  //
  //  1. TS FFI: compile_session.ts:31 emits `String(napi_external)` =
  //     "[object Object]"; the C++ side throws stoull("no conversion").
  //     `koffi.address(_h)` returns the WRAPPER's address, not the
  //     inner ControlParam* the kernel expects (tropical_c.cpp's
  //     TropicalParam wraps a ControlParam* member). The fix needs a
  //     dedicated `tropical_param_address(handle)` C export that
  //     returns `static_cast<TropicalParam*>(p)->param`, OR a layout
  //     change to TropicalParam.
  //
  //  2. TS emit: emit_resolved.ts emits a bare `param` operand; the
  //     C++ JIT (OrcJitEngine.cpp:624) returns null for it because
  //     params must be wrapped in a SmoothParam / TriggerParam
  //     instruction before they can flow into arithmetic. emit_wasm.ts
  //     does this; emit_resolved.ts doesn't yet.
  //
  //  3. C++ runtime state: FlatRuntime::load_plan never populates
  //     KernelState::param_ptrs or trigger_params from the parsed
  //     plan. The kernel's param_ptrs GEP indexes into a zero-length
  //     vector → segfault. The per-buffer trigger snapshot loop also
  //     iterates an empty vector.
  //
  // All three need to land together for these tests to pass.
  // ─────────────────────────────────────────────────────────────

  test.skip('(D) param fan-out + non-aliasing: shared cutoff updates lockstep, drive unchanged', () => {
    // Three Accum instances: a1 and a2 both wire param('cutoff') as
    // their `x` input; a3 wires param('drive'). Output is a1+a2+a3.
    //
    // Fan-out: changing cutoff moves a1's and a2's outputs together.
    // Non-aliasing: changing cutoff must NOT move a3's output.
    //
    // Both properties together rule out the alternative bug where the
    // materializer aliased all params to one slot.
    const session = makeSession(8)
    loadBuiltins(session)
    loadProgramAsType(ACCUM, session)
    loadJSON({
      schema: 'tropical_program_2',
      name: 'patch',
      body: { op: 'block', decls: [
        { op: 'paramDecl', name: 'cutoff', value: 0.0, time_const: 0 } as unknown as ExprNode,
        { op: 'paramDecl', name: 'drive',  value: 0.0, time_const: 0 } as unknown as ExprNode,
        { op: 'instanceDecl', name: 'a1', program: 'Accum',
          inputs: { x: { op: 'param', name: 'cutoff' } } },
        { op: 'instanceDecl', name: 'a2', program: 'Accum',
          inputs: { x: { op: 'param', name: 'cutoff' } } },
        { op: 'instanceDecl', name: 'a3', program: 'Accum',
          inputs: { x: { op: 'param', name: 'drive' } } },
      ]},
      audio_outputs: [
        { instance: 'a1', output: 'out' },
        { instance: 'a2', output: 'out' },
        { instance: 'a3', output: 'out' },
      ],
    }, session)

    applySessionWiring(session)
    session.graph.primeJit()

    // Buffer 1: cutoff = drive = 0. All accums stay 0 (no input).
    session.graph.process()
    const buf0 = new Float64Array(session.graph.outputBuffer)
    for (let i = 0; i < buf0.length; i++) expect(buf0[i]).toBe(0)

    // Buffer 2: set cutoff = 1.0; drive stays 0.
    session.paramRegistry.get('cutoff')!.value = 1.0
    session.graph.process()
    const buf1 = new Float64Array(session.graph.outputBuffer)

    // Buffer 3 — measure a1 / a2 / a3 separately by routing only one
    // at a time would require rewiring; instead, use the linearity of
    // the audio mix. With cutoff=1, drive=0, time_const=0 (instant
    // smoothing): a1 and a2 each accumulate +1 per sample, a3 stays 0.
    // Mix = (a1 + a2 + a3) / 20 with crossfade on the first buffer
    // after re-priming. After enough samples, mix dominantly reflects
    // 2 * cutoff_accum / 20.
    //
    // Simpler check: after buf2, the mix's last sample should be
    // roughly (a1+a2)/20 ≈ 2*8/20 = 0.8 if smoothing instant.
    // The crossfade window slightly damps the leading samples, so
    // assert just that buf2 grows over buf1: if cutoff fan-out works,
    // both a1 and a2 track cutoff and the mix climbs.
    const buf1Last = buf1[buf1.length - 1]
    expect(buf1Last).toBeGreaterThan(0)

    // Now set drive = 1.0 and re-process. With both cutoff and drive
    // at 1.0, all three accums advance; mix grows further. Compare to
    // what cutoff alone produced — drive's contribution is purely
    // additive.
    session.paramRegistry.get('drive')!.value = 1.0
    session.graph.process()
    const buf2 = new Float64Array(session.graph.outputBuffer)
    expect(buf2[buf2.length - 1]).toBeGreaterThan(buf1[buf1.length - 1])

    session.graph.dispose()
  })

  test.skip('(D) trigger as gate_input: trigger-driven gate matches always-true gate on the buffer it fired in', () => {
    // Gateable instance whose gate_input is `{op:'trigger', name:'go'}`.
    // Fire the trigger before process(); the runtime snapshots the
    // trigger value once per buffer (FlatRuntime.hpp:105-111 does
    // value.exchange(0.0)) and the kernel reads `frame_value` every
    // sample with no per-sample reset. So a fired trigger reads as
    // 1.0 for ALL samples of that buffer (not just sample 0). The
    // reference is therefore a literal-true gate, not a sample-0-only
    // gate — both should produce identical Accum behavior across the
    // whole buffer in which the trigger fires.
    const triggerSession = makeSession(8)
    loadBuiltins(triggerSession)
    loadProgramAsType(ACCUM, triggerSession)
    loadJSON({
      schema: 'tropical_program_2',
      name: 'patch',
      body: { op: 'block', decls: [
        { op: 'paramDecl', name: 'go', type: 'trigger' } as unknown as ExprNode,
        { op: 'instanceDecl', name: 'a1', program: 'Accum',
          inputs: { x: 1.0 }, gateable: true,
          gate_input: { op: 'trigger', name: 'go' } },
      ]},
      audio_outputs: [{ instance: 'a1', output: 'out' }],
    }, triggerSession)
    applySessionWiring(triggerSession)
    triggerSession.graph.primeJit()
    // Fire the trigger before the buffer runs. The runtime snapshots
    // the value once per buffer; gate stays true for all 8 samples.
    triggerSession.triggerRegistry.get('go')!.fire()
    triggerSession.graph.process()
    const triggerOut = new Float64Array(triggerSession.graph.outputBuffer)

    const literalSession = makeSession(8)
    loadBuiltins(literalSession)
    loadProgramAsType(ACCUM, literalSession)
    // Reference: gate_input = literal true (whole buffer on), matching
    // the trigger's per-buffer snapshot semantics described above.
    loadJSON({
      schema: 'tropical_program_2',
      name: 'patch',
      body: { op: 'block', decls: [
        { op: 'instanceDecl', name: 'a1', program: 'Accum',
          inputs: { x: 1.0 }, gateable: true,
          gate_input: true },
      ]},
      audio_outputs: [{ instance: 'a1', output: 'out' }],
    }, literalSession)
    applySessionWiring(literalSession)
    literalSession.graph.primeJit()
    literalSession.graph.process()
    const literalOut = new Float64Array(literalSession.graph.outputBuffer)

    // Both should produce the same output (gateable Accum: sample 0
    // gate true → out updates to 1; samples 1+ gate false → out holds).
    for (let i = 0; i < triggerOut.length; i++) {
      expect(triggerOut[i]).toBeCloseTo(literalOut[i], 10)
    }

    triggerSession.graph.dispose()
    literalSession.graph.dispose()
  })

  test('gate chain (a2 gated on a1 output): JIT matches interpreter', () => {
    // Two gateable instances where the second's gate is derived from the
    // first's output. Verifies gate-dependency tracking in the topo graph
    // and correct hot-swap behaviour when one gate toggles the other.
    const session = makeSession(32)
    loadBuiltins(session)
    loadProgramAsType(ACCUM, session)

    // a1 gates on (sample_index % 6) < 3 — alternates live/skip in 3-sample runs.
    const a1Gate: ExprNode = {
      op: 'lt',
      args: [{ op: 'mod', args: [{ op: 'sampleIndex' }, 6] }, 3],
    }
    // a2's gate depends on a1's output: live only when a1 has accumulated
    // past some threshold.
    const a2Gate: ExprNode = {
      op: 'gt',
      args: [{ op: 'ref', instance: 'a1', output: 'out' }, 1.5],
    }

    loadJSON({
      schema: 'tropical_program_2',
      name: 'patch',
      body: { op: 'block', decls: [
        { op: 'instanceDecl', name: 'a1', program: 'Accum',
          inputs: { x: 1.0 }, gateable: true, gate_input: a1Gate },
        { op: 'instanceDecl', name: 'a2', program: 'Accum',
          inputs: { x: 1.0 }, gateable: true, gate_input: a2Gate },
      ]},
      audio_outputs: [{ instance: 'a2', output: 'out' }],
    }, session)

    applySessionWiring(session)
    session.graph.primeJit()
    session.graph.process()
    const jit = new Float64Array(session.graph.outputBuffer)
    const interp = interpretSession(session, jit.length)

    for (let i = 0; i < jit.length; i++) {
      expect(jit[i]).toBeCloseTo(interp[i], 10)
    }
    session.graph.dispose()
  })
})
