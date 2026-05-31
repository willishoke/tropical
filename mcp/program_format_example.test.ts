/**
 * Regression guard for the canonical program-format doc example.
 *
 * The `tropical://program-format` resource is what the build-patch prompt tells
 * agents to fetch before writing any patch — so a rotted example is a rotted
 * agent. This pins the example (which once taught deprecated `audio_outputs`
 * and a non-existent `sin` op) to: no deprecated fields, loads cleanly, makes
 * sound.
 *
 * Needs the native dylib (loadJSON → JIT). Runs in CI's build-and-test job
 * (which builds libtropical before `bun test`); excluded from the pure-TS
 * subset like apply_plan.test.ts.
 */
import { test, expect } from 'bun:test'
import { makeSession, loadJSON } from '../compiler/session.js'
import { loadStdlib as loadBuiltins } from '../compiler/program.js'
import * as b from '../compiler/runtime/bindings.js'
import { PROGRAM_FORMAT_EXAMPLE } from './program_format_example.js'

test('program-format example uses no deprecated file-root fields', () => {
  const s = JSON.stringify(PROGRAM_FORMAT_EXAMPLE)
  expect(s).not.toContain('audio_outputs')
  // Output must be wired via a body dac.out outputAssign.
  expect(s).toContain('dac.out')
})

test('program-format example loads and produces non-zero, non-exploding audio', () => {
  const session = makeSession(256)
  loadBuiltins(session)
  // Clone — loadJSON may mutate its input, and the doc renders the original.
  loadJSON(structuredClone(PROGRAM_FORMAT_EXAMPLE), session)

  const runtime = session.runtime
  let peak = 0
  for (let i = 0; i < 4410; i++) {
    b.tropical_runtime_process(runtime._h)
    const buf = runtime.outputBuffer
    for (let j = 0; j < buf.length; j++) peak = Math.max(peak, Math.abs(buf[j]))
  }
  expect(peak).toBeGreaterThan(0)   // not silent
  expect(peak).toBeLessThan(100)    // not exploding/NaN
})
