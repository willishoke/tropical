/**
 * Freeze golden FlatPlan fixtures for the unified-IR refactor (Phase A).
 *
 * For each input program (patches that load cleanly + a set of synthetic
 * scenarios exercising the stdlib features), runs the current v1 pipeline
 * and emits a JSON fixture containing:
 *   { input: <tropical_program_1 JSON>, expected_plan: <tropical_plan_4 JSON> }
 *
 * Run once, commit. After the refactor lands, the round-trip test loads each
 * fixture, migrates the input to v2 in-memory, routes through the new
 * pipeline, and asserts byte-equality against expected_plan.
 */

import { readFileSync, writeFileSync, mkdirSync, readdirSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

import { makeSession, loadJSON } from '../compiler/session.js'
import { loadStdlib } from '../compiler/program.js'
import { flattenSession } from '../compiler/flatten.js'
import type { ProgramJSON } from '../compiler/program.js'

const __dirname = dirname(fileURLToPath(import.meta.url))
const REPO = join(__dirname, '..')
const OUT = join(REPO, 'compiler/__fixtures__/flat_plan')
mkdirSync(OUT, { recursive: true })

interface Fixture { input: ProgramJSON; expected_plan: unknown }

function freeze(name: string, input: ProgramJSON): void {
  const session = makeSession()
  loadStdlib(session)
  loadJSON(input as unknown as { schema: string; [k: string]: unknown }, session)
  const plan = flattenSession(session)
  const fixture: Fixture = { input, expected_plan: plan }
  writeFileSync(join(OUT, `${name}.json`), JSON.stringify(fixture, null, 2))
  const planStr = JSON.stringify(plan)
  console.log(`  ${name}: ${plan.instructions.length} instrs, ${plan.register_count} regs, ${planStr.length} bytes`)
}

// ── Real patches that load cleanly ─────────────────────────────
console.log('Patches:')
const patchesDir = join(REPO, 'patches')
for (const file of readdirSync(patchesDir)) {
  if (!file.endsWith('.json')) continue
  const name = file.slice(0, -5)
  try {
    const input = JSON.parse(readFileSync(join(patchesDir, file), 'utf-8')) as ProgramJSON
    freeze(`patch_${name}`, input)
  } catch (e) {
    console.log(`  patch_${name}: SKIP (${(e as Error).message.slice(0, 80)})`)
  }
}

// ── Synthetic scenarios exercising stdlib features ─────────────
//
// Each scenario is a minimal top-level program that instantiates one or
// more stdlib types and routes the first output to audio. The goal is
// coverage of compilation pathways (nested instances, delays, type
// params, array ports, combinators, inline nested programs).

const scenarios: Array<[string, ProgramJSON]> = [
  // Simple leaf: Sin with a literal freq input
  ['stdlib_sin', {
    schema: 'tropical_program_1',
    name: 'stdlib_sin_scenario',
    instances: { osc: { program: 'Sin', inputs: { x: 440 } } },
    audio_outputs: [{ instance: 'osc', output: 'out' }],
  }],
  // SinOsc — oscillator with internal state
  ['stdlib_sinosc', {
    schema: 'tropical_program_1',
    name: 'stdlib_sinosc_scenario',
    instances: { osc: { program: 'SinOsc', inputs: { freq: 440 } } },
    audio_outputs: [{ instance: 'osc', output: 'sine' }],
  }],
  // OnePole — nested instances (Tanh x2)
  ['stdlib_onepole', {
    schema: 'tropical_program_1',
    name: 'stdlib_onepole_scenario',
    instances: {
      osc: { program: 'SinOsc', inputs: { freq: 440 } },
      filt: { program: 'OnePole', inputs: { input: { op: 'ref', instance: 'osc', output: 'sine' }, g: 0.3 } },
    },
    audio_outputs: [{ instance: 'filt', output: 'out' }],
  }],
  // LadderFilter — four OnePoles + SoftClip nested
  ['stdlib_ladder', {
    schema: 'tropical_program_1',
    name: 'stdlib_ladder_scenario',
    instances: {
      osc: { program: 'BlepSaw', inputs: { freq: 220 } },
      filt: { program: 'LadderFilter', inputs: {
        input: { op: 'ref', instance: 'osc', output: 'saw' },
        cutoff: 1200, resonance: 0.7, drive: 1.0,
      } },
    },
    audio_outputs: [{ instance: 'filt', output: 'lp' }],
  }],
  // Phaser — inline `programs` map (nested _allpassStage definition)
  ['stdlib_phaser', {
    schema: 'tropical_program_1',
    name: 'stdlib_phaser_scenario',
    instances: {
      osc: { program: 'BlepSaw', inputs: { freq: 110 } },
      fx: { program: 'Phaser', inputs: {
        input: { op: 'ref', instance: 'osc', output: 'saw' },
        feedback: 0.5, lfo_speed: 0.3,
      } },
    },
    audio_outputs: [{ instance: 'fx', output: 'output' }],
  }],
  // Sequencer — generic with type_args
  ['stdlib_sequencer', {
    schema: 'tropical_program_1',
    name: 'stdlib_sequencer_scenario',
    instances: {
      clk: { program: 'Clock', inputs: { freq: 4 } },
      seq: {
        program: 'Sequencer', type_args: { N: 4 },
        inputs: {
          clock: { op: 'ref', instance: 'clk', output: 'output' },
          values: [200, 300, 400, 500],
        },
      },
      osc: { program: 'SinOsc', inputs: { freq: { op: 'ref', instance: 'seq', output: 'value' } } },
    },
    audio_outputs: [{ instance: 'osc', output: 'sine' }],
  }],
  // Delay — generic array state (fixed N-sample delay line)
  ['stdlib_delay', {
    schema: 'tropical_program_1',
    name: 'stdlib_delay_scenario',
    instances: {
      osc: { program: 'SinOsc', inputs: { freq: 440 } },
      dly: { program: 'Delay', type_args: { N: 1000 }, inputs: {
        x: { op: 'ref', instance: 'osc', output: 'sine' },
      } },
    },
    audio_outputs: [{ instance: 'dly', output: 'y' }],
  }],
  // NoiseLFSR + BitCrusher — pure integer/bit ops
  ['stdlib_noise_crush', {
    schema: 'tropical_program_1',
    name: 'stdlib_noise_crush_scenario',
    instances: {
      n: { program: 'NoiseLFSR', inputs: {} },
      bc: { program: 'BitCrusher', inputs: {
        audio: { op: 'ref', instance: 'n', output: 'out' },
        bit_depth: 6,
      } },
    },
    audio_outputs: [{ instance: 'bc', output: 'output' }],
  }],
  // VCA + CrossFade — pure scalar composition
  ['stdlib_vca_xfade', {
    schema: 'tropical_program_1',
    name: 'stdlib_vca_xfade_scenario',
    instances: {
      a: { program: 'SinOsc', inputs: { freq: 220 } },
      b: { program: 'SinOsc', inputs: { freq: 330 } },
      xf: { program: 'CrossFade', inputs: {
        a: { op: 'ref', instance: 'a', output: 'sine' },
        b: { op: 'ref', instance: 'b', output: 'sine' },
        mix: 0.5,
      } },
      vca: { program: 'VCA', inputs: {
        audio: { op: 'ref', instance: 'xf', output: 'out' },
        cv: 0.6,
      } },
    },
    audio_outputs: [{ instance: 'vca', output: 'out' }],
  }],
]

console.log('Synthetic scenarios:')
for (const [name, input] of scenarios) freeze(name, input)

console.log(`\nWrote ${readdirSync(OUT).length} fixtures to ${OUT}`)
