/**
 * build_patches.ts — precompile curated patches for the web demo.
 *
 * Patches are defined inline from stdlib programs (SinOsc, OnePole, etc.),
 * flattened to tropical_plan_4 JSON, and written to
 * web/dist/patches/<slug>.plan.json. Browser loads them via fetch().
 *
 *   bun web/build_patches.ts
 */

import { writeFileSync, mkdirSync } from 'fs'
import { dirname, resolve, join } from 'path'
import { fileURLToPath } from 'url'
import { makeSession, loadJSON } from '../compiler/session.js'
import { loadStdlib } from '../compiler/program.js'
import { flattenSession } from '../compiler/flatten.js'

const __dirname = dirname(fileURLToPath(import.meta.url))
const distDir = resolve(__dirname, 'dist/patches')

type Patch = { slug: string; title: string; description: string; program: unknown }

const PATCHES: Patch[] = [
  {
    slug: 'pure-sine-440',
    title: 'Pure Sine 440',
    description: 'A plain SinOsc at 440 Hz. The smallest thing tropical does.',
    program: {
      schema: 'tropical_program_2',
      name: 'pure_sine_440',
      body: { op: 'block', decls: [
        { op: 'instance_decl', name: 'osc', program: 'SinOsc', inputs: { freq: 440 } },
      ]},
      audio_outputs: [{ instance: 'osc', output: 'sine' }],
    },
  },
  {
    slug: 'sine-lowpass',
    title: 'Filtered Sine',
    description: 'SinOsc at 220 Hz through a OnePole lowpass at 1 kHz.',
    program: {
      schema: 'tropical_program_2',
      name: 'sine_lowpass',
      body: { op: 'block', decls: [
        { op: 'instance_decl', name: 'osc', program: 'SinOsc', inputs: { freq: 220 } },
        { op: 'instance_decl', name: 'lp',  program: 'OnePole', inputs: {
          signal: { op: 'ref', instance: 'osc', output: 'sine' },
          cutoff: 1000,
        }},
      ]},
      audio_outputs: [{ instance: 'lp', output: 'out' }],
    },
  },
  {
    slug: 'ladder-sawlike',
    title: 'Ladder Filter',
    description: 'SinOsc (tri-like at 110 Hz) driving a Moog-style LadderFilter.',
    program: {
      schema: 'tropical_program_2',
      name: 'ladder_saw',
      body: { op: 'block', decls: [
        { op: 'instance_decl', name: 'osc', program: 'SinOsc', inputs: { freq: 110 } },
        { op: 'instance_decl', name: 'lp',  program: 'LadderFilter', inputs: {
          signal: { op: 'ref', instance: 'osc', output: 'sine' },
          cutoff: 1200,
          resonance: 0.7,
        }},
      ]},
      audio_outputs: [{ instance: 'lp', output: 'out' }],
    },
  },
  {
    slug: 'noise-crush',
    title: 'Crushed Noise',
    description: 'NoiseLFSR through a BitCrusher — rough digital hiss.',
    program: {
      schema: 'tropical_program_2',
      name: 'noise_crush',
      body: { op: 'block', decls: [
        { op: 'instance_decl', name: 'noise', program: 'NoiseLFSR' },
        { op: 'instance_decl', name: 'crush', program: 'BitCrusher', inputs: {
          signal: { op: 'ref', instance: 'noise', output: 'out' },
          bits: 4,
          rate: 8000,
        }},
      ]},
      audio_outputs: [{ instance: 'crush', output: 'out' }],
    },
  },
  {
    slug: 'sine-phaser',
    title: 'Phaser Sine',
    description: 'SinOsc at 165 Hz through a Phaser with LFO sweep.',
    program: {
      schema: 'tropical_program_2',
      name: 'sine_phaser',
      body: { op: 'block', decls: [
        { op: 'instance_decl', name: 'osc', program: 'SinOsc', inputs: { freq: 165 } },
        { op: 'instance_decl', name: 'ph',  program: 'Phaser', inputs: {
          signal: { op: 'ref', instance: 'osc', output: 'sine' },
          rate: 0.3,
          depth: 0.8,
          feedback: 0.5,
        }},
      ]},
      audio_outputs: [{ instance: 'ph', output: 'out' }],
    },
  },
  {
    slug: 'sequencer-saw',
    title: 'Sequenced Saw',
    description: 'A 4-step sequencer advancing a BlepSaw through a OnePole.',
    program: {
      schema: 'tropical_program_2',
      name: 'sequencer_saw',
      body: { op: 'block', decls: [
        { op: 'instance_decl', name: 'clk',  program: 'Clock', inputs: { freq: 4 } },
        { op: 'instance_decl', name: 'seq',  program: 'Sequencer', type_args: { N: 4 }, inputs: {
          tick:   { op: 'ref', instance: 'clk', output: 'pulse' },
          values: { op: 'array_literal', shape: [4], values: [110, 165, 147, 220] },
        }},
        { op: 'instance_decl', name: 'osc',  program: 'BlepSaw', inputs: {
          freq: { op: 'ref', instance: 'seq', output: 'value' },
        }},
        { op: 'instance_decl', name: 'lp',   program: 'OnePole', inputs: {
          signal: { op: 'ref', instance: 'osc', output: 'saw' },
          cutoff: 1500,
        }},
      ]},
      audio_outputs: [{ instance: 'lp', output: 'out' }],
    },
  },
]

mkdirSync(distDir, { recursive: true })

const manifest: Array<{ slug: string; title: string; description: string; planPath: string }> = []

for (const patch of PATCHES) {
  try {
    const session = makeSession(1024)
    loadStdlib(session)
    loadJSON(patch.program as { schema: string; [k: string]: unknown }, session)
    const plan = flattenSession(session)
    const out = JSON.stringify(plan)
    const outPath = join(distDir, `${patch.slug}.plan.json`)
    writeFileSync(outPath, out, 'utf-8')
    manifest.push({ slug: patch.slug, title: patch.title, description: patch.description, planPath: `patches/${patch.slug}.plan.json` })
    // eslint-disable-next-line no-console
    console.log(`built ${patch.slug.padEnd(22)} ${out.length.toString().padStart(8)} bytes`)
    session.runtime.dispose()
  } catch (err) {
    // eslint-disable-next-line no-console
    console.warn(`SKIP  ${patch.slug}: ${(err as Error).message}`)
  }
}

writeFileSync(join(distDir, 'index.json'), JSON.stringify(manifest, null, 2), 'utf-8')
// eslint-disable-next-line no-console
console.log(`\nmanifest: ${manifest.length} patches -> ${join(distDir, 'index.json')}`)
