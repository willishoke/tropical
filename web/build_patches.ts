/**
 * build_patches.ts — precompile curated patches for the web demo.
 *
 * Patches are defined inline from stdlib programs (SinOsc, OnePole, etc.),
 * flattened to tropical_plan_4 JSON, and written to
 * web/dist/patches/<slug>.plan.json. Browser loads them via fetch().
 *
 *   bun web/build_patches.ts
 */

import { writeFileSync, mkdirSync, readdirSync, unlinkSync, existsSync } from 'fs'
import { dirname, resolve, join } from 'path'
import { fileURLToPath } from 'url'
import { makeSession, loadJSON } from '../compiler/session.js'
import { loadStdlib } from '../compiler/program.js'
import { flattenSession } from '../compiler/flatten.js'

const __dirname = dirname(fileURLToPath(import.meta.url))
const distDir = resolve(__dirname, 'dist/patches')

type Patch = { slug: string; title: string; description: string; program: unknown }

// Curated list. Patches that use programs with delay lines or `pow`
// (OnePole, LadderFilter, Phaser, BitCrusher, ...) are excluded on this
// branch because the TS flattener on origin/main emits unresolved
// delay_value / pow ops; both native and WASM backends render them as
// zero. See project_testing_gaps.md.
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
    slug: 'ring-mod',
    title: 'Ring Mod',
    description: 'Two SinOscs (220 Hz × 331 Hz) multiplied through a VCA — dissonant partials.',
    program: {
      schema: 'tropical_program_2',
      name: 'ring_mod',
      body: { op: 'block', decls: [
        { op: 'instance_decl', name: 'a', program: 'SinOsc', inputs: { freq: 220 } },
        { op: 'instance_decl', name: 'b', program: 'SinOsc', inputs: { freq: 331 } },
        { op: 'instance_decl', name: 'm', program: 'VCA', inputs: {
          audio: { op: 'ref', instance: 'a', output: 'sine' },
          cv:    { op: 'ref', instance: 'b', output: 'sine' },
        }},
      ]},
      audio_outputs: [{ instance: 'm', output: 'out' }],
    },
  },
  {
    slug: 'fm-pair',
    title: 'FM Pair',
    description: 'Simple 2-op FM: a 73 Hz modulator shaping a ~220 Hz carrier.',
    program: {
      schema: 'tropical_program_2',
      name: 'fm_pair',
      body: { op: 'block', decls: [
        { op: 'instance_decl', name: 'mod', program: 'SinOsc', inputs: { freq: 73 } },
        { op: 'instance_decl', name: 'car', program: 'SinOsc', inputs: {
          freq: { op: 'add', args: [220, { op: 'mul', args: [80, { op: 'ref', instance: 'mod', output: 'sine' }] }] },
        }},
      ]},
      audio_outputs: [{ instance: 'car', output: 'sine' }],
    },
  },
  {
    slug: 'fm-stack',
    title: 'FM Stack',
    description: '3-op FM with a slow LFO on the modulator depth — evolves gently.',
    program: {
      schema: 'tropical_program_2',
      name: 'fm_stack',
      body: { op: 'block', decls: [
        { op: 'instance_decl', name: 'm1', program: 'SinOsc', inputs: { freq: 0.5 } },
        { op: 'instance_decl', name: 'm2', program: 'SinOsc', inputs: {
          freq: { op: 'add', args: [11, { op: 'mul', args: [5, { op: 'ref', instance: 'm1', output: 'sine' }] }] },
        }},
        { op: 'instance_decl', name: 'c',  program: 'SinOsc', inputs: {
          freq: { op: 'add', args: [220, { op: 'mul', args: [200, { op: 'ref', instance: 'm2', output: 'sine' }] }] },
        }},
      ]},
      audio_outputs: [{ instance: 'c', output: 'sine' }],
    },
  },
  {
    slug: 'blepsaw',
    title: 'Band-Limited Saw',
    description: 'A 110 Hz BlepSaw — anti-aliased classic subtractive waveform.',
    program: {
      schema: 'tropical_program_2',
      name: 'blepsaw',
      body: { op: 'block', decls: [
        { op: 'instance_decl', name: 'o', program: 'BlepSaw', inputs: { freq: 110 } },
      ]},
      audio_outputs: [{ instance: 'o', output: 'saw' }],
    },
  },
]

mkdirSync(distDir, { recursive: true })

// Clear any stale .plan.json from a prior build (keep index.json; we rewrite it).
if (existsSync(distDir)) {
  for (const f of readdirSync(distDir)) {
    if (f.endsWith('.plan.json')) unlinkSync(join(distDir, f))
  }
}

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
