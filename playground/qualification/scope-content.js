#!/usr/bin/env node
'use strict'

const assert = require('node:assert/strict')
const { spawn } = require('node:child_process')
const { tmpdir } = require('node:os')
const { join, resolve } = require('node:path')
const { performance } = require('node:perf_hooks')

const scene = require('../scene.js')
const phaseLock = require('../renderer/phase-lock.js')
const scopeProfile = require('../renderer/scope-profile.js')
const scopeView = require('../renderer/scope-view.js')
const { JsonLineRpcClient } = require('../rpc-client.js')

const repo = resolve(__dirname, '../..')
const enginePath = process.argv[2]
  ? resolve(process.argv[2])
  : join(repo, 'lean/.lake/build/bin/frontend')
const socketPath = join(tmpdir(), `tropical-scope-content-${process.pid}.sock`)
const engine = spawn(enginePath, ['--serve', socketPath], {
  cwd: repo,
  stdio: ['ignore', 'ignore', 'inherit'],
})
const client = new JsonLineRpcClient({ path: socketPath })
client.start()

const count = scopeProfile.displaySamples
  + scopeProfile.searchSamples + scopeProfile.warmupSamples
const ages = [0.08, 0.25, 0.5, 1, 2, 3.8]

function peak(values) {
  return values.reduce((maximum, value) => Math.max(maximum, Math.abs(value)), 0)
}

function percentile(values, fraction) {
  const sorted = [...values].sort((a, b) => a - b)
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * fraction))]
}

async function probe() {
  await client.call('load_patch_graph', scene.buildSceneGraph())
  const taps = (await client.call('list_scope_taps')).taps || []
  const slots = new Map(taps.map((tap) => [tap.name, tap.slot]))
  scene.SCOPE_TAPS.forEach((name) => assert.ok(slots.has(name), `missing ${name}`))

  const observations = []
  const rpcMs = []
  for (let chordIndex = 0; chordIndex < scene.CHORDS.length; chordIndex += 1) {
    const chord = scene.CHORDS[chordIndex]
    const chordSlots = chord.voices.map((_voice, voiceIndex) => (
      slots.get(scene.scopeModeId(chordIndex, voiceIndex))
    ))
    for (const age of ages) {
      await client.call('set_param', {
        name: 'master.tau_base',
        value: chord.startStep * scene.STEP_SECONDS + age,
      })
      const before = performance.now()
      const response = await client.call('render_window', {
        start: 0,
        count,
        point_budget: scopeProfile.pointBudget,
        slots: chordSlots,
      })
      rpcMs.push(performance.now() - before)
      assert.equal(response.stride, 1)
      response.values.forEach((values, voiceIndex) => {
        const raw = values.slice(scopeProfile.warmupSamples)
        const frequency = scene.voiceFrequency(chord, chord.voices[voiceIndex])
        const cycles = scopeView.visibleCycles(frequency, scene.BASE_HZ)
        const frame = phaseLock.centeredWindow(
          raw,
          scopeProfile.displaySamples,
          cycles * scene.SAMPLE_RATE / frequency,
        )
        const distinct = new Set(raw.map((value) => value.toPrecision(15))).size
        assert.ok(
          distinct > raw.length * 0.95,
          `${chord.label}/${voiceIndex} became constant (${distinct}/${raw.length})`,
        )
        assert.equal(frame.active, true, `${chord.label}/${voiceIndex} became silent`)
        assert.equal(frame.locked, true, `${chord.label}/${voiceIndex} lost phase lock`)
        const lockError = Math.abs(frame.centerValue) / frame.peak
        assert.ok(lockError < 1e-12, `${chord.label}/${voiceIndex} lock error ${lockError}`)
        observations.push({
          chord: chord.label,
          voice: voiceIndex,
          age,
          distinct,
          peak: peak(raw),
          cycles,
          lock_error: lockError,
        })
      })
    }
  }

  for (const chord of scene.CHORDS) {
    for (let voice = 0; voice < chord.voices.length; voice += 1) {
      const series = observations.filter((entry) => (
        entry.chord === chord.label && entry.voice === voice
      ))
      assert.ok(series[2].peak > series[1].peak, `${chord.label}/${voice} has no attack`)
      assert.ok(series[5].peak < series[3].peak, `${chord.label}/${voice} has no decay`)
    }
  }

  return {
    pass: true,
    taps: scene.SCOPE_TAPS.length,
    observations: observations.length,
    minimum_distinct_samples: Math.min(...observations.map((entry) => entry.distinct)),
    maximum_relative_lock_error: Math.max(...observations.map((entry) => entry.lock_error)),
    scope_rpc_ms: {
      p50: percentile(rpcMs, 0.5),
      p95: percentile(rpcMs, 0.95),
      p99: percentile(rpcMs, 0.99),
      max: Math.max(...rpcMs),
    },
  }
}

function finish(error) {
  client.close(error || new Error('scope content probe complete'))
  engine.kill()
  if (error) {
    console.error(error)
    process.exitCode = 1
  }
}

probe().then((result) => {
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`)
  finish()
}).catch(finish)
