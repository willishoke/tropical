#!/usr/bin/env node
'use strict'

const assert = require('node:assert/strict')
const { spawn } = require('node:child_process')
const { tmpdir } = require('node:os')
const { join, resolve } = require('node:path')
const { performance } = require('node:perf_hooks')

const scene = require('../scene.js')
const scopeFrame = require('../renderer/scope-frame.js')
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
const ages = [0.12, 0.25, 0.5, 1, 2, 3.8]
const playbackPositions = [0, 1, 2].map((loop) => (
  count + loop * scene.SCENE_SECONDS * scene.SAMPLE_RATE
))

function peak(values) {
  return values.reduce((maximum, value) => Math.max(maximum, Math.abs(value)), 0)
}

function percentile(values, fraction) {
  const sorted = [...values].sort((a, b) => a - b)
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * fraction))]
}

function maximumDifference(left, right) {
  assert.equal(left.length, right.length)
  return left.reduce(
    (maximum, value, index) => Math.max(maximum, Math.abs(value - right[index])),
    0,
  )
}

async function probe() {
  await client.call('load_patch_graph', scene.buildSceneGraph())
  const taps = (await client.call('list_scope_taps')).taps || []
  const slots = new Map(taps.map((tap) => [tap.name, tap.slot]))
  scene.SCOPE_TAPS.forEach((name) => assert.ok(slots.has(name), `missing ${name}`))

  const observations = []
  const rpcMs = []
  const loopReferences = new Map()
  for (let chordIndex = 0; chordIndex < scene.CHORDS.length; chordIndex += 1) {
    const chord = scene.CHORDS[chordIndex]
    const chordSlots = chord.voices.map((_voice, voiceIndex) => (
      slots.get(scene.scopeModeId(chordIndex, voiceIndex))
    ))
    for (const age of ages) {
      for (let loop = 0; loop < playbackPositions.length; loop += 1) {
        const position = playbackPositions[loop]
        const audibleSceneTime = chord.startStep * scene.STEP_SECONDS + age
        const tauBase = audibleSceneTime - position / scene.SAMPLE_RATE
        await client.call('set_param', {
          name: 'master.tau_base',
          value: tauBase,
        })
        const before = performance.now()
        const response = await client.call('render_window', {
          start: position - count,
          count,
          point_budget: scopeProfile.pointBudget,
          slots: chordSlots,
        })
        rpcMs.push(performance.now() - before)
        assert.equal(response.stride, 1)
        response.values.forEach((values, voiceIndex) => {
          const raw = values.slice(scopeProfile.warmupSamples)
          const frequency = scene.voiceFrequency(chord, chord.voices[voiceIndex])
          const frame = scopeFrame.fromProjection({
            values,
            responseStart: response.start,
            stride: response.stride,
            warmupSamples: scopeProfile.warmupSamples,
            displaySamples: scopeProfile.displaySamples,
            frequency,
            chordIndex,
            voiceIndex,
            tauBase,
            velocity: 1,
            playbackPosition: position,
          })
          const { cycles, displayEnvelope } = frame
          const displayed = frame.values.map((value) => (
            scopeView.applyEnvelope(value, displayEnvelope)
          ))
          const expectedCarrier = frame.values.map((_value, index) => (
            Math.sin(
              Math.PI * 2 * cycles
                * (index - (frame.values.length - 1) / 2)
                / (frame.values.length - 1),
            )
          ))
          const distinct = new Set(raw.map((value) => value.toPrecision(15))).size
          assert.ok(
            distinct > raw.length * 0.95,
            `${chord.label}/${voiceIndex} became constant (${distinct}/${raw.length})`,
          )
          assert.equal(frame.active, true, `${chord.label}/${voiceIndex} became silent`)
          assert.equal(frame.locked, true, `${chord.label}/${voiceIndex} lost phase lock`)
          const lockError = Math.abs(frame.centerValue) / frame.peak
          const cycleShapeError = maximumDifference(frame.values, expectedCarrier)
          const amplitudeError = Math.abs(frame.peak - 1)
          assert.ok(lockError < 1e-12, `${chord.label}/${voiceIndex} lock error ${lockError}`)
          assert.ok(
            cycleShapeError < 1e-3,
            `${chord.label}/${voiceIndex} cycle shape error ${cycleShapeError}`,
          )
          assert.ok(
            amplitudeError < 1e-3,
            `${chord.label}/${voiceIndex} carrier amplitude error ${amplitudeError}`,
          )
          const key = `${chordIndex}:${voiceIndex}:${age}`
          const reference = loopReferences.get(key)
          const loopError = reference ? maximumDifference(displayed, reference) : 0
          const relativeLoopError = displayEnvelope > 0
            ? loopError / displayEnvelope : loopError
          if (!reference) loopReferences.set(key, displayed)
          assert.ok(
            relativeLoopError < 1e-5,
            `${chord.label}/${voiceIndex} loop ${loop} drifted by ${relativeLoopError}`,
          )
          observations.push({
            chord: chord.label,
            voice: voiceIndex,
            age,
            loop,
            distinct,
            raw_peak: peak(raw),
            display_envelope: displayEnvelope,
            cycles,
            lock_error: lockError,
            cycle_shape_error: cycleShapeError,
            carrier_amplitude_error: amplitudeError,
            loop_error: loopError,
            relative_loop_error: relativeLoopError,
          })
        })
      }
    }
  }

  for (const chord of scene.CHORDS) {
    for (let voice = 0; voice < chord.voices.length; voice += 1) {
      const series = observations.filter((entry) => (
        entry.chord === chord.label && entry.voice === voice && entry.loop === 0
      ))
      assert.ok(
        series[2].display_envelope > series[1].display_envelope,
        `${chord.label}/${voice} has no attack`,
      )
      assert.ok(
        series[5].display_envelope < series[3].display_envelope,
        `${chord.label}/${voice} has no decay`,
      )
    }
  }

  return {
    pass: true,
    taps: scene.SCOPE_TAPS.length,
    observations: observations.length,
    scene_loops: playbackPositions.length,
    audio_started: false,
    minimum_distinct_samples: Math.min(...observations.map((entry) => entry.distinct)),
    maximum_relative_lock_error: Math.max(...observations.map((entry) => entry.lock_error)),
    maximum_cycle_shape_error: Math.max(...observations.map((entry) => (
      entry.cycle_shape_error
    ))),
    maximum_carrier_amplitude_error: Math.max(...observations.map((entry) => (
      entry.carrier_amplitude_error
    ))),
    maximum_loop_error: Math.max(...observations.map((entry) => entry.loop_error)),
    maximum_relative_loop_error: Math.max(...observations.map((entry) => (
      entry.relative_loop_error
    ))),
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
