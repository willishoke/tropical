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
const socketPath = join(tmpdir(), `tropical-scope-live-${process.pid}.sock`)
const engine = spawn(enginePath, ['--serve', socketPath], {
  cwd: repo,
  env: {
    ...process.env,
    TROPICAL_BACKEND: 'metal',
    TROPICAL_BUFFER_LENGTH: '128',
    TROPICAL_METAL_RENDER_TILE_FRAMES: '512',
    TROPICAL_QUALIFICATION: '1',
  },
  stdio: ['ignore', 'ignore', 'inherit'],
})
const control = new JsonLineRpcClient({ path: socketPath })
const scope = new JsonLineRpcClient({ path: socketPath })
control.start()
scope.start()

const requestCount = scopeProfile.displaySamples
  + scopeProfile.searchSamples + scopeProfile.warmupSamples
const frameCount = 90

const sleep = (milliseconds) => new Promise((resolveSleep) => (
  setTimeout(resolveSleep, milliseconds)
))

function maximumDifference(left, right) {
  assert.equal(left.length, right.length)
  return left.reduce(
    (maximum, value, index) => Math.max(maximum, Math.abs(value - right[index])),
    0,
  )
}

async function probeChord(chordIndex, slots) {
  const chord = scene.CHORDS[chordIndex]
  const observedPosition = (await control.call('playback_position')).position
  const desiredSceneTime = chord.startStep * scene.STEP_SECONDS + 0.5
  const tauBase = desiredSceneTime - observedPosition / scene.SAMPLE_RATE
  await control.call('set_param', { name: 'master.tau_base', value: tauBase })
  await sleep(80)

  const references = []
  let maximumShapeDelta = 0
  let maximumCycleError = 0
  let maximumAmplitudeError = 0
  let maximumLockError = 0
  let previousPosition = -1
  let nextFrame = performance.now()

  for (let observation = 0; observation < frameCount; observation += 1) {
    const position = (await scope.call('playback_position')).position
    assert.ok(position > previousPosition, `${chord.label} playback clock stopped`)
    previousPosition = position
    const response = await scope.call('render_window', {
      start: Math.max(0, position - requestCount),
      count: requestCount,
      point_budget: scopeProfile.pointBudget,
      slots,
    })
    assert.equal(response.preempted, false)
    assert.equal(response.stride, 1)

    response.values.forEach((values, voiceIndex) => {
      const frequency = scene.voiceFrequency(chord, chord.voices[voiceIndex])
      const frame = scopeFrame.fromProjection({
        values,
        responseStart: response.start,
        stride: response.stride,
        warmupSamples: scopeProfile.warmupSamples,
        displaySamples: scopeProfile.displaySamples,
        timeCompression: scopeProfile.timeCompression,
        frequency,
        chordIndex,
        voiceIndex,
        tauBase,
        velocity: 1,
        playbackPosition: position,
      })
      const rawPeak = values.reduce(
        (maximum, value) => Math.max(maximum, Math.abs(value)), 0,
      )
      assert.equal(
        frame.active,
        true,
        `${chord.label}/${voiceIndex} frame ${observation} inactive; raw peak ${rawPeak}; position ${position}; tau ${tauBase}; control ${response.control_version}`,
      )
      assert.equal(
        frame.locked,
        true,
        `${chord.label}/${voiceIndex} frame ${observation} unlocked; raw peak ${rawPeak}; position ${position}; tau ${tauBase}; control ${response.control_version}`,
      )

      const expected = frame.values.map((_value, index) => Math.sin(
        Math.PI * 2 * frame.cycles
          * (index - (frame.values.length - 1) / 2)
          / (frame.values.length - 1),
      ))
      maximumCycleError = Math.max(
        maximumCycleError, maximumDifference(frame.values, expected),
      )
      maximumLockError = Math.max(
        maximumLockError, Math.abs(frame.centerValue) / frame.peak,
      )
      maximumAmplitudeError = Math.max(
        maximumAmplitudeError, Math.abs(frame.peak - 1),
      )
      if (references[voiceIndex]) {
        maximumShapeDelta = Math.max(
          maximumShapeDelta,
          maximumDifference(frame.values, references[voiceIndex]),
        )
      } else {
        references[voiceIndex] = frame.values
      }
    })

    nextFrame += 1000 / scopeProfile.fps
    await sleep(Math.max(0, nextFrame - performance.now()))
  }

  assert.ok(maximumShapeDelta < 1e-3, `${chord.label} shape jitter ${maximumShapeDelta}`)
  assert.ok(maximumCycleError < 1e-3, `${chord.label} cycle error ${maximumCycleError}`)
  assert.ok(maximumAmplitudeError < 1e-3, `${chord.label} amplitude jitter ${maximumAmplitudeError}`)
  assert.ok(maximumLockError < 1e-12, `${chord.label} lock error ${maximumLockError}`)
  return {
    chord: chord.label,
    frames: frameCount,
    maximum_frame_shape_delta: maximumShapeDelta,
    maximum_cycle_shape_error: maximumCycleError,
    maximum_carrier_amplitude_error: maximumAmplitudeError,
    maximum_relative_lock_error: maximumLockError,
  }
}

async function probe() {
  await control.call('load_patch_graph', scene.buildSceneGraph())
  await control.call('start_audio')
  for (const name of scene.PRIME_SLOTS) {
    const controlMeta = scene.CONTROLS.find((item) => item.slot === name)
    await control.call('set_param', { name, value: controlMeta.value })
  }
  const taps = (await control.call('list_scope_taps')).taps || []
  const tapSlots = new Map(taps.map((tap) => [tap.name, tap.slot]))
  const chords = []
  for (let chordIndex = 0; chordIndex < scene.CHORDS.length; chordIndex += 1) {
    const slots = scene.CHORDS[chordIndex].voices.map((_voice, voiceIndex) => (
      tapSlots.get(scene.scopeModeId(chordIndex, voiceIndex))
    ))
    chords.push(await probeChord(chordIndex, slots))
  }

  const capture = await control.call('qualification_capture_next_block')
  assert.ok(capture.samples.every((value) => value === 0), 'muted output was nonzero')
  const telemetry = await control.call('get_telemetry')
  const audio = await control.call('audio_status')
  assert.equal(audio.stats.underrunCount, 0)
  assert.equal(audio.stats.overrunCount, 0)
  return {
    pass: true,
    audio_output: 'muted',
    captured_nonzero_samples: capture.samples.filter((value) => value !== 0).length,
    chords,
    scope_preemptions: telemetry.scope_preemptions,
  }
}

function finish(error) {
  control.close(error || new Error('live scope probe complete'))
  scope.close(error || new Error('live scope probe complete'))
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
