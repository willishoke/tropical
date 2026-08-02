#!/usr/bin/env node
'use strict'

const { spawn, execFileSync } = require('node:child_process')
const { createHash } = require('node:crypto')
const { mkdirSync, createWriteStream, writeFileSync } = require('node:fs')
const os = require('node:os')
const { tmpdir } = os
const { basename, join, resolve } = require('node:path')
const { performance } = require('node:perf_hooks')

const scene = require('../scene.js')
const scopeProfile = require('../renderer/scope-profile.js')
const { LatestValueSender } = require('../renderer/latest-value-sender.js')
const { JsonLineRpcClient } = require('../rpc-client.js')
const { summarize } = require('./metrics.js')
const { assembleCaptureTimeline, writeMonoFloat64Pcm24 } = require('./wav.js')

const REPO = resolve(__dirname, '../..')
const DEFAULT_ENGINE = join(REPO, 'lean/.lake/build/bin/frontend')
const SCOPE_SAMPLES = scopeProfile.displaySamples
  + scopeProfile.searchSamples + scopeProfile.warmupSamples

function parseArgs(argv) {
  const options = {
    mode: 'smoke',
    durationSeconds: 90,
    engine: process.env.TROPICAL_ENGINE || DEFAULT_ENGINE,
    output: join(REPO, 'benchmarks/demo_release/data'),
    deviceQuantum: 128,
    renderQuantum: 512,
    exerciseFlow: true,
    muteOutput: false,
  }
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index]
    const next = () => argv[++index]
    if (arg === '--mode') options.mode = next()
    else if (arg === '--duration-seconds') options.durationSeconds = Number(next())
    else if (arg === '--engine') options.engine = resolve(next())
    else if (arg === '--output') options.output = resolve(next())
    else if (arg === '--device-quantum') options.deviceQuantum = Number(next())
    else if (arg === '--render-quantum') options.renderQuantum = Number(next())
    else if (arg === '--skip-flow-gesture') options.exerciseFlow = false
    else if (arg === '--mute-output') options.muteOutput = true
    else if (arg === '--help') options.help = true
    else throw new Error(`unknown argument: ${arg}`)
  }
  if (options.mode === 'adversarial' && !argv.includes('--duration-seconds')) {
    options.durationSeconds = 600
  }
  if (options.mode === 'normal' && !argv.includes('--duration-seconds')) {
    options.durationSeconds = 1800
  }
  return options
}

function usage() {
  return `usage: node playground/qualification/run.js [options]

  --mode smoke|adversarial|normal   workload (default smoke)
  --duration-seconds N             total measured duration (default 90)
  --device-quantum N               Bdev (default 128)
  --render-quantum N               Rgpu (default 512)
  --skip-flow-gesture              omit the pause/reverse/resume FLOW probe
  --mute-output                    run the device clock with graph output at zero
  --engine PATH                    frontend executable
  --output DIR                     evidence directory`
}

const sleep = (milliseconds) => new Promise((resolveSleep) => setTimeout(resolveSleep, milliseconds))
const sha256 = (value) => createHash('sha256').update(value).digest('hex')
const wallMs = () => performance.now()

function git(args) {
  try {
    return execFileSync('git', args, { cwd: REPO, encoding: 'utf8' }).trim()
  } catch { return null }
}

function command(executable, args = [], cwd = REPO) {
  try {
    return execFileSync(executable, args, {
      cwd,
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
    }).trim()
  } catch { return null }
}

function isoRunId() {
  return new Date().toISOString().replace(/[:.]/g, '-').replace('T', '_').replace('Z', '')
}

async function main() {
  const options = parseArgs(process.argv.slice(2))
  if (options.help) {
    process.stdout.write(`${usage()}\n`)
    return
  }
  if (!Number.isFinite(options.durationSeconds) || options.durationSeconds <= 0) {
    throw new Error('duration must be positive')
  }
  if (options.renderQuantum % options.deviceQuantum !== 0) {
    throw new Error('Rgpu must be a multiple of Bdev')
  }

  mkdirSync(options.output, { recursive: true })
  const runId = `${isoRunId()}-${options.mode}-b${options.deviceQuantum}-r${options.renderQuantum}`
  const prefix = join(options.output, runId)
  const eventPath = `${prefix}.jsonl`
  const summaryPath = `${prefix}.summary.json`
  const manifestPath = `${prefix}.manifest.json`
  const capturePath = `${prefix}.capture.wav`
  const events = createWriteStream(eventPath, { flags: 'wx' })
  const event = (kind, data = {}) => {
    events.write(`${JSON.stringify({ schema: 1, kind, wall_time: new Date().toISOString(), ...data })}\n`)
  }

  const graph = scene.buildSceneGraph()
  const graphJson = JSON.stringify(graph)
  const socketPath = join(tmpdir(), `tropical-qualification-${process.pid}.sock`)
  const engineStderr = []
  const engine = spawn(options.engine, ['--serve', socketPath], {
    cwd: REPO,
    env: {
      ...process.env,
      TROPICAL_BACKEND: 'metal',
      TROPICAL_BUFFER_LENGTH: String(options.deviceQuantum),
      TROPICAL_METAL_RENDER_TILE_FRAMES: String(options.renderQuantum),
      TROPICAL_QUALIFICATION: '1',
    },
    stdio: ['ignore', 'ignore', 'pipe'],
  })
  engine.stderr.on('data', (chunk) => {
    const text = chunk.toString()
    engineStderr.push(text)
    event('engine_stderr', { text })
  })

  const control = new JsonLineRpcClient({ path: socketPath })
  const scope = new JsonLineRpcClient({ path: socketPath })
  const diagnostics = new JsonLineRpcClient({ path: socketPath })
  control.start()
  scope.start()
  diagnostics.start()
  const writes = []
  const scopes = []
  const captures = []
  let scopeRunning = false
  let engineExited = false
  engine.once('exit', (code, signal) => {
    engineExited = true
    event('engine_exit', { code, signal })
    control.close(new Error(`engine exited ${code ?? signal}`))
    scope.close(new Error(`engine exited ${code ?? signal}`))
    diagnostics.close(new Error(`engine exited ${code ?? signal}`))
  })

  const stopEngine = () => {
    control.close()
    scope.close()
    diagnostics.close()
    try { engine.kill('SIGTERM') } catch {}
  }
  for (const signal of ['SIGINT', 'SIGTERM', 'SIGHUP']) {
    process.once(signal, () => { stopEngine(); process.exitCode = 130 })
  }

  const callTimed = async (client, method, params = {}) => {
    const begin = wallMs()
    const result = await client.call(method, params)
    return { result, begin, end: wallMs() }
  }

  const sendParam = async (_key, update) => {
    const dispatchStartMs = wallMs()
    const response = await control.call('set_param', {
      name: update.name,
      value: update.value,
    })
    const publishedMs = wallMs()
    let audibleMs = null
    const audibleDeadline = publishedMs + 1000
    while (wallMs() < audibleDeadline) {
      const telemetry = await diagnostics.call('get_telemetry')
      if (Number(telemetry?.metal?.acknowledged_epoch || 0)
          >= Number(response.epoch_id || 0)) {
        audibleMs = wallMs()
        break
      }
      await sleep(1)
    }
    if (audibleMs === null) {
      throw new Error(`epoch ${response.epoch_id} was not audibly acknowledged`)
    }
    const record = {
      gestureId: update.gestureId,
      generatedMs: update.generatedMs,
      dispatchStartMs,
      dispatchEndMs: publishedMs,
      audibleMs,
      name: update.name,
      value: update.value,
      phase: update.phase,
      dispatched: true,
      epoch_id: response.epoch_id,
      device_activation_frame: response.device_activation_frame,
      activation: { ...response.activation, audible: true },
    }
    writes.push(record)
    event('param_dispatch', record)
  }
  const senderErrors = []
  const sender = new LatestValueSender(
    sendParam,
    (error, key, update) => {
      senderErrors.push({ error: error.message, key, update })
      event('param_error', { error: error.message, key, update })
    },
    (update) => update.value,
  )
  const gestureExpectations = []
  let gestureSequence = 0
  let lastControlConvergedMs = null

  async function gesture(name, values, cadenceMs, kind = 'sweep') {
    if (!values.length) throw new Error('gesture must contain a value')
    const gestureId = `${name}:${++gestureSequence}`
    const generated = []
    for (let index = 0; index < values.length; index += 1) {
      const update = {
        gestureId,
        generatedMs: wallMs(),
        name,
        value: values[index],
        phase: index === 0 ? 'first' : index === values.length - 1 ? 'final' : 'interior',
      }
      generated.push(update)
      writes.push({ ...update, dispatched: false })
      sender.submit(name, update)
      if (index + 1 < values.length) await sleep(cadenceMs)
    }
    await sender.whenIdle(name)
    lastControlConvergedMs = wallMs()
    let turningValue = null
    for (let index = 1; index + 1 < values.length; index += 1) {
      const before = Math.sign(values[index] - values[index - 1])
      const after = Math.sign(values[index + 1] - values[index])
      if (before && after && before !== after) {
        turningValue = values[index]
        break
      }
    }
    gestureExpectations.push({
      gestureId,
      name,
      kind,
      firstValue: values[0],
      finalValue: values.at(-1),
      turningValue,
      firstGeneratedMs: generated[0].generatedMs,
      finalGeneratedMs: generated.at(-1).generatedMs,
    })
    return generated
  }

  function logSweep(low, high, count) {
    return Array.from({ length: count }, (_, index) => (
      low * Math.pow(high / low, index / Math.max(1, count - 1))
    ))
  }

  async function captureBurst(blockCount) {
    for (let index = 0; index < blockCount; index += 1) {
      try {
        const block = await control.call('qualification_capture_next_block')
        captures.push(block)
        event('capture_block', block)
      } catch (error) {
        event('capture_error', { message: error.message })
        break
      }
    }
  }

  async function scopeLoop(tapSlots) {
    const periodMs = 1000 / scopeProfile.fps
    let nextFrame = wallMs()
    while (scopeRunning) {
      const frameStart = wallMs()
      const frameBeganIdle = !sender.isBusy()
      try {
        const position = (await scope.call('playback_position')).position
        const response = await scope.call('render_window', {
          start: Math.max(0, position - SCOPE_SAMPLES),
          count: SCOPE_SAMPLES,
          point_budget: scopeProfile.pointBudget,
          slots: tapSlots,
        })
        const record = {
          wallMs: frameStart,
          durationMs: wallMs() - frameStart,
          idle: frameBeganIdle && !sender.isBusy(),
          preempted: Boolean(response.preempted),
          stride: response.stride,
          count: response.count,
        }
        scopes.push(record)
        event('scope_frame', record)
      } catch (error) {
        const record = { wallMs: frameStart, durationMs: wallMs() - frameStart, error: error.message }
        scopes.push(record)
        event('scope_frame', record)
      }
      nextFrame += periodMs
      await sleep(Math.max(0, nextFrame - wallMs()))
    }
  }

  let finalTelemetry = null
  let finalAudioStatus = null
  let summary = null
  let failure = null
  const startedMs = wallMs()
  try {
    event('run_start', { runId, options, graph_sha256: sha256(graphJson) })
    const loaded = await callTimed(control, 'load_patch_graph', graph)
    event('graph_loaded', { duration_ms: loaded.end - loaded.begin, report: loaded.result })
    const audio = await callTimed(control, 'start_audio')
    event('audio_started', { duration_ms: audio.end - audio.begin, response: audio.result })

    // Prime exactly the coefficient families the scene declares, while its
    // final level remains boot-muted.
    for (const name of scene.PRIME_SLOTS) {
      const controlMeta = scene.CONTROLS.find((item) => item.slot === name)
      event('prime_start', { name, value: controlMeta.value })
      await control.call('set_param', { name, value: controlMeta.value })
      event('prime_end', { name, value: controlMeta.value })
    }
    const position = (await control.call('playback_position')).position
    await control.call('set_param', {
      name: 'master.tau_base',
      value: -position / scene.SAMPLE_RATE,
    })
    const level = scene.CONTROLS.find((item) => item.slot === 'levelControl.value')
    await control.call('set_param', {
      name: level.slot,
      value: options.muteOutput ? 0 : level.value,
    })
    const taps = await control.call('list_scope_taps')
    const tapMap = new Map(taps.taps.map((tap) => [tap.name, tap.slot]))
    const qualificationTaps = scene.CHORDS[0].voices.map((_voice, voiceIndex) => (
      scene.scopeModeId(0, voiceIndex)
    ))
    const tapSlots = qualificationTaps.map((name) => {
      if (!tapMap.has(name)) throw new Error(`missing scope tap: ${name}`)
      return tapMap.get(name)
    })

    scopeRunning = true
    const scopePromise = scopeLoop(tapSlots)
    await sleep(250)
    const measuredStartMs = wallMs()

    const capturePromise = captureBurst(32)
    await gesture('veilControl.value', logSweep(120, 5200, 168), 12)
    await gesture('veilControl.value', [5200, 4200, 2600, 900, 260, 780], 12)
    await gesture(
      'veilControl.value',
      [780, 1400, 3000, 5000, 4200, 2400, 780],
      8,
      'reversal',
    )
    await gesture('edgeControl.value', Array.from({ length: 64 }, (_, i) => 0.1 + 0.85 * i / 63), 12)
    await gesture('spaceLevel.value', [0, 0.2, 0.45, 0.75, 1, 1.3, 1], 16)
    await gesture(
      'positionControl.value',
      [1, 0.5, 0, -0.5, -1, -0.25, 0.5, 1],
      16,
      'reversal',
    )
    if (options.exerciseFlow) {
      event('flow_gesture_start', { values: [0, -1, 1], hold_ms: [120, 160] })
      const stopped = await control.call('set_param', {
        name: 'master.velocity',
        value: 0,
      })
      event('flow_gesture_activation', { value: 0, response: stopped })
      await sleep(120)
      const reversed = await control.call('set_param', {
        name: 'master.velocity',
        value: -1,
      })
      event('flow_gesture_activation', { value: -1, response: reversed })
      await sleep(160)
      const resumed = await control.call('set_param', {
        name: 'master.velocity',
        value: 1,
      })
      event('flow_gesture_activation', { value: 1, response: resumed })
      event('flow_gesture_end', {})
    }
    lastControlConvergedMs = wallMs()
    await capturePromise

    const measuredDeadline = measuredStartMs + options.durationSeconds * 1000
    let direction = 1
    while (wallMs() < measuredDeadline) {
      const values = direction > 0
        ? logSweep(160, 4800, 48)
        : logSweep(4800, 160, 48)
      await gesture('veilControl.value', values, options.mode === 'normal' ? 28 : 12)
      direction *= -1
      const telemetry = await control.call('get_telemetry')
      const status = await control.call('audio_status')
      event('health', { telemetry, audio_status: status })
      await sleep(options.mode === 'normal' ? 1200 : 100)
    }
    // Give the live scope loop its complete contractual resume window even
    // when a deliberately short diagnostic ends immediately after a control.
    if (lastControlConvergedMs !== null) {
      const resumeDeadline = lastControlConvergedMs + 150
      while (wallMs() < resumeDeadline && !scopes.some((record) => (
        record.wallMs >= lastControlConvergedMs
        && !record.preempted
        && !record.error
      ))) await sleep(5)
    }
    scopeRunning = false
    await scopePromise
    finalTelemetry = await scope.call('get_telemetry')
    finalAudioStatus = await scope.call('audio_status')
    summary = summarize({
      writes,
      scopes,
      captures,
      telemetry: finalTelemetry,
      audioStatus: finalAudioStatus,
      expectSilence: options.muteOutput,
    })
    summary.run_id = runId
    summary.mode = options.mode
    summary.duration_seconds = (wallMs() - startedMs) / 1000
    summary.measured_duration_seconds = (wallMs() - measuredStartMs) / 1000
    summary.sender_errors = senderErrors
    summary.final_telemetry = finalTelemetry
    summary.final_audio_status = finalAudioStatus

    const dispatched = writes.filter((record) => record.dispatched)
    const gestureDelivery = gestureExpectations.map((expected) => {
      const delivered = dispatched.filter(
        (record) => record.gestureId === expected.gestureId,
      )
      const first = delivered[0]
      const final = delivered.at(-1)
      return {
        ...expected,
        dispatched_count: delivered.length,
        first_delivered: first?.value === expected.firstValue,
        final_delivered: final?.value === expected.finalValue,
        turning_delivered: expected.turningValue === null
          || delivered.some((record) => record.value === expected.turningValue),
        final_scheduled_ms: final
          ? final.dispatchStartMs - expected.finalGeneratedMs
          : null,
        final_audible_ms: final?.audibleMs
          ? final.audibleMs - expected.finalGeneratedMs
          : null,
      }
    })
    summary.gesture_delivery = gestureDelivery
    const lastCutoff = dispatched.filter((record) => record.name === 'veilControl.value').at(-1)
    summary.final_cutoff = lastCutoff?.value ?? null
    const resumed = scopes.find((record) => (
      lastControlConvergedMs !== null
      && record.wallMs >= lastControlConvergedMs
      && !record.preempted
      && !record.error
    ))
    summary.scope_resume_latency_ms = resumed
      ? resumed.wallMs - lastControlConvergedMs
      : null
    const within = (value, limit) => Number.isFinite(value) && value <= limit
    const cutoffDelivery = gestureDelivery.filter(
      (delivery) => delivery.name === 'veilControl.value',
    )
    const reversalDelivery = gestureDelivery.filter(
      (delivery) => delivery.kind === 'reversal',
    )
    const gates = {
      no_faults: summary.faults.length === 0 && senderErrors.length === 0,
      control_p95: within(summary.scheduled_write_latency_ms.p95, 25),
      control_p99: within(summary.scheduled_write_latency_ms.p99, 35),
      audible_p95: within(summary.audible_activation_latency_ms.p95, 50),
      audible_max: within(summary.audible_activation_latency_ms.max, 75),
      scope_p99: within(summary.scope_rpc_ms.p99, 20),
      scope_interval_p95: within(
        summary.scope_frame_interval_ms.p95,
        (1000 / scopeProfile.fps) * 1.25,
      ),
      scope_average_fps: Number.isFinite(summary.scope_idle_fps)
        && summary.scope_idle_fps >= scopeProfile.fps * 0.95,
      scope_resume: within(summary.scope_resume_latency_ms, 150),
      first_final_delivery: cutoffDelivery.length > 0
        && cutoffDelivery.every((delivery) => (
          delivery.first_delivered && delivery.final_delivered
        )),
      reversal_delivery: reversalDelivery.length > 0
        && reversalDelivery.every((delivery) => delivery.turning_delivered),
      final_scheduled: cutoffDelivery.length > 0
        && cutoffDelivery.every((delivery) => within(delivery.final_scheduled_ms, 35)),
      final_audible: cutoffDelivery.length > 0
        && cutoffDelivery.every((delivery) => within(delivery.final_audible_ms, 75)),
      final_value: Number.isFinite(summary.final_cutoff)
        && summary.final_cutoff === cutoffDelivery.at(-1)?.finalValue,
      engine_alive: !engineExited,
    }
    summary.gates = gates
    summary.pass = Object.values(gates).every(Boolean)

    const timeline = assembleCaptureTimeline(captures)
    summary.capture_timeline = {
      start_sample_index: timeline.start,
      sample_count: timeline.samples.length,
      gap_samples: timeline.gapSamples,
      overlap_samples: timeline.overlapSamples,
    }
    if (!options.muteOutput && timeline.samples.length
      && !summary.capture_faults.nonfinite_sample_count
      && !summary.capture_faults.clamped_sample_count) {
      try {
        summary.capture_wav = writeMonoFloat64Pcm24(capturePath, timeline.samples, scene.SAMPLE_RATE)
      } catch (error) {
        summary.capture_wav_error = error.message
      }
    }
    if (!summary.pass) failure = new Error(`qualification gates failed: ${JSON.stringify(gates)}`)
  } catch (error) {
    failure = error
    scopeRunning = false
    event('run_error', { message: error.message, stack: error.stack })
  } finally {
    if (!finalTelemetry && !engineExited) {
      // A timed-out data-plane call occupies only its originating socket
      // reader. Diagnose it through the independent scope connection.
      try { finalTelemetry = await scope.call('get_telemetry') } catch {}
    }
    if (!finalAudioStatus && !engineExited) {
      try { finalAudioStatus = await scope.call('audio_status') } catch {}
    }
    const manifest = {
      schema: 1,
      run_id: runId,
      commit: git(['rev-parse', 'HEAD']),
      worktree_status: git(['status', '--short']),
      graph_sha256: sha256(graphJson),
      scene_file: 'playground/scene.js',
      scope_profile: scopeProfile,
      options,
      platform: process.platform,
      arch: process.arch,
      machine: {
        hostname: os.hostname(),
        os_release: os.release(),
        os_version: os.version(),
        cpu: os.cpus()[0]?.model ?? null,
        cpu_count: os.cpus().length,
        memory_bytes: os.totalmem(),
      },
      toolchain: {
        macos: command('sw_vers', ['-productVersion']),
        clang: command('clang', ['--version'])?.split('\n')[0] ?? null,
        cmake: command('cmake', ['--version'])?.split('\n')[0] ?? null,
        lean: command(
          join(process.env.HOME || '', '.elan/bin/lake'),
          ['env', 'lean', '--version'],
          join(REPO, 'lean'),
        ),
        lake: command(
          join(process.env.HOME || '', '.elan/bin/lake'),
          ['--version'],
          join(REPO, 'lean'),
        ),
      },
      node: process.version,
      engine: basename(options.engine),
      audio_device: finalAudioStatus?.device ?? null,
      negotiated_buffer_length: finalTelemetry?.buffer_length ?? null,
      sample_rate: scene.SAMPLE_RATE,
      engine_stderr_sha256: sha256(engineStderr.join('')),
    }
    writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`)
    writeFileSync(summaryPath, `${JSON.stringify(summary || {
      pass: false,
      error: failure?.message,
      final_telemetry: finalTelemetry,
      final_audio_status: finalAudioStatus,
    }, null, 2)}\n`)
    event('run_end', { pass: Boolean(summary?.pass), error: failure?.message })
    await new Promise((resolveEnd) => events.end(resolveEnd))
    stopEngine()
  }
  process.stdout.write(`${summaryPath}\n`)
  if (failure) throw failure
}

main().catch((error) => {
  process.stderr.write(`${error.stack || error}\n`)
  process.exitCode = 1
})
