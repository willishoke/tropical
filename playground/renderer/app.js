// A deliberately fixed instrument: twenty exact-ratio strings unfold as one
// scene. The UI does not pretend to be a patcher or a sequencer. Its sixteen
// moments are places in one closed-form signal, and seeking is a clock rebase.
'use strict'

const rpc = (method, params = {}) => window.tropical.call(method, params)
const scene = window.ModalScene
const GRAPH = scene.buildSceneGraph()
const CONTROLS = scene.CONTROLS.map((control) => ({ ...control }))
const SCOPE_PROFILE = window.ModalScopeProfile

const SCOPES = [
  {
    tap: 'veil',
    label: 'STRINGS / RESONANT VEIL',
    color: '--strings',
    trigger: true,
  },
  {
    tap: 'room',
    label: 'ROOM / METAL RETURN',
    color: '--room',
    trigger: false,
  },
]

const DISPLAY_SAMPLES = SCOPE_PROFILE.displaySamples
const SEARCH_SAMPLES = SCOPE_PROFILE.searchSamples
const WARMUP_SAMPLES = SCOPE_PROFILE.warmupSamples
const SCOPE_POINT_BUDGET = SCOPE_PROFILE.pointBudget
const SCOPE_FPS = SCOPE_PROFILE.fps
const SLIDER_RESOLUTION = 1000

const clamp = (value, low, high) => Math.max(low, Math.min(high, value))
const controlBySlot = new Map(CONTROLS.map((control) => [control.slot, control]))

const transport = {
  velocity: controlBySlot.get('master.velocity').value,
  tauBase: 0,
  lastNonzeroVelocity: 1,
  playbackPosition: 0,
  sceneTime: 0,
  rebasing: false,
}

let selectedControl = 0
const committedValues = new Map(
  CONTROLS.map((control) => [control.slot, control.value]),
)
let loopStarted = false
const wells = []
const stepElements = []
const chordElements = []
const scopeArbiter = new window.ScopeArbiter({
  quietMs: SCOPE_PROFILE.resumeQuietMs,
})

function normalizedValue(control) {
  if (control.log) {
    return Math.log(control.value / control.min)
      / Math.log(control.max / control.min)
  }
  return (control.value - control.min) / (control.max - control.min)
}

function valueFromNormalized(control, normalized) {
  if (control.log) {
    return control.min * Math.pow(control.max / control.min, normalized)
  }
  return control.min + (control.max - control.min) * normalized
}

function quantize(control, value) {
  if (!control.step) return value
  const steps = Math.round((value - control.min) / control.step)
  return control.min + steps * control.step
}

function formatValue(control) {
  const value = Math.abs(control.value) < 5e-7 ? 0 : control.value
  let digits = 2
  if (control.slot === 'veilControl.value') digits = 0
  else if (Math.abs(value) < 1) digits = 2
  else if (Math.abs(value) >= 10) digits = 1
  const number = value.toFixed(digits)
  return control.unit ? `${number} ${control.unit}` : number
}

function setSelectedControl(index, focus = false) {
  selectedControl = (index + CONTROLS.length) % CONTROLS.length
  document.querySelectorAll('.control').forEach((row, rowIndex) => {
    row.classList.toggle('selected', rowIndex === selectedControl)
  })
  if (focus) {
    document.querySelector(`.control[data-index="${selectedControl}"] input`)?.focus()
  }
}

function updateControlElement(index) {
  const control = CONTROLS[index]
  const row = document.querySelector(`.control[data-index="${index}"]`)
  if (!row) return
  const slider = row.querySelector('input')
  const normalized = clamp(normalizedValue(control), 0, 1)
  slider.value = String(Math.round(normalized * SLIDER_RESOLUTION))
  slider.style.setProperty('--fill', `${(normalized * 100).toFixed(2)}%`)
  row.querySelector('.control-value').textContent = formatValue(control)
}

async function commitParamWrite(_key, update) {
  const { control, targetValue } = update
  if (control.transport) {
    const response = await rpc('set_param', {
      name: control.slot,
      value: targetValue,
    })
    const effective = (
      response.effective_sample_index
      ?? response.observed_sample_index
      ?? transport.playbackPosition
    )
    transport.tauBase = response.tau_base ?? (
      transport.tauBase
      + (transport.velocity - targetValue) * effective / scene.SAMPLE_RATE
    )
    transport.velocity = targetValue
    if (Math.abs(targetValue) > 1e-6) {
      transport.lastNonzeroVelocity = targetValue
    }
    renderTransport()
  } else {
    const slots = control.slots || [control.slot]
    for (const slot of slots) {
      await rpc('set_param', { name: slot, value: targetValue })
    }
  }
  committedValues.set(control.slot, targetValue)
}

function handleParamWriteError(error, _key, update) {
  const { control, targetValue } = update
  if (control.value === targetValue) {
    control.value = committedValues.get(control.slot) ?? control.value
    updateControlElement(CONTROLS.indexOf(control))
  }
  showFault(error)
}

const paramSender = new window.LatestValueSender(
  commitParamWrite,
  handleParamWriteError,
  (update) => update.targetValue,
  (busy) => scopeArbiter.setSenderBusy(busy),
)

function enqueueParamWrite(control, nextValue) {
  const targetValue = clamp(quantize(control, nextValue), control.min, control.max)
  control.value = targetValue
  updateControlElement(CONTROLS.indexOf(control))
  paramSender.submit(control.slot, { control, targetValue })
  scopeArbiter.setSenderBusy(paramSender.isBusy())
}

function nudgeControl(control, direction, fine) {
  let next
  if (control.log) {
    const ratio = Math.pow(
      control.max / control.min,
      direction * (fine ? 0.003 : 0.018),
    )
    next = control.value * ratio
  } else {
    const step = control.step
      || (control.max - control.min) * (fine ? 0.002 : 0.012)
    next = control.value + direction * step * (fine ? 0.2 : 1)
  }
  enqueueParamWrite(control, next)
}

function buildControls() {
  const root = document.getElementById('controls')
  root.innerHTML = ''

  CONTROLS.forEach((control, index) => {
    const row = document.createElement('div')
    row.className = `control${control.transport ? ' transport-control' : ''}`
    row.dataset.index = String(index)
    row.innerHTML = `
      <span class="control-label">${control.label}</span>
      <input type="range" min="0" max="${SLIDER_RESOLUTION}" step="1"
        aria-label="${control.label}">
      <span class="control-value"></span>
      <span class="control-help">${control.help}</span>
    `
    const slider = row.querySelector('input')
    slider.addEventListener('pointerdown', (event) => {
      setSelectedControl(index)
      scopeArbiter.begin(`pointer:${event.pointerId}`)
      try { slider.setPointerCapture(event.pointerId) } catch {}
    })
    slider.addEventListener('lostpointercapture', (event) => {
      scopeArbiter.end(`pointer:${event.pointerId}`)
    })
    slider.addEventListener('focus', () => setSelectedControl(index))
    slider.addEventListener('input', () => {
      const normalized = Number(slider.value) / SLIDER_RESOLUTION
      enqueueParamWrite(control, valueFromNormalized(control, normalized))
    })
    row.addEventListener('click', (event) => {
      setSelectedControl(index)
      if (event.target !== slider) slider.focus()
    })
    root.appendChild(row)
    updateControlElement(index)
  })

  setSelectedControl(0)
}

function buildSequence() {
  const steps = document.getElementById('steps')
  const chords = document.getElementById('chord-strip')

  for (let index = 0; index < scene.PATTERN_STEPS; index += 1) {
    const chord = scene.CHORDS.find((candidate) => candidate.startStep === index)
    const button = document.createElement('button')
    button.type = 'button'
    button.className = `step${chord ? ' downbeat' : ''}`
    button.title = chord
      ? `${chord.label} · ${scene.ratioLabel(chord)}`
      : `moment ${String(index + 1).padStart(2, '0')}`
    button.innerHTML = `<span class="ordinal">${
      chord ? chord.label : String(index + 1).padStart(2, '0')
    }</span>`
    button.addEventListener('click', () => seekScene(index * scene.STEP_SECONDS))
    steps.appendChild(button)
    stepElements.push(button)
  }

  scene.CHORDS.forEach((chord) => {
    const button = document.createElement('button')
    button.type = 'button'
    button.className = 'chord'
    button.innerHTML = `
      <span class="chord-name">${chord.label} / ${chord.name}</span>
      <span class="chord-ratios">${scene.ratioLabel(chord)}</span>
    `
    button.title = `seek ${chord.startStep * scene.STEP_SECONDS} s`
    button.addEventListener('click', () => {
      seekScene(chord.startStep * scene.STEP_SECONDS)
    })
    chords.appendChild(button)
    chordElements.push(button)
  })
}

function buildScopes() {
  const root = document.getElementById('scopes')
  SCOPES.forEach((scope) => {
    const well = document.createElement('article')
    well.className = 'scope'
    well.innerHTML = `
      <div class="scope-head">
        <span>${scope.label}</span>
        <span class="scope-meta">scope.${scope.tap} · waiting</span>
      </div>
    `
    const canvas = document.createElement('canvas')
    well.appendChild(canvas)
    root.appendChild(well)
    wells.push({
      ...scope,
      canvas,
      meta: well.querySelector('.scope-meta'),
      slot: null,
      peak: 1e-6,
    })
  })
}

function findTrigger(values, searchLength, hysteresis = 0.025) {
  const limit = Math.min(searchLength, values.length)
  let armed = (values[0] ?? 0) < -hysteresis
  for (let index = 1; index < limit; index += 1) {
    if (values[index] < -hysteresis) armed = true
    else if (armed && values[index - 1] < 0 && values[index] >= 0) return index
  }
  return 0
}

function resizeCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1
  const width = Math.max(1, Math.round(canvas.clientWidth * dpr))
  const height = Math.max(1, Math.round(canvas.clientHeight * dpr))
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width
    canvas.height = height
  }
  return { width, height, dpr }
}

function drawTrace(well, values) {
  const { canvas } = well
  const { width, height, dpr } = resizeCanvas(canvas)
  const context = canvas.getContext('2d')
  const palette = getComputedStyle(document.documentElement)
  const ground = palette.getPropertyValue('--surface')
  const grid = palette.getPropertyValue('--rule')
  const trace = palette.getPropertyValue(well.color)

  context.fillStyle = ground
  context.fillRect(0, 0, width, height)
  context.strokeStyle = grid
  context.lineWidth = 1

  context.beginPath()
  context.moveTo(0, Math.round(height / 2) + 0.5)
  context.lineTo(width, Math.round(height / 2) + 0.5)
  context.stroke()

  for (let division = 1; division < 8; division += 1) {
    const x = Math.round(width * division / 8) + 0.5
    context.beginPath()
    context.moveTo(x, 0)
    context.lineTo(x, height)
    context.stroke()
  }

  if (!values || values.length < 2) {
    well.meta.textContent = `scope.${well.tap} · no signal`
    return
  }

  let framePeak = 1e-7
  values.forEach((value) => { framePeak = Math.max(framePeak, Math.abs(value)) })
  well.peak = Math.max(framePeak, well.peak * 0.91)
  const scale = Math.max(framePeak, well.peak * 0.72, 1e-7)

  context.strokeStyle = trace
  context.lineWidth = Math.max(1, dpr)
  context.beginPath()
  values.forEach((value, index) => {
    const x = index * width / (values.length - 1)
    const y = height / 2 - value / (scale * 1.12) * (height / 2 - 5 * dpr)
    if (index === 0) context.moveTo(x, y)
    else context.lineTo(x, y)
  })
  context.stroke()

  const milliseconds = values.length / scene.SAMPLE_RATE * 1000
  well.meta.textContent = (
    `scope.${well.tap} · ${milliseconds.toFixed(1)} ms · auto ±${formatPeak(framePeak)}`
  )
}

function formatPeak(peak) {
  if (peak >= 0.1) return peak.toFixed(2)
  if (peak >= 0.001) return peak.toFixed(3)
  return peak.toExponential(1)
}

function renderSequence(time) {
  const inPhrase = time >= 0 && time < scene.PATTERN_SECONDS
  const activeStep = inPhrase
    ? clamp(Math.floor(time / scene.STEP_SECONDS), 0, scene.PATTERN_STEPS - 1)
    : -1
  stepElements.forEach((element, index) => {
    element.classList.toggle('active', index === activeStep)
  })

  let activeChord = -1
  scene.CHORDS.forEach((chord, index) => {
    const next = scene.CHORDS[index + 1]
    const chordStart = chord.startStep * scene.STEP_SECONDS
    const chordEnd = next
      ? next.startStep * scene.STEP_SECONDS
      : scene.PATTERN_SECONDS
    if (time >= chordStart && time < chordEnd) activeChord = index
  })
  chordElements.forEach((element, index) => {
    element.classList.toggle('active', index === activeChord)
  })

  document.getElementById('tail-state').textContent = inPhrase
    ? `moment ${String(activeStep + 1).padStart(2, '0')} / 16`
    : `returning / ${scene.SCENE_SECONDS} s`
}

function renderTransport() {
  const velocity = transport.velocity
  const play = document.getElementById('play-button')
  if (Math.abs(velocity) < 1e-6) play.textContent = '▶ FLOW'
  else if (velocity < 0) play.textContent = '◀ REVERSE'
  else play.textContent = 'Ⅱ HOLD'

  const time = clamp(transport.sceneTime, 0, scene.SCENE_SECONDS)
  document.getElementById('clock').innerHTML = (
    `<span class="dim">τ</span> ${time.toFixed(3).padStart(6, '0')}`
    + ` <span class="dim">/ ${scene.SCENE_SECONDS.toFixed(3)}</span>`
  )
}

async function seekScene(targetSeconds) {
  if (transport.rebasing) return
  transport.rebasing = true
  scopeArbiter.begin('rebase')
  try {
    const position = (await rpc('playback_position')).position
    const target = clamp(targetSeconds, 0, scene.SCENE_SECONDS - 1 / scene.SAMPLE_RATE)
    const base = target - transport.velocity * position / scene.SAMPLE_RATE
    await rpc('set_param', { name: 'master.tau_base', value: base })
    transport.playbackPosition = position
    transport.tauBase = base
    transport.sceneTime = target
    renderSequence(target)
    renderTransport()
  } catch (error) {
    showFault(error)
  } finally {
    transport.rebasing = false
    scopeArbiter.end('rebase')
  }
}

function toggleFlow() {
  const control = controlBySlot.get('master.velocity')
  const next = Math.abs(control.value) < 1e-6
    ? transport.lastNonzeroVelocity
    : 0
  enqueueParamWrite(control, next)
}

function reverseFlow() {
  const control = controlBySlot.get('master.velocity')
  const magnitude = Math.max(Math.abs(control.value), Math.abs(transport.lastNonzeroVelocity), 1)
  enqueueParamWrite(control, control.value < 0 ? magnitude : -magnitude)
}

function installInteraction() {
  document.getElementById('restart-button').addEventListener('click', () => seekScene(0))
  document.getElementById('reverse-button').addEventListener('click', reverseFlow)
  document.getElementById('play-button').addEventListener('click', toggleFlow)

  window.addEventListener('keydown', (event) => {
    const control = CONTROLS[selectedControl]
    if (event.key === 'ArrowUp') setSelectedControl(selectedControl - 1, true)
    else if (event.key === 'ArrowDown') setSelectedControl(selectedControl + 1, true)
    else if (event.key === 'ArrowLeft') {
      scopeArbiter.begin(`key:${event.code}`)
      nudgeControl(control, -1, event.shiftKey)
    } else if (event.key === 'ArrowRight') {
      scopeArbiter.begin(`key:${event.code}`)
      nudgeControl(control, 1, event.shiftKey)
    }
    else if (event.key === ' ') toggleFlow()
    else if (event.key.toLowerCase() === 'r') reverseFlow()
    else if (event.key === 'Enter') seekScene(0)
    else return
    event.preventDefault()
  })
  window.addEventListener('keyup', (event) => {
    scopeArbiter.end(`key:${event.code}`)
  })
  for (const eventName of ['pointerup', 'pointercancel']) {
    window.addEventListener(eventName, (event) => {
      scopeArbiter.end(`pointer:${event.pointerId}`)
    })
  }
  window.addEventListener('blur', () => scopeArbiter.clearTransient())
}

function showFault(error) {
  scopeArbiter.fault()
  const status = document.getElementById('status')
  status.classList.add('fault')
  status.textContent = `fault / ${error?.message || String(error)}`
}

async function renderFrame() {
  try {
    const position = (await rpc('playback_position')).position
    transport.playbackPosition = position
    transport.sceneTime = (
      transport.tauBase
      + transport.velocity * position / scene.SAMPLE_RATE
    )

    if (!transport.rebasing && transport.velocity > 0
      && transport.sceneTime >= scene.SCENE_SECONDS) {
      await seekScene(0)
    } else if (!transport.rebasing && transport.velocity < 0
      && transport.sceneTime <= 0) {
      await seekScene(scene.SCENE_SECONDS - 1 / scene.SAMPLE_RATE)
    }

    renderSequence(transport.sceneTime)
    renderTransport()

    scopeArbiter.setSenderBusy(paramSender.isBusy())
    if (scopeArbiter.isHeld()) {
      window.setTimeout(renderFrame, 1000 / SCOPE_FPS)
      return
    }

    const live = wells.filter((well) => well.slot !== null)
    const count = DISPLAY_SAMPLES + SEARCH_SAMPLES + WARMUP_SAMPLES
    const start = Math.max(0, position - count)
    const response = await rpc('render_window', {
      start,
      count,
      point_budget: SCOPE_POINT_BUDGET,
      slots: live.map((well) => well.slot),
    })

    scopeArbiter.setSenderBusy(paramSender.isBusy())
    if (response.preempted || scopeArbiter.isHeld()) {
      window.setTimeout(renderFrame, 1000 / SCOPE_FPS)
      return
    }

    const stride = response.stride ?? 1
    const warmupPoints = Math.ceil(WARMUP_SAMPLES / stride)
    const searchPoints = Math.ceil(SEARCH_SAMPLES / stride)
    const displayPoints = Math.ceil(DISPLAY_SAMPLES / stride)

    live.forEach((well, resultIndex) => {
      const values = (response.values[resultIndex] || []).slice(warmupPoints)
      const offset = well.trigger
        ? findTrigger(values, searchPoints)
        : Math.min(searchPoints, Math.max(0, values.length - displayPoints))
      drawTrace(well, values.slice(offset, offset + displayPoints))
    })
  } catch {
    // Compilation/hot-swap transitions can make one random-access read stale.
    // The next frame is authoritative.
  }
  window.setTimeout(renderFrame, 1000 / SCOPE_FPS)
}

async function boot() {
  buildSequence()
  buildScopes()
  buildControls()
  installInteraction()

  const status = document.getElementById('status')
  try {
    status.textContent = 'compiling / twenty strings + four metal returns'
    await rpc('load_patch_graph', GRAPH)

    let audio = 'audio live'
    let audioStarted = false
    try {
      await rpc('start_audio')
      audioStarted = true
    } catch {
      audio = 'no audio device / scopes live'
    }

    if (audioStarted) {
      status.textContent = 'priming / modal controls'
      // First-use coefficient pages and Metal command paths can add tens of
      // milliseconds even though steady-state activations are short. Exercise
      // the three coefficient families at their no-op defaults while the graph
      // is boot-muted; user gestures then see the warm path.
      for (const slot of scene.PRIME_SLOTS) {
        await rpc('set_param', { name: slot, value: controlBySlot.get(slot).value })
      }
    }
    if (audioStarted) {
      const position = (await rpc('playback_position')).position
      const base = -transport.velocity * position / scene.SAMPLE_RATE
      await rpc('set_param', { name: 'master.tau_base', value: base })
      transport.playbackPosition = position
      transport.tauBase = base
      transport.sceneTime = 0
    }
    await rpc('set_param', {
      name: 'levelControl.value',
      value: controlBySlot.get('levelControl.value').value,
    })

    const response = await rpc('list_scope_taps')
    const taps = new Map((response.taps || []).map((tap) => [tap.name, tap.slot]))
    wells.forEach((well) => { well.slot = taps.get(well.tap) ?? null })
    const missing = wells.filter((well) => well.slot === null)

    status.classList.toggle('fault', missing.length > 0)
    status.textContent = missing.length
      ? `${audio} / missing ${missing.map((well) => `scope.${well.tap}`).join(', ')}`
      : `${audio} / 44.1 kHz / twenty strings + wet metal`

    if (!loopStarted) {
      loopStarted = true
      renderFrame()
    }
  } catch (error) {
    showFault(error)
  }
}

boot()
