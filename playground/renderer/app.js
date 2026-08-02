// A deliberately fixed instrument: twenty exact-ratio strings unfold as one
// scene. The UI does not pretend to be a patcher or a sequencer. Its sixteen
// moments are places in one closed-form signal, and seeking is a clock rebase.
'use strict'

const rpc = (method, params = {}) => window.tropical.call(method, params)
const scene = window.ModalScene
const GRAPH = scene.buildSceneGraph()
const CONTROLS = scene.CONTROLS.map((control) => ({ ...control }))
const SCOPE_PROFILE = window.ModalScopeProfile
const phaseLock = window.ModalPhaseLock
const scopeView = window.ModalScopeView

const MODE_COLORS = [
  '--mode-1', '--mode-2', '--mode-3', '--mode-4', '--mode-5',
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
const scopeSlots = new Map()
const stepElements = []
const chordElements = []

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
)

function enqueueParamWrite(control, nextValue) {
  const targetValue = clamp(quantize(control, nextValue), control.min, control.max)
  control.value = targetValue
  updateControlElement(CONTROLS.indexOf(control))
  paramSender.submit(control.slot, { control, targetValue })
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
      try { slider.setPointerCapture(event.pointerId) } catch {}
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
  const element = document.createElement('article')
  element.className = 'scope'
  element.innerHTML = `
    <div class="scope-head">
      <span>STRING MODES / PHASE VIEW</span>
      <span class="scope-meta">five independent locks · waiting</span>
    </div>
  `
  const canvas = document.createElement('canvas')
  element.appendChild(canvas)
  root.appendChild(element)
  wells.push({
    canvas,
    meta: element.querySelector('.scope-meta'),
    traces: MODE_COLORS.map((color, voiceIndex) => ({ color, voiceIndex })),
  })
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

function drawModeOverlay(well, frames, chord) {
  const { canvas } = well
  const { width, height, dpr } = resizeCanvas(canvas)
  const context = canvas.getContext('2d')
  const palette = getComputedStyle(document.documentElement)
  const ground = palette.getPropertyValue('--surface')
  const grid = palette.getPropertyValue('--rule')

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

  const visibleFrames = scopeView.visibleFrames(frames)
  if (visibleFrames.length === 0) {
    well.meta.textContent = 'five independent locks · no signal'
    return
  }

  // A fixed volts/div calibration is the point: changing modal amplitude must
  // move the trace. Per-frame peak normalization made a decaying, phase-locked
  // sinusoid mathematically indistinguishable from a frozen one.
  const scale = scene.SCOPE_FULL_SCALE * 1.08

  visibleFrames.forEach((frame) => {
    context.strokeStyle = palette.getPropertyValue(frame.trace.color)
    context.lineWidth = Math.max(1, dpr)
    context.beginPath()
    frame.values.forEach((value, index) => {
      const x = index * width / (frame.values.length - 1)
      const y = height / 2 - scopeView.normalizedAmplitude(value, scale)
        * (height / 2 - 5 * dpr)
      if (index === 0) context.moveTo(x, y)
      else context.lineTo(x, y)
    })
    context.stroke()
  })

  const milliseconds = (
    frames[0].values.length * frames[0].stride / scene.SAMPLE_RATE * 1000
  )
  const ratios = chord.voices.map((voice) => scene.absoluteRatio(chord, voice).join('/'))
  const envelope = scopeView.envelopeFraction(frames, scene.SCOPE_FULL_SCALE)
  well.meta.textContent = (
    `${chord.label} · ${ratios.join(' · ')} · env ${Math.round(envelope * 100)}%`
    + ` · ${milliseconds.toFixed(1)} ms`
  )
}

function chordIndexAt(time) {
  return scene.CHORDS.findIndex((chord, index) => {
    const next = scene.CHORDS[index + 1]
    return time >= chord.startStep * scene.STEP_SECONDS
      && time < (next ? next.startStep * scene.STEP_SECONDS : scene.PATTERN_SECONDS)
  })
}

function renderSequence(time) {
  const inPhrase = time >= 0 && time < scene.PATTERN_SECONDS
  const activeStep = inPhrase
    ? clamp(Math.floor(time / scene.STEP_SECONDS), 0, scene.PATTERN_STEPS - 1)
    : -1
  stepElements.forEach((element, index) => {
    element.classList.toggle('active', index === activeStep)
  })

  const activeChord = chordIndexAt(time)
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
      nudgeControl(control, -1, event.shiftKey)
    } else if (event.key === 'ArrowRight') {
      nudgeControl(control, 1, event.shiftKey)
    }
    else if (event.key === ' ') toggleFlow()
    else if (event.key.toLowerCase() === 'r') reverseFlow()
    else if (event.key === 'Enter') seekScene(0)
    else return
    event.preventDefault()
  })
}

function showFault(error) {
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

    const chordIndex = Math.max(0, chordIndexAt(transport.sceneTime))
    const chord = scene.CHORDS[chordIndex]
    const well = wells[0]
    const live = well.traces.map((trace) => {
      const tap = scene.scopeModeId(chordIndex, trace.voiceIndex)
      return { ...trace, tap, slot: scopeSlots.get(tap) }
    }).filter((trace) => trace.slot !== undefined)
    const count = DISPLAY_SAMPLES + SEARCH_SAMPLES + WARMUP_SAMPLES
    const start = Math.max(0, position - count)
    const response = await rpc('render_window', {
      start,
      count,
      point_budget: SCOPE_POINT_BUDGET,
      slots: live.map((well) => well.slot),
    })

    if (response.preempted) {
      window.setTimeout(renderFrame, 1000 / SCOPE_FPS)
      return
    }

    const stride = response.stride ?? 1
    const warmupPoints = Math.ceil(WARMUP_SAMPLES / stride)
    const searchPoints = Math.ceil(SEARCH_SAMPLES / stride)
    const displayPoints = Math.ceil(DISPLAY_SAMPLES / stride)

    const frames = live.map((trace, resultIndex) => {
      const values = (response.values[resultIndex] || []).slice(warmupPoints)
      return {
        trace,
        stride,
        ...phaseLock.window(values, searchPoints, displayPoints),
      }
    })
    drawModeOverlay(well, frames, chord)
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
      // the declared coefficient families at their no-op defaults while the graph
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
    scopeSlots.clear()
    taps.forEach((slot, name) => scopeSlots.set(name, slot))
    const missing = scene.SCOPE_TAPS.filter((name) => !scopeSlots.has(name))

    status.classList.toggle('fault', missing.length > 0)
    status.textContent = missing.length
      ? `${audio} / missing ${missing.map((name) => `scope.${name}`).join(', ')}`
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
