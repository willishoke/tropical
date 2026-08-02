const test = require('node:test')
const assert = require('node:assert/strict')

const scene = require('./scene.js')

test('scene is one acyclic-looking fixed graph with unique ids', () => {
  const graph = scene.buildSceneGraph()
  const ids = graph.nodes.map((node) => node.id)
  assert.equal(new Set(ids).size, ids.length)
  assert.equal(graph.out, 'out')
  assert.ok(ids.includes(graph.out))
  assert.equal(graph.nodes.filter((node) => node.kind === 'string').length, 40)
  assert.deepEqual(graph.taps, scene.SCOPE_TAPS)
  assert.equal(graph.taps.length, 20)
  assert.equal(graph.nodes.filter((node) => node.kind === 'modalmix').length, 0)
  assert.equal(graph.nodes.filter((node) => node.kind === 'reverb').length, 0)
  assert.equal(graph.nodes.filter((node) => node.kind === 'groupedroom').length, 0)
  assert.equal(graph.nodes.filter((node) => node.kind === 'groupedroomcache').length, 1)
  assert.equal(graph.nodes.filter((node) => node.kind === 'filter').length, 4)
  assert.equal(graph.nodes.filter((node) => node.kind === 'glideknob').length, 4)
  assert.equal(graph.nodes.filter((node) => node.kind === 'knob').length, 3)
  assert.deepEqual(graph.nodes.find((node) => node.id === 'veil').in.in,
    ['veil1', 'veil2', 'veil3', 'veil4'])
  assert.equal(graph.nodes.find((node) => node.id === 'room').params.profile,
    'clouds-current-radii-mono-v1-scene-cache')
  assert.deepEqual(graph.nodes.find((node) => node.id === 'room').in.position,
    ['positionControl'])
  assert.deepEqual(graph.nodes.find((node) => node.id === 'impact').in.in,
    scene.PIZZICATO_BEAT_IDS)
  assert.equal(
    graph.nodes.find((node) => node.id === 'pizzicatoRoomCompensation').params.value,
    scene.PIZZICATO_ROOM_SEND,
  )
  assert.deepEqual(graph.nodes.find((node) => node.id === 'compensatedRoom').in.in,
    ['room', 'pizzicatoRoomCompensation'])
  assert.deepEqual(graph.nodes.find((node) => node.id === 'wet').in.in,
    ['compensatedRoom', 'spaceLevel'])
  assert.deepEqual(graph.nodes.find((node) => node.id === 'leveled').in.in,
    ['heard', 'levelControl'])
  assert.equal(graph.nodes.find((node) => node.id === 'levelControl').params.value, 0)
  assert.equal(scene.CONTROLS.find((control) =>
    control.slot === 'levelControl.value').value, 0.72)
  assert.deepEqual(
    scene.CONTROLS.find((control) => control.slot === 'positionControl.value'),
    {
      slot: 'positionControl.value',
      label: 'position',
      min: -1,
      max: 1,
      step: 0.05,
      value: 1,
      unit: '',
      help: 'move the room from a future-aware pre-tail to its forward decay',
    },
  )
  assert.equal(scene.CONTROLS.find((control) =>
    control.slot === 'spaceLevel.value').value, 1)
  assert.equal(scene.CONTROLS.find((control) =>
    control.slot === 'spaceLevel.value').max, 1.5)
  assert.deepEqual(graph.nodes.find((node) => node.id === 'out').in.in,
    ['leveled'])
  for (let index = 0; index < 4; index++) {
    const chord = graph.nodes.find((node) => node.id === `chord${index + 1}`)
    assert.equal(chord.params.modes.length, 30)
    assert.equal(
      chord.params.t,
      scene.CHORDS[index].startStep * scene.STEP_SECONDS + 0.06,
    )
    assert.equal(graph.nodes.find((node) => node.id === `room${index + 1}`), undefined)
    assert.deepEqual(
      graph.nodes.find((node) => node.id === `veil${index + 1}`).in.cutoff,
      ['veilControl'],
    )
    for (let voiceIndex = 0; voiceIndex < 5; voiceIndex++) {
      const projection = graph.nodes.find(
        (node) => node.id === scene.scopeModeId(index, voiceIndex),
      )
      assert.equal(projection.params.modes.length, 2)
      assert.equal(projection.params.t, chord.params.t)
      assert.ok(Math.abs(
        projection.params.modes[0][0]
        - scene.voiceFrequency(scene.CHORDS[index], scene.CHORDS[index].voices[voiceIndex]),
      ) < 1e-8)
      assert.equal(projection.params.modes[1][0], projection.params.modes[0][0])
      assert.equal(projection.params.modes[1][2], -projection.params.modes[0][2])
    }
  }
  for (let beatIndex = 0; beatIndex < scene.PATTERN_STEPS; beatIndex++) {
    const hit = graph.nodes.find((node) => node.id === scene.pizzicatoBeatId(beatIndex))
    assert.equal(
      hit.params.t,
      beatIndex * scene.STEP_SECONDS + scene.STRING_ATTACK_DELAY,
    )
    assert.equal(hit.params.modes.length, beatIndex % 4 === 0 ? 20 : 10)
    assert.equal(
      graph.nodes.find((node) => node.id === `${scene.pizzicatoBeatId(beatIndex)}Room`),
      undefined,
    )
  }
  assert.deepEqual(graph.nodes.find((node) => node.id === 'body').in.in,
    ['veil', 'wet', 'impact'])
})

test('amplitudes glide while modal coefficients stay hoistable', () => {
  const graph = scene.buildSceneGraph()
  const byId = new Map(graph.nodes.map((node) => [node.id, node]))
  for (const id of [
    'presence', 'spaceLevel', 'positionControl', 'levelControl',
  ]) {
    assert.equal(byId.get(id)?.kind, 'glideknob')
    assert.ok(scene.CONTROLS.some((control) => control.slot === `${id}.value`))
  }
  for (const id of ['veilControl', 'edgeControl']) {
    assert.equal(byId.get(id)?.kind, 'knob')
    assert.ok(scene.CONTROLS.some((control) => control.slot === `${id}.value`))
  }
  assert.ok(scene.CONTROLS.some((control) => control.slot === 'master.velocity'))
  assert.deepEqual(scene.PRIME_SLOTS, ['veilControl.value', 'edgeControl.value'])
  assert.ok(!scene.PRIME_SLOTS.includes('positionControl.value'))
})

test('all chord anchors are rational multiples of the common 55 Hz root', () => {
  const expectedRatios = new Map([
    ['A−11', ['1/1', '3/2', '12/5', '18/5', '16/3']],
    ['D7·9', ['4/3', '2/1', '7/3', '10/3', '6/1']],
    ['FΔ♯11', ['8/5', '12/5', '3/1', '4/1', '9/2']],
    ['E9', ['3/2', '21/8', '15/4', '9/2', '27/4']],
  ])
  for (const chord of scene.CHORDS) {
    assert.deepEqual(
      chord.voices.map((voice) => scene.absoluteRatio(chord, voice).join('/')),
      expectedRatios.get(chord.label),
    )
    for (const voice of chord.voices) {
      const [numerator, denominator] = scene.absoluteRatio(chord, voice)
      const expected = scene.BASE_HZ * numerator / denominator
      assert.equal(scene.voiceFrequency(chord, voice), expected)
      assert.ok(Number.isInteger(numerator))
      assert.ok(Number.isInteger(denominator))
    }
  }
})

test('each string partial blooms causally from an exact amplitude pair', () => {
  for (const chord of scene.CHORDS) {
    for (let voiceIndex = 0; voiceIndex < chord.voices.length; voiceIndex++) {
      const frequency = scene.voiceFrequency(chord, chord.voices[voiceIndex])
      const rows = scene.stringModes(frequency, chord.startStep / 4, voiceIndex)
      assert.equal(rows.length, 6)
      for (let row = 0; row < rows.length; row += 2) {
        assert.equal(rows[row][0], rows[row + 1][0])
        assert.equal(rows[row][2], -rows[row + 1][2])
        assert.equal(rows[row][3], rows[row + 1][3])
        assert.ok(rows[row][1] < rows[row + 1][1])
      }
    }
  }
})

test('scope volts-per-division is the exact shared fundamental envelope peak', () => {
  const peaks = scene.CHORDS.flatMap((chord, chordIndex) => (
    chord.voices.map((_voice, voiceIndex) => (
      scene.scopeEnvelopePeak(chordIndex, voiceIndex)
    ))
  ))
  assert.equal(scene.SCOPE_FULL_SCALE, Math.max(...peaks))
  assert.ok(scene.SCOPE_FULL_SCALE > 0.04)
  assert.ok(scene.SCOPE_FULL_SCALE < 0.041)
  peaks.forEach((peak) => assert.ok(peak <= scene.SCOPE_FULL_SCALE))
})

test('scope envelope follows the exact paired mode at audible-now', () => {
  scene.CHORDS.forEach((chord, chordIndex) => {
    chord.voices.forEach((_voice, voiceIndex) => {
      const strike = chord.startStep * scene.STEP_SECONDS + scene.STRING_ATTACK_DELAY
      const rows = scene.stringModes(
        scene.voiceFrequency(chord, chord.voices[voiceIndex]), chordIndex, voiceIndex,
      )
      const peakAge = Math.log(rows[1][1] / rows[0][1]) / (rows[1][1] - rows[0][1])
      assert.equal(scene.scopeEnvelopeAt(chordIndex, voiceIndex, strike), 0)
      assert.ok(Math.abs(
        scene.scopeEnvelopeAt(chordIndex, voiceIndex, strike + peakAge)
        - scene.scopeEnvelopePeak(chordIndex, voiceIndex)
      ) < 1e-15)
    })
  })
})

test('each chord-derived pizzicato beat is a causal harmonic attack pair', () => {
  assert.equal(scene.PIZZICATO_SECTION_GAIN, 3)
  assert.equal(scene.PIZZICATO_ROOM_SEND, 3)
  scene.CHORDS.forEach((chord) => {
    for (let beatInChord = 0; beatInChord < 4; beatInChord++) {
      const rows = scene.pizzicatoBeatModes(chord, beatInChord)
      assert.equal(rows.length, beatInChord === 0 ? 20 : 10)
      const voiceCount = beatInChord === 0 ? 2 : 1
      assert.equal(rows.length, voiceCount * 5 * 2)
      for (let row = 0; row < rows.length; row += 2) {
        assert.equal(rows[row][0], rows[row + 1][0])
        assert.equal(rows[row][2], -rows[row + 1][2])
        assert.equal(rows[row][3], rows[row + 1][3])
        assert.ok(rows[row][1] < rows[row + 1][1])
      }
    }
  })
})

test('pizzicato attack timing retains the approved prompt onset', () => {
  const rows = scene.pizzicatoModes(55, 1, 0)
  const values = Array.from({ length: Math.round(0.1 * scene.SAMPLE_RATE) }, (_, index) => {
    const time = index / scene.SAMPLE_RATE
    return rows.reduce((sum, [frequency, decay, amplitude, phase]) => (
      sum + amplitude * Math.exp(-decay * time)
        * Math.cos(2 * Math.PI * frequency * time + phase)
    ), 0)
  })
  const peak = Math.max(...values.map(Math.abs))
  const onsetIndex = values.findIndex((value) => Math.abs(value) >= 0.01 * peak)
  const peakIndex = values.findIndex((value) => Math.abs(value) === peak)
  assert.ok(onsetIndex / scene.SAMPLE_RATE * 1000 < 2)
  assert.ok(peakIndex / scene.SAMPLE_RATE * 1000 > 14)
  assert.ok(peakIndex / scene.SAMPLE_RATE * 1000 < 17)
  assert.equal(values[0], 0)
  for (let row = 0; row < rows.length; row += 2) {
    assert.equal(rows[row][0], rows[row + 1][0])
    assert.equal(rows[row][2], -rows[row + 1][2])
    assert.equal(rows[row][3], rows[row + 1][3])
    assert.ok(rows[row][1] < rows[row + 1][1])
  }
})

test('the scene loops exactly at the end of the sixteen-step phrase', () => {
  assert.equal(scene.PATTERN_STEPS, 16)
  assert.equal(scene.PATTERN_SECONDS, 16)
  assert.equal(scene.SCENE_SECONDS, scene.PATTERN_SECONDS)
})
