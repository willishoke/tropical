const test = require('node:test')
const assert = require('node:assert/strict')

const scene = require('./scene.js')

test('scene is one acyclic-looking fixed graph with unique ids', () => {
  const graph = scene.buildSceneGraph()
  const ids = graph.nodes.map((node) => node.id)
  assert.equal(new Set(ids).size, ids.length)
  assert.equal(graph.out, 'out')
  assert.ok(ids.includes(graph.out))
  assert.equal(graph.nodes.filter((node) => node.kind === 'string').length, 8)
  assert.equal(graph.nodes.filter((node) => node.kind === 'modalmix').length, 0)
  assert.equal(graph.nodes.filter((node) => node.kind === 'reverb').length, 8)
  assert.equal(graph.nodes.filter((node) => node.kind === 'filter').length, 4)
  assert.equal(graph.nodes.filter((node) => node.kind === 'glideknob').length, 3)
  assert.equal(graph.nodes.filter((node) => node.kind === 'knob').length, 3)
  assert.deepEqual(graph.nodes.find((node) => node.id === 'veil').in.in,
    ['veil1', 'veil2', 'veil3', 'veil4'])
  assert.deepEqual(graph.nodes.find((node) => node.id === 'room').in.in,
    [
      'room1', 'room2', 'room3', 'room4',
      'metalRoom1', 'metalRoom2', 'metalRoom3', 'metalRoom4',
    ])
  assert.deepEqual(graph.nodes.find((node) => node.id === 'leveled').in.in,
    ['heard', 'levelControl'])
  assert.equal(graph.nodes.find((node) => node.id === 'levelControl').params.value, 0)
  assert.equal(scene.CONTROLS.find((control) =>
    control.slot === 'levelControl.value').value, 0.72)
  assert.deepEqual(graph.nodes.find((node) => node.id === 'out').in.in,
    ['leveled'])
  for (let index = 0; index < 4; index++) {
    const chord = graph.nodes.find((node) => node.id === `chord${index + 1}`)
    const room = graph.nodes.find((node) => node.id === `room${index + 1}`)
    assert.equal(chord.params.modes.length, 30)
    assert.equal(
      chord.params.t,
      scene.CHORDS[index].startStep * scene.STEP_SECONDS + 0.06,
    )
    assert.equal(room.params.modes, 12)
    assert.deepEqual(room.in.in, [`chord${index + 1}`])
    assert.deepEqual(room.in.rt60, ['lengthControl'])
    assert.deepEqual(
      graph.nodes.find((node) => node.id === `veil${index + 1}`).in.cutoff,
      ['veilControl'],
    )
  }
  for (let index = 0; index < 4; index++) {
    const hit = graph.nodes.find((node) => node.id === `metalHit${index + 1}`)
    const room = graph.nodes.find((node) => node.id === `metalRoom${index + 1}`)
    assert.equal(hit.params.t, index * 4 * scene.STEP_SECONDS + 2 * scene.STEP_SECONDS)
    assert.equal(hit.params.modes.length, 12)
    assert.equal(room.params.modes, 16)
    assert.deepEqual(room.in.in, [`metalHit${index + 1}`])
    assert.deepEqual(room.in.rt60, ['lengthControl'])
  }
  assert.deepEqual(graph.nodes.find((node) => node.id === 'body').in.in,
    ['veil', 'wet', 'impact'])
})

test('amplitudes glide while modal coefficients stay hoistable', () => {
  const graph = scene.buildSceneGraph()
  const byId = new Map(graph.nodes.map((node) => [node.id, node]))
  for (const id of [
    'presence', 'spaceLevel', 'levelControl',
  ]) {
    assert.equal(byId.get(id)?.kind, 'glideknob')
    assert.ok(scene.CONTROLS.some((control) => control.slot === `${id}.value`))
  }
  for (const id of ['veilControl', 'edgeControl', 'lengthControl']) {
    assert.equal(byId.get(id)?.kind, 'knob')
    assert.ok(scene.CONTROLS.some((control) => control.slot === `${id}.value`))
  }
  assert.ok(scene.CONTROLS.some((control) => control.slot === 'master.velocity'))
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

test('each metal hit is a causal inharmonic attack pair', () => {
  for (let hit = 0; hit < 4; hit++) {
    const rows = scene.snareModes(hit)
    assert.equal(rows.length, 12)
    for (let row = 0; row < rows.length; row += 2) {
      assert.equal(rows[row][0], rows[row + 1][0])
      assert.equal(rows[row][2], -rows[row + 1][2])
      assert.equal(rows[row][3], rows[row + 1][3])
      assert.ok(rows[row][1] < rows[row + 1][1])
    }
  }
})

test('the scene loops exactly at the end of the sixteen-step phrase', () => {
  assert.equal(scene.PATTERN_STEPS, 16)
  assert.equal(scene.PATTERN_SECONDS, 16)
  assert.equal(scene.SCENE_SECONDS, scene.PATTERN_SECONDS)
})
