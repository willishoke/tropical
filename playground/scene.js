(function installModalScene(root, factory) {
  const scene = factory()
  if (typeof module !== 'undefined' && module.exports) module.exports = scene
  root.ModalScene = scene
})(typeof globalThis !== 'undefined' ? globalThis : this, function makeModalScene() {
  'use strict'

  const TWO_PI = Math.PI * 2
  const GOLDEN_ANGLE = 2.399963229728653
  const BASE_HZ = 55
  const STEP_SECONDS = 1
  const PATTERN_STEPS = 16
  const PATTERN_SECONDS = STEP_SECONDS * PATTERN_STEPS
  const SCENE_SECONDS = PATTERN_SECONDS
  const SAMPLE_RATE = 44100
  const STRING_ATTACK_DELAY = 0.06
  const PIZZICATO_SECTION_GAIN = 3
  const PIZZICATO_ROOM_SEND = 3

  const CHORDS = [
    {
      label: 'A−11',
      name: 'minor eleven',
      root: [1, 1],
      startStep: 0,
      voices: [
        [[1, 1], 0],
        [[3, 2], 0],
        [[6, 5], 1],
        [[9, 5], 1],
        [[4, 3], 2],
      ],
    },
    {
      label: 'D7·9',
      name: 'harmonic nine',
      root: [4, 3],
      startStep: 4,
      voices: [
        [[1, 1], 0],
        [[3, 2], 0],
        [[7, 4], 0],
        [[5, 4], 1],
        [[9, 8], 2],
      ],
    },
    {
      label: 'FΔ♯11',
      name: 'bright crossing',
      root: [8, 5],
      startStep: 8,
      voices: [
        [[1, 1], 0],
        [[3, 2], 0],
        [[15, 8], 0],
        [[5, 4], 1],
        [[45, 32], 1],
      ],
    },
    {
      label: 'E9',
      name: 'harmonic return',
      root: [3, 2],
      startStep: 12,
      voices: [
        [[1, 1], 0],
        [[7, 4], 0],
        [[5, 4], 1],
        [[3, 2], 1],
        [[9, 8], 2],
      ],
    },
  ]

  const VOICE_LEVELS = [1, 0.9, 0.77, 0.68, 0.6]

  function gcd(a, b) {
    let x = Math.abs(a)
    let y = Math.abs(b)
    while (y) [x, y] = [y, x % y]
    return x || 1
  }

  function absoluteRatio(chord, voice) {
    const [[num, den], octave] = voice
    const numerator = chord.root[0] * num * (2 ** octave)
    const denominator = chord.root[1] * den
    const common = gcd(numerator, denominator)
    return [numerator / common, denominator / common]
  }

  function ratioLabel(chord) {
    return chord.voices
      .map((voice) => absoluteRatio(chord, voice).join('/'))
      .join(' · ')
  }

  function voiceFrequency(chord, voice) {
    const [num, den] = absoluteRatio(chord, voice)
    return BASE_HZ * num / den
  }

  function rounded(value, places = 9) {
    return Number(value.toFixed(places))
  }

  // A slow sampled-chord gesture expressed entirely as modal rows. Every
  // partial is a +a/-a decay pair: it starts at zero, blooms in, and returns
  // to silence. Frequencies stay on the exact harmonic lattice of the JI
  // anchor; there is no equal-tempered detune hiding in the timbre.
  function stringModes(fundamental, chordIndex, voiceIndex) {
    const rows = []
    for (let partial = 1; partial <= 3; partial++) {
      const amplitude = (
        0.055
        * VOICE_LEVELS[voiceIndex]
        / Math.pow(partial, 1.22)
      )
      const slowDecay = 0.42 + 0.14 * (partial - 1)
      const attackDecay = slowDecay + 4.8 + 0.7 * (partial - 1)
      const phase = (
        GOLDEN_ANGLE
        * (1 + partial + 5 * voiceIndex + 29 * chordIndex)
      ) % TWO_PI
      const frequency = rounded(fundamental * partial)
      rows.push([
        frequency,
        rounded(slowDecay, 6),
        rounded(amplitude, 9),
        rounded(phase, 9),
      ])
      rows.push([
        frequency,
        rounded(attackDecay, 6),
        rounded(-amplitude, 9),
        rounded(phase, 9),
      ])
    }
    return rows
  }

  function chordModes(chord, chordIndex) {
    return chord.voices.flatMap((voice, voiceIndex) => (
      stringModes(voiceFrequency(chord, voice), chordIndex, voiceIndex)
    ))
  }

  // The scope uses one shared, immutable volts/div calibration. Derive it from
  // the exact paired exponential carried by each projected fundamental so the
  // canvas cannot normalize away the audible attack/decay envelope.
  function pairedEnvelopePeak(slowDecay, fastDecay, amplitude) {
    if (!(slowDecay > 0) || !(fastDecay > slowDecay)) return 0
    const peakTime = Math.log(fastDecay / slowDecay) / (fastDecay - slowDecay)
    return Math.abs(amplitude) * (
      Math.exp(-slowDecay * peakTime) - Math.exp(-fastDecay * peakTime)
    )
  }

  function scopeEnvelopePeak(chordIndex, voiceIndex) {
    const chord = CHORDS[chordIndex]
    const voice = chord?.voices[voiceIndex]
    if (!chord || !voice) return 0
    const rows = stringModes(voiceFrequency(chord, voice), chordIndex, voiceIndex)
    return pairedEnvelopePeak(rows[0][1], rows[1][1], rows[0][2])
  }

  function scopeEnvelopeAt(chordIndex, voiceIndex, sceneTime) {
    const chord = CHORDS[chordIndex]
    const voice = chord?.voices[voiceIndex]
    if (!chord || !voice || !Number.isFinite(sceneTime)) return 0
    const age = sceneTime - (
      chord.startStep * STEP_SECONDS + STRING_ATTACK_DELAY
    )
    if (!(age > 0)) return 0
    const rows = stringModes(voiceFrequency(chord, voice), chordIndex, voiceIndex)
    return Math.abs(rows[0][2]) * (
      Math.exp(-rows[0][1] * age) - Math.exp(-rows[1][1] * age)
    )
  }

  function scopeEnvelopeSamples(
    chordIndex, voiceIndex, firstSceneTime, sceneStep, count,
  ) {
    const chord = CHORDS[chordIndex]
    const voice = chord?.voices[voiceIndex]
    if (!chord || !voice || !Number.isFinite(firstSceneTime)
      || !Number.isFinite(sceneStep) || !(count > 0)) return []
    const rows = stringModes(voiceFrequency(chord, voice), chordIndex, voiceIndex)
    const amplitude = Math.abs(rows[0][2])
    const slowDecay = rows[0][1]
    const fastDecay = rows[1][1]
    const strikeTime = chord.startStep * STEP_SECONDS + STRING_ATTACK_DELAY
    return Array.from({ length: Math.floor(count) }, (_unused, index) => {
      const age = firstSceneTime + index * sceneStep - strikeTime
      if (!(age > 0)) return 0
      return amplitude * (
        Math.exp(-slowDecay * age) - Math.exp(-fastDecay * age)
      )
    })
  }

  const SCOPE_FULL_SCALE = Math.max(...CHORDS.flatMap((chord, chordIndex) => (
    chord.voices.map((_voice, voiceIndex) => scopeEnvelopePeak(chordIndex, voiceIndex))
  )))

  // A short, warm pizzicato derived from the exact chord lattice. The shared
  // section gain is the listening-approved +9.5 dB lift; the room has a second
  // explicit compensation gain below because this low-register source excites
  // the fitted transfer much less strongly than the retired Metal cloud did.
  function pizzicatoModes(fundamental, velocity, voiceIndex) {
    const rows = []
    for (let partial = 1; partial <= 5; partial += 1) {
      const amplitude = (
        0.04
        * PIZZICATO_SECTION_GAIN
        * velocity
        / Math.pow(partial, 1.42)
      )
      const bodyDecay = 9.5 + 3.5 * (partial - 1)
      const attackDecay = bodyDecay + 118 + 24 * (partial - 1)
      const phase = GOLDEN_ANGLE * (voiceIndex + 3 * partial) % TWO_PI
      const frequency = rounded(fundamental * partial)
      rows.push([
        frequency,
        rounded(bodyDecay, 6),
        rounded(amplitude, 9),
        rounded(phase, 9),
      ])
      rows.push([
        frequency,
        rounded(attackDecay, 6),
        rounded(-amplitude, 9),
        rounded(phase, 9),
      ])
    }
    return rows
  }

  const GHOST_VOICES = [2, 4, 1]
  const GHOST_VELOCITIES = [0.44, 0.52, 0.4]

  function pizzicatoBeatModes(chord, beatInChord) {
    const voices = beatInChord === 0
      ? [[0, 0.72], [3, 0.64]]
      : [[GHOST_VOICES[beatInChord - 1], GHOST_VELOCITIES[beatInChord - 1]]]
    return voices.flatMap(([voiceIndex, velocity]) => (
      pizzicatoModes(
        voiceFrequency(chord, chord.voices[voiceIndex]),
        velocity,
        voiceIndex,
      )
    ))
  }

  const pizzicatoBeatId = (beatIndex) => `pizzicatoBeat${beatIndex + 1}`
  const PIZZICATO_BEAT_IDS = Array.from(
    { length: PATTERN_STEPS },
    (_unused, beatIndex) => pizzicatoBeatId(beatIndex),
  )

  const scopeModeId = (chordIndex, voiceIndex) => (
    `scopeChord${chordIndex + 1}Mode${voiceIndex + 1}`
  )
  const SCOPE_TAPS = CHORDS.flatMap((chord, chordIndex) => (
    chord.voices.map((_voice, voiceIndex) => scopeModeId(chordIndex, voiceIndex))
  ))

  function buildSceneGraph() {
    const nodes = []
    const veilIds = []
    const hitIds = []

    nodes.push(
      {
        id: 'veilControl',
        kind: 'knob',
        params: { value: 780 },
        sel: {},
        in: {},
      },
      {
        id: 'edgeControl',
        kind: 'knob',
        params: { value: 0.7 },
        sel: {},
        in: {},
      },
      {
        id: 'positionControl',
        // POSITION is a sample-rate equal-power pan between the profile's
        // analytic reverse and forward arms. It does not rebuild an epoch.
        kind: 'glideknob',
        params: { value: 1 },
        sel: {},
        in: {},
      },
      {
        id: 'levelControl',
        kind: 'glideknob',
        // Boot-muted while the modal coefficient paths are primed. The UI's
        // authored level is applied immediately before the scene is rebased to
        // zero, so no false first chord leaks during warm-up.
        params: { value: 0 },
        sel: {},
        in: {},
      },
    )

    CHORDS.forEach((chord, chordIndex) => {
      const number = chordIndex + 1
      const sourceId = `chord${number}`
      const veilId = `veil${number}`
      const strikeTime = chord.startStep * STEP_SECONDS + STRING_ATTACK_DELAY

      // A chord is one modal bank with five exact-ratio voices. Distinct
      // chords must remain distinct modal islands: modalMix is a pole union
      // with one shared strike anchor, so merging the score there would erase
      // every timestamp after the first.
      nodes.push(
        {
          id: sourceId,
          kind: 'string',
          params: {
            t: rounded(strikeTime, 6),
            modes: chordModes(chord, chordIndex),
          },
          sel: {},
          in: {},
        },
        {
          id: veilId,
          kind: 'filter',
          params: { cutoff: 780, resonance: 0.7 },
          sel: {},
          in: {
            in: [sourceId],
            cutoff: ['veilControl'],
            resonance: ['edgeControl'],
          },
        },
      )

      // Scope projections are deliberately disconnected from the audible
      // root. Each is one fundamental decay pair, giving the phase view five
      // simple modes it can lock independently without re-evaluating every
      // visible graph node as a tap.
      chord.voices.forEach((voice, voiceIndex) => {
        const fundamental = voiceFrequency(chord, voice)
        nodes.push({
          id: scopeModeId(chordIndex, voiceIndex),
          kind: 'string',
          params: {
            t: rounded(strikeTime, 6),
            modes: stringModes(fundamental, chordIndex, voiceIndex).slice(0, 2),
          },
          sel: {},
          in: {},
        })
      })
      veilIds.push(veilId)
    })

    // The selected percussion score is one chord-derived pizzicato per beat:
    // an open dyad on each chord downbeat, then three lighter single-note
    // ghosts. Its strike offset matches the string lane so chord and pluck
    // arrive as one gesture.
    PIZZICATO_BEAT_IDS.forEach((hitId, beatIndex) => {
      const chordIndex = Math.floor(beatIndex / 4)
      const beatInChord = beatIndex % 4
      const strikeTime = beatIndex * STEP_SECONDS + STRING_ATTACK_DELAY
      nodes.push(
        {
          id: hitId,
          kind: 'string',
          params: {
            t: rounded(strikeTime, 6),
            modes: pizzicatoBeatModes(CHORDS[chordIndex], beatInChord),
          },
          sel: {},
          in: {},
        },
      )
      hitIds.push(hitId)
    })

    nodes.push(
      // These are ordinary signal sums. Each modal branch is realized on its
      // own side of the boundary, so all sixteen beat anchors survive.
      {
        id: 'veil',
        kind: 'mix',
        params: {},
        sel: {},
        in: { in: veilIds },
      },
      {
        id: 'room',
        kind: 'groupedroomcache',
        params: { profile: 'clouds-current-radii-mono-v1-scene-cache' },
        sel: {},
        in: { position: ['positionControl'] },
      },
      {
        id: 'spaceLevel',
        kind: 'glideknob',
        params: { value: 1 },
        sel: {},
        in: {},
      },
      {
        id: 'pizzicatoRoomCompensation',
        kind: 'knob',
        params: { value: PIZZICATO_ROOM_SEND },
        sel: {},
        in: {},
      },
      {
        id: 'compensatedRoom',
        kind: 'ring',
        params: {},
        sel: {},
        in: { in: ['room', 'pizzicatoRoomCompensation'] },
      },
      {
        id: 'wet',
        kind: 'ring',
        params: {},
        sel: {},
        in: { in: ['compensatedRoom', 'spaceLevel'] },
      },
      {
        id: 'impact',
        kind: 'mix',
        params: {},
        sel: {},
        in: { in: hitIds },
      },
      {
        id: 'body',
        kind: 'mix',
        params: {},
        sel: {},
        in: { in: ['veil', 'wet', 'impact'] },
      },
      {
        id: 'presence',
        kind: 'glideknob',
        params: { value: 0.82 },
        sel: {},
        in: {},
      },
      {
        id: 'heard',
        kind: 'ring',
        params: {},
        sel: {},
        in: { in: ['body', 'presence'] },
      },
      {
        id: 'leveled',
        kind: 'ring',
        params: {},
        sel: {},
        in: { in: ['heard', 'levelControl'] },
      },
      {
        id: 'out',
        kind: 'out',
        params: {},
        sel: {},
        in: { in: ['leveled'] },
      },
    )

    return { nodes, out: 'out', taps: SCOPE_TAPS }
  }

  const CONTROLS = [
    {
      slot: 'presence.value',
      label: 'presence',
      min: 0,
      max: 1.2,
      step: 0.02,
      value: 0.82,
      unit: '',
      help: 'let the chord body enter or disappear',
    },
    {
      slot: 'veilControl.value',
      label: 'veil',
      min: 120,
      max: 5200,
      step: 0,
      log: true,
      value: 780,
      unit: 'Hz',
      help: 'move the low-pass veil across the strings',
    },
    {
      slot: 'edgeControl.value',
      label: 'edge',
      min: 0,
      max: 0.98,
      step: 0.02,
      value: 0.7,
      unit: '',
      help: 'draw a narrow luminous edge at the veil',
    },
    {
      slot: 'spaceLevel.value',
      label: 'room',
      min: 0,
      max: 1.5,
      step: 0.05,
      value: 1,
      unit: '×',
      help: 'raise the temporal pizzicato field against the strings',
    },
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
    {
      slot: 'master.velocity',
      label: 'flow',
      min: -1.25,
      max: 1.25,
      step: 0.05,
      value: 1,
      unit: '×',
      help: 'reverse, arrest, or varispeed the whole scene',
      transport: true,
    },
    {
      slot: 'levelControl.value',
      label: 'level',
      min: 0,
      max: 1.2,
      step: 0.02,
      value: 0.72,
      unit: '×',
      help: 'final listening level',
    },
  ]

  // First-use modal coefficient families exercised while `levelControl` keeps
  // the boot graph muted. The app and noninteractive qualification runner both
  // consume this list; neither maintains a second scene-specific copy.
  const PRIME_SLOTS = [
    'veilControl.value',
    'edgeControl.value',
  ]

  return {
    BASE_HZ,
    STEP_SECONDS,
    PATTERN_STEPS,
    PATTERN_SECONDS,
    SCENE_SECONDS,
    SAMPLE_RATE,
    STRING_ATTACK_DELAY,
    PIZZICATO_SECTION_GAIN,
    PIZZICATO_ROOM_SEND,
    CHORDS,
    CONTROLS,
    PRIME_SLOTS,
    SCOPE_TAPS,
    SCOPE_FULL_SCALE,
    PIZZICATO_BEAT_IDS,
    absoluteRatio,
    ratioLabel,
    voiceFrequency,
    stringModes,
    chordModes,
    pairedEnvelopePeak,
    scopeEnvelopePeak,
    scopeEnvelopeAt,
    scopeEnvelopeSamples,
    pizzicatoModes,
    pizzicatoBeatModes,
    pizzicatoBeatId,
    scopeModeId,
    buildSceneGraph,
  }
})
