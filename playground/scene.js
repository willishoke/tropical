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

  // A short, inharmonic metal cloud for the wet-room witness. Like the string
  // attack, each row is paired with an equal negative fast decay, so the hit is
  // causal and click-free while remaining a finite closed-form modal bank.
  // The frequencies are intentionally non-latticed: this is percussion, not a
  // tempered pitch pretending to be noise.
  const SNARE_FREQUENCIES = [
    211, 433, 887, 1511, 2837, 5081,
  ]

  function snareModes(hitIndex) {
    const rows = []
    SNARE_FREQUENCIES.forEach((frequency, index) => {
      const amplitude = 0.065 * Math.pow(1 + index, -0.32)
      const bodyDecay = 7.5 + 1.15 * index
      const attackDecay = bodyDecay + 88 + 3.7 * index
      const phase = GOLDEN_ANGLE * (index + 11 * hitIndex + 1) % TWO_PI
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
    })
    return rows
  }

  function buildSceneGraph() {
    const nodes = []
    const veilIds = []
    const roomIds = []
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
        id: 'lengthControl',
        // RT60 changes remain epoch-rate: making the pole coordinate itself a
        // sample-rate glide defeats stage-0 coefficient hoisting. The Metal
        // control sender still applies latest-value backpressure to this row.
        kind: 'knob',
        params: { value: 8 },
        sel: {},
        in: {},
      },
      {
        id: 'levelControl',
        kind: 'glideknob',
        // Boot-muted while the Metal coefficient paths are primed. The UI's
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
      const roomId = `room${number}`
      const strikeTime = chord.startStep * STEP_SECONDS + 0.06

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
        {
          id: roomId,
          kind: 'reverb',
          params: {
            rt60: 8,
            dir: 0,
            sway: 0.12,
            rate: 0.21,
            modes: 12,
          },
          sel: {},
          // A pre-veil send: the room acts directly on this modal chord bank.
          in: { in: [sourceId], rt60: ['lengthControl'] },
        },
      )
      veilIds.push(veilId)
      roomIds.push(roomId)
    })

    // One hard wet hit halfway between each chord. The direct modal cloud is
    // retained at low level by its authored residues; the same source crosses
    // a denser room so the return is unmistakably long, metallic, and spatial.
    for (let hitIndex = 0; hitIndex < 4; hitIndex += 1) {
      const number = hitIndex + 1
      const hitId = `metalHit${number}`
      const roomId = `metalRoom${number}`
      const strikeTime = hitIndex * 4 * STEP_SECONDS + 2 * STEP_SECONDS
      nodes.push(
        {
          id: hitId,
          kind: 'string',
          params: {
            t: rounded(strikeTime, 6),
            modes: snareModes(hitIndex),
          },
          sel: {},
          in: {},
        },
        {
          id: roomId,
          kind: 'reverb',
          params: {
            rt60: 8,
            dir: 0,
            sway: 0.18,
            rate: 0.31,
            modes: 16,
          },
          sel: {},
          in: { in: [hitId], rt60: ['lengthControl'] },
        },
      )
      hitIds.push(hitId)
      roomIds.push(roomId)
    }

    nodes.push(
      // These are ordinary signal sums. Each modal branch is realized on its
      // own side of the boundary, so all four strike anchors survive.
      {
        id: 'veil',
        kind: 'mix',
        params: {},
        sel: {},
        in: { in: veilIds },
      },
      {
        id: 'room',
        kind: 'mix',
        params: {},
        sel: {},
        in: { in: roomIds },
      },
      {
        id: 'spaceLevel',
        kind: 'glideknob',
        params: { value: 36 },
        sel: {},
        in: {},
      },
      {
        id: 'wet',
        kind: 'ring',
        params: {},
        sel: {},
        in: { in: ['room', 'spaceLevel'] },
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

    return { nodes, out: 'out', taps: true }
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
      max: 80,
      step: 1,
      value: 36,
      unit: '×',
      help: 'raise the long wet metal field against the strings',
    },
    {
      slot: 'lengthControl.value',
      label: 'length',
      min: 1.2,
      max: 14,
      step: 0,
      log: true,
      value: 8,
      unit: 's',
      help: 'change how long the room remembers each sonority',
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

  return {
    BASE_HZ,
    STEP_SECONDS,
    PATTERN_STEPS,
    PATTERN_SECONDS,
    SCENE_SECONDS,
    SAMPLE_RATE,
    CHORDS,
    CONTROLS,
    absoluteRatio,
    ratioLabel,
    voiceFrequency,
    stringModes,
    chordModes,
    snareModes,
    buildSceneGraph,
  }
})
