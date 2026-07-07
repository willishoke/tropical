// Reversible τ-scrub — reverse reverb via the arrow moat, on the modern
// (arrowemit / `load_patch_graph`) path.
//
// The stateful stdlib this demo was born on (VelocityClock / AnchoredPhase /
// Smooth) was retired by the cf-only fork; the live-param machinery moved into
// the EmitArrow patch-graph (see lean/Tropical/Playground.lean). So this is now
// a fixed patch-GRAPH driven live over the same three RPCs the playground uses:
//
//   set_param          — a raw slot write (steps at block rate)
//   set_param_glide    — a closed-form smoothstep ramp (click-free, stateless)
//   set_param_freq     — a phase-anchored freq change (waveform stays continuous)
//   set_param_velocity — the GLOBAL time-warp scrub (re-bases τ for continuity)
//
// The graph (downstream-only, in the playground's node vocabulary):
//
//   plk ─► cmb ─► sfl ─► out
//
//   plk  PluckedMorphOsc — a MorphOsc with a closed-form pluck envelope baked in.
//        THE dynamic content: forward it's a fast-attack/slow-decay pluck; reversed
//        (Time warp < 0) a slow swell into a hard cut — the reversed-tape cue.
//   cmb  one-sided resonant comb — the voice read at a decaying series of clock
//        offsets k·delay. Because the voice is plucked, each tap is a delayed
//        PLUCKED copy: an audible echo (delay < 0) or PRE-echo (delay > 0, reading
//        the future — impossible on a stream). One-sided ⇒ time-asymmetric ⇒ the
//        tail flips (echo ↔ pre-echo) when the master clock reverses.
//   sfl  through-zero flange — motion over the whole tail.
//
// The global Time warp scrubs the master clock every generator reads, so one knob
// runs the entire closed-form patch — voice, envelope, every comb tap — forward,
// frozen, or backward, exactly. The slide (`normalize`) pushes each effect's warp
// up onto the generators' clocks; nothing holds history.
//
// Every knob below is a live `param:<id>.<knob>` slot — turning it drives the
// running kernel with NO recompile (the topology is fixed).

// A knob exposed on the TUI panel. `slot` is the engine param name
// (`<nodeId>.<knob>`); `mode` picks which RPC drives it, matching how
// Playground.lean allocates the slot (glided → ramp slots, anchored → #phase
// slot, else a raw slot).
export type ParamMode = 'live' | 'glide' | 'freq' | 'velocity'

export type ParamSpec = {
  slot: string
  label: string
  min: number
  max: number
  step: number
  value: number
  unit: string
  mode: ParamMode
  group: 'transport' | 'voice' | 'space' | 'motion'
}

// The panel. Values are in the graph's native units (Hz, seconds, unitless) —
// exactly what the knob defaults below carry, so the seeded slots and these
// agree name-for-name and unit-for-unit.
export const PARAMS: ParamSpec[] = [
  { slot: 'master.velocity', label: 'Time warp', min: -2, max: 2, step: 0.05, value: 1, unit: '×', mode: 'velocity', group: 'transport' },
  { slot: 'plk.freq',       label: 'Voice pitch', min: 40,     max: 440,  step: 1,      value: 110,   unit: 'Hz', mode: 'freq',  group: 'voice' },
  { slot: 'plk.morph',      label: 'Morph',       min: 0,      max: 1,    step: 0.02,   value: 0.3,   unit: '',   mode: 'glide', group: 'voice' },
  { slot: 'plk.event_rate', label: 'Pluck rate',  min: 0.5,    max: 8,    step: 0.1,    value: 2.5,   unit: '/s', mode: 'glide', group: 'voice' },
  { slot: 'cmb.delay',      label: 'Tap (±=echo/pre)', min: -0.03, max: 0.03, step: 0.001, value: 0.012, unit: 's', mode: 'glide', group: 'space' },
  { slot: 'cmb.decay',      label: 'Tail decay',  min: 0.2,    max: 0.9,  step: 0.01,   value: 0.7,   unit: '',   mode: 'glide', group: 'space' },
  { slot: 'sfl.depth',      label: 'Flange depth', min: 0.0002, max: 0.02, step: 0.0002, value: 0.006, unit: 's', mode: 'glide', group: 'motion' },
  { slot: 'sfl.rate',       label: 'Flange rate', min: 0.02,   max: 6,    step: 0.02,   value: 0.3,   unit: 'Hz', mode: 'live',  group: 'motion' },
]

// A patch-graph node, shaped for `load_patch_graph` (see Playground.decodeGraph):
// `{ id, kind, params, sel, in: { port: [srcId, …] } }`. Ports absent from `in`
// default to empty (open) inlets.
type GraphNode = {
  id: string
  kind: string
  params: Record<string, number>
  sel: Record<string, string>
  in: Record<string, string[]>
}

// The fixed graph. Node kinds and knob names are 1:1 with Playground's `KINDS`.
// `taps: true` asks the engine to materialize a `render_window`-readable tap per
// node, so the scope TUI (`tui/scope`) can attach and inspect any point live.
export function buildGraph() {
  const nodes: GraphNode[] = [
    { id: 'plk', kind: 'pluck',   params: { freq: 110, morph: 0.3, event_rate: 2.5 }, sel: {}, in: { freq: [] } },
    { id: 'cmb', kind: 'comb',    params: { delay: 0.012, decay: 0.7 },   sel: {}, in: { in: ['plk'] } },
    { id: 'sfl', kind: 'sflange', params: { depth: 0.006, rate: 0.3 },    sel: {}, in: { in: ['cmb'], mod: [] } },
    { id: 'out', kind: 'out',     params: {},                             sel: {}, in: { in: ['sfl'] } },
  ]
  return { nodes, out: 'out', taps: true }
}
