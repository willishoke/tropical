// Reversible τ-scrub + closed-form reverse reverb, anchored-phase voice,
// CONTINUOUS pluck envelope (no discontinuity → no impulse-train scratch).
//
//   VelocityClock(velocity) → τ
//        ├ AnchoredPhase(f0, τ)         → Φ   (pitch phase)
//        └ AnchoredPhase(event_rate, τ) → Ψ   (event phase)
//   tap k:  ModalVoice(Φ + f0·kD) · env(Ψ + event_rate·kD) · gᵏ ,  Σ → SoftClip → dac
//
// env(ψ) = 17.6·f·(1−f)⁶ with f=frac(ψ): a smooth skewed pulse — 0 at both ends
// of the period (continuous across the wrap, so NO step → no aliasing comb),
// asymmetric (fast rise, slow decay) so it still reverses audibly. No BLEP
// needed because there's no discontinuity to band-limit.
//
// All params are smoothed (audible glide) AND snap-settle, so between knob
// moves f0/event_rate are exactly constant → AnchoredPhase is closed-form →
// random access + exact reverse; during a move it glides (no squelch).

const ref = (instance: string, output: string) => ({ op: 'ref', instance, output })
const par = (name: string) => ({ op: 'param', name })
const add = (a: any, b: any) => ({ op: 'add', args: [a, b] })
const sub = (a: any, b: any) => ({ op: 'sub', args: [a, b] })
const mul = (a: any, b: any) => ({ op: 'mul', args: [a, b] })
const floor = (a: any) => ({ op: 'floor', args: [a] })
const frac = (x: any) => sub(x, floor(x))

const K = 10 // future reverb taps (k = 1..K); k = 0 is the dry event

export type ParamSpec = {
  name: string
  label: string
  min: number
  max: number
  step: number
  value: number
  unit: string
  group: 'transport' | 'voice' | 'reverb' | 'master'
}

export const PARAMS: ParamSpec[] = [
  { name: 'velocity', label: 'Velocity', min: -2, max: 2, step: 0.05, value: 1, unit: '×', group: 'transport' },
  { name: 'f0', label: 'Voice pitch', min: 40, max: 440, step: 1, value: 110, unit: 'Hz', group: 'voice' },
  { name: 'event_rate', label: 'Event rate', min: 0.2, max: 6, step: 0.1, value: 1.0, unit: '/s', group: 'voice' },
  { name: 'rev_time', label: 'Tap spacing', min: 5, max: 120, step: 1, value: 45, unit: 'ms', group: 'reverb' },
  { name: 'rev_decay', label: 'Decay (g)', min: 0.3, max: 0.95, step: 0.01, value: 0.72, unit: '', group: 'reverb' },
  { name: 'rev_amount', label: 'Reverb amt', min: 0, max: 1.2, step: 0.01, value: 0.7, unit: '', group: 'reverb' },
  { name: 'master_gain', label: 'Drive', min: 0, max: 1.5, step: 0.01, value: 0.6, unit: '', group: 'master' },
]

const GLIDE = 0.0007 // ~30 ms one-pole coefficient (clearly audible, then snaps)

export function buildProgram() {
  const paramDecls = PARAMS.map((p) => ({ op: 'paramDecl', name: p.name, value: p.value }))
  const smoothers = PARAMS.map((p) => ({
    op: 'instanceDecl', name: `sm_${p.name}`, program: 'Smooth', inputs: { x: par(p.name), rate: GLIDE },
  }))
  const s = (name: string) => ref(`sm_${name}`, 'out') // smoothed param ref

  const Phi = ref('pphase', 'phase')
  const Psi = ref('ephase', 'phase')
  const D = mul(s('rev_time'), 0.001)
  const g = s('rev_decay')
  const gPow = (k: number) => { let e: any = 1; for (let j = 0; j < k; j++) e = mul(g, e); return e }

  const voicePhaseAt = (k: number) => (k === 0 ? Phi : add(Phi, mul(s('f0'), mul(k, D))))
  const evtPhaseAt = (k: number) => (k === 0 ? Psi : add(Psi, mul(s('event_rate'), mul(k, D))))
  // continuous skewed-pulse envelope: 17.6·f·(1−f)⁶, f = frac(event phase)
  const envExpr = (k: number) => {
    const f = frac(evtPhaseAt(k))
    const u = sub(1, f)
    const u2 = mul(u, u)
    const u6 = mul(mul(u2, u2), u2)
    return mul(17.6, mul(f, u6))
  }

  const dry = mul(ref('v0', 'out'), envExpr(0))
  let revSum: any = null
  for (let k = 1; k <= K; k++) {
    const term = mul(gPow(k), mul(ref(`v${k}`, 'out'), envExpr(k)))
    revSum = revSum === null ? term : add(revSum, term)
  }
  const total = add(dry, mul(s('rev_amount'), revSum))

  // ModalVoice reused as phase reader: f0=1, tau = the phase
  const voices = Array.from({ length: K + 1 }, (_, k) => ({
    op: 'instanceDecl', name: `v${k}`, program: 'ModalVoice', inputs: { tau: voicePhaseAt(k), f0: 1 },
  }))

  const instances = [
    { op: 'instanceDecl', name: 'clock', program: 'VelocityClock', inputs: { velocity: s('velocity') } },
    { op: 'instanceDecl', name: 'pphase', program: 'AnchoredPhase', inputs: { rate: s('f0'), tau: ref('clock', 'tau') } },
    { op: 'instanceDecl', name: 'ephase', program: 'AnchoredPhase', inputs: { rate: s('event_rate'), tau: ref('clock', 'tau') } },
    ...smoothers,
    ...voices,
    { op: 'instanceDecl', name: 'master', program: 'SoftClip', inputs: { input: mul(total, s('master_gain')), drive: 1 } },
  ]

  return {
    schema: 'tropical_program_2',
    name: 'AnchoredReverseReverb',
    body: {
      op: 'block',
      decls: [...paramDecls, ...instances],
      assigns: [{ op: 'outputAssign', name: 'dac.out', expr: ref('master', 'out') }],
    },
  }
}
