# Host param dispatch — the normative contract

Every runtime host (the C++ FlatRuntime, a Swift/Metal host, the wasm player)
implements param writes by reading `param_disciplines` from the plan manifest
and dispatching per slot. **No client ever chooses a write verb** — the
discipline is a fact about the compiled patch, carried by the plan, exactly
like `slot_names` and `slot_defaults`. A host that implements this section is
conformant; hosts are gated against each other differentially (same write
sequence → identical slot trajectories → identical audio).

The reference implementation is the Lean control plane
(`lean/Tropical/Engine.lean`, `handleSetParamGlide` / `handleSetParamFreq` /
`handleSetParamVelocity`). Where this document and that code disagree, this
document is the intent and the code is the bug.

## The manifest field

```json
"param_disciplines": [
  { "name": "master.velocity", "discipline": "velocity",
    "companions": ["master.tau_base"] },
  { "name": "sf.depth", "discipline": "glide", "glide_dur_sec": 0.02,
    "companions": ["sf.depth#v0", "sf.depth#v1", "sf.depth#t0"] },
  { "name": "osc.freq", "discipline": "anchor",
    "companions": ["osc.freq#phase"] },
  { "name": "res.decay", "discipline": "raw" }
]
```

Omitted when empty; a host that does not read it treats every write as `raw`
(old plans, old hosts: both unaffected). Slot names are `param:<name>`.

All disciplines below are **stateless re-anchorings**: the "state" they touch
lives in param slots the kernel reads as pure functions of τ — navigable,
transferable by name across hot-swap, exactly like everything else. A write
needs two ambient values from the host: `now` (the first output sample at
which the transaction's captured generation is audible) and `SR` (the
kernel's sample rate).

For a callback starting at `C`, synchronous JIT/Metal uses `now = C`. A
steady-state future-block Metal pipeline of depth `D` uses `now = C + D·B`,
where `B` is the buffer length. Fresh kernels, hot-swaps, and clock-jump
re-primes are exceptions: their ring is rebuilt from the current capture, so
`now = C` at every depth. Publication racing a callback is linearized against
generation capture: a transaction that wins is stamped for that callback; a
transaction that loses is recomputed for the next capture. The audio callback
must not wait, retry, allocate, or emit silence because control is publishing.

Production dispatch evidence reports both `observed_sample_index` (the last
completed callback boundary visible when dispatch began, diagnostic only) and
`effective_sample_index` (the normative `now` used by discipline math and the
first audible output index). Qualification replays the latter.

## `raw`

Write `value` to `param:<name>`. Nothing else.

## `glide` — closed-form smoothstep re-anchor

The kernel evaluates
`f(τ) = v0 + (v1 − v0) · s²(3 − 2s)`, `s = clamp((τ − t0)/dur, 0, 1)`,
`dur = glide_dur_sec · SR` samples, from the three companion slots. A write
re-anchors the ramp so it departs from the CURRENT value (no jump):

1. read `v0, v1, t0`;
2. `curr = f(now)` (the formula above, evaluated at `now`);
3. write `v0 := curr`, `v1 := target`, `t0 := now`.

The base slot `param:<name>` is not written (glided knobs have no base slot;
the companions are the value).

## `anchor` — phase-anchored frequency

The oscillator's phase is `f·τ + φ_off`; changing `f` alone jumps the phase by
`Δf·τ` — a click that grows with τ. The write bumps the `#phase` companion by
the phase the change would have jumped, using the phasor's own quantized
increments so the correction is exact against the kernel's arithmetic:

1. `inc₀ = ⌊f_current · 2³² / SR⌋`, `inc₁ = ⌊target · 2³² / SR⌋`;
2. `Δφ = (inc₀ − inc₁) · now / 2³²` (cycles);
3. `φ := frac(φ + Δφ)` (wrap to `[0, 1)`), then write `param:<name> := target`.

If the `#phase` companion is absent from the loaded plan, degrade to `raw`.

## `velocity` — master-clock re-base

The master clock is `M(n) = tau_base·SR·2³² + velocity·2³²·n`; changing
`velocity` alone jumps `M` by `Δv·n`. The write re-bases the origin so `M`
stays value-continuous:

1. `tau_base += (v_current − target) · now / SR` (the companion slot);
2. write `param:<name> := target`.

## Conformance

- The Lean control plane and the C++ socket data-plane dispatch from the SAME
  manifest table; a differential gate drives an identical write sequence
  through both and requires identical slot trajectories.
- A host that rounds differently, evaluates the glide with a different
  polynomial, or applies the phase bump un-quantized will pass casual
  listening and fail the differential — that is what the gate is for.
- Current hosts expose `observed_sample_index` and `effective_sample_index`;
  the retired `applied_sample_index` key in retained historical rows meant the
  completed boundary observed by the older host and is not current evidence.
