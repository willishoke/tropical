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
  { "name": "sf.depth", "discipline": "glide", "glide_dur_sec": 0.01,
    "companions": ["sf.depth#v0", "sf.depth#v1", "sf.depth#t0",
      "sf.depth#t0#u0", "sf.depth#t0#u1",
      "sf.depth#t0#u2", "sf.depth#t0#u3"] },
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
needs two ambient values from the host: `now` (the exact source-sample epoch
at which the transaction's captured generation becomes audible) and `SR`
(the kernel's sample rate).

For the JIT, a control transaction captured at callback boundary `C` uses
`now = C`. Metal is a two-phase handoff: the host reserves a future physical
device frame, maps that frame to an exact source epoch `E`, evaluates every
discipline companion at `E`, and gives one immutable snapshot to the render
worker. The old snapshot is audible strictly before `E`; the new snapshot is
audible starting at `E`. If the worker cannot safely publish the prepared
window for the reserved frame, it retargets a later physical frame and the
host recomputes the whole transaction at the new `E`; it never reuses
companions calculated for the missed epoch.

Live controls and clock jumps reserve one render tile of lead and publish once
that prime tile is ready; the worker fills the rest of the inactive bank during
the lead. Fresh loads and graph hot-swaps retain a complete-bank lead and prime
because replacing executable topology has a larger safety requirement.
Running audio serializes control epochs through callback acknowledgement so a
later glide cannot reuse the projected state of a cancelled epoch. With the DAC
stopped, unclaimed epochs may coalesce because no callback can acknowledge them.

Raw writes, glides, anchors, velocity changes, clock jumps, and hot-swaps all
use the same exact-epoch rule. Device frames remain monotonic through a clock
jump while the source coordinate changes to the requested index. Publication
racing the callback is linearized by a bounded activation descriptor read.
An unstable descriptor is deferred to a later callback; a missing or
mismatched prepared tile is replaced with fault silence and a sticky
diagnostic. The callback does not wait, retry, allocate, submit Metal work, or
pack control state.

Production dispatch evidence reports both `observed_sample_index` (the last
completed source boundary visible when dispatch began, diagnostic only) and
`effective_sample_index` (the normative exact epoch `E` used by discipline
math and the first audible output index). Qualification replays the latter.

## `raw`

Write `value` to `param:<name>`. Nothing else.

## `glide` — closed-form smoothstep re-anchor

The kernel evaluates
`f(τ) = v0 + (v1 − v0) · s²(3 − 2s)`,
`s = clamp(float(τ − t0_exact)/dur, 0, 1)`,
`dur = glide_dur_sec · SR` samples. `t0_exact` is reconstructed as a signed
i64 from four little-endian 16-bit limbs. Each limb is exactly representable
in both the JIT's f64 slots and Metal's f32 snapshot, so the absolute
timestamps are subtracted on the integer clock rail before the bounded
elapsed delta enters float arithmetic. `#t0` remains the host-readable scalar
companion and backward-compatible fallback for plans without limbs. A write
re-anchors the ramp so it departs from the CURRENT value (no jump):

1. read `v0`, `v1`, and `t0_exact` (or scalar `t0` for a legacy plan);
2. `curr = f(now)` (the formula above, evaluated at `now`);
3. write `v0 := curr`, `v1 := target`, `t0 := now`, and the four exact
   16-bit limbs of `now`.

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
