# Host param dispatch — the normative contract

Every runtime host (the C++ FlatRuntime, a Swift/Metal host, the wasm player)
implements param writes by reading `param_disciplines` from the plan manifest
and dispatching per slot. **No client ever chooses a write verb** — the
discipline is a fact about the compiled patch, carried by the plan, exactly
like `slot_names` and `slot_defaults`. A host that implements this section is
conformant; hosts are gated against each other differentially (same write
sequence → identical slot trajectories → identical audio).

The executor is the C++ host, `FlatRuntime::dispatch_param_sync`
(`engine/runtime/FlatRuntime.cpp`), reached by the socket data plane directly
and by the Lean engine's `set_param` through its host port
(`tropical_runtime_set_param`). The compiler declares; the host executes.
`tests/web/param_dispatch_conformance.test.ts` pins both routes against the
closed forms below. Where this document and that code disagree, this
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
  { "name": "osc.freq", "discipline": "anchor_clock",
    "companions": ["osc.freq#phase"], "clock": "master.velocity" },
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

## `anchor`, `anchor_clock` — phase-anchored frequency

The oscillator's phase on ℤ/2³² is `Φ(inc, C) + off`, with
`Φ(inc, C) = ⌊inc · C / 2¹⁶⌋ mod 2³²`, `C` the Q48.16 clock the phasor reads,
`inc = trunc(f · 2³² / SR)` and `off = trunc(φ · 2³²)`. Changing `f` alone
jumps the phase by `Φ(inc₁, C) − Φ(inc₀, C)` — a click that grows with `C`.
The write moves the `#phase` companion by exactly that jump, in the kernel's
own integer arithmetic (truncating conversions, products wrapping modulo
2⁶⁴, arithmetic shifts):

1. `C` = the clock at `now`. For `anchor` it is the sample clock `now · 2¹⁶`.
   For `anchor_clock` it is the master clock
   `trunc(tau_base · SR · 2¹⁶) + scaleClockQ(trunc(velocity · 2³²), now · 2¹⁶)`,
   where `"clock"` names the velocity param and that param's `velocity`
   discipline names the origin (`tau_base`) as its companion
   (`EmitArrow.scaleClockQ` is the exact product);
2. `off₁ = (off₀ + Φ(inc₀, C) − Φ(inc₁, C)) mod 2³²`;
3. write `#phase := off₁ / 2³²` (a multiple of 2⁻³² in `[0, 1)`, exact in an
   f64 slot), then `param:<name> := target`.

A `clock` appears exactly on an `anchor_clock` (an empty string is still a
clock), and must name a param that carries the `velocity` discipline with an
origin companion, both with slots in the plan. A discipline outside the five
named here is not written as something else. What each reader refuses, when
it reads the manifest and before any write:

| Reader | Unknown discipline | Clock off an `anchor_clock`, or missing on one | Clock that is not a velocity param with an origin |
|---|---|---|---|
| `FlatPlan.disciplineError?` (`PlanDecode`, `Kernel.ofPlan`) | yes | yes | yes |
| C++ plan parser (`NumericProgramParser`) | yes | yes | yes |
| Web `ParamWriter` constructor (port disciplines) | yes | yes | yes |
| `Kernel.Discipline.ofWire` (one port entry) | yes | yes | no: it sees one entry; the nested plan's decode checks the table |

The C++ parser also records the clock's velocity and origin slots on the
`anchor_clock` entry, so a native write looks nothing up and has no failure
path of its own.

The web writer knows `glide` and `velocity` but does not write them: a write
to such a param is refused by name. The playground declares `anchor_clock`
on every anchored knob, and its compile report (`load_patch_graph`) names
each param's discipline from the same table; plain `anchor` keeps its
sample-clock meaning. `get_vocabulary` names the authored port class
(`anchor`); the plan and the compile report name the discipline a host
dispatches (`anchor_clock`).

On the f64 lanes (the JIT and wasm) the phase is then bit-exactly continuous
at `now`: the oscillator continues from its pre-write phase with the new
increment. `tropicaltest --anchor-only` checks this in the rendered audio,
against both an exact model of the kernel (lockstep reference) and the
kernel itself (the first audible sample after the write equals the old
oscillator's sample at `now`). Known exceptions:

- Metal (known issue, both `anchor` and `anchor_clock`): slots reach the
  kernel as f32, and a slot-driven increment `trunc(f·2³²/SR)` is computed
  on the f32 value path in-kernel (`Ir/EmitMsl.lean`: only constant-folded
  landings carry exact i64 increments). The kernel's increment then differs
  from the host's f64 one by some `δ`, and the phase step at a write is
  about `(δ₁ − δ₀)·n / 2³²` cycles at sample `n`: it grows with τ, and an
  hour in it can be as large as a raw write's jump. The fix is either an
  exact increment on Metal (the param landing on the i64 rail) or refusing
  anchored disciplines on Metal plans; neither is built.
- `harmonic` partials: partial `k` takes `⌊k·P / 2¹⁶⌋` of the fundamental's
  phase with the clock's fraction bits kept (`EmitArrow.Spectral`), so when
  the clock has fraction bits (velocity ≠ 1 or a fractional origin) partial
  `k ≥ 2` can step by up to about `k` LSBs at the write. The fundamental is
  exact.
- Warped generators (`fm`, `delay`, `reverse`) read the source at a warped
  clock: a re-tune rewrites the source's history, so a tap at shift `s`
  steps by `Δinc·s`. That is the intended semantics.

If the `#phase` companion is absent from the loaded plan, degrade to `raw`.

A hot-swap does not carry `#phase`: the fresh plan seeds it with 0, so an
oscillator retuned by anchored writes jumps at the next structural edit.

### Patch-bay params

A `tropical_program_2` patch (the JSON front door, `save`, every MCP
session) gets `anchor` from two declarations, one by the type and one by the
patch, and the session compile checks one against the other.

The registered type declares, on its frequency input, which sibling inputs
are that phasor's phase offset and clock (`InputDecl.anchor?`, a
`PortAnchor`; the type's catalog entry carries it as the input's `anchor`). The stdlib
`ClockPhasor` declares `freq: { phase: "offset", clock: "clk" }`: its phase
is `Φ(trunc(freq·2³²/SR), clk) + trunc(offset·2³²)`, the formula above.

The patch opts a param in with `"discipline": "anchor"` on its `paramDecl`
(absent or `"raw"` is raw; `save` writes it back):

```json
{ "op": "paramDecl", "name": "pitch", "value": 220, "discipline": "anchor" },
{ "op": "instanceDecl", "name": "ph", "program": "ClockPhasor",
  "inputs": { "clk": { "op": "clock" }, "freq": { "op": "param", "name": "pitch" } } }
```

The compile (`Session.anchorParams`) then adds a companion param
`pitch#phase` (seeded 0), wires it into `ph.offset`, and lists
`{ name: "pitch", discipline: "anchor", companions: ["pitch#phase"] }` in
the plan. The rule is strict: an anchored param is read only bare, and only
by inputs that declare an `anchor`. The check sees one hop from the param and
cannot look inside a type, so any other reader could integrate the value as a
frequency (a `FixedSinOsc.freq`, which declares no anchor, or a `ClockPhasor`
fed through a VCA) and click on every retune unannounced. The compile
refuses, naming the wire:

- an integer param (`int32`/`int`): written raw, as `portError?` says;
- a read through an expression, or through a lifted wire (the `__wire_N`
  program an array literal becomes), into any input;
- a bare read into an input that declares no `anchor`;
- a consumer whose clock input is not wired to `{op:'clock'}` (τ).
  `anchor` is the sample-clock identity `C = now·2¹⁶`; a `ClockPhasor`
  reading a scrubbed or warped clock steps by a different amount. Such a
  phasor's pitch param is written raw (no declaration). `anchor_clock`, the
  master-clock form, needs a velocity param with an origin companion, which
  patch-bay programs do not declare; it is not declarable on a `paramDecl`
  today, nor are `glide` and `velocity`;
- a consumer whose phase input is wired (other than to its declared
  default), or claimed by another anchored param: the companion takes it;
- any read of the companion's name (`pitch#phase`), declared or not;
- the same param declared twice with different disciplines.

A pitch param that must also drive something else (a filter cutoff, a gain)
needs a second, raw param. A param declared `anchor` that nothing reads gets
no companion and no table entry: it is raw until wired, so an edit session
is not refused while its phasor is being rewired.

A sound relaxation would need two things not built: a transitive dependency
check over the session graph, for reads that reach an anchor-declaring input
through other instances; and an `anchor?` on every input a type integrates
as a frequency — at least `FixedSinOsc`, `MorphOsc` and `PluckedMorphOsc`
(the playground already anchors these through `phasePort`), plus a decision
on exported types whose phase input is not exposed.

A param fanned out bare to several anchored inputs, all on τ, shares one
companion: the bump depends on the param's value and the clock only. The
companion is not a session param: `list_params` and `save` do not show it,
and `set_param` refuses it by name. Like the playground's, an anchored
param's companion reseeds to 0 on a hot-swap, so after a retune the next
structural edit jumps (a raw param does not).

The playground does not read this declaration yet: its port table
(`Playground/VocabularyMetadata.lean`, `discipline := .anchor`) and its voice
table (`Playground/Vocabulary.lean`, `phasePort`) say the same thing for its
own node kinds, and its generators read the master clock (`anchor_clock`).
`export_program` carries `anchor?` onto an exported input when the inner
phasor's phase and clock inputs are exposed too (under their exposed names);
otherwise the exported frequency input declares no anchor.

## `velocity` — master-clock re-base

The master clock is the Q48.16 value
`M(n) = trunc(tau_base · SR · 2¹⁶) + scaleClockQ(trunc(velocity · 2³²), n · 2¹⁶)`
(`Playground/Decode.lean`, `masterClock`): `tau_base·SR + velocity·n` samples,
with 16 fraction bits. Changing `velocity` alone jumps `M` by `Δv·n` samples.
The write re-bases the origin so `M` stays value-continuous:

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
