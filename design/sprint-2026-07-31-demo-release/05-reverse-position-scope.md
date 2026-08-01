# Reverse-room temporal position scope

Date: 2026-08-01

Status: exact feasibility demonstrated and POSITION selected for production.
Native-rate asset, mono scope, implementation seam, and qualification are
governed by `06-room-position-production-handoff.md`; stereo costs below remain
feasibility evidence rather than the production budget.

## Product claim

The dry modal scene continues forward while the room field can be scrubbed from
a future-aware reverse pre-tail, through a two-sided bloom, to the accepted
causal forward tail. This is materially different from reversing playback of
the whole scene.

The proposed live control is:

```text
POSITION = -1   classic reverse-reverb pre-tail
POSITION =  0   equal-power two-sided bloom
POSITION = +1   accepted forward room
```

It replaces the removed `LENGTH` row. `ROOM` remains the wet level and `FLOW`
remains whole-scene transport direction.

## Exact operation

For a causal source mode and one grouped room lattice,

```text
x[m] = C p^m u[m]
h[k] = r^k c[k mod P] u[k]
```

the requested operation

```text
reverse(x) -> h -> reverse(result)
```

equals `x` convolved with the anti-causal kernel `h[-k]`. At output coordinate
`n` relative to the source anchor, define

```text
a = max(n, 0)
d = max(-n, 0)
z = p r
B[R] = sum_(s=0..P-1) c[(R+s) mod P] z^s
```

Then the infinite stable future sum is

```text
y_reverse[n] = C p^a r^d B[d mod P] / (1 - z^P)
```

`B` is a frozen source/group cyclic-prefix table. Runtime work is one indexed
read and bounded complex arithmetic per source/group pair. No reverse buffer,
sequential delay state, or future sample generation is required; the future
source is the analytic modal continuation already present in the graph.

The continuous control uses an equal-power temporal pan:

```text
g_reverse = sqrt((1 - POSITION) / 2)
g_forward = sqrt((1 + POSITION) / 2)
wet = g_reverse * reverse + g_forward * forward
```

This moves the wet-energy centroid continuously. It does not claim that the
middle is one rigidly translated tail; the center is deliberately two-sided.

## Proof results

The exact 12-group `current_radii` fitted asset and exact 12-row Metal source
were tested over a 12-second source and room window.

| Gate | Result |
|---|---:|
| Analytic mid vs literal reverse-source/reverb/reverse, relative NRMSE | `1.88e-10` |
| Analytic mid max absolute error | `1.73e-11` |
| Analytic stereo side vs literal operation, relative NRMSE | `2.55e-10` |
| Analytic stereo side max absolute error | `1.88e-11` |
| Existing whole-composite mirror vs classic operation, NRMSE | `1.0` |

The `1e-10` residual is the finite 12-second reference truncation floor. The
incumbent `dir` lowering is decisively the wrong operation: it mirrors the
already-composed source plus room, whereas the requested effect keeps the
source forward and reverses only the room kernel. It must not be reused.

Measured mono wet placement:

| POSITION | Pre-event energy | Energy centroid relative to hit |
|---:|---:|---:|
| `-1.0` | `56.86%` | `-112 ms` |
| `-0.5` | `51.94%` | `-46 ms` |
| `0.0` | `35.84%` | `+63 ms` |
| `+0.5` | `17.31%` | `+174 ms` |
| `+1.0` | `0%` | `+249 ms` |

The reverse wet field retains some post-anchor energy because the authored
metal source itself has a causal modal body. That is correct for the classic
operation, not an error or a hidden dry path.

The `-112 ms` reverse centroid is an audible-strength risk: exactness alone does
not prove the selling moment. If listening finds it too subtle, the admissible
next levers are a longer fitted room decay or a more transient-only Metal room
send. Neither requires the incorrect whole-composite mirror, and both must be
auditioned before changing the accepted dry hit.

## Repository feasibility

The current production compiler already demonstrates every primitive needed by
the new lowering:

- signed relative clocks and audible negative-time evaluation;
- causal/anti-causal gates and live direction mixing;
- banked table reductions in JIT and Metal;
- 20 ms stateless glided parameters;
- arbitrary seek/render-window coordinates.

The existing `tropicaltest` suite passed `122/122`, including `negative-clock`,
`banks-as-data-dir`, `modal-direction`, `modal-patch`, MSL bank emission, live
slot, and seek/address gates. This proves backend expressibility, not the new
room's performance.

## Cost and implementation choice

### Direct analytic path — preferred

| Cost | One direction | Live intermediate POSITION |
|---|---:|---:|
| Source/group arithmetic per Metal hit | 144 | 288 equivalent |
| Four Metal hits | 576 | 1,152 equivalent |
| Mono forward prefix data | 0.94 MiB | retained |
| Mono reverse cyclic-prefix data | 0.94 MiB | retained |
| Stereo forward + reverse prefix data | — | 3.76 MiB |

Implementation should use one source/group loop with a two-direction result so
period, source-pole, and loop overhead are shared. A dynamic parameter normally
keeps both arms present even at an endpoint; endpoint specialization is not
part of the performance claim.

### Fixed-scene cached fallback

If the direct dual path misses the Metal reserve, render immutable forward and
reverse wet bases for the exact 16-second stereo Metal lane at load time. Two
float32 stereo bases require 11,289,600 bytes (10.77 MiB). POSITION then costs
two addressable reads, two multiplies, and one sum per output sample. This is
valid only because the Metal source, hit anchors, and room are frozen; changing
any of them requires regenerating the bases.

The fallback remains state-free and seekable, but the direct analytic path is
the stronger product demonstration and gets the first cost probe.

## Demo timing

Metal hits are anchored at 2, 6, 10, and 14 seconds. The first reverse tail is
necessarily clipped by the scene's zero boundary. The selling gesture should
target the second hit:

1. leave POSITION at `+1` for the first forward hit;
2. between approximately 3 and 4.5 seconds, scrub POSITION toward `-1`;
3. the analytically known pre-tail grows toward the hit at 6 seconds while the
   chords and transport continue forward;
4. return through `0` toward `+1` before a later hit to expose the continuum.

A control write cannot revise samples the DAC already emitted. It can reveal
the remainder of an upcoming known pre-tail immediately and all of the next
scheduled tail. Unscheduled live input would require lookahead and is outside
the claim.

## Implementation blocks

1. **Grouped reverse asset and oracle**
   - Serialize mid/side cyclic future tables alongside forward prefixes.
   - Keep the Python literal-operation oracle and frozen hashes.
2. **Lean grouped-transfer lowering**
   - Add the new grouped room representation required by DR-12.
   - Emit forward and classic anti-causal arms in one banked source/group loop.
   - Do not route through incumbent `modalBankSigDir`.
3. **POSITION parameter**
   - Add one `glideknob`, range `[-1,+1]`, default `+1`.
   - Replace the removed LENGTH row; preserve ROOM and FLOW semantics.
4. **Scene route**
   - Apply full bidirectional grouped room to the four fixed Metal hits only.
   - Keep the string-room lane cheaper and forward-only for this sprint.
5. **Metal cost fork**
   - Measure the direct dual arm on the release Mac.
   - If it misses reserve, use the 10.77 MiB fixed-scene basis fallback without
     changing audible semantics.
6. **Qualification and presentation**
   - Add endpoint, midpoint, monotonic-placement, seek-before-hit, gesture,
     JIT/Metal parity, fault, headroom, and listening gates.
   - Script the second-hit selling gesture in the demo notes.

## Hard gates

- `POSITION=+1` matches the accepted forward room within backend tolerance.
- `POSITION=-1` matches the literal classic reference, not whole-output mirror.
- Dry Metal remains causal and forward at every POSITION.
- Five fixed POSITION values move pre-event energy and centroid monotonically.
- A seek to four seconds at `POSITION=-1` produces finite nonzero wet output
  before the six-second dry hit.
- A live scrub is click-free and preserves the first/reversal/final gesture
  contract.
- Direct or cached path passes the existing zero-fault Metal smoke and soak.
- Final stereo scene capture passes user listening.
- The reverse endpoint is unmistakable against the dry hit; if it is not, stop
  integration and audition the longer-room/transient-send fork.

## Evidence

- Reproducer: `benchmarks/demo_release/room_audition/run_reverse_position.py`
- Measurements: `benchmarks/demo_release/room_audition/reverse_position_out/reverse_position_summary.json`
- Mono and stereo auditions: `benchmarks/demo_release/room_audition/reverse_position_out/*.wav`
