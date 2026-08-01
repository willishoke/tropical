# Structured room-recovery feasibility

Date: 2026-08-01

Status: exploratory result; no product integration authorized.

## Question

Can a convincing stateful delay reverb be recovered economically inside
Tropical's state-free, random-access modal realm, or does an exact modal
representation necessarily pay one runtime oscillator per delay-memory sample?

The earlier room auditions did not answer this. They tested hand-authored,
independent continuous-time poles. This experiment instead derives every pole
and residue from a known stateful delay topology and then looks for algebraic
structure after decomposition.

## Literature constraints

The experiment follows three published observations:

1. An FDN's exact discrete modal order is the sum of its delay lengths. Schlecht
   and Habets give a direct modal decomposition for large FDNs and report that
   modal frequency and residue distributions expose useful structure.
2. Mode count alone is not a colorlessness criterion. Heldmann and Schlecht
   connect perceived coloration to modal excitation distribution; their survey
   cites at least 3,000 complex modes over 80 Hz–10 kHz for equally excited
   synthetic modal reverberation, with 10,000 modes used as the reference.
3. ERA and vector fitting are appropriate black-box reduction tools, but an
   exact pole/residue teacher is stronger evidence for this first pass. No
   identification error is mixed into the structural test.

Primary sources:

- S. J. Schlecht and E. A. P. Habets, “Modal Decomposition of Feedback Delay
  Networks,” IEEE TSP 67(20), 2019, DOI `10.1109/TSP.2019.2937286`.
- J. Heldmann and S. J. Schlecht, “The Role of Modal Excitation in Colorless
  Reverberation,” DAFx, 2021.
- J.-N. Juang and R. S. Pappa, “An Eigensystem Realization Algorithm for Modal
  Parameter Identification and Model Reduction,” 1985.
- B. Gustavsen and A. Semlyen, “Rational Approximation of Frequency Domain
  Responses by Vector Fitting,” IEEE TPWRD 14(3), 1999.

## Teacher

The LTI teacher is intentionally simple and inspectable:

```text
8 parallel homogeneous feedback combs
    -> 4 serial Schroeder allpasses
```

At 32 kHz it contains 9,133 delay samples and therefore 9,133 exact complex
discrete-time modes. The comb feedback gains share a four-second RT60 and the
allpass gain is 0.70. This is Freeverb-style rather than a claim of byte-level
Freeverb identity; damping filters and stereo spread are excluded so the exact
modal decomposition remains auditable.

The official Mutable Clouds render remains the perceptual gold reference. The
teacher is the mathematically controlled system used to test recoverability.

## Three representations

### Stateful teacher

The ordinary implementation performs eight feedback-comb updates and four
allpass updates for every sequential sample. It is cheap and diffuse, but it
cannot answer an arbitrary time coordinate without reconstructing prior state.

### Diagonal modal expansion

Partial fractions expand the teacher into all 9,133 poles and residues. The
result matches the stateful impulse response to `6.01e-13` relative L2 error.
This validates the decomposition while also demonstrating the bad cost model:
one independent runtime term per exact pole.

Truncating that expansion is not a good recovery strategy. Energy-ranked and
frequency-stratified subsets were tested at 64, 256, 1,024, and 3,072 complex
modes. Below 1,024 modes, spectral flatness collapses and the omitted modes
destroy the cancellations that create the teacher's delayed causal buildup.
At 3,072 modes the stratified response begins to recover diffuse statistics,
but it remains a large and temporally inaccurate runtime bank.

### Pole-lattice collection

Each scalar delay denominator contributes a complete uniform pole lattice with
one shared radius. For one such group,

```text
sum_k residue[k] * pole[k]^n = radius^n * carrier[n mod period]
```

where `carrier` is a compile-time inverse DFT of the group's residues. Thus an
entire delay's poles become one exponential envelope and one periodic table
read—not hundreds of oscillators.

For a source mode `x[n] = C p^n`, write the collected room group as
`h[k] = r^k c[k mod P]`, and let

```text
n = K P + R
z = r / p
Q = z^P
A(R) = sum_(s=0..R) c[s] z^s
A*   = A(P - 1)
```

Then its complete zero-state convolution is

```text
C p^n * (
  A(R)       * (1 - Q^(K + 1)) / (1 - Q)
  + (A*-A(R)) * (1 - Q^K)       / (1 - Q)
)
```

The apparent overflow for `|Q| > 1` is removed algebraically by evaluating
`p^n - r^(P L) p^(n-P L)` before division. Exact resonance uses the smooth
geometric limit `L p^n`.

The required `A(R)` values are build-time prefix tables. Random access reads
one entry per source-mode/room-group pair; cost is independent of delay length,
elapsed time, and requested coordinate.

## Results

| Result | Measurement |
|---|---:|
| Full modal vs stateful teacher, relative L2 | `6.01e-13` |
| Structured comb vs stateful comb, relative L2 | `1.20e-13` |
| Structured full teacher vs exact modal convolution, relative L2 | `2.72e-13` |
| Exact diagonal complex modes | 9,133 |
| Collected pole-lattice groups | 12 |
| Authored real source rows | 12 |
| Runtime source/group interactions | 144 |
| Runtime term reduction | 63.4× |
| Carrier-table samples | 10,266 |
| Source-specific prefix data at complex64 | 985,536 bytes |

The frozen demo scene contains 168 authored modal rows at 114 distinct
`(frequency, decay)` pole coordinates. Applying this full-strength teacher to
every row would require 2,016 source/group interactions. Prefix tables can be
shared across rows with the same pole, so their complex64 footprint would be
9,362,592 bytes (8.93 MiB); differing amplitudes, phases, and start times do
not require new tables. They do remain distinct runtime branches once their
anchors differ.

The master design already reserves the complete early field for the four Metal
hits. That narrower application needs 576 interactions across those hits and
reuses the same 985,536-byte prefix set measured here. Strings can retain a
cheaper late-field treatment. These are representation counts, not a GPU timing
claim; the next prototype must measure the generated expression and table-read
cost on the target path.

The official Clouds mono impulse measured spectral flatness `0.2976` at
100–300 ms and `0.1075` at 500–1000 ms. The stateful/fully recovered teacher
measured `0.3507` and `0.1882`, respectively. These numbers establish diffuse
behavior, not perceptual equivalence; listening remains mandatory.

The 3,072-mode stratified truncation measured `0.3112` and `0.0481`. It reaches
the early-field statistic while still losing more than half the late spectral
flatness and most of the exact temporal waveform. This is useful negative
evidence against ordinary pole pruning.

## Feasibility verdict

**Positive, with a narrower and more interesting representation than a modal
bank.** A fixed LTI delay reverb acting on fixed modal sources can be recovered
exactly, state-free, and random-access by collecting structured pole lattices.
For this teacher the runtime term count falls from 9,133 to 144 for the metal
source without approximation.

The result is topology-specific. It applies when the transfer decomposes into
scalar delay-denominator lattices, including this parallel-comb/serial-allpass
family. It does not yet establish the same collection for a generally coupled
FDN, arbitrary live input, live delay geometry, nonlinear quantization, or
Clouds' modulated delays.

The likely demo architecture, if listening accepts the teacher, is therefore a
new fixed **grouped modal transfer**: periodic carrier and source-specific prefix
tables plus closed-form quotient-time evaluation. It is not thousands of
`room_modes`, and it is not a hidden stateful bus.

## Next gates

1. Listening: Clouds mono gold vs exact recovered teacher. If the teacher is
   not perceptually massive, algebraic success alone is irrelevant.
2. Cost prototype: emit the 12-group/12-row formula through the current Metal
   expression/table path and measure one hit, then the four-hit 576-interaction
   scene application. The Python timing is not a realtime predictor.
3. Mutable proximity: either derive a collectable Dattorro-like LTI topology or
   tune the collectable teacher against Clouds' time/frequency targets.
4. Smear: treat slow modulation as a separate closed-form extension only after
   the frozen LTI core passes listening.

Reproducer and exact measurements:
`benchmarks/demo_release/room_audition/run_structured_recovery.py` and
`recovery_out/recovery_summary.json`.
