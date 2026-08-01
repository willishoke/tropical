# Clouds-nearer grouped carrier fit

Date: 2026-08-01

Status: 32 kHz `current_radii` mono listening accepted for MVP; production
regeneration and integration are governed by
`06-room-position-production-handoff.md`.

## Question

Can the accepted exact grouped-modal representation move materially closer to
Mutable Clouds without increasing the 12-group runtime structure?

## Method

The 12 periods proven in the structured recovery experiment are retained. Each
complete pole lattice already permits arbitrary residues, equivalently an
arbitrary real periodic carrier. Instead of deriving those carriers from the
Freeverb-style topology, this experiment fits their 10,266 samples offline to
the official Clouds mono impulse with log-time-weighted least squares.

This is still a stable fixed LTI modal transfer. The fit changes frozen carrier
assets, not the random-access evaluator. Analytic convolution against the exact
12-row Metal source matches ordinary convolution to `3.46e-13` relative L2.

Clouds' side waveform is deliberately nonstationary because its tank and delay
reads are modulated. Direct stationary waveform fitting therefore projected
almost none of that channel. The stereo candidate instead preserves each
lattice's fitted modal magnitudes and deterministically scrambles residue phase
for the side carrier, scaled to the gold side/mid energy in the diffuse field.
This targets spatial statistics honestly rather than claiming to recover the
LFO trajectory. The side evaluator matches ordinary convolution to `3.62e-13`.

## Results

| Measurement | Clouds gold | Freeverb teacher | Fitted grouped candidate |
|---|---:|---:|---:|
| Echo density at 20 ms | `0.2608` | `0.0286` | `0.2537` |
| Echo density at 100 ms | `0.6146` | `0.8540` | `0.8075` |
| Echo density at 500 ms | `1.0469` | `0.9147` | `0.9826` |
| Spectral flatness, 100–300 ms | `0.2976` | `0.3507` | `0.2410` |
| Spectral flatness, 500–1000 ms | `0.1075` | `0.1882` | `0.1000` |
| L/R correlation, 100–300 ms | `0.0032` | mono | `0.0132` |
| Side/mid RMS, 100–300 ms | `0.9968` | mono | `0.9869` |
| L/R correlation, 500–1000 ms | `0.0153` | mono | `0.0042` |
| Side/mid RMS, 500–1000 ms | `0.9848` | mono | `0.9958` |

The direct waveform NRMSE remains high. That is expected and desirable as a
scope boundary: a stationary kernel should match the decay, density, spectrum,
and spatial distribution, not overfit one realization of a modulated noise-like
tail.

A second decay-ladder fit slightly improved the sampled echo-density score but
reduced late spectral flatness to `0.0494` and worsened time/band energy
distance. It remains a negative control; the current-radii fit is the audition
candidate.

## Cost

| Cost | Mono | Stereo |
|---|---:|---:|
| Pole-lattice groups | 12 | 12 shared |
| Source/group interactions per Metal hit | 144 | 144 vector-output |
| Four-hit interactions | 576 | 576 vector-output |
| Carrier samples | 10,266 | 20,532 scalar values |
| Source-specific complex64 prefix data | 985,536 bytes | 1,971,072 bytes |

Stereo shares quotient, pole power, and geometric evaluations, but it requires
two prefix reads and two accumulators. These are representation counts, not a
Metal timing result.

## Design consequence

The highest-leverage representation is now a **reference-fitted grouped modal
transfer**, not a literal modalization of a canonical reverb topology. Complete
lattice residues are frozen offline assets. A deterministic side-phase asset
can provide spatial diffusion without another pole bank or live modulation.

The user accepted the mono `current_radii` impulse as the MVP room on
2026-08-01. Product integration still requires:

1. the 12-group vector-output formula passes the target Metal cost probe;
2. the fixed kernel remains convincing at all four authored hit anchors; and
3. the integrated stereo/final-scene capture passes listening.

The transfer has infinite mathematical support; the 12-second WAV is only an
audition truncation. Lengthening RT60 changes the per-group radii and requires a
new carrier fit, but does not add groups, interactions, or table samples.
Continuously changing room size is different: lattice periods and prefix tables
are structural. The bounded post-MVP path is a small bank of precomputed sizes
with output morphing, not live mutation of one exact lattice.

Reproducer and assets:

- `benchmarks/demo_release/room_audition/run_grouped_clouds_fit.py`
- `benchmarks/demo_release/room_audition/grouped_fit_out/grouped_fit_summary.json`
- `benchmarks/demo_release/room_audition/grouped_fit_out/*.npz`
- `benchmarks/demo_release/room_audition/grouped_fit_out/*.wav`
