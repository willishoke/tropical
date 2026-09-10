# Modal source through a resonant lowpass — the composition benchmark

The bare-oscillator table (`../faust_comparison`) prices per-voice synthesis;
nothing in it exercises COMPOSITION, which is where tropical's design is
supposed to pay. This fixture measures composition directly: P voices of an
M-partial modal source, each through a 2-pole resonant lowpass.

## Run

```sh
make build && make lean
brew install faust
benchmarks/modal_filter/run.py            # everything
benchmarks/modal_filter/run.py cost       # metrics 1+2 only
benchmarks/modal_filter/run.py sweep      # metric 3 only
```

## Method

Three metrics, per the design handoff:

1. **Marginal cost of the filter** — `cost(source→filter) − cost(source)`,
   per system. Cancels each system's source realization and isolates what
   adding a filter costs. tropical composes the filter into the modal
   coefficient plane (pole/residue partial fractions, run at CONTROL-WRITE
   time in the coefficient kernel); Faust's `fi.svf` filters the summed
   signal at O(1) per sample.
2. **Total cost**, alongside — marginal alone is exactly the framing that
   would make this a rigged benchmark.
3. **Quality under cutoff sweep**, each system against ITS OWN finer-grained
   reference:
   - tropical: the same graph with the knob written every SAMPLE
     (`diffcli render-sweep --buffer 1`) vs written once per block. Prices
     coefficient-update quantization — settled knobs are coefficient-time.
     The static row is exact (`inf`): the closed-form filter has no
     discretization to be wrong about.
   - Faust: the same dsp at 4× the rate, compared at common instants (the
     sweep runs on absolute time). Prices the svf's discretization; its
     static row is the floor to read the sweep rows against.

## Fairness commitments

- **Per-voice filters on both sides** (each voice its own lowpass).
- **`fi.svf`, not a direct-form biquad** — direct-form under modulation is a
  strawman.
- **The source mirrors `resonatorBank`'s partial law exactly**
  (freq `k·f0`, decay `decay·(1+0.4k)`, amp `k^-1.1`; filter Q =
  `0.55·80^res`) but in each system's own paradigm: Faust idiomatic-recurrent
  (os.osc accumulators, one-pole envelopes), tropical closed-form. Metric 1
  differences the source out; metric 2 carries the paradigm difference and
  says so. The metric-3 Faust variant uses a CLOSED-FORM source so the 4×
  reference isolates the filter (a recurrent source decorrelates between
  rates and swamps the measurement).
- **Losses are published next to wins.** See `findings.md`.

## Traps hit while building this (kept so nobody re-hits them)

- **Faust's `ma.SR` clamps to [1, 192000].** An "8×" render at 352.8 kHz
  silently computes absolute time against 192 kHz and produces a different
  signal entirely — fully uncorrelated, −4.5 dB at STATIC cutoff, M- and
  Q-independent. The tell: `cos(...)` sampling to exactly 0.5000. The
  reference is 4× (176.4 kHz), under the clamp.
- **A recurrent source is not rate-portable.** os.osc phase accumulators and
  one-pole envelopes drift between sample rates over a 2 s render; only an
  absolute-time closed-form source lets two rates be compared sample-wise.
- **Marginal-by-differencing assumes compiler separability.** Faust's bare
  M=64 source optimizes to ~1140 ns/partial while the filtered variant sits
  at ~2250 (identical vector-op counts) — the marginal at that point measures
  the optimizer's mood, not the svf. Read the M=16 marginal for the svf's
  real cost.
