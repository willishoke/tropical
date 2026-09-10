# Modal source → resonant lowpass — findings

**Date:** 2026-09-05 · **Host:** Apple M1 Pro, macOS 26.3 · **Faust:** 2.85.9,
Homebrew LLVM 22.1.7 (the same LLVM the tropical JIT uses)
**Fixture:** P voices × M-partial modal source (`resonatorBank`'s law) → 2-pole
resonant lowpass (`filter`, Q≈4.92 ↔ `fi.svf.lp`), 512-frame blocks @ 44.1 kHz
**Rows:** [`data/`](data/) (two full runs; spreads quoted below are across runs)
**Compiler state:** `feat/modal-s0-compose` — settled coefficients + WS3b, i.e.
the composition runs at control-write time in the coefficient kernel.

## Metric 1 — marginal cost of the filter

```
                       tropical                      faust (fi.svf)
  P=1 M=16    +17.2k..22.9k ns  (+15..22%)     +4.0k ns (+12.5%)   [stable]
  P=1 M=64    +31.8k..46.8k ns   (+8..12%)     UNRELIABLE (see below)
  P=8 M=16    +221k..222k ns    (+28.4%)       −25k..+17k ns       [swings]
  P=8 M=64    +230k..269k ns     (+7.4..8.7%)  +144k..−54k ns      [swings]
```

**tropical's marginal is flat in M.** Cleanest at P=8 (best noise floor):
27.7k ns/voice at M=16 vs 28.8k at M=64 — the partial count quadrupled and the
filter's audio-rate price did not move, so the marginal FRACTION falls
(28.4% → 7.4%). This is the handoff's central prediction — "a fixed additive
constant, approaching zero as a fraction as M grows" — measured true. It was
false by ~65× before the composition landed in the coefficient kernel
(`preflight/FINDINGS.md`); the per-sample residue is 249 instructions.

**Faust's marginal at larger shapes is not measurable by differencing.** The
compiled binaries are bimodal: bare-vs-filtered at P=1 M=64 read
72.8k/131.2k in one compile session and 126.5k/72.3k in the next — the same
two values, swapped — with identical vector-op counts. Per-binary layout
luck; medians over blocks cannot average away what is fixed at compile time.
The one stable cell (P=1 M=16: +4.0k ns, +12.5%) is the svf's real per-voice
price, and it is CHEAP — a handful of FLOPs per sample.

Reading them together honestly: per voice, tropical's filter marginal
(~28k ns/block ≈ 55 ns/sample) is ~7× Faust's (~4k ≈ 8 ns/sample) in absolute
terms — the coefficient kernel's composition is not free, and this bench
re-materializes nothing (a static filter pays it once, but the audio-side
column/select reads remain. Where tropical wins is SCALING: the price does
not grow with the source's richness, while metric 2 shows where each system's
total actually lands.

## Metric 2 — totals (the number a user pays)

```
                bare              filtered
  P=1 M=64   tropical 408k     tropical 454k
             faust     73..127k faust    72..131k     (bimodal, see above)
  P=8 M=64   tropical 3.12M    tropical 3.35M
             faust    349k     faust    493k
```

Faust's TOTALS are ~4–7× cheaper on this fixture shape. The gap is the
SOURCE, not the filter: os.osc is a table oscillator (~13 effective bits —
see `../faust_comparison/findings.md`) with one-pole envelopes, against
tropical's closed-form exp + Q31 sine per partial (~28 bits, random-access,
warp-closed). Same accuracy-class trade as the bare-oscillator benchmark, and
it dominates the totals here exactly as it did there.

## Metric 3 — quality under cutoff sweep (±2 octaves), SNR dB vs own reference

```
   rate Hz   tropical b512   tropical b64   faust 1x vs 4x
      0          inf             inf            38.5     ← its floor
      2         10.3            26.9            38.2
     20          0.2             9.2            42.0
```

**Both halves of the original prediction ("expect tropical flat, Faust
degrading with rate") are refuted.**

- Faust's svf does NOT degrade with sweep rate: per-sample coefficient
  updates track a ±2-octave 20 Hz sweep as well as it renders anything. Its
  ~38–42 dB is a discretization floor vs its own 4× render, present at
  STATIC cutoff — the baseline to read its rows against, not a sweep artifact.
- tropical is EXACT when static — `inf` is not a rounding of "very good":
  block size cannot change a closed-form filter's output because there is no
  discretization anywhere in it. But under a swept knob it pays coefficient-
  update quantization hard: 10.3 dB at 2 Hz with block-rate (86 Hz) updates,
  0.2 dB at 20 Hz. Finer updates (64-sample) buy ~17 dB. This is the settle
  tradeoff priced: settled knobs are coefficient-time by design
  (`Bank.settled?`), and a per-sample modulation path exists in the compiler
  (the un-settleable decline) but the vocabulary today has no way to wire an
  LFO into `filter.cutoff` — the knob is control-plane only.

**Verdict by use:** static and stepped filters — tropical, exactly. Fast
continuous cutoff sweeps — Faust, clearly, until either the update
granularity is raised (the b64 column is the dial) or a per-sample modal
modulation path is exposed. The two systems fail in orthogonal ways: exact
but stepped vs smooth but discretized.

## What went which way (the handoff's closing question)

- metric 1: tropical — the composition claim is real, flat marginal in M.
- metric 2: Faust — the source paradigm gap (accuracy-classed) dominates.
- metric 3: split — tropical exact at rest, Faust better under sweeps.

The traps hit while building this (ma.SR's 192 kHz clamp, recurrent sources
not being rate-portable, marginal-by-differencing vs compiler bimodality) are
in [`README.md`](README.md).
