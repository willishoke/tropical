# Phaser

An N-stage phaser built from a series chain of first-order allpass filters, all sharing a single LFO-swept coefficient. The allpass chain delays different frequencies by different amounts without attenuating any of them; when the phase-shifted signal is mixed back with the dry signal at 50/50, frequencies where the allpass contributes 180° of shift cancel to form deep notches. The LFO continuously sweeps those notch positions up and down in frequency, producing the characteristic "swoosh."

`N` is a compile-time integer type parameter (default 4). The `Phaser16` stdlib program is an equivalent fixed-16-stage version written out explicitly.

## Signal flow

```mermaid
flowchart LR
  in([input]) --> MIX
  in --> SUM
  feedback([feedback]) --> SUM
  lfo_speed([lfo_speed]) --> PH["Phasor"]

  subgraph lfo_block ["LFO"]
    PH -->|phase| TWOPI["× 2π"] --> SIN["Sin"] --> COEF["a = 0.6 + 0.35·lfo"]
  end

  SUM(("+ fb·feedback")) --> AP

  subgraph chain ["allpass chain (stage 0 … N−1)"]
    AP["stage k: −a·x + x_prev[k] + a·y_prev[k]"] --> NEXT["stage k+1 …"]
    X[("reg x_prev[k]")] -. "next x_prev[k]" .-> X
    Y[("reg y_prev[k]")] -. "next y_prev[k]" .-> Y
  end

  NEXT -->|ys[N−1]| MIX(("0.5·dry + 0.5·wet"))
  NEXT -->|ys[N−1]| FB[("reg fb")] -. "next fb" .-> FB
  FB --> SUM
  MIX --> output([output])
  SIN -->|out| lfo([lfo])
  COEF --> AP
```

## Internals

**Allpass stage.** Each of the N stages implements the first-order allpass:

```
y_k = -a · x_k + x_prev[k] + a · y_prev[k]
```

where `x_k` is the stage's input (output of the previous stage, or the chain input for k=0), `x_prev[k]` is the one-sample-delayed version of `x_k`, and `y_prev[k]` is the one-sample-delayed output. This corresponds to the transfer function:

```
H(z) = (a + z⁻¹) / (1 + a·z⁻¹)
```

which has unit magnitude at all frequencies — it only shifts phase. The phase shift at frequency f is `−2·arctan(a·sin(ω) / (1 + a·cos(ω)))` where `ω = 2πf/rate`.

**LFO and coefficient sweep.** The LFO runs at `lfo_speed` Hz via a `Phasor` (sawtooth phase accumulator) fed through `Sin` with the full-cycle scaling `6.283185307179586` (exact double representation of 2π). The allpass coefficient is:

```
a = 0.6 + 0.35 · lfo_sin
```

Since `lfo_sin` ∈ [−1, 1], `a` sweeps between 0.25 and 0.95. At `a = 0.25` the allpass corner is near ¼ of Nyquist; at `a = 0.95` it moves close to DC, sweeping the notch positions across the audible range each LFO cycle.

**Chain via `scan`.** The N stages are not unrolled separately — they are expressed as a single `scan` over a `generate(N, (i) => i)` index array. `scan` folds left: it threads an accumulator (the chain input, then each stage's output) through the per-stage lambda, producing the array of all intermediate stage outputs `ys`. This makes the stage count a compile-time parameter `N` with no code change.

**State arrays.** `x_prev` and `y_prev` are length-N arrays initialized to `zeros(N)`. Each sample the `next` assignments update them:

- `next y_prev` is set to the full `ys` array — the output of each stage at the current sample.
- `next x_prev` is built by `generate(N, (i) => select(i > 0, ys[clamp(i-1, 0, N-1)], input + feedback * fb))`: stage 0's input is the chain input; stage k's input (k > 0) is the previous stage's current output `ys[k-1]`.

**Feedback.** Register `fb` holds the previous sample of `ys[N-1]` (the final stage output). It is added to the current input before the chain: `input + feedback * fb`. Higher `feedback` deepens and narrows the notches, approaching self-oscillation as `feedback → 1`.

**Output mix.** `output = 0.5 * input + 0.5 * ys[N-1]` is a fixed 50/50 dry/wet blend. The notch depth and phase-cancellation character are determined by this ratio; at exactly 0.5 the notches are theoretically infinitely deep for unit-magnitude allpass stages.

**Repeated `let` blocks.** The `a` and `ys` expressions are spelled out three times in the source (for `output`, `next fb`, `next y_prev`, and `next x_prev`). The strata compiler's `arrayLower` pass unrolls each independently; no shared CSE across `next` assignments is performed at the source level. This is a known verbosity of the array-combinator form — the explicit `Phaser16` variant avoids the repetition by instantiating named `_allpassStage` inner programs.

## Source

```tropical
program Phaser<N: int = 4>(
  input = 0,
  feedback = 0.4,
  lfo_speed = 0.2
) -> (output, lfo) {
  reg fb = 0
  reg x_prev = zeros(N)
  reg y_prev = zeros(N)
  lfo_ph = Phasor(freq: lfo_speed)
  lfo_sin = Sin(x: 6.283185307179586 * lfo_ph.phase)
  output = let {
    a: 0.6 + 0.35 * lfo_sin.out,
    ys: scan(generate(N, (i) => i), input + feedback * fb, (acc, i) =>
           -a * acc + x_prev[i] + a * y_prev[i])
  } in 0.5 * input + 0.5 * ys[N - 1]
  lfo = lfo_sin.out
  next fb = let {
    a: 0.6 + 0.35 * lfo_sin.out,
    ys: scan(generate(N, (i) => i), input + feedback * fb, (acc, i) =>
           -a * acc + x_prev[i] + a * y_prev[i])
  } in ys[N - 1]
  next y_prev = let { a: 0.6 + 0.35 * lfo_sin.out } in
    scan(generate(N, (i) => i), input + feedback * fb, (acc, i) =>
      -a * acc + x_prev[i] + a * y_prev[i])
  next x_prev = let {
    a: 0.6 + 0.35 * lfo_sin.out,
    ys: scan(generate(N, (i) => i), input + feedback * fb, (acc, i) =>
           -a * acc + x_prev[i] + a * y_prev[i])
  } in generate(N, (i) =>
    select(i > 0, ys[clamp(i - 1, 0, N - 1)], input + feedback * fb))
}
```
