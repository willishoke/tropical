# stdlib/

The 33 built-in DSP program types, written in tropical's `.trop` surface
syntax. Loaded by `loadStdlib()` (`compiler/program.ts`); for the browser
build the same files are inlined into `compiler/stdlib_bundled.ts` by
`web/bundle_stdlib.ts`.

Every program in this directory is *just code* — no privileged primitives.
Math functions are polynomial approximations defined in `.trop`; filters
compose from one-pole sections; effects are graphs of these. Swap a file
to change the math.

## Compilation

Each `.trop` file goes through the same pipeline as a user-defined program:

```
parse  →  elaborate  →  strata  →  ResolvedProgram
```

The strata pipeline (`compiler/ir/strata.ts`) progressively retires
structure — type parameters, sum types, cycles, instance nesting,
shapes, combinators — until the program is a graph of scalar decls
ready for any backend (`compileResolved` → JIT, `interpret_resolved`,
`emit_wasm`). See `compiler/CLAUDE.md` for the per-stratum description.

## Catalog

The catalogue is stratified into four tiers. Each tier may use the
tiers below it but never the tiers above. Reading top-down, every row
is *strictly poorer in primitives* than its dependencies — the same
direction the strata pipeline runs.

```
                    ┌───────────────┐
                    │  Composites   │   Phaser, LadderFilter, Bubble, …
                    └───────┬───────┘
                            │ instantiates
                    ┌───────▼───────┐
                    │ DSP primitives│   OnePole, Delay<N>, SinOsc, …
                    └───────┬───────┘
                            │ may call
                    ┌───────▼───────┐
                    │     Math      │   Sin, Tanh, Exp, Log, …
                    └───────┬───────┘
                            │ written in
                    ┌───────▼───────┐
                    │   Builtins    │   +  *  clamp  select  ldexp  …
                    └───────────────┘
```

### Builtins

The closed set of expression-IR ops the parser knows about. These are
the only things in the system that aren't written in `.trop`. The
authoritative list is `WireFormatOp` in `compiler/expr.ts`; the
user-facing subset (the ops a `.trop` author can write) is:

| Group | Ops |
|-------|-----|
| Arithmetic       | `+` `-` `*` `/` `%` `//` `ldexp` |
| Comparison       | `<` `<=` `>` `>=` `==` `!=` |
| Logical          | `&&` `\|\|` `!` |
| Bitwise          | `&` `\|` `^` `<<` `>>` `~` |
| Unary math       | unary `-`, `abs` `sqrt` `floor` `ceil` `round` `floatExponent` |
| Conversions      | `toInt` `toBool` `toFloat` |
| Selection        | `select(cond, a, b)`, `clamp(x, lo, hi)` |
| Arrays           | array literals `[…]`, `index`, `arraySet` |
| Ambient leaves   | numeric/bool constants, `sampleRate()`, `sampleIndex()` |
| Combinators      | `let`, `fold`, `scan`, `generate`, `iterate`, `chain`, `map2`, `zipWith` (lowered by `arrayLower`) |
| ADTs             | `tag`, `match` (lowered by `sumLower`) |

Wiring/structural ops (`ref`, `param`, `delay`, the `*Decl` family,
etc.) exist in `WireFormatOp` but aren't callable as functions in
source — the parser produces them from syntax.

### Math

Pure scalar functions over the builtins. Polynomial / rational
approximations live in the `.trop` source — change the coefficients
and the JIT picks up the new approximation on the next build.

| Type   | Ports | Notes |
|--------|-------|-------|
| `Sin`  | `(x: float) → out` | Range-reduce by `n = round(x/π)`, sign flip on odd `n`, 5-term Horner on `r²`. |
| `Cos`  | `(x: float) → out` | `Sin(x + π/2)`. |
| `Tanh` | `(x: float) → out` | Padé `c·(27 + c²) / (27 + 9c²)` after `clamp(x, -3, 3)`. |
| `Exp`  | `(x: float) → out` | Cody-Waite range reduction `clamp(x, -87, 88)`, 6-term Horner, final `ldexp(_, n)`. |
| `Log`  | `(x: float) → out` | `floatExponent` for the integer part, 14-term polynomial on `m - 1`. |
| `Pow`  | `(x: float, y: float) → out` | `Exp(y · Log(x))`. |

### DSP primitives

Foundational signal-processing blocks. Each does one job and may call
into Math. They're the leaves of any patch graph — the one shared
sub-primitive is `Phasor`, the phase core the oscillators compose.
Sub-grouped by role.

#### Filters and shapers

| Type         | Ports |
|--------------|-------|
| `OnePole`    | `(input: signal, g: float) → out`. One-pole IIR with `Tanh` saturation on input and state. |
| `SoftClip`   | `(input: signal, drive: float) → out`. `Tanh(drive · input)`. |
| `SVF`        | `(input, cutoff: freq, q: float) → (lp, bp, hp)`. ZDF state-variable filter. |
| `BitCrusher` | `(audio, bit_depth, sample_rate_hz) → output`. Quantization + sample-rate decimation. |

#### Delays

| Type           | Ports |
|----------------|-------|
| `AllpassDelay` | `(input: signal, coeff: float) → out`. First-order allpass, transposed direct form II. |
| `CombDelay`    | `(input: signal, feedback: float) → out`. Single-tap feedback comb. |
| `Delay<N>`     | `<N: int = 44100>(x) → y breaks_cycles`. Generic ring buffer of length `N`; the `breaks_cycles` flag tells `traceCycles` it's a feedback-safe boundary. |

#### Oscillators

All oscillators share a phase-accumulating core, `Phasor` — `phase` advances
by `freq/sampleRate` each sample and wraps to `[0,1)`. This is correct under
frequency modulation (the absolute-sample-count form `sampleIndex · freq` is
not: modulating `freq` makes instantaneous frequency blow up by the unbounded
`sampleIndex` multiplier and smears into broadband noise).

| Type     | Ports |
|----------|-------|
| `Phasor` | `(freq: freq) → phase: unipolar`. The shared accumulating-phase core; `reg p; next p = (p + freq/sr) % 1`. |
| `SinOsc` | `(freq: freq) → sine`. `Sin(2π · Phasor(freq).phase)`. |
| `BlepSaw`| `(freq: freq) → saw`. Polynomial-BLEP-corrected sawtooth (no aliasing at the discontinuity), over `Phasor`'s phase. |

#### Noise

| Type         | Ports |
|--------------|-------|
| `WhiteNoise` | `() → out: float`. xorshift64 noise. |
| `NoiseLFSR`  | `(clock) → out: signal`. 16-bit LFSR clocked by an external trigger. |

#### Envelopes and triggers

| Type           | Ports |
|----------------|-------|
| `EnvExpDecay`  | `(trigger: signal, decay: float) → env`. Sum-typed delay (`Idle | Decaying(level)`) — the canonical example of a sum that `sum_lower` decomposes into a tag register plus one scalar slot per (variant, field). |
| `TriggerRamp`  | `(trigger: signal) → (frames: float, edge: float)`. Sum-typed delay (`Quiescent | Counting(n)`) that emits a frame counter from each rising edge. |
| `SampleHold`   | `(trigger: signal, input: signal) → value`. Captures `input` on rising edge of `trigger`; holds otherwise. |
| `PoissonEvent` | `(rate: float) → trigger: signal`. xorshift-based stochastic event generator at the requested rate. |

#### Sequencing and control

| Type           | Ports |
|----------------|-------|
| `Sequencer<N>` | `<N: int = 8>(clock: unipolar, values: float[N]) → value`. Edge-triggered step sequencer over an `N`-element array. |
| `Clock`        | `(freq: freq, ratios_in: float[1]) → (output: unipolar, ratios_out: float[1])`. Master clock + ratio-array fan-out. |
| `VCA`          | `(audio: float, cv: float) → out`. Multiplicative gain. |
| `CrossFade`    | `(a: signal, b: signal, mix: unipolar) → out`. Linear two-channel mix. |

### Composites

Programs whose body wires together other DSP types — patches in the
stdlib's clothing. The `inline_instances` stratum splices these
subgraphs into a single flat body before emit; the JIT sees no
"composite" / "primitive" distinction, only scalar instructions.

| Type                 | Ports | Composed of |
|----------------------|-------|-------------|
| `LadderFilter`       | `(input, cutoff: freq, resonance: unipolar, drive) → (lp, bp, hp, notch)` | 4× `OnePole` + `Tanh` |
| `Phaser`             | `(input, feedback, lfo_speed) → (output, lfo)` | 4× inline `_allpassStage` + `Phasor`→`Sin` LFO |
| `Phaser16`           | `(input, feedback, lfo_speed) → (output, lfo)` | 16× inline `_allpassStage` + `Phasor`→`Sin` LFO |
| `Bubble`             | `(trigger, radius, q, sigma, decay_scale, amp_scale, attack_g) → out` | `SampleHold` + `TriggerRamp` + `Exp` + `EnvExpDecay` + `SVF` |
| `BubbleCloud`        | `(trigger, radius, q, sigma, decay_scale, amp_scale) → out` | 8× `Bubble`, integer round-robin |
| `Seq4MinorTranspose` | `(trigger: unipolar) → freq: freq` | `Sequencer<N=4>` over a fixed minor-key value array |

## Generic programs

Two stdlib programs declare type parameters:

- `Delay<N: int = 44100>` — array of `N` samples; specialized at instance time via `type_args: { N: 8 }` etc.
- `Sequencer<N: int = 8>` — `N`-step value sequencer.

Specialization happens in the `specialize` stratum: the concrete `N` is
substituted into every `ShapeDim` and expression-position `TypeParamRef`,
producing a fresh `ResolvedProgram` per `(template, args)` pair.

## Surface syntax

Anchor points for `.trop` syntax:

- `compiler/parse/lexer.ts`, `expressions.ts`, `statements.ts`,
  `declarations.ts` — the four parsing layers.
- `compiler/parse/lower_bounds.ts` — desugaring of bounds annotations
  (`signal[-1, 1]`, `unipolar`, `freq`, etc.) to `clamp` ops at parse time.
- `compiler/parse/print.ts` — pretty-printer used by the round-trip
  test in `compiler/parse/stdlib_round_trip.test.ts`. Every file in
  this directory must round-trip through the printer.

## Adding a program

1. Create `stdlib/MyType.trop` in surface syntax.
2. `loadStdlib()` discovers it by filename on next session boot.
3. Run `bun run scripts/validate_stdlib.ts` to confirm it parses,
   elaborates, and lowers cleanly.
4. No C++ changes unless you need a new expression op.
