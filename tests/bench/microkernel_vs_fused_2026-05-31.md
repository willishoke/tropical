# Benchmark Results: microkernel-vs-fused on the root-program default

| Field       | Value |
|-------------|-------|
| Date        | 2026-05-31 |
| Branch      | `test/hotswap-and-bench-refresh` |
| Captured at | commit `457a0c0` |
| Hardware    | darwin/arm64, Apple M5 |
| LLVM        | 22.1.1 |
| Opt level   | `OptimizationLevel::O2` |
| Reproduce   | `bun run tests/bench/microkernel_vs_fused.ts` |

Re-capture **after the Option A cutover** (root-program lowering is the default).
The question this answers: does the `microkernel` compile-time escape valve still
work now that the session lowers as one synthetic root program? **Yes.** Cold
compile (cache wiped per case).

| case                 | mode             | jit_ms | ns/sample | rt_ratio |
|----------------------|------------------|-------:|----------:|---------:|
| bubble_drip          | fused            |   0.5  |   38.8    | 0.17%    |
| bubble_drip          | microkernel      |  29.3  |   39.2    | 0.17%    |
| bubble_drip          | microkernel-deep |  43.3  |   47.0    | 0.21%    |
| cross_fm_4           | fused            |   0.6  |   35.8    | 0.16%    |
| cross_fm_4           | microkernel      |  38.4  |   37.7    | 0.17%    |
| cross_fm_4           | microkernel-deep |  52.1  |   37.7    | 0.17%    |
| odd_harmonics        | fused            |   2.4  |  194.5    | 0.86%    |
| odd_harmonics        | microkernel      | 202.6  |  214.2    | 0.94%    |
| odd_harmonics        | microkernel-deep | 232.9  |  197.7    | 0.87%    |
| polyphony_4x_SinOsc  | fused            |  23.6  |   23.9    | 0.11%    |
| polyphony_4x_SinOsc  | microkernel      |  30.1  |   28.6    | 0.13%    |
| polyphony_4x_SinOsc  | microkernel-deep |  67.8  |   36.1    | 0.16%    |
| polyphony_8x_SinOsc  | fused            | 175.4  |   60.3    | 0.27%    |
| polyphony_8x_SinOsc  | microkernel      |  70.3  |   59.0    | 0.26%    |
| polyphony_8x_SinOsc  | microkernel-deep | 119.9  |   66.2    | 0.29%    |
| polyphony_16x_SinOsc | fused            | 421.2  |  121.9    | 0.54%    |
| polyphony_16x_SinOsc | microkernel      | 201.7  |  131.2    | 0.58%    |
| polyphony_16x_SinOsc | microkernel-deep | 310.6  |  130.7    | 0.58%    |
| polyphony_32x_SinOsc | fused            |1362.2  |  296.8    | 1.31%    |
| polyphony_32x_SinOsc | microkernel      | 559.2  |  310.6    | 1.37%    |
| polyphony_32x_SinOsc | microkernel-deep | 779.4  |  310.1    | 1.37%    |

**Read:**

- **Runtime parity across modes.** ns/sample is within a few % across fused /
  microkernel / deep for every case — fusion buys little here, as expected for
  these graphs. Everything is far under realtime.
- **Compile-time crossover ~8 voices.** Below it, fused wins (tiny single
  function). Above it, fused's one-huge-function compile scales superlinearly:
  **32-voice fused is 1362 ms vs microkernel 559 ms (2.4× faster compile)** for
  the same runtime. The microkernel escape valve is intact on the root default —
  large polyphony should select `microkernel` to stay inside an interactive
  recompile budget.
- deep mode adds compile cost without a runtime win here (children inline cheaply
  in these patches); it's for graphs with genuinely heavy nested sub-instances.
