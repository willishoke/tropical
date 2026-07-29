# Metal qualification — sprint handoff

- **Sprint:** 2026-07-28 through 2026-08-10
- **Lane:** E — Apple GPU runtime qualification
- **DRI:** Apple-runtime lane
- **Supervisor:** Staff engineer
- **Status:** Complete — scoped qualification failure recorded; no live configuration declared supported
- **Master:** [Staff engineer sprint handoff](00-staff-engineer-master-handoff.md)
- **Depends on:** Lane D owns general performance reporting; this lane owns
  Metal-specific reliability and latency evidence.
- **Must not overlap:** No plan/IR redesign and no automatic backend-selection
  policy.

## Outcome — configuration evidence complete, live release blocked

The latency matrix and short validation landed, and the final long row ran
exactly once. On the canonical M1 Pro, B128/D3 missed its 2.902494 ms
deadline; B512/D3 later missed its 11.609977 ms deadline at the scheduled
clock-jump re-prime after 450.050 measured seconds. B256 was not tested.
Accordingly, this sprint declares no supported pipelined live-Metal
configuration. The retained brief below is the qualification protocol; the
raw rows and current decision are in
[`benchmarks/metal_live/findings.md`](../../benchmarks/metal_live/findings.md).

## Mission

Close the three explicit gaps in the July 7 live-Metal findings:

1. run a meaningful soak for leaks, autorelease behavior, dropout tails, and
   long-coordinate numeric drift;
2. sweep live buffer lengths instead of relying on the boot default;
3. measure parameter-change latency through the future-block pipeline.

The result is a qualified operating envelope for Metal, not a universal claim
that Metal is the best backend.

## Product contract under test

The pipelined Metal path renders future blocks because a Tropical kernel is a
pure function of coordinate and a captured slot snapshot. This removes GPU
submission jitter from the callback, but it introduces bounded control
latency:

```text
control latency ≤ pipeline depth × block length / sample rate
```

The sprint must measure the actual distribution and verify behavior across:

- ordinary raw slot writes;
- glided parameters;
- anchored frequency writes;
- master velocity/time re-basing;
- clock jumps;
- kernel hot-swap.

Audio-position latency and control latency must remain distinct in all reports.

## Test matrix

### Hardware and software

Required:

- the canonical Apple Silicon development machine;
- current macOS and Metal toolchain recorded in the report;
- LLVM/JIT dual load enabled;
- no dedicated/headless GPU assumption.

Optional if available:

- a second Apple Silicon generation;
- a machine under deliberate compositor/display load.

Do not delay the required matrix waiting for optional hardware.

### Patch classes

- pure sine correctness canary;
- through-zero flanger;
- modal fixed banks at 128 and 512 partials;
- four-ring playground circuit;
- one dynamic-count bank;
- one patch with frequent stage-0 coefficient refreshes.

### Runtime dimensions

- block lengths: 128, 256, 512;
- pipeline depths: 1, 2, 3 where supported;
- JIT reference plus Metal;
- minimum soak:
  - 30 minutes at the heaviest realtime-safe configuration;
  - 10 minutes each at the other block sizes;
- one mid-soak hot-swap;
- periodic raw, glide, anchor, and velocity writes;
- periodic clock jumps.

## Required harness changes

The existing engine boot path fixes buffer length too early for the desired
sweep. Add the smallest test/control-plane mechanism needed to select block
length before runtime/DAC construction.

Constraints:

- default behavior remains unchanged;
- invalid values refuse clearly;
- production protocol schemas do not change unless separately approved;
- the setting is visible in diagnostics;
- tests do not require changing global source constants.

Add a non-interactive soak harness under:

```text
benchmarks/metal_live/
```

or:

```text
engine/tests/
```

according to whether it is a manual qualification run or a short deterministic
CTest. Long soak execution remains manual; a shortened smoke belongs in the
automated suite.

## Measurements

Capture:

- callback average, p95, p99, and max;
- dropout/underrun count;
- sustained Metal block interval;
- pipeline queue depth over time;
- parameter write timestamp;
- first output sample reflecting the write;
- predicted versus observed control latency;
- hot-swap publication time and callback tail;
- process resident memory at intervals;
- Metal resource/object counts where observable;
- CPU use;
- SNR/max error against JIT at start, midpoint, and end;
- SNR at a large coordinate after clock jump;
- any re-prime duration.

Use an impulse-like, unambiguous parameter mutation for latency measurement,
not a knob whose slow glide makes onset detection subjective.

## Correctness thresholds

Reuse current per-fixture Metal-vs-JIT thresholds. This lane may tighten a
threshold after evidence, but may not loosen one without staff approval and a
documented numeric reason.

Minimum runtime expectations:

- zero dropouts in the required 30-minute pipelined soak;
- no monotonic memory growth after warm-up;
- callback p99 below 50% of its deadline;
- measured control latency no greater than the predicted pipeline bound plus
  one block of timestamp/detection tolerance;
- clock jump and hot-swap re-prime without stale future blocks;
- no time-dependent SNR deterioration on the fixed-point clock path.

## Owned files

Primary ownership:

- `benchmarks/metal_live/`
- `engine/metal/MetalKernel.hpp`
- `engine/metal/MetalKernel.mm`
- `engine/tests/test_metal_kernel.cpp`
- Metal-specific portions of `engine/runtime/FlatRuntime.*`
- a new Metal qualification findings document

Coordinate any shared `FlatRuntime` edit with Lane F. Lane E may not change
generic plan parsing or state compatibility behavior.

## Work plan

### Day 1: harness design

- Freeze matrix, timestamps, and thresholds.
- Identify the smallest block-length selection seam.
- Reproduce the July 7 128/512 reference rows.

### Days 2–3: controllability

- Land block-length selection.
- Land short deterministic pipeline latency instrumentation.
- Add a stale-future test around parameter writes and clock re-prime.

### Day 4: short sweep

- Run all block sizes and depths for 60 seconds.
- Fix harness defects before the long soak.
- Verify JIT/Metal correctness at start and end.

### Day 5: first integration checkpoint

- Submit runtime changes for review.
- Run C++ tests and full validation.
- Freeze the code before long qualification unless a correctness defect is
  found.

### Days 6–7: soak and latency matrix

- Run the 30-minute heavy soak.
- Run 10-minute supporting rows.
- Collect raw latency samples for each parameter discipline.

### Day 8: contention and failure probes

- Run one ordinary desktop-contention case.
- Test invalid buffer/depth configuration.
- Test hot-swap and clock jump while the future queue is full.

### Day 9: findings and recommendation

Choose an operating recommendation by patch class and control-latency budget.
Do not encode it as automatic policy.

### Day 10: qualification freeze

- Commit raw data and findings.
- Give Lane A approved product wording.
- Give Lane C final evidence links and open obligations.

## Acceptance gates

1. Buffer length is selectable for qualification without changing defaults.
2. A short pipeline/re-prime smoke runs automatically on supported macOS.
3. Required soak rows complete with raw data.
4. Parameter latency is empirically measured for all four disciplines.
5. Stale future blocks cannot survive a clock jump or hot-swap.
6. Memory behavior is reported, including warm-up.
7. Metal-vs-JIT evidence is captured at multiple times/coordinates.
8. Findings state the tested hardware and do not generalize beyond it.
9. `make validate` and Metal C++ tests are green.

## Non-goals

- No CUDA/NVPTX work.
- No automatic CPU/GPU cost model.
- No dynamic per-block backend switching.
- No lower numeric accuracy to gain speed.
- No requirement that long hardware soak run in Linux CI.
- No new modal synthesis feature.

## Stop and escalate

Immediately stop and notify the staff engineer if:

- the pipelined path emits stale audio after a write, jump, or swap;
- any required soak drops a block;
- memory grows monotonically after the warm-up window;
- SNR degrades with elapsed coordinate on a fixed-point patch;
- a Metal fix changes JIT or wasm output;
- a generic `FlatRuntime` change conflicts with compatibility ownership.

A qualification failure is not permission to weaken the test. Land the
reproducer and make the failure visible.

## Handoff package

Leave:

- automated short smoke;
- manual soak harness;
- raw samples and environment manifest;
- qualified buffer/depth table;
- measured control-latency distributions;
- explicit open hardware/runtime risks for Lane C and the staff engineer.
