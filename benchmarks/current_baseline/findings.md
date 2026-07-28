# Current performance baseline findings

**Status:** Schema-3 exact-product baseline complete. **S-06 staff decision:
accept the measured baseline; no tiered preview or compile optimization now.**

The qualifying raw dataset is
[`full-m1pro-20260728-exact-product-schema3.jsonl`](data/full-m1pro-20260728-exact-product-schema3.jsonl).
Its SHA-256 is
`a5792b58edf026ae3d56133179e4618a5a3387a99d808adbad8609e5fa9aefb1`.
It was recorded from clean commit
`b0c89b953bee0a103878d20cd8f0012bc20f801b` on 2026-07-28 UTC, on the
canonical Apple M1 Pro (10 CPU cores, 16 GPU cores, 16 GB), macOS 26.3,
LLVM 22.1.7, RelWithDebInfo, 44.1 kHz, B=512.

The earlier schema-2 row remains historical evidence for its benchmark-owned
six-node approximation. It is rejected for flagship product and S-06
decisions: it omitted the address path, source oscillators, current ring
parameters/default partial semantics, reverb, filter, and taps.

## Exact product fixture and byte evidence

The schema-3 flagship is not copied into the matrix. The harness reads the
strict-JSON `GRAPH` declaration directly from `playground/renderer/app.js`.
The normalized graph SHA-256 is
`ea6779becbc520ab83c3d71fb56335f367083b637cee6fd679a8109309660a81`.
Its 11 nodes preserve:

- the `o1`/`o2` source mix and `adr` address path;
- four address-driven rings at 110/165/220/330 Hz with decay 4;
- omitted `partials`, exercising the renderer's current default of six;
- modal mix → reverb (`rt60=2`) → filter
  (`cutoff=800`, `resonance=0.5`) → output; and
- `taps: true`.

The two derived edits retain their complete normalized graphs in the raw row.
One adds an address-driven 440 Hz fifth ring; the other changes only `r1` from
the default six partials to an explicit seven.

Every cold and warm repeat retains byte count and SHA-256 for manifest, audio
LLVM IR, coefficient LLVM IR, and MSL. All six main generations and all six
generations for each edit are byte-identical within their series. The main
product artifacts are:

| artifact | bytes | SHA-256 |
|---|---:|---|
| manifest | 2,234,615 | `bbb63f79dcb63f500f527e020179430ffca35ea662596f5bc98e128900929b81` |
| audio LLVM IR | 940,453 | `89952639ee13acc5770b2f9f9b656eb14ba3554296756a31933cc20692c70bca` |
| coefficient LLVM IR | 5,899,218 | `86d51ff9e003afde080fc2999152f199ec2b7a53cc026dd3d4dddd606fe1b00e` |
| MSL | 915,251 | `14df9aded5e6bb9305e4e462e286c1e2a80d6ccaccc73b5fa1b69de30b1e4923` |

The graph produces 11,039 plan instructions, 95,446 registers, 601 slots,
22 array slots, and 14 coefficient-array slots. These are substantially
different from the rejected approximation and explain why its sub-500 ms
claim was not product evidence.

## Flagship structural editing

Each entry is a complete `render-graph` generation/load/publication subprocess
wall. Every cold and warm column has three raw samples. Cold repeats use fresh
benchmark-owned caches; each warm repeat is a new process reusing only its
paired cold cache.

| exact-product operation | cold raw (ms) | cold median | cold range / median | warm raw (ms) | warm median | warm range / median |
|---|---|---:|---:|---|---:|---:|
| full generation | 5288.650, 5253.585, 5278.919 | 5278.919 ms | 0.66% | 1397.467, 1397.915, 1392.085 | 1397.467 ms | 0.42% |
| add addressed fifth ring | 6291.362, 6193.188, 6208.277 | 6208.277 ms | 1.58% | 1634.664, 1624.635, 1617.873 | 1624.635 ms | 1.03% |
| make default partials 6 → 7 | 5446.444, 5429.437, 5444.909 | 5444.909 ms | 0.31% | 1421.681, 1424.694, 1427.792 | 1424.694 ms | 0.43% |

The exact-product walls are accepted measurements: approximately 1.4–1.6 s
warm and 5–6 s cold across the full generation and two representative edits.
Their low variance makes them suitable as a machine-local baseline.

## S-06 staff decision

Structural topology editing is outside the product's primary live-performance
loop. The measured 1.4–1.6 s warm and 5–6 s cold walls are therefore retained
as an exact-product baseline, not promoted into a 500 ms release gate.

No tiered preview path or compile optimization is planned from this result.
That decision does not weaken numerics, change the product graph, or reinterpret
the measurements; it aligns architecture work with the active live-performance
requirements.

The rejected alternative is spending architecture on an inactive product
requirement. If structural topology editing enters the primary live loop later,
these exact measurements provide the baseline for setting a product-owned
target and reconsidering tiering or compilation work.

## Bank capacity and live count

The banked audio plan remains exactly 137 instructions from K=16 through
K=512. Capacity-dependent coefficient materialization grows the full
generation wall. Runtime medians below use warm offline probes; Metal is
synchronous (D=0).

| fixed bank | plan instructions | generation cold / warm | JIT ms / block | Metal ms / block |
|---|---:|---:|---:|---:|
| K=16 | 137 | 335.119 / 318.202 ms | 0.0602 | 0.2795 |
| K=64 | 137 | 394.984 / 367.701 ms | 0.2502 | 0.3879 |
| K=128 | 137 | 470.330 / 437.529 ms | 0.5053 | 0.5491 |
| K=256 | 137 | 680.889 / 570.716 ms | 1.0041 | 0.7993 |
| K=512 | 137 | 1113.322 / 850.243 ms | 2.0070 | 1.3862 |

The synchronous Metal crossover on this machine remains between 128 and 256
partials. The exact product itself is faster per block on warm JIT
(1.947 ms median) than synchronous Metal (2.948 ms median), so the bank
crossover must not be generalized into automatic backend selection.

The dynamic K=512 fixture holds generation cost and emitted structure broadly
flat across live counts while execution tracks the live trip count:

| live count | generation cold / warm | JIT ms / block | Metal ms / block |
|---:|---:|---:|---:|
| 16 | 1120.000 / 849.770 ms | 0.1378 | 0.4249 |
| 128 | 1113.443 / 851.669 ms | 0.5612 | 0.5555 |
| 512 | 1101.207 / 837.907 ms | 1.9967 | 1.5084 |

## Runtime and evidence integrity

- The 512-sample deadline is 11.610 ms; its half-deadline is 5.805 ms.
  Across all 192 individual 100-block probes, the worst process p99 was
  3.503 ms (exact-product cold Metal). No probe crossed the half-deadline.
- The worst raw slot-write p99 was 1.547 ms (nested bank), below one block.
- The row retains 19,200 raw process samples and 1,512 raw write samples.
  Every probe reports zero non-finite outputs, ownership failures, reference
  ownership failures, and overruns.
- All 16 main generation series and both repeated flagship edit series report
  byte-identical emitted artifacts, backed by the complete retained digest
  matrices rather than a boolean assertion alone.
- The ordinary user kernel-cache inventory was identical before and after:
  936 files, 11,182,832 bytes, whole-tree SHA-256
  `6b6bfb2b752574c513481e317965a0e507756f41e4d5cca37c0871bc4c39fd3d`.
  The added, removed, and modified lists are all empty.

Two coarse cold walls cross the handoff's 20% escalation line, and remain
visible rather than being discarded:

- pure fixed-sine frontend plan: 1649.639, 1284.155, 1291.873 ms
  (28.29% range/median);
- gong graph preparation: 535.795, 385.072, 383.862 ms
  (39.46% range/median).

Each is one high first sample with a tight following pair and tight warm
series. The cause is not isolated by current instrumentation, so neither row
is used for S-06. The exact-product and edit series that do determine S-06
remain below 1.6% range/median.

## Candidate regression budgets

- Exact flagship topology→publication: retain the accepted approximately
  1.4–1.6 s warm and 5–6 s cold measurements as the machine-local comparison
  baseline. There is no 500 ms release gate for the existing full path.
- 512-bank cold topology→publication: median ≤2 s.
- Banked audio plan instruction growth from K=16 to K=512: zero.
- JIT/Metal block p99: below 50% of the 11.61 ms B=512 deadline.
- Raw slot-write p99: below one 512-sample block.
- Emitted artifacts must remain byte-stable within a repeated run; every
  repeat must retain artifact byte counts and SHA-256 values.
- A qualifying run must preserve the complete ordinary-cache tree digest and
  retain attributable path changes if it does not.

## Limitations

- Program fixtures expose separate coarse `diffcli compile`, IR-emission, and
  MSL-emission subprocess walls; authoring/lowering cannot be split from those
  frontend processes without invasive instrumentation.
- Graph topology-to-publication is one public wall covering compile, split,
  emit, dual-load, and publication. ORC load likewise covers IR parse,
  optimization/code generation, add-module, and lookup.
- Arena node count and standalone stage-0 split time remain unavailable.
- The nested-bank fixture remains noninteractive at 4293.925 ms cold and
  2197.608 ms warm; it needs a measured product scenario before it affects
  product architecture.
- Baseline Metal probes are offline synchronous D=0 measurements, not live
  callback or parameter-latency qualification. No DAC was opened by Lane D.
- One M1 Pro is machine-local evidence, not a universal Apple GPU crossover.
