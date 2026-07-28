# Current performance baseline findings

**Status:** Frozen schema-2 baseline complete; S-06 recommendation proposed
for staff sign-off.

The accepted raw row is
[`full-m1pro-20260727-schema2.jsonl`](data/full-m1pro-20260727-schema2.jsonl).
It was recorded from clean commit
`c80981ac2479361258d2952678bfdf12fca92672` on 2026-07-28 UTC, on the
canonical Apple M1 Pro (10 CPU cores, 16 GPU cores, 16 GB), macOS 26.3,
LLVM 22.1.7, RelWithDebInfo, 44.1 kHz, B=512. The previous schema-1 row
remains historical runtime evidence; it is not used for structural-edit
decisions.

## Flagship structural editing

Each entry is a complete `render-graph` generation/load/publication subprocess
wall. Every cold and warm column has three raw samples. Cold repeats use fresh
benchmark-owned caches; each warm repeat is a new process reusing only its
paired cold cache.

| four-ring operation | cold raw (ms) | cold median | cold range / median | warm raw (ms) | warm median | warm range / median |
|---|---|---:|---:|---|---:|---:|
| full generation | 387.015, 386.663, 383.256 | 386.663 ms | 0.97% | 359.111, 357.337, 355.748 | 357.337 ms | 0.94% |
| add fifth ring | 402.129, 397.071, 406.558 | 402.129 ms | 2.36% | 375.754, 384.839, 374.063 | 375.754 ms | 2.87% |
| baked capacity 16 → 17 | 384.860, 381.339, 383.419 | 383.419 ms | 0.92% | 362.411, 358.132, 361.082 | 361.082 ms | 1.19% |

All measured flagship edits are below the frozen 500 ms warm target and the
2 s cold hypothesis. The graph decoder's `sel` object is inert, so the second
edit is honestly the current baked modal-capacity surrogate, not a no-op
selector claim.

## Bank capacity and live count

The banked audio plan remains exactly 137 instructions from K=16 through
K=512. Capacity-dependent coefficient materialization still grows the full
generation wall. Runtime medians below use the warm probes; Metal is
synchronous (D=0).

| fixed bank | plan instructions | generation cold / warm | JIT ms / block | Metal ms / block |
|---|---:|---:|---:|---:|
| K=16 | 137 | 333.894 / 311.535 ms | 0.0602 | 0.2960 |
| K=64 | 137 | 393.075 / 362.836 ms | 0.2502 | 0.3914 |
| K=128 | 137 | 468.064 / 430.247 ms | 0.5042 | 0.5484 |
| K=256 | 137 | 676.761 / 567.253 ms | 1.0053 | 0.7957 |
| K=512 | 137 | 1108.133 / 842.631 ms | 2.0067 | 1.3859 |

The synchronous Metal crossover on this machine is between 128 and 256
partials. Small patches remain faster on JIT because fixed GPU dispatch cost
dominates.

The dynamic K=512 fixture holds generation cost and emitted structure flat
across live counts while execution tracks the live trip count:

| live count | generation cold / warm | JIT ms / block | Metal ms / block |
|---:|---:|---:|---:|
| 16 | 1108.016 / 843.190 ms | 0.1379 | 0.3175 |
| 128 | 1105.709 / 846.362 ms | 0.5568 | 0.5517 |
| 512 | 1105.327 / 843.934 ms | 2.0112 | 1.5119 |

## Runtime budgets and evidence integrity

- The 512-sample deadline is 11.610 ms; its half-deadline is 5.805 ms.
  Across every individual 100-block probe, the worst JIT p99 was 2.239 ms
  and the worst synchronous Metal p99 was 1.657 ms. No probe crossed the
  half-deadline.
- The worst raw slot-write p99 was 1.506 ms (nested bank), below one block.
- The row retains 19,200 raw process samples and 1,512 raw write samples.
  Every probe reports zero non-finite outputs and zero runtime ownership
  failures.
- All 16 main generation series and both repeated flagship edit series report
  byte-identical emitted artifacts across their cold/warm repeats. Instruction,
  line, and byte metrics match the prior schema-1 row for all 16 fixtures.
  The older row did not retain content hashes, so cross-row byte identity
  beyond those frozen metrics cannot be reconstructed.
- The ordinary user kernel cache was unchanged: file count and root mtime are
  identical before and after the run.

One retained noise point crosses the handoff's 20% escalation line by 0.40
percentage points: the first-ever cold fixed-sine `diffcli compile` sample was
1553.003 ms versus 1295.940 and 1288.642 ms (20.40% range/median). The paired
IR/MSL subprocess walls were not inflated, its warm range was 0.57%, and no
later full-generation or flagship-edit wall exceeded 3.56%. This is consistent
with one-time suite/process startup rather than fixture variance; no product
decision below depends on that coarse program-frontend sample.

## S-06 recommendation proposal

**Propose: no tiered preview path for the current four-ring flagship next
sprint.**

- Reject **tiering recommended** for the flagship: both explicit structural
  edits and the unchanged full graph are below the agreed warm and cold
  budgets, with less than 3% range/median.
- Reject **more isolation needed** for this decision: the target distributions
  are repeated, low-variance, cache-isolated, byte-stable, and preserve all raw
  samples.
- Do not generalize this decision to every composition. The nested-bank
  fixture is noninteractive at 4361.973 ms cold and 2214.665 ms warm, with a
  19 MB coefficient module and 2.866 s cold ORC load. If that composition
  becomes an editing target, investigate its coefficient-generation/load wall
  separately and return with a measured product scenario before proposing
  tiering.

This is a Lane-D evidence proposal, not the signed S-06 staff decision.

## Candidate regression budgets

- Four-ring topology→publication: median ≤500 ms warm; always retain raw
  cold/warm repeats and report range/median.
- 512-bank cold topology→publication: median ≤2 s.
- Banked audio plan instruction growth from K=16 to K=512: zero.
- JIT block p99: below 50% of the 11.61 ms B=512 deadline.
- Raw slot-write p99: below one 512-sample block.
- Emitted artifacts must remain byte-stable within a repeated run; frozen
  artifact metrics must not drift without an explained compiler change.

## Limitations

- Program fixtures expose separate coarse `diffcli compile`, IR-emission, and
  MSL-emission subprocess walls (about 1.3 s each for the small programs);
  authoring/lowering cannot be split from those frontend processes without
  invasive instrumentation.
- ORC load remains one public wall covering IR parse, optimization/codegen,
  add-module, and lookup.
- Arena node count and standalone stage-0 split time remain unavailable.
- Baseline Metal probes are offline synchronous D=0 measurements, not live
  callback or parameter-latency qualification. Those claims belong to
  `../metal_live/`.
- One M1 Pro is machine-local evidence, not a universal Apple GPU crossover.
