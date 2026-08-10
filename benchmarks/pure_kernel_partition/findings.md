# Findings: scratch liveness and pure kernel partitioning

Measured 2026-08-09 on an Apple M1 Pro at 44.1 kHz. Metal safe math was enabled. Latency
figures are synchronous command-buffer completion times over 1,000 blocks for production
graphs and 300 blocks for the isolated partition benchmark.

## 1. Liveness removes graph-width memory growth

The production compiler's linear-scan allocator maps logical array slots whose instruction
live ranges do not overlap onto one physical threadgroup arena. These graphs contain two,
three, or four independent `resonator → reverb → reverb` branches, followed by `modalmix`
and `out`:

| Branches | Graph nodes | Logical array floats | Physical peak floats | Total scratch | Median, 512f | p99 | Overruns |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 8 | 6,996 | 3,548 | 22,688 B | 5.337 ms | 5.902 ms | 0 / 1,000 |
| 3 | 11 | 10,440 | 3,548 | 22,688 B | 10.198 ms | 11.025 ms | 1 / 1,000 |
| 4 | 14 | 13,884 | 3,548 | 22,688 B | 13.527 ms | 14.585 ms | 1,000 / 1,000 |

The 512-frame deadline is 11.610 ms. Scratch remains constant as independent branches are
added, even though the number of logical arrays and registers grows. The M1 Pro crossover in
this deliberately heavy topology is arithmetic throughput between three and four branches,
not the 32 KiB threadgroup-memory ceiling. The single outlier in the three-branch run means it
is a useful capacity marker, not a production-safe budget.

## 2. A pure multi-kernel split crosses the hard ceiling cheaply

The standalone benchmark makes the resource conflict explicit. Each modal branch uses 5,600
floats (22,400 bytes) of cooperative scratch. A kernel that keeps both arrays live requests
44,800 bytes and Metal rejects its pipeline against the M1 Pro's 32,768-byte maximum. Two
producer kernels plus a mixer succeed; their intermediate buffers use
`MTLResourceStorageModePrivate`, and all three dispatches are in one command buffer, so there
is no CPU readback or stateful fallback.

| Frames | Deadline | Fused-reuse median / p99 | Partitioned median / p99 | Median split cost |
|---:|---:|---:|---:|---:|
| 128 | 2.902 ms | 0.446 / 0.829 ms | 0.465 / 0.712 ms | 0.019 ms |
| 256 | 5.805 ms | 0.711 / 0.830 ms | 0.722 / 0.828 ms | 0.011 ms |
| 512 | 11.610 ms | 0.938 / 1.080 ms | 0.951 / 1.077 ms | 0.013 ms |

The two paths agree exactly in safe math mode for this fixture. The measured median cost of
the extra producer/mix dispatches is 11–19 microseconds, small compared with both the kernel
work and the audio deadlines.

## Read

This is evidence for a two-level strategy:

1. Reuse scratch by liveness inside a kernel. It is the right default and makes independent
   graph width memory-neutral.
2. When a genuinely overlapping live set still exceeds the hardware ceiling, cut only at a
   pure boundary, keep intermediates GPU-private, and submit the stages as one command buffer.

The first level is implemented in the production Plan/MSL path. The second result is an
execution-model proof of concept, not yet an automatic Plan partitioner; the next production
step is to define partition metadata and live-value buffer bindings in the emitted ABI.
