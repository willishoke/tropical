# Pure multi-kernel partition proof of concept

This benchmark isolates the escape hatch for a pure graph whose combined cooperative
working set is larger than Metal's per-threadgroup memory limit. It compares two equivalent
executions:

- `fused_reuse` evaluates both independent modal branches in one kernel, reusing one
  threadgroup arena between them.
- `partitioned` evaluates each branch in its own kernel, writes two device-only intermediate
  buffers, and mixes them in a third kernel. All three dispatches are encoded into one Metal
  command buffer; no intermediate reaches the CPU.

Each branch uses 5,600 floats (22,400 bytes) of threadgroup storage. A deliberately
unpartitioned kernel that requires both branch arenas at once uses 44,800 bytes and should
be rejected on hardware with a 32 KiB threadgroup-memory limit. This distinguishes the hard
per-kernel resource ceiling from total graph complexity: a larger pure graph can remain on
the GPU as long as each partition fits.

Run on Apple Silicon:

```sh
make run
```

The program checks the fused and partitioned outputs for numerical agreement, reports the
pipeline compiler's static threadgroup-memory sizes, and measures synchronous block latency
for 128, 256, and 512 sample buffers. It uses Metal's safe math mode, matching the production
backend rather than relaxing the numerical contract for a prettier benchmark.

See [findings.md](findings.md) for the M1 Pro result and the companion measurements from
real emitted modal graphs.
