# compile_time — LTO link cost on precompiled bitcodes

**Question.** If we cache the per-kernel compile work (Stage A) and
only re-run the LTO link step (Stage B) on patch edits, is Stage B
fast enough to make "add an instance and rewire it" feel
interactive? Target: a few hundred ms for the link step at a large N.

This re-frames the earlier monolithic-vs-separate question into the
two-stage incremental-build story:

```
Stage A (one-time per kernel, cached across edits):
    clang -c -flto[=thin] -O2 kernel.c -o kernel.bc
Stage B (re-runs whenever the patch graph changes):
    clang -flto[=thin] -O2 kernel0.bc kernel1.bc ... main.bc -o out
```

If most kernels are unchanged on a small edit, Stage A only re-runs
for the modified/added kernel(s) — cheap. Stage B reruns over all N
bitcodes — expensive (this is what we measure).

The earlier `inlining_across_modules/` benchmark established that LTO
is *necessary* if you compile separately: without it, separate modules
pay ~2.2× runtime. So the question is not "do we need LTO?" but "is
LTO link fast enough to be the per-edit cost?"

## How it works

`gen.py` writes C source for either layout. `bench.py` measures, per N:

- `mono_nolto` / `mono_lto` — full monolithic rebuild (baseline).
- `full_bc_total` / `full_lto_link` — Stage A + Stage B with FullLTO
  (`-flto`). Stage A timed but reported separately; Stage B is the
  headline.
- `thin_bc_total` / `thin_lto_link` — same with ThinLTO (`-flto=thin`).

3 repeats, min reported. 15s hard timeout per clang invocation;
larger N skipped if a variant times out.

```
python3 bench.py        # full sweep
python3 gen.py separate 256 /tmp/foo   # inspect generated source
```

Uses `/opt/homebrew/opt/llvm/bin/clang` (LLVM 20.1).

## Results

| N | mono_nolto | mono_lto | full_bc_total | full_link | thin_bc_total | thin_link |
|---:|---:|---:|---:|---:|---:|---:|
| 16   | 125 ms | 124 ms | 435 ms  | 440 ms   | 458 ms  | 444 ms |
| 64   | 137 ms | 147 ms | 1585 ms | 1402 ms  | 1591 ms | 1399 ms |
| 256  | 168 ms | 310 ms | 6186 ms | 5347 ms  | 6160 ms | 5223 ms |
| 512  | 227 ms | 685 ms | 12500 ms | 10897 ms | 12300 ms | 10108 ms |
| 1024 | 377 ms | 1983 ms | 24666 ms | timeout | 24544 ms | timeout |
| 2048 | 851 ms | 8310 ms | — | — | — | — |
| 4096 | 2252 ms | timeout | — | — | — | — |

Three findings — the third one is the one that matters.

### 1. Stage A is expensive (~24 ms per kernel TU)

Compiling each kernel `.c` to LTO bitcode costs ~24 ms per file at
this scale — almost entirely clang frontend overhead. For Tropical's
ORC JIT path that overhead is *not real*: we already generate LLVM
IR directly without parsing C. The right Stage A measurement for
Tropical is "time to run the per-instance O2 pipeline on a single
function's IR," which is much smaller.

### 2. ThinLTO link time is not meaningfully faster than FullLTO

At every N the ThinLTO and FullLTO link rows are within ~5% of each
other. The naïve invocation `clang -flto=thin *.bc -o out` does **not**
exercise ThinLTO's actual incremental story. Clang's driver does the
ThinLTO codegen itself before invoking `ld`, then hands `ld` already-
compiled `.o` files. Passing `-Wl,-cache_path_lto,...` populates the
cache (verified: 259 entries written for N=256) but the work it would
cache has already happened by then. ThinLTO's per-bitcode caching
requires a three-stage explicit pipeline (index → per-bitcode codegen
→ link), which I didn't build out here.

So: this benchmark *doesn't* rule out ThinLTO. It does say that ThinLTO
won't help out of the box — you have to drive it explicitly.

### 3. The LTO link is the bottleneck, and it's an order of magnitude slower than monolithic full rebuild

The headline comparison:

| N | mono_lto (full rebuild) | full_lto_link (link only) | ratio |
|---:|---:|---:|---:|
| 16   | 124 ms  | 440 ms   | 3.5× |
| 64   | 147 ms  | 1402 ms  | 9.5× |
| 256  | 310 ms  | 5347 ms  | 17.2× |
| 512  | 685 ms  | 10897 ms | 15.9× |
| 1024 | 1983 ms | timeout  | — |

For "few hundred ms target," **monolithic full rebuild beats LTO-link-
only for any N up to ~256**. The incremental story doesn't pay off:
caching Stage A doesn't help because Stage B is doing strictly more
work than just compiling one TU from scratch — it loads N bitcodes,
merges them, then runs the LTO pipeline (which is heavier than the
per-module pipeline).

## Implications for Tropical

The user-facing hope was: "absorb the bulk upfront, then small edits
trigger only the cheap link." With off-the-shelf clang LTO, that
inverts — the link is the bulk.

Three directions, in increasing order of investment:

**(a) Stick with monolithic compile in ORC.** Today's M11 fractal
architecture emits one LLVM function with N nested basic blocks. The
relevant baseline from this benchmark is the `mono_lto` column with
the clang frontend cost stripped out (since ORC starts from IR). At
N=1024 the clang version is ~2 s, dominated by frontend + per-module
O2. Stripped of frontend, that's likely a few hundred ms in pure ORC
— roughly what the parent benchmark predicted. **For sessions ≤ ~256
instances, full rebuild on every edit is plausibly already fast
enough.**

**(b) ThinLTO with explicit pipeline + caching.** If we go down the
separate-modules path anyway (for hot-swap granularity, structural
naming, etc.), the next experiment is the explicit ThinLTO workflow:
build a summary index, per-bitcode codegen with `-fthinlto-index`,
cache the per-bitcode `.o`. The unknown is how much the index/codegen
step costs *with* caching when one input changes — none of this
benchmark's data answers that.

**(c) Custom incremental layer above ORC.** Cache per-instance IR
post-O2, stitch by linking-without-optimizing, optionally run one
lightweight inliner pass for cross-instance work. This bypasses LTO
entirely. More work to build but avoids the linker's per-bitcode
codegen overhead, and gives us control over what cross-module
optimization is actually worth doing.

The data points at (a) being the right default for now and (c) being
the path to scale beyond ~256 instances. (b) is worth a follow-up
experiment if (c) feels like too much custom infrastructure.

## Caveats

- All measurements through clang/ld64 on macOS arm64. The LLVM ORC JIT
  path doesn't share the same code paths as clang+ld — Stage A's
  frontend cost in particular is illusory for ORC. Stage B's
  optimization cost is more directly translatable, but ORC doesn't go
  through ld at all, and an ORC-based LTO would use the LLVM LTO API
  directly with potentially different scaling.
- ThinLTO results here reflect the simple-invocation case. Properly
  pipelined ThinLTO with caching has not been tested.
- 3-repeat min is enough to filter system jitter but not enough to
  separate fine-grained pipeline effects. The ratios (10×+) are robust
  signal; the small differences between FullLTO and ThinLTO at this
  scale are not.

## Reproducing

```
cd benchmarks/llvm/compile_time
python3 bench.py
```

Roughly 3 minutes for the full sweep up to N=1024 (everything past
that skips because earlier rows hit the 15s timeout).
