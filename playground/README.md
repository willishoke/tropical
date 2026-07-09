# tropical · demo

One fixed circuit, four scopes, every knob live, no state anywhere:

```
o1, o2 ──Σ──► ADDRESS ──► RING I·II·III·IV ──Σ──► FILTER ──► out
```

The oscillator bank is the hand: its summed waveform is the **time-address**
that scrubs four modal rings (a signal driving a clock is a warp). The rings
compose through the filter's exact conjugate pole pair by the residue
calculus. The four wells are **random-access reads of the running kernel**
(`render_window`) — the same closed-form function the audio thread evaluates,
at any τ, while audio dispatches pipelined on the **Metal backend** (the JIT
dual-loads as the scope reference). The whole client is `main.js` (socket
plumbing), `preload.js` (one verb), and `renderer/` — the instrument is the
protocol; the app just draws it.

```
make lean && make build          # engine (Metal on by default)
cd playground && bun install && bunx electron .
```

`TROPICAL_DEMO_JIT=1 bunx electron .` runs the CPU JIT instead.

## Finding: symbolic-composition compile scaling (2026-07-08)

The circuit's modal REVERB originally hit a compile wall super-linear in
voice modes. Profiling (mid-compile `sample`, twice) put 100% of the time in
the strata passes — `InlineInstances.runE → mapExprId`, then
`ArrayLower.runE` — walking the shared expression DAG as a TREE: a per-node
rewrite costing O(arena) each, called O(arena) times. The walkers had
dropped the TS ports' identity memos under a justification ("the unmemoized
walk is O(output tree), the same bound as encoding") that Phase B silently
invalidated when it removed the tree encoding.

**Fix (landed on this branch): memoized, DAG-shaped strata walks** —
`mapExprId` memoizes per hook-set application; ArrayLower/SumLower carry
pass-wide memos of their needs-lowering predicates and short-circuit
invariant subgraphs (id-identical because `eintern` hash-conses). Gated by
the audio goldens byte-for-byte (68/68). Measured cache-cold, same machine:

| graph | before | after |
|---|---|---|
| 1 ring (6 modes) → reverb(32) | ~7 s (warm) | 18.8 s |
| 1 ring → reverb → filter | 40 s | 24.5 s |
| 1 ring → **filter → reverb** | >120 s | 63.0 s (order still matters: small-into-big LAST) |
| 2 rings → reverb → filter | 235 s | 40.2 s |
| 4 rings → reverb (any order) | >300 s, never returned | 56.8 s |
| 4 rings → reverb → filter (the demo circuit) | — | 75.9 s cold / **10.8 s** kernel-cache warm |

**Round 2 (same day): two more walls, same disease.** Profiling the
residual found (a) the warm load 100% inside `EmitArrow.lowerSig` — the
combinators share `Sig` subterms by Lean object reference (a pointer-DAG),
and the lowering walked it structurally, paying the expanded tree just for
`eintern` to collapse it back; and (b) the emitted kernel was a slot
machine — 255k lines, ONE basic block, 49k loads + 30.6k stores against
21k flops, every temp round-tripping through the caller-provided `%temps`
array, which escapes, so LLVM's machine scheduler + regalloc ground on
80k unremovable memory ops (and the indirection hid ~40% redundant amp
arithmetic from CSE). Fixes: pointer-identity memo on `lowerSig`;
`EmitLlvm` keeps temps as SSA values (the shape `EmitMsl` already used),
falling back to the zero-initialized `%temps` only for never-written reads.
Same gates (goldens 68/68, wasm≡JIT, Metal SNR, MCP). Bonus: shipped wasm
patches halved in size. Cache-cold, same machine:

| graph | round 1 | round 2 |
|---|---|---|
| 1 ring → reverb(32) | 18.8 s | 5.8 s |
| 1 ring → reverb → filter | 24.5 s | 3.8 s |
| 1 ring → filter → reverb | 63.0 s | 5.2 s (order asymmetry gone) |
| 2 rings → reverb → filter | 40.2 s | 6.8 s |
| 4 rings → reverb | 56.8 s | 14.6 s |
| 4 rings → reverb → filter (the demo circuit) | 75.9 s / 10.8 s warm | **17.0 s cold / 5.3 s warm** |

The residue: cold is LLVM codegen on the real (post-CSE) arithmetic —
one ~16k-flop block — and warm is the engine-side load (kernel-object
cache + Apple's MSL compile). Both shrink with the panel's remaining
cuts: (2) stage-0 uniform hoisting — a binding-time index on the
expression type, τ-free subgraphs (the composed amps: ~90% of the
surviving flops, essentially all 2.3k fdivs) erased into a one-sample
coefficient kernel run at param writes, results landing in slots (also
upgrades GPU amp accuracy from in-kernel f32 to host f64 constants);
(3) banks as tables — mode loops over array slots, making compile time
O(1) in mode count.
