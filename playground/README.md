Status: Current

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
at any τ, while an off-RT worker prepares exact-epoch **Metal** tiles for the
audio callback (the JIT dual-loads as the scope reference). The whole client is
`main.js` (socket
plumbing), `preload.js` (one verb), and `renderer/` — the instrument is the
protocol; the app just draws it.

```
make lean && make build          # engine (Metal on by default)
cd playground && bun install && bunx electron .
```

`TROPICAL_DEMO_JIT=1 bunx electron .` runs the CPU JIT instead.

## Finding: symbolic-composition compile scaling (2026-07-08)

> Historical measurement record — not a current performance baseline. These
> rounds predate banks-as-data and later compiler/runtime changes. Keep them for
> causal history; use
> [`benchmarks/current_baseline/findings.md`](../benchmarks/current_baseline/findings.md)
> for current claims.

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
cuts: (2) stage-0 uniform hoisting (landed — round 3 below);
(3) banks as tables — mode loops over array slots, making compile time
O(1) in mode count.

## Round 3: stage-0 uniform hoisting (2026-07-08)

Landed as a plan-level binding-time split (`Tropical.Ir.Stage0`, applied
by `StagedLoad` at every kernel load): a forward dataflow pass over the
flat instruction stream in emit order marks each instruction `fold`
(const/rate-only — stays put, both emitters fold it in f64), `s0`
(τ-independent but param-slot-derived — hoisted), or `s1` (per-sample).
The `s0` instructions — the composed modal amplitudes, ~90% of the
demo's flops — move verbatim into a one-sample **coefficient kernel**
(the same `tropical_plan_6`/`EmitLlvm` pipeline, a second module) that
the engine runs once at load and after every slot write; boundary values
cross to the audio kernel through fresh `coef:<n>` module slots.

Two findings the gates forced, in the order they bit:

- **The coefficient kernel must be compiled dumb end-to-end.** At the
  JIT's default level, the split made cold compiles WORSE (25.6 s):
  profiling showed the wall was never "the kernel is big" but "the
  backend is superlinear" — the loop vectorizer's VPlan churn, then
  SelectionDAG scheduling + greedy regalloc on one huge block. O0 IR
  passes alone made it worse still (88 s — bigger input to the same
  backend). The fix is a sibling LLJIT at `CodeGenOptLevel::None`
  (fast ISel, linear scheduler, regalloc-fast): the coefficient kernel
  runs once per knob write, its codegen quality is irrelevant.
- **`fold` must stay behind, wholesale.** EmitMsl's emit-time f64
  constant folding propagates through in-kernel slot write→read, so
  hoisting const-only chains (or their slot writes) demoted exact f64
  GPU literals to f32 host-slot crossings — pure-sine fell from >140 to
  ~109 dB SNR. With `fold` untouched, const-only patches are identity
  plans; only live-knob graphs split, and their GPU amps IMPROVE
  (one f32 rounding of a host f64 value instead of thousands of f32
  in-kernel flops).

Measured cache-cold (JIT path), same machine — and note the scaling in
mode count is now nearly flat, because everything that scales with
modes lives in the coefficient kernel:

| graph | round 2 | round 3 |
|---|---|---|
| 1 ring → reverb(32) | 5.8 s | 4.5 s |
| 1 ring → reverb → filter | 3.8 s | 2.5 s |
| 1 ring → filter → reverb | 5.2 s | 3.8 s |
| 2 rings → reverb → filter | 6.8 s | 3.3 s |
| 4 rings → reverb | 14.6 s | 2.6 s |
| 4 rings → reverb → filter (the demo circuit) | 17.0 s cold / 5.3 s warm | **5.2 s cold / 3.1 s warm** |

Gates: audio goldens 68/68 byte-identical cold-cache (hoisting moved no
output bit), wasm≡JIT, metal_vs_jit 8/8 (~140 dB flat). modal_heavy64
stays at 92.5 dB: its params bind as per-program FFI operands (stage-1
by rule — that path has no re-run hook), so its amps never hoist; the
predicted GPU-amp-accuracy win applies to the session/knob path the
demo actually uses. `TROPICAL_STAGE0=0` disables the split for A/B.
