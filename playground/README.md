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

The wall has MOVED, not vanished: post-fix `sample` shows 100% of the
remaining cold time inside LLVM's machine scheduler
(`ScheduleDAGInstrs::buildSchedGraph` / `mayAlias`) compiling the one giant
unrolled kernel — strata+emit is now ~10 s for the full circuit. The
remaining causes are the ones the panel named: the composed amps are
slot-UNIFORM (functions of params only, never τ) yet live in the per-sample
kernel, and the mode banks are unrolled syntax instead of data. Next cuts,
in order: (2) stage-0 uniform hoisting — a binding-time index on the
expression type, τ-free subgraphs erased into a one-sample coefficient
kernel run at param writes, results landing in slots (also upgrades GPU amp
accuracy from in-kernel f32 to host f64 constants); (3) banks as tables —
mode loops over array slots, making compile time O(1) in mode count.
