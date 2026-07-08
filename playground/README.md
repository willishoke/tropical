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

The circuit wants a modal REVERB after the filter; it's absent because the
residue composition currently hits a compile wall that scales super-linearly
in voice modes (measured, warm engine, this machine):

| graph | compile |
|---|---|
| 1 ring (6 modes) → reverb(32) | ~7 s |
| 1 ring → reverb → filter | 40 s |
| 1 ring → **filter → reverb** | >120 s (order matters: compose small-into-big LAST) |
| 2 rings → reverb → filter | 235 s |
| 4 rings → reverb (any order) | >300 s |
| 4 rings → filter (2-pole) | **17 s** (shipped) |
| … + comb (7 taps) | 105 s (each tap re-evaluates the whole upstream bank) |

Two distinct causes worth separating: (a) some pass walks the shared
expression DAG as a TREE (cost tracks the unshared tree size — the comb's 7×
and the addr-warp's embedding multiply), and (b) the composed amps are
slot-UNIFORM (functions of params only, never τ) yet are recompiled into the
kernel and re-evaluated per sample. Host-side uniform hoisting — evaluate amp
rationals on the control plane at param-write time, land them in slots —
fixes both the compile wall and the per-sample cost, and was already the
deferred item in the Metal plan. The reverb returns with it.
