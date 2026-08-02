Status: prototype

# tropical / night scene 01

A single closed-form scene, presented as a small modal pocket instrument.
Twenty string voices move through four perfect just-intonation sonorities and
return immediately to the beginning after the sixteenth moment.
There is no note-entry mode, patching view, voice allocator, or wiring diagram.

The sixteen clickable moments occupy the complete 16-second scene. The host
loops and seeks by rebasing the scene clock, so the graph itself remains one
fixed, total signal.

## Sonorities

Every frequency is an integer ratio of A1 = 55 Hz. The displayed ratios are
absolute, not tempered pitch-class annotations.

| moment | name | ratios from 55 Hz |
|---|---|---|
| 01 | A−11 / minor eleven | 1/1 · 3/2 · 12/5 · 18/5 · 16/3 |
| 05 | D7·9 / harmonic nine | 4/3 · 2/1 · 7/3 · 10/3 · 6/1 |
| 09 | FΔ♯11 / bright crossing | 8/5 · 12/5 · 3/1 · 4/1 · 9/2 |
| 13 | E9 / harmonic return | 3/2 · 21/8 · 15/4 · 9/2 · 27/4 |

Each voice is three harmonic partials. Every partial is represented by a
matched positive/negative decay pair, making a finite attack followed by a
decay without introducing state or an envelope primitive.

A chord-derived pizzicato lands on every beat. Each chord opens with the
listening-approved two-note downbeat voicing, followed by three lighter
single-note ghosts. Only that causal transient lane enters the frozen grouped
Clouds room. `POSITION` moves its wet field between a future-aware pre-tail and
forward decay; the direct pizzicato always remains causal.

## Surface

- Click any of the sixteen moments, or a chord caption, to seek.
- Use the seven rows directly with the mouse.
- Arrow up/down chooses a row; left/right changes it; Shift makes a fine move.
- Space holds or resumes the scene. `R` reverses it. Enter returns to zero.
- `STRINGS / RESONANT VEIL` is the filtered direct string field.
- `ROOM / PIZZICATO RETURN` is the wet modal room before its return-level control.

`presence`, `veil`, `edge`, `room`, and `position` shape the scene. `flow` is a
continuous local clock velocity, including zero and reverse. `level` is the
final relative listening gain. Amplitude rows use a shared closed-form 20 ms
glide. `POSITION` is a 20 ms equal-power glide from `-1` (pre-tail) through `0`
(two-sided bloom) to `+1` (forward decay). `veil` and `edge` remain epoch-rate
modal controls so their coefficient banks stay hoistable; continuous Metal
handoffs dezipper the resulting output seam over one 128-sample demo quantum
instead of stepping from the old waveform to the new one.

## Run

```sh
make lean
make build
cd playground
bun install
bun run start
```

The app uses the production Metal path by default. The CPU JIT remains available
as a diagnostic fallback:

```sh
TROPICAL_DEMO_JIT=1 bun run start
```

Each chord remains its own modal island until the signal boundary. Each of the
sixteen pizzicato beats does too, preserving every strike anchor (a modal pole
union has only one). The string lane has no room send. The room is the bounded
5.38 MiB fixed-scene path: two native-rate float32 bases generated from the
fixed score and the accepted `clouds-current-radii-mono-v1` grouped carriers,
then checked against the infinite analytic grouped equations.

All filters read the shared `veil` and `edge` controls; the cached room reads
the glided `position` control. A gesture is one parameter write,
not a sequential fan-out across the chord islands. The renderer sends
the first drag value immediately; while that request is in flight it retains
the latest value and one direction-reversal point for that row. A fast
out-and-back gesture therefore remains audible without allowing obsolete drag
positions to accumulate in an unbounded FIFO.

The demo opts into `Bdev = 128`, `Rgpu = 512`. Continuous epochs reserve two
device quanta ahead (5.8 ms at 44.1 kHz), publish after their first exact tile,
and fill the other three staging tiles asynchronously. Boot-muted no-op writes
prime the two coefficient families before the scene is rebased to zero.
`TROPICAL_DEMO_QUANTUM` can override both quanta for qualification.
