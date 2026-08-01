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

Four sparse inharmonic metal hits land halfway between the chords. Each is a
finite causal attack pair sent into a longer 16-mode room; the room supplies
the dense ringing field without turning the exciter itself into a large bank.

## Surface

- Click any of the sixteen moments, or a chord caption, to seek.
- Use the seven rows directly with the mouse.
- Arrow up/down chooses a row; left/right changes it; Shift makes a fine move.
- Space holds or resumes the scene. `R` reverses it. Enter returns to zero.
- `STRINGS / RESONANT VEIL` is the filtered direct string field.
- `ROOM / METAL RETURN` is the wet modal room before its return-level control.

`presence`, `veil`, `edge`, `room`, and `length` shape the scene. `flow` is a
continuous local clock velocity, including zero and reverse. `level` is the
final relative listening gain. Amplitude rows use a shared closed-form 20 ms
glide. `veil`, `edge`, and `length` remain epoch-rate modal controls so their
coefficient banks stay hoistable; continuous Metal handoffs dezipper the
resulting output seam over one 128-sample demo quantum instead of stepping
from the old waveform to the new one.

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

Each chord remains its own modal island until the signal boundary; this
preserves all four strike anchors (a modal pole union has only one). Its
pre-veil send therefore acts directly on the modal chord bank. String branches
request 12 room modes and metal-hit branches request 16; reverb graphs that
omit the structural `modes` field retain the engine's existing 32-mode default.

All filters and room branches read three shared control nodes
(`veil`, `edge`, and `length`). A gesture is one parameter write and one Metal
epoch, not a sequential fan-out across the chord islands. The renderer sends
the first drag value immediately; while that request is in flight it retains
the latest value and one direction-reversal point for that row. A fast
out-and-back gesture therefore remains audible without allowing obsolete drag
positions to accumulate in an unbounded FIFO.

The demo opts into `Bdev = Rgpu = 128`. Continuous epochs reserve two render
quanta ahead (5.8 ms at 44.1 kHz), publish after their first exact tile, and
fill the other three staging tiles asynchronously. Boot-muted no-op writes
prime the three coefficient families before the scene is rebased to zero.
`TROPICAL_DEMO_QUANTUM` can override both quanta for qualification.
