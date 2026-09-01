# Cliff probes

The instruments behind "The runtime kernel-size cliff: resolved" in
[`../findings.md`](../findings.md). Each isolates one variable in the emitted
kernel while holding the others fixed; run them from the repo root with
`make build && make lean` already done.

| probe | what it varies | what it holds fixed |
|---|---|---|
| `slotknee.py <stages> <counts>` | voice count, and slots-per-voice via a 1- or 2-stage voice | the compilation route |
| `threshold.py <stages> <counts>` | voice count, stock vs page-rebased | reports `__text` for both |
| `rebase3.py <in.ll> <out.ll>` | slot addressing: page-relative bases behind a zero-instruction `asm ""` identity | the instruction stream |
| `pad.py <in.ll> <out.ll> <n>` | executed code size, via independent integer adds | the real work and the data footprint |
| `stride.py <in.ll> <in.json> <out.ll> <out.json> <k>` | slots working-set bytes | the instruction stream, byte for byte |
| `bankknee.py <partials>` | partial count on the `resonator` bank | the banked lowering |

`rebase3.py`, `pad.py` and `stride.py` rewrite an `audio.ll` dumped by
`TROPICAL_STAGE0_DUMP`; feed the result to `build/tropical_runtime_bench --ir`
with the matching manifest.

Two traps worth keeping:

- A page rebase written as plain GEPs **does not survive `-O2`** — InstCombine
  reassociates it back onto one base. `rebase3.py` hides the page base behind
  an opaque identity for exactly this reason; an earlier attempt without it
  measured "no effect" for a rewrite that was not there by codegen time.
- The JIT emits objects with no asm parser, so string inline asm
  (`asm "nop"`) aborts with *"Inline asm not supported by this streamer"*.
  Only empty-body asm is usable as an optimisation barrier; `pad.py` pads with
  real `add`s instead.
