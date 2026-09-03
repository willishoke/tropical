# Sine probes

The instruments behind "Reading 2 resolved" in
[`../findings.md`](../findings.md). Run from the repo root with
`make build && make lean` already done.

| probe | what it measures |
|---|---|
| `loopdiff.py` | marginal per-voice instructions in the PER-SAMPLE LOOP BODY, tropical vs Faust F3, by differencing N=64 -> 128 |
| `disasm.py` | the same difference over the whole function — superseded by `loopdiff.py`, kept because it is what exposed the preheader trap |
| `conststore.py <stages> <counts>` | what deleting the dead constant slot stores is worth (answer: it is negative) |
| `accuracy.py` | max error of both sine kernels vs libm — what F1's speed actually costs |

- **F1 interpolates nothing.** Faust's `os.osc` compiles to a TRUNCATED
  65536-entry table lookup, not the interpolated one the earlier findings
  text claimed. It is 88 dB less accurate than tropical's polynomial
  (`accuracy.py`), so its ~750 ns/voice is a different point on the
  accuracy/cost curve, not a gap to close.

## Traps

- **Count the loop body, not the function.** Both compilers hoist
  block-invariant work into a preheader that runs once per 512-frame block.
  tropical's `freq * 2^32 / sampleRate` is an `fdiv` per voice sitting in
  `loop_body.lr.ph`, not in the sample loop. Counting the whole function
  prices it as if it were hot and inflates tropical's per-voice figure by
  ~7 instructions. `disasm.py` makes this mistake; `loopdiff.py` does not.

- **Faust's static counts are ~3x its dynamic ones.** clang emits three
  copies of the frame loop — unrolled-by-2, a scalar fallback for the
  aliasing case, and a remainder — so N=64 shows 192 `bl _sin`. Divide by
  the number of loop copies, or difference loop bodies as `loopdiff.py`
  does.

- **The frozen `tests/golden/stdlib/*.hash` goldens are STRUCTURE, not
  audio.** They freeze wire+port shape, so any builder change that emits
  fewer nodes moves them, and they cannot testify to numerical equivalence.
  The gates that actually render and compare bits are `bootstrap-sin`,
  `fixedsin-longtau`, `negative-clock`, and `arrow-law/*-fixedpoint`.

- **Fewer instructions is not always faster, and neither is less code.**
  Deleting the ~1 dead constant store per voice per sample shrinks `__text`
  by ~19% and makes the kernel 5-10% SLOWER, reproducibly, at N=64-192.
  Measured before being believed; see `conststore.py` and the findings.

- **`slotknee.run` cannot build the fixture much past N=192.** Its patch
  builder nests `add` dicts and the JSON encoder hits its recursion limit;
  `benchmarks/faust_comparison/run.py` avoids this by emitting the
  expression as text. Raising the recursion limit risks the C stack instead.
