# HashNoise

Stateless, reversible, random-access noise — the closed-form twin of
`WhiteNoise`. Where `WhiteNoise` marches a `reg state` xorshift PRNG one
step per sample (so its value depends on *how it got here*), `HashNoise`
is a pure **hash of the time coordinate**: `out = hash(⌊pos⌋)`. Because
it is a function of the coordinate and nothing else, it is

- **stateless** — no register, so it passes `cfOnly` and fuses freely;
- **random-access** — evaluable at any `pos`, forward or backward;
- **reversible** — equal `pos` ⟹ equal output, so a patch driven by a
  symmetric coordinate renders a bit-exact palindrome. Scrub the
  coordinate backward and the noise replays exactly (frozen noise you can
  rewind), instead of the irreversible forward-only walk of a PRNG.

The aperiodicity is in the **hash**, not in an accumulated register: a
splitmix-style finalizer (two rounds of xorshift → odd 64-bit multiply,
then a final xorshift) gives full avalanche, so consecutive `pos`
decorrelate (measured lag-1..3 autocorrelation ≈ 0) and the spectrum is
flat-ish white. The register was never the noise — the entropy of the
sequence was, and a good hash of the index supplies it without state.

`pos` is the coordinate in **sample units** (the sample-resolution
position, e.g. `toFloat(sampleIndex())` for forward time, or a symmetric
triangle for a reversible scrub). Driving `pos` with a sub-sample-rate
coordinate quantizes the noise (a sample-and-hold of the hash), which is a
feature: low-rate `pos` gives stepped/colored noise for free.

For richer/longer-period or spectrally-shaped noise, XOR several
`HashNoise` reads at coprime per-tap offsets — each is a pure coordinate
read, so the combination stays stateless and reversible (the
memoized-short-LFSR-XOR idea, with the hash standing in for the table).

## Source

```tropical
program HashNoise(pos: float = 0) -> (out: float) {
  out = let { n: toInt(pos) } in
    let { h1: n ^ n >> 31 } in
    let { h2: h1 * 2685821657736338717 } in
    let { h3: h2 ^ h2 >> 27 } in
    let { h4: h3 * 2685821657736338717 } in
    let { h5: h4 ^ h4 >> 33 } in (h5 & 65535) * 2 / 65535 - 1
}
```
