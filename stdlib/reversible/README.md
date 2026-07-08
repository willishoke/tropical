# stdlib/reversible/

A small suite exploring **reversible, closed-form-in-τ synthesis**: programs
that are pure functions of a navigable time coordinate `tau`, so that running
time backward runs the *sound* backward — exactly. The discipline is simple
and enforced by one rule: confine all state to a single signed-velocity time
clock, and build everything else as a closed-form function of its `tau`.
Reverse that one clock and the whole patch reverses, because nothing
downstream holds history.

These compile and run through the same pipeline as the base stdlib (parse →
elaborate → strata → JIT/wasm). They are picked up by `make parse-all`
(extended to scan this subdirectory) and registered in
`stdlib/parsed/manifest.json`.

## The programs

| Program | Role |
|---|---|
| `ScrubClock` | The host transport made explicit — the Q32.32 integer clock `clk` advances by a *signed* `velocity`. The live "finger on the tape": drive `param:velocity` negative to reverse, zero to freeze. |
| `ModalVoice` | A closed-form modal voice — a sum of undamped, incommensurate sine partials, a pure function of the integer clock. Conservative (undamped) so it reverses cleanly and stays bounded. |
| `ReversibleComb` | A comb/flange built from *offset reads* of the source — `clk`, `clk − Δ`, and `clk + Δ` (the future tap, offsets landed on the integer clock). The closed-form analogue of a delay line; reads ahead because the source has a computable future. |
| `ReversibleProbe` | The reversibility witness. Drives a palindromic integer clock from the sample counter (forward to `half`, then back) and feeds the comb. Its render is a **bit-exact palindrome** about sample `half`. |

## The test

`tropicaltest` (the built binary — never `lake exe`, see the repo CLAUDE.md) renders `patches/reversible_probe.json` (a one-instance
patch over `ReversibleProbe`) for `2·half` samples and asserts the output is a
bit-exact palindrome about `half` — i.e. `out[half + k] == out[half − k]` for
every `k`. Equal clock ⟹ equal output, sample for sample; a single mismatched
pair would mean a register leaked in and broke purity. The witness is exact
(not within-epsilon) because the palindromic clock is a *coordinate* computed
by a symmetric formula, not an accumulated state.

## Build

These programs are picked up by `make parse-all` (`parseAllVerb` scans this
subdirectory after the base stdlib, so their dependencies — `Sin` etc. —
resolve first) and registered in `stdlib/parsed/manifest.json`. The parsed
bridge is fully regenerable: re-running `make parse-all` is idempotent.
