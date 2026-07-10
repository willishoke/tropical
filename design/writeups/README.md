# Writeups

Standalone technical explainers — each one takes a single idea about what
Tropical is or does and explains it from the ground up, for a reader who
does not already live in the codebase. These are distinct from the design
notes and handoffs elsewhere in `design/`: those are working documents for
the build; these are meant to be *read*.

Figures are committed as vector SVG under `figures/` and regenerated from
their source scripts; the house aesthetic is thin grayscale lines,
Computer-Modern serif, high data-ink ratio.

## Index

Reading order runs foundation-first; each effect leans on the atom below it.

- [The sawtooth is a line](sawtooth-is-a-line.md) — the foundational atom:
  the simplest closed-form signal is a linear function of time read modulo
  one, and the classic waveforms are memoryless shapings of that one ramp.
- [Through-zero flanging](through-zero-flanging.md) — why the canonical
  tape-studio effect cannot be done at zero latency on a sample stream,
  and why it is free for a closed-form kernel.
- [Reverse reverb](reverse-reverb.md) — why the reverberant swell that
  arrives before a sound has always been an offline studio trick, and how
  a closed-form kernel plays it live at zero latency.
- [Phase alignment by anchoring](phase-anchoring.md) — how an oscilloscope
  makes a waveform stand still, and why the century-old trigger dissolves
  into an exact subtraction once every signal is a function of time.
