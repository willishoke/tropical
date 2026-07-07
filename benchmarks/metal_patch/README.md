# ModalHeavy64 on Metal

A genuine tropical patch (64-partial modal voice) realized on real Metal hardware in f32,
scored against tropical's own f64 render for fitness (its own goldens, not bit-exactness),
and benchmarked for realtime latency vs a CPU-f64 baseline. This is the de-risk experiment
for `design/metal-emitter-evaluation.md` — run before committing to a full `EmitMsl` backend.

- `modal_heavy64.json` — the patch, authored in `tropical_program_2` (compiles via `diffcli`).
- `modal_params.h` — the patch's coefficients, generated for the MSL/CPU kernels.
- `modal_patch.mm` — f32 Metal kernel (per-thread = per-sample) + pipelined dispatch +
  CPU-f64 baseline + fitness scoring.

See `findings.md` for results and `make run` to reproduce (needs `mh64_ref.bin`, see there).
