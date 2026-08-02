# Fixed-room audition

Rendered WAVs are reproducible local listening evidence and are intentionally
Git-ignored. The repository tracks the generators, JSON summaries/profiles,
and the accepted compact mono carrier `.npz` asset. Run the commands below to regenerate
the referenced WAV sets; the native 44.1 kHz production asset defined by the
handoff will be tracked separately.

This directory is the pre-integration listening gate for the modal-pocket demo's
fixed room. It reproduces the scene's exact first metal hit, compares the
incumbent 16-mode generated room with three bounded early/late candidates, and
exports linearly loudness-matched 24-bit WAVs. No limiter or dynamics processor
is used.

Run from the repository root:

```sh
uv run --with numpy python benchmarks/demo_release/room_audition/run.py
```

The generated files live in `out/`:

- `dry_metal.wav`
- `incumbent_wet.wav` and `incumbent_mix.wav`
- one wet-only and one mix WAV for each candidate
- `profiles.json`, containing the exact frozen modal rows
- `summary.json` and `summary.md`, containing levels and room-gate measures

The audition is intentionally separate from scene integration. One candidate
must be selected by listening before its rows, early taps, predelay, and gains
are admitted to the production graph.

## Reference-first second pass

The first three modal candidates were rejected at listening review as resonant
objects rather than massive rooms. `run_reference.py` renders the official
Mutable Instruments Clouds reverb at frozen Eurorack commit
`08460a69a7e1f7a81c5a2abcc7189c9a6b7208d4`. It processes at Clouds' 32 kHz
rate, exports stereo and mono-fold wet/mix references at 44.1 kHz, and records
echo-density, spectral-flatness, and stereo-correlation targets.

```sh
uv run --with numpy python benchmarks/demo_release/room_audition/run_reference.py \
  --clouds-root /path/to/pichenettes/eurorack
```

This is an audition benchmark only. The Mutable stateful reverb is not linked
into the product, and no rejected modal candidate is integrated.

The host build defines the official source's `TEST` switch only to select its
portable C++ saturation helpers in place of Cortex-M `ssat`/`usat` assembly.
The reverb topology, delay memory format, coefficients, LFOs, and processing
code are otherwise the frozen official implementation.

## Modal capacity and hybrid pass

`run_modal_match.py` projects the Clouds mono impulse into deterministic modal
bloom shells at 256, 1,024, and 4,096 frequency cells. It also renders bounded
experiments using the repository's closed-form swept-delay idiom and a finite
projection of Clouds' four input diffusers. These are capacity/architecture
probes, not production presets.

```sh
uv run --with numpy python benchmarks/demo_release/room_audition/run_modal_match.py \
  --clouds-root /path/to/pichenettes/eurorack
```

Listen in this order:

1. `reference_out/clouds_gold_mix_stereo.wav` — the gold standard.
2. `reference_out/clouds_gold_mix_mono.wav` — isolates what the current mono
   DAC contract loses.
3. `match_out/modal_cloud_1024_mix.wav` — stationary modal baseline at a much
   larger capacity than v1.
4. `match_out/modal_cloud_1024_smear_wide_3_mix.wav` — strongest closed-form
   smear probe.
5. `match_out/modal_cloud_1024_diffused_smear_mutable_rate_mix.wav` — finite
   four-diffuser plus Mutable-rate smear probe.

All comparisons share one wet RMS target, one wet/dry mix ratio, and one common
headroom gain. `match_summary.json` records the target and candidate features;
`modal_profiles.json` records the exact rows and hybrid parameters.

## Structured stateful-reverb recovery

`run_structured_recovery.py` is the feasibility experiment for recovering a
stateful delay reverb without flattening every delay sample into an independent
runtime oscillator. It derives and validates all 9,133 modes of an inspectable
Freeverb-style LTI teacher, tests ordinary modal truncation, and then collects
complete delay pole lattices into periodic carriers with analytic convolution
against the exact modal metal source.

```sh
uv run --with numpy python \
  benchmarks/demo_release/room_audition/run_structured_recovery.py \
  --clouds-root /path/to/pichenettes/eurorack
```

The mathematical result is exact to the reported floating-point floor. Listen
to these files before drawing the product conclusion:

1. `recovery_out/clouds_gold_mix_mono.wav`
2. `recovery_out/freeverb_teacher_mix.wav`
3. `recovery_out/freeverb_stratified_1024_mix.wav`
4. `recovery_out/freeverb_stratified_3072_mix.wav`

The full teacher WAV is also the output of the structured 12-group recovery;
the two paths differ by only `2.72e-13` relative L2, so exporting duplicate WAVs
would add no evidence. See `03-room-recovery-feasibility.md` for the derivation,
scope boundary, and decision implications.

For the reverb tails without the Metal source or dry mix, use the separately
loudness-matched wet impulse-response set:

1. `recovery_out/clouds_gold_impulse_mono.wav`
2. `recovery_out/freeverb_teacher_impulse.wav`
3. `recovery_out/freeverb_stratified_1024_impulse.wav`
4. `recovery_out/freeverb_stratified_3072_impulse.wav`

`freeverb_energy_1024_impulse.wav` is also retained as the energy-ranked
negative-control variant. All impulse responses are 12-second, 24-bit mono
44.1 kHz files. They are active-RMS matched to Clouds and share one peak-safe
headroom gain; no source signal is embedded.

## Clouds-nearer grouped carrier fit

`run_grouped_clouds_fit.py` retains the exact 12-group evaluator but fits the
complete lattice carriers offline to the Clouds impulse. It also builds a
stereo side asset by deterministic residue-phase scrambling; this preserves
the fitted modal magnitude distribution while approximating Clouds' spatial
decorrelation without live modulation or more pole groups.

```sh
uv run --with numpy python \
  benchmarks/demo_release/room_audition/run_grouped_clouds_fit.py \
  --clouds-root /path/to/pichenettes/eurorack
```

Listen first to the mono impulse progression:

1. `grouped_fit_out/clouds_gold_impulse.wav` — official stateful control.
2. `grouped_fit_out/freeverb_teacher_impulse.wav` — accepted exact grouped
   baseline.
3. `grouped_fit_out/grouped_fit_current_radii_impulse.wav` — fixed-cost fitted
   candidate.

Then compare the native stereo fields:

1. `grouped_fit_out/clouds_gold_stereo_impulse.wav`
2. `grouped_fit_out/grouped_fit_current_radii_stereo_impulse.wav`

The corresponding `*_wet.wav` files replace the impulse with the exact Metal
source. `grouped_fit_decay_ladder_impulse.wav` is a negative-control fit. The
summary records exact provenance, metrics, cost, hashes, and structured-versus-
ordinary convolution error.

## Classic reverse-room POSITION

`run_reverse_position.py` proves the requested classic operation against a
literal finite reverse-source/reverb/reverse reference. It also renders the
continuous temporal-position control using the fitted mid/side carrier assets.
This is not the incumbent whole-output direction mirror; the measurements show
that mirror has `1.0` NRMSE against the classic operation.

```sh
uv run --with numpy python \
  benchmarks/demo_release/room_audition/run_reverse_position.py
```

The event is anchored at six seconds. Listen to:

1. `reverse_position_out/position_1p0_stereo_mix.wav` — forward endpoint.
2. `reverse_position_out/position_0p0_stereo_mix.wav` — two-sided midpoint.
3. `reverse_position_out/position_minus_1p0_stereo_mix.wav` — classic reverse
   endpoint.
4. `reverse_position_out/position_scrub_forward_to_reverse_stereo_mix.wav` —
   a live-style scrub completed while the six-second event is still upcoming.

Wet-only and intermediate `±0.5` files are included. See
`05-reverse-position-scope.md` for the proof, cost fork, demo timing, and hard
integration gates.

## Native 44.1 kHz production gate

Gate 0 renders Clouds once at its frozen 32 kHz rate, resamples the complete
target once before optimization, then fits and evaluates the frozen twelve
native periods directly at 44.1 kHz:

```sh
uv run --with numpy python \
  benchmarks/demo_release/room_audition/run_grouped_clouds_fit.py \
  --sample-rate 44100 \
  --output-dir benchmarks/demo_release/room_audition/native_rate_out \
  --clouds-root /path/to/pichenettes/eurorack

uv run --with numpy python \
  benchmarks/demo_release/room_audition/run_reverse_position.py \
  --sample-rate 44100 \
  --asset benchmarks/demo_release/room_audition/native_rate_out/grouped_fit_current_radii.npz \
  --output-dir benchmarks/demo_release/room_audition/native_rate_out \
  --mono

uv run --with numpy python \
  benchmarks/demo_release/room_audition/generate_grouped_room_asset.py
```

The last command emits the tracked `.tgrm` payload and JSON manifest under
`playground/assets/grouped-room/`. The manifest documents the fixed v1 binary
header, aligned metadata, source/group table order, input and payload hashes,
oracle results, and listening status. Native audition WAVs remain ignored;
the accepted mono carrier `.npz`, summaries, production payload, and manifest
are tracked.

The production runtime probe compiles a real `groupedroom` graph, loads that
tracked asset through Plan 6, renders the five endpoint/fractional fixtures,
and compares the result with the analytic grouped-room oracle:

```sh
uv run --with numpy python \
  benchmarks/demo_release/room_audition/run_grouped_room_runtime_probe.py \
  --backend jit

# Run on a release Mac with Metal available.
uv run --with numpy python \
  benchmarks/demo_release/room_audition/run_grouped_room_runtime_probe.py \
  --backend both
```

The enforced worst-case absolute-error limits are `1e-9` for JIT and `1e-5`
for Metal.

## Selected fixed-scene cache fallback

The release-Mac direct evaluator reserve failure is retained under
`benchmarks/demo_release/data/`. Regenerate the selected two-arm, 16-second
float32 basis directly from `playground/scene.js`, then run its endpoint and
FLOW gate:

```sh
uv run --with numpy python \
  benchmarks/demo_release/room_audition/generate_grouped_room_scene_cache.py

uv run --with numpy python \
  benchmarks/demo_release/room_audition/run_grouped_room_scene_cache_probe.py
```

The 5,644,800-byte payload contains causal then classic-reverse mono arrays.
Runtime addressing is cyclic linear interpolation from the master scene
coordinate; POSITION retains the direct evaluator's equal-power mix. Integer
endpoints agree with the direct JIT at `2.62e-8` relative error, and the frozen
fractional/stopped/reverse FLOW matrix stays below `1.74%` NRMSE.

## Release listening set

Generate the continuous, authored-level listening artifacts from a clean
candidate with:

```sh
uv run --with numpy python \
  benchmarks/demo_release/room_audition/generate_grouped_room_release_wavs.py
```

The renderer verifies the scene/cache hashes, renders dry and static endpoint
graphs through the native JIT, applies the runtime equal-power law to the
documented POSITION choreography, and refuses parity or PCM-headroom failure.
It applies no normalization or limiter. Listen in this order:

1. `release_out/wet_position_plus_1.wav` — causal wet endpoint.
2. `release_out/wet_position_minus_1.wav` — classic reverse wet endpoint.
3. `release_out/wet_position_scrub.wav` — forward/reverse choreography only.
4. `release_out/final_scene_dry.wav` — strings and causal dry impacts.
5. `release_out/final_scene_position_choreography.wav` — final review capture.

`release_out/release_listening_summary.json` records the clean source commit,
graph/cache hashes, endpoint parity, gain staging, peak/RMS values, and every
WAV hash. The summary deliberately leaves headphone/monitor approval pending
until a user listens to the final capture.
