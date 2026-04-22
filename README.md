# tropical

Realtime audio synthesis driven by Claude Code over MCP.

Describe a program — oscillators, filters, envelopes, effects, wiring — and tropical compiles the entire signal graph into a single native kernel via LLVM ORC JIT. No interpreter, no module boundaries at runtime. Every wiring change hot-swaps a new kernel without interrupting playback.

## Install

```bash
brew install <tap>/tropical    # macOS — coming soon
```

Or [build from source](INSTALL.md). Requires LLVM >= 15, CMake, and Bun.

### Connect with Claude Code

tropical ships with `.mcp.json` — open the repo in Claude Code and the `tropical` toolset is available immediately. Then:

> Load `patches/compressor_harmonics.json` and start audio.

That's it. The MCP server handles compilation, kernel loading, and audio output. Claude can also build patches from scratch:

> Define a sine oscillator, run it through a ladder filter with resonance at 0.8, and start audio.

### MCP tools

The server exposes 16 tools covering the full workflow:

**Program management** — `define_program`, `add_instance`, `remove_instance`, `list_programs`, `list_instances`, `get_info`

**Wiring** — `wire` (batched set/remove in a single recompile), `list_wiring`

**Output** — `set_output` (declaratively set the full output list)

**Control** — `set_param`, `list_params`

**Program I/O** — `load`, `save`, `merge`

**Audio** — `start_audio`, `stop_audio`, `audio_status`

Every wiring mutation triggers a full recompile and atomic kernel swap.

## Programs

19 built-in DSP program types, all defined as human-readable JSON and compiled to native code at runtime:

| Program | What it does |
|---------|-------------|
| **Sin / Cos / Tanh** | Polynomial approximations (7th-order minimax for sin/cos, Padé for tanh) |
| **Exp / Log / Pow** | Cody-Waite + Horner polynomial for exp; exponent extraction + Remez for log; `exp(y · log(x))` for pow |
| **OnePole** | One-pole lowpass filter with tanh saturation |
| **AllpassDelay** | First-order allpass, transposed direct form II |
| **CombDelay** | Feedback comb filter |
| **SoftClip** | Soft clipper (tanh waveshaper) |
| **CrossFade** | Linear crossfade between two signals |
| **LadderFilter** | 4-pole Moog-style resonant filter (composed from 4 OnePole instances) |
| **Clock** | Clock/trigger generator with ratio array |
| **VCA** | Voltage-controlled amplifier |
| **Phaser / Phaser16** | 4 or 16 stage allpass phaser |
| **BitCrusher** | Bit depth and sample rate reduction |
| **NoiseLFSR** | Linear feedback shift register noise |
| **Delay** | Fixed-length delay line, parameterized by `N` samples (e.g. `{program:"Delay", type_args:{N:8}}`) |

Complex types compose from simpler ones — LadderFilter is a few hundred lines of JSON using OnePole and Sin instances, not an opaque blob. Even transcendentals are programs: swap `stdlib/Sin.json` to change the approximation. New program types can be defined at runtime via `define_program` — no rebuild required.

## Patches

JSON files in `patches/`. Examples include cross-FM synthesis, acid noise, and microtonal sequencing.

Program format (`tropical_program_2`):

```json
{
  "schema": "tropical_program_2",
  "name": "Example",
  "body": {
    "op": "block",
    "decls": [
      { "op": "program_decl", "name": "Sine", "program": {
        "op": "program",
        "name": "Sine",
        "ports": {
          "inputs": [{ "name": "freq", "default": 440 }],
          "outputs": [{ "name": "out" }]
        },
        "body": {
          "op": "block",
          "decls": [
            { "op": "reg_decl", "name": "phase", "init": 0 },
            { "op": "instance_decl", "name": "sin1", "program": "Sin", "inputs": {
              "x": { "op": "mul", "args": [6.283185307179586, { "op": "reg", "name": "phase" }] }
            }}
          ],
          "assigns": [
            { "op": "output_assign", "name": "out", "expr": { "op": "nested_out", "ref": "sin1", "output": "out" } },
            { "op": "next_update", "target": { "kind": "reg", "name": "phase" }, "expr": {
              "op": "mod", "args": [
                { "op": "add", "args": [
                  { "op": "reg", "name": "phase" },
                  { "op": "div", "args": [{ "op": "input", "name": "freq" }, { "op": "sample_rate" }] }
                ]},
                1
              ]
            }}
          ]
        }
      }},
      { "op": "instance_decl", "name": "osc", "program": "Sine", "inputs": { "freq": 440 } },
      { "op": "instance_decl", "name": "filt", "program": "LadderFilter", "inputs": {
        "input": { "op": "ref", "instance": "osc", "output": "out" },
        "cutoff": 2000, "resonance": 0.7
      }}
    ],
    "assigns": []
  },
  "audio_outputs": [
    { "instance": "filt", "output": "lp" }
  ]
}
```

## How it works

TypeScript defines programs as symbolic expression trees. The compiler flattens all instances into a single instruction stream, lowers array operations to scalar primitives, and emits a typed program. This crosses a stable C API (via koffi FFI) as JSON, where the C++ engine JIT-compiles it to a native kernel using LLVM ORC. The kernel runs per-sample in an audio callback.

Rewiring a connection recompiles the entire program and atomically swaps the kernel — state is transferred by name, so registers and delay lines survive the swap. No click, no gap. Feedback loops (A→B→A or A→A) resolve automatically with a one-sample delay, just like hardware propagation — no special configuration needed.

See `design/architecture.md` for the full technical reference.

## Development

```bash
make build                                          # build C++ engine
cmake --build build -j4 && ctest --test-dir build   # C++ tests (JIT + C API)
bun test                                            # TypeScript compiler tests
```

## Troubleshooting

**JIT compilation failure** — JIT failures are fatal; there is no interpreter fallback. Check that your LLVM installation matches what CMakeLists.txt expects. Stale cached kernels can also cause issues — clear `~/.cache/tropical/kernels/` and rebuild.

**No audio output** — Verify your default output device is set correctly in System Settings. `audio_status` reports device info and callback stats.

## License

MIT
