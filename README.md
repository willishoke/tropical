# tropical

Realtime audio synthesis driven by Claude Code over MCP.

Describe a patch — oscillators, filters, envelopes, effects, wiring — and tropical compiles the entire signal graph into a single native kernel via LLVM ORC JIT. No interpreter, no module boundaries at runtime. Every wiring change hot-swaps a new kernel without interrupting playback.

## Install

```bash
brew install <tap>/tropical    # macOS — coming soon
```

Or [build from source](INSTALL.md). Requires LLVM >= 15, CMake, and Bun.

### Connect with Claude Code

tropical ships with `.mcp.json` — open the repo in Claude Code and the `tropical` toolset is available immediately. Then:

> Load `patches/compressor_harmonics.json` and start audio.

That's it. The MCP server handles compilation, kernel loading, and audio output. Claude can also build patches from scratch:

> Make a simple subtractive synth — sawtooth oscillator into a ladder filter, with an envelope on the cutoff. 200 Hz, slow filter sweep. Start audio.

### MCP tools

The server exposes 20 tools covering the full workflow:

**Build** — `instantiate_module`, `define_module`, `remove_module`, `list_module_types`, `list_modules`, `get_module_info`

**Wire** — `connect_modules`, `disconnect_modules`, `set_module_input`, `set_inputs_batch`, `list_inputs`

**Output** — `add_graph_output`, `remove_graph_output`

**Control** — `set_param`, `list_params`

**Patch I/O** — `load_patch`, `merge_patch`, `save_patch`

**Audio** — `start_audio`, `stop_audio`, `audio_status`

Every wiring mutation triggers a full recompile and atomic kernel swap.

## Modules

14 built-in DSP modules, all defined in TypeScript and compiled to native code at runtime:

| Module | What it does |
|--------|-------------|
| **VCO** | Band-limited oscillator (saw, tri, sin, square) with FM |
| **Clock** | Clock/trigger generator with ratio array |
| **ADEnvelope** | Attack-decay envelope, polyBLAMP antialiased |
| **ADSREnvelope** | Full ADSR envelope, polyBLAMP antialiased |
| **VCA** | Voltage-controlled amplifier |
| **LadderFilter** | 4-pole Moog-style resonant filter |
| **Reverb** | Algorithmic reverb |
| **Phaser / Phaser16** | 4 or 16 stage allpass phaser |
| **Compressor** | Dynamics compressor with sidechain input |
| **BassDrum** | Synthesized kick drum |
| **BitCrusher** | Bit depth and sample rate reduction |
| **NoiseLFSR** | Linear feedback shift register noise |
| **TopoWaveguide** | 2D waveguide mesh physical model |
| **Delay** | Fixed-length delay lines (8 to 44100 samples) |

New module types can be defined at runtime via `define_module` — no rebuild required.

## Patches

JSON files in `patches/`. Two good starting points:

- **`compressor_harmonics.json`** — Ten VCOs at the odd harmonics of 40 Hz, each gated by its own compressor/envelope pair at a different clock rate. Spectral animation through dynamics.
- **`31tet_otonal_seq.json`** — Five VCOs tuned to the overtone series in 31-tone equal temperament, with a slow transposition sequence and a sub-bass voice.

Patch format (`tropical_patch_1`):

```json
{
  "schema": "tropical_patch_1",
  "modules": [
    {"type": "VCO", "name": "VCO1"},
    {"type": "VCA", "name": "VCA1"}
  ],
  "outputs": [
    {"module": "VCA1", "output": "out"}
  ],
  "input_exprs": [
    {"module": "VCO1", "input": "freq", "expr": 440},
    {"module": "VCA1", "input": "audio", "expr": {"op": "ref", "module": "VCO1", "output": "saw"}},
    {"module": "VCA1", "input": "cv", "expr": 0.3}
  ]
}
```

## How it works

TypeScript defines modules as symbolic expression trees. The compiler flattens all modules into a single instruction stream, lowers array operations to scalar primitives, and emits a typed program. This crosses a stable C API (via koffi FFI) as JSON, where the C++ engine JIT-compiles it to a native kernel using LLVM ORC. The kernel runs per-sample in an audio callback.

Rewiring a connection recompiles the entire patch and atomically swaps the kernel — state is transferred by name, so registers and delay lines survive the swap. No click, no gap.

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
