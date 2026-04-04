# egress

A C++ library for algorithmic and generative audio synthesis. You build a graph of modules — oscillators, envelopes, effects, sequencers — wire them together with expressions, and the engine runs them in realtime at 44.1 kHz through your audio interface. Load the 31-TET patch and you hear five microtonal sine voices stepping slowly through an otonal sequence, smeared by a 16-stage phaser and a long reverb tail.

Modules are defined using built-in or user-defined operations; connections are expressed using a symbolic expression syntax. An LLVM ORC JIT backend compiles module definitions to native code for realtime execution. Currently only tested on macOS.

## Getting started

Build the core and start the MCP server:

```bash
make build
make mcp-ts
```

Then connect any MCP-compatible client — Claude or otherwise — and load a patch:

> Load `patches/31tet_otonal_seq.json` and start audio.

The MCP server exposes `load_patch` and `start_audio` tools; the client will call them automatically. Within a few seconds you should hear five microtonal sine voices cycling slowly through a 31-TET otonal sequence, passing through a 16-stage phaser and a long reverb tail.

Two patches are included to start:

- **`31tet_otonal_seq.json`** — Five VCOs tuned to the overtone series in 31-tone equal temperament, with a slow global transposition sequence and a sub-bass voice. Additive drone synthesis; long reverb. Good for getting a feel for expression-based routing and multi-voice patches.
- **`compressor_harmonics.json`** — Ten VCOs at the odd harmonics of 40 Hz (the spectral content of a square wave), each gated by its own compressor/envelope pair at a different clock rate. Spectral animation through dynamics rather than amplitude — each partial opens and closes independently.

## Build

```bash
make build
```

Configures and builds the C++ core via CMake. Requires [LLVM](https://llvm.org) (ORC JIT).

## Graph

`Graph` stores modules, per-input expression trees, outputs, and the output buffer. Each input is represented by a single expression tree whose leaves are literals or references to module outputs. `Graph::process()` evaluates those input expressions sample-by-sample, processes modules, and mixes selected outputs into the output buffer.

Top-level module references always read previous-sample outputs, so connected modules observe a single-sample boundary even when graph-level worker threads are enabled.

Modules expose named inputs and outputs plus an optional register bank. After each sample, the runtime applies register updates and resets inputs to their default values.

## Patches

Patches are JSON files describing a graph: modules, input expressions, outputs, and parameter values. Sample patches live in `patches/`.

The canonical format uses `input_exprs` to describe routing — each input holds an expression tree whose leaves reference other module outputs:

```json
"input_exprs": {
  "freq": { "op": "ref", "module": "Clock1", "output": "out" },
  "amp":  { "op": "mul", "a": { "op": "ref", "module": "Env1", "output": "out" }, "b": 0.8 }
}
```

The legacy `connections` array is deprecated. Replace it with `input_exprs` entries on the receiving module.

## C API

A stable C API is exposed in `src/c_api/egress_c.h`. This is the integration point for language bindings and external tools.

## JIT

Module kernels are compiled to native code on first use via LLVM ORC JIT and cached on disk across process restarts. The JIT uses a static type system (float/int/bool) derived from expression structure.

## Testing

```bash
make debug
```

Builds and runs the C++ test suite. Tests cover module processing, expression evaluation, and the JIT pipeline. No audio device is required.

## Profiling

```bash
make profile
```

Builds with timing instrumentation. Profile stats are accessible via the C API and at runtime through the MCP server.

## MCP Server (experimental)

An MCP server in `tui/` exposes the full graph API as tools, allowing MCP-compatible clients to build and manipulate patches programmatically. Requires [Bun](https://bun.sh).

```bash
make mcp-ts
```

**The TUI and MCP server are experimental and unsupported.** The C++ core and C API are the stable surface.

## Troubleshooting

**JIT compilation failure** — JIT failures are fatal; there is no interpreter fallback. If the engine throws on startup, check that your LLVM installation matches the version expected by the build (see `CMakeLists.txt`). Stale cached kernels can also cause issues; clear `~/.cache/egress/kernels/` and rebuild.

## License

Free use of `egress` is permitted under the terms of the MIT License.
