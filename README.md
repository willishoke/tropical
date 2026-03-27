# egress

## Intro

`egress` is a C++ library for realtime audio synthesis. Modules are defined using built-in or user-defined operations, and connections are expressed using a symbolic expression syntax. JIT compilation for module definitions ensures fast realtime execution with native audio playback. Currently only tested on macOS.

![demo](./img/testchaos.png)

## Build

```bash
make build
```

Configures and builds the C++ core via CMake. Requires [LLVM](https://llvm.org) (ORC JIT).

## TUI and MCP server

The primary interface is a TypeScript TUI and MCP server in `tui/`. Requires [Bun](https://bun.sh).

```bash
make tui-ts    # launch the full-screen Ink/React TUI
make mcp-ts    # launch the MCP server standalone on stdio
```

The TUI is a full-screen terminal interface for building and manipulating patches interactively — create modules, wire connections, set parameters, and control playback.

The MCP server exposes the full egress graph API as MCP tools, so any MCP-compatible client (including Claude) can build and manipulate patches programmatically at runtime.

Sample patches live in `patches/`.

## Graph

`Graph` stores modules, per-input expression trees, output taps, and the output buffer. Each input is represented by a single expression tree whose leaves are literals or references to module outputs. `Graph::process()` evaluates those input expressions sample-by-sample, processes modules, and mixes selected outputs into the output buffer. Top-level module references always read previous-sample outputs, so connected modules have a single-sample boundary even when graph-level worker threads are enabled.

Modules expose named inputs and outputs plus an optional register bank. After each sample, the runtime applies register updates and resets inputs to their default values. Output signals are clipped to `[-10.0, 10.0]`; most patches are expected to stay in the bipolar `[-5.0, 5.0]` range.

## Testing

Tests can be compiled with `make debug`. The `test` directory contains scripts for visualizing test outputs.

## License

Free use of `egress` is permitted under the terms of the MIT License.
