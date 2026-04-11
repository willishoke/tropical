# mcp/

MCP server — the primary agent interface. Runs on stdio via `@modelcontextprotocol/sdk`.

## Running

```bash
make mcp-ts    # build C++ core + launch MCP server
```

Also configured in `.mcp.json` for Claude Code integration.

## Layout

```
server.ts       MCP server: session management, tool definitions, request handlers
patch.test.ts   Patch round-trip tests
```

## How it works

The server maintains a `SessionState` (from `compiler/patch.ts`) containing the type registry, program instances, wiring expressions, graph outputs, control parameters, and a `Runtime` handle.

Every mutation that affects the signal graph calls `wire()`, which runs the full compilation pipeline: `flattenPatch()` → `JSON.stringify()` → `runtime.loadPlan()`. This recompiles and hot-swaps the kernel. Errors during compilation are caught and returned as tool error responses.

## Tools

### Program management
- `define_program` — register a reusable DSP program type (accepts ProgramJSON)
- `add_instance` — create a named instance of a registered program type
- `remove_instance` — remove an instance and cascade-clean wiring
- `list_programs` — list all registered program types with ports and defaults
- `list_instances` — list all live instances
- `get_info` — detailed info about an instance (ports, wiring, registers)

### Wiring
- `wire` — set and/or remove input wiring in a single recompile. Replaces connect_modules, disconnect_modules, set_module_input, set_inputs_batch.
- `list_wiring` — show current input expressions, optionally filtered by instance

### Audio output
- `set_output` — declaratively set the full audio output list (replaces add/remove_graph_output)

### Control parameters
- `set_param` — set a named smoothed or trigger parameter value
- `list_params` — list all registered parameters

### Program I/O
- `load` — load a `tropical_program_1` or `tropical_patch_1` JSON (replaces session)
- `save` — serialize current session to `tropical_program_1` JSON
- `merge` — merge a program into the current session (additive)

### Audio control
- `start_audio` — open audio device and begin playback
- `stop_audio` — stop playback
- `audio_status` — device info, callback stats, running state

### Deprecated aliases
Old tool names (`define_module`, `instantiate_module`, `connect_modules`, `load_patch`, etc.) still work via an alias layer for backward compatibility. Descriptions are prefixed with `[deprecated]`.
