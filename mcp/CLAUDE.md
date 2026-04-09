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

The server maintains a `SessionState` (from `compiler/patch.ts`) containing the type registry, module instances, wiring expressions, graph outputs, control parameters, and a `Runtime` handle.

Every mutation that affects the signal graph calls `wire()`, which runs the full compilation pipeline: `flattenPatch()` → `JSON.stringify()` → `runtime.loadPlan()`. This recompiles and hot-swaps the kernel. Errors during compilation are caught and returned as tool error responses.

`set_inputs_batch` batches multiple wiring changes into a single recompile.

## Tools

### Module management
- `define_module` — register a new module type from JSON definition
- `instantiate_module` — create a named instance of a registered type
- `remove_module` — remove an instance and its wiring
- `list_module_types` — list all registered types
- `list_modules` — list all live instances
- `get_module_info` — type info, inputs, outputs for a module

### Wiring
- `connect_modules` — wire source output → destination input (as ref expression)
- `disconnect_modules` — remove a connection
- `set_module_input` — set an input to an arbitrary expression (literal, ref, op tree)
- `set_inputs_batch` — set multiple inputs in one recompile
- `list_inputs` — show current input expressions for a module

### Audio graph
- `add_graph_output` — add a module output to the audio mix
- `remove_graph_output` — remove from mix

### Control parameters
- `set_param` — set a named smoothed or trigger parameter value
- `list_params` — list all registered parameters

### Patch I/O
- `load_patch` — load a `tropical_patch_1` JSON file (replaces session)
- `merge_patch` — merge a patch into the current session (additive)
- `save_patch` — serialize current session to `tropical_patch_1` JSON

### Audio control
- `start_audio` — open audio device and begin playback
- `stop_audio` — stop playback
- `audio_status` — device info, callback stats, running state
