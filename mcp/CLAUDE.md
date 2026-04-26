# mcp/

MCP server — the primary agent interface. Runs on stdio via `@modelcontextprotocol/sdk`.

## Running

```bash
make mcp-ts    # build C++ core + launch MCP server
```

Also configured in `.mcp.json` for Claude Code integration.

## Layout

```
server.ts      MCP server: session management, tool definitions, request handlers
test_patch.ts  Standalone CLI smoke-tester: bun run mcp/test_patch.ts <patch.json> [n_frames]
```

## How it works

The server maintains a `SessionState` (from `compiler/session.ts`) containing the type registry, program instances, wiring expressions, graph outputs, control parameters, and a `Runtime` handle.

Every mutation that affects the signal graph calls `wire()`, which runs the full compilation pipeline: `flattenSession()` → `JSON.stringify()` → `runtime.loadPlan()`. This recompiles and hot-swaps the kernel. Errors during compilation are caught and returned as tool error responses.

## Tools

### Program management
- `define_program` — register a reusable DSP program type (accepts a `tropical_program_2` object); generic programs declaring `type_params` are stored as templates and monomorphize on instantiation
- `add_instance` — create a named instance of a registered program type; pass `type_args` (e.g. `{"N": 8}`) for generic programs
- `remove_instance` — remove an instance and cascade-clean wiring
- `list_programs` — list all registered program types with ports and defaults; generic programs also surface their `type_params`
- `list_instances` — list all live instances (includes `type_args` for generic instances)
- `get_info` — detailed info about an instance (ports, wiring, registers, `type_args`)

### Export
- `export_program` — crystallize session instances into a reusable program type. Specify input/output mappings; current wiring becomes defaults. Optionally removes exported instances from the session.

### Wiring
- `wire` — set and/or remove input wiring in a single recompile. Audio output uses the same tool: wire to `instance: "dac", input: "out"` with a ref-shaped expression. Multiple wires to dac.out sum into the mono output bus; remove with `{instance: "dac", input: "out"}` clears all dac wires.
- `list_wiring` — show current input expressions, optionally filtered by instance

### Control parameters
- `set_param` — set a named smoothed or trigger parameter value
- `list_params` — list all registered parameters

### Program I/O
- `load` — load a `tropical_program_2` JSON (replaces session)
- `save` — serialize current session to `tropical_program_2` JSON
- `merge` — merge a program into the current session (additive)

### Audio control
- `start_audio` — open audio device and begin playback
- `stop_audio` — stop playback
- `audio_status` — device info, callback stats, running state

