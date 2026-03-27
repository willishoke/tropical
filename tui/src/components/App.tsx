/**
 * App.tsx — Root Ink component for the egress TUI.
 *
 * Layout (vertical, full-height):
 *   ┌─────────────────────┐
 *   │  OutputLog (flex:1) │
 *   ├─────────────────────┤  Divider
 *   │ ❯ <InputBar>        │
 *   └─────────────────────┘  Divider
 *
 * Manages:
 *   - MCP connection lifecycle (connect on mount, close on unmount)
 *   - Append-only log entries
 *   - Command history
 *   - Cached module names (updated after list_modules)
 *   - Command dispatch (built-ins + MCP tools)
 */

import React, { useState, useEffect, useCallback, useRef } from "react";
import { Box, Text, useApp, useStdout } from "ink";
import { stringify as yamlStringify } from "yaml";
import { readFileSync, existsSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, join } from "path";
import { connectMcp } from "../mcp.js";
import type { McpClient, ToolMap, JsonSchema } from "../mcp.js";
import type { LogEntry } from "../types.js";
import { OutputLog } from "./OutputLog.js";
import { Divider } from "./Divider.js";
import { InputBar } from "./InputBar.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatResult(data: unknown): string {
  if (data !== null && typeof data === "object") {
    return yamlStringify(data, { indent: 2 }).trimEnd();
  }
  return String(data);
}

function coerceArg(raw: string, schemaType: string | undefined): unknown {
  switch (schemaType) {
    case "integer":
      return parseInt(raw, 10);
    case "number":
      return parseFloat(raw);
    case "boolean":
      return ["true", "1", "yes"].includes(raw.toLowerCase());
    default:
      return raw;
  }
}

function buildArgDict(
  rawArgs: string[],
  schema: JsonSchema
): Record<string, unknown> {
  const props = schema.properties ?? {};
  const paramNames = Object.keys(props);
  const result: Record<string, unknown> = {};
  for (let i = 0; i < Math.min(rawArgs.length, paramNames.length); i++) {
    const name = paramNames[i];
    result[name] = coerceArg(rawArgs[i], props[name]?.type);
  }
  return result;
}

const __dirname = dirname(fileURLToPath(import.meta.url));
const DEMO_PATCH_PATH = join(__dirname, "..", "..", "..", "egress", "demo_patch.json");

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface Props {
  bgColor: string;
}

export function App({ bgColor: _bgColor }: Props): React.ReactElement {
  const { exit } = useApp();
  const { stdout } = useStdout();

  const [entries, setEntries] = useState<LogEntry[]>([]);
  const [history, setHistory] = useState<string[]>([]);
  const [cachedModuleNames, setCachedModuleNames] = useState<string[]>([]);
  const [tools, setTools] = useState<ToolMap>(new Map());
  const clientRef = useRef<McpClient | null>(null);

  const appendEntry = useCallback((entry: LogEntry) => {
    setEntries((prev) => [...prev, entry]);
  }, []);

  // -------------------------------------------------------------------------
  // MCP lifecycle
  // -------------------------------------------------------------------------

  useEffect(() => {
    let cancelled = false;

    async function init() {
      appendEntry({ kind: "info", text: "Connecting to egress MCP server…" });
      try {
        const client = await connectMcp();
        if (cancelled) {
          await client.close();
          return;
        }
        clientRef.current = client;
        setTools(client.tools);
        appendEntry({
          kind: "info",
          text: `Connected. ${client.tools.size} tools available.`,
        });

        // Load demo patch if present.
        if (existsSync(DEMO_PATCH_PATH) && client.tools.has("load_patch")) {
          try {
            const patch = JSON.parse(readFileSync(DEMO_PATCH_PATH, "utf-8"));
            const result = await client.callTool("load_patch", { patch });
            appendEntry({ kind: "success", text: formatResult(result) });
          } catch (err) {
            appendEntry({
              kind: "error",
              text: `Demo patch load failed: ${err}`,
            });
          }
        }
      } catch (err) {
        if (!cancelled) {
          appendEntry({
            kind: "error",
            text: `MCP connection error: ${err}`,
          });
        }
      }
    }

    init();

    return () => {
      cancelled = true;
      clientRef.current?.close().catch(() => {});
      clientRef.current = null;
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // -------------------------------------------------------------------------
  // Help handler
  // -------------------------------------------------------------------------

  const handleHelp = useCallback(
    (args: string[]) => {
      if (tools.size === 0) {
        appendEntry({ kind: "error", text: "No tools loaded (MCP server offline?)." });
        return;
      }
      if (!args.length) {
        const lines = ["Available tools:"];
        for (const [name, tool] of [...tools].sort(([a], [b]) => a.localeCompare(b))) {
          lines.push(`  ${name}  ${tool.description ?? ""}`);
        }
        appendEntry({ kind: "info", text: lines.join("\n") });
      } else {
        const toolName = args[0];
        const tool = tools.get(toolName);
        if (!tool) {
          appendEntry({ kind: "error", text: `Unknown tool: ${toolName}` });
          return;
        }
        const props = tool.inputSchema.properties ?? {};
        const required = tool.inputSchema.required ?? [];
        const lines = [toolName, `  ${tool.description ?? "(no description)"}`];
        if (Object.keys(props).length) {
          lines.push("  Arguments:");
          for (const [pname, prop] of Object.entries(props)) {
            const req = required.includes(pname) ? "*" : " ";
            lines.push(
              `    ${req} ${pname}  [${prop.type ?? "any"}]  ${prop.description ?? ""}`
            );
          }
        }
        appendEntry({ kind: "info", text: lines.join("\n") });
      }
    },
    [tools, appendEntry]
  );

  // -------------------------------------------------------------------------
  // Tab completion
  // -------------------------------------------------------------------------

  const handleTab = useCallback(
    (value: string, setValue: (v: string) => void) => {
      const parts = value.trim().split(/\s+/);
      const toolNames = [...tools.keys()].sort();

      if (!value.trim()) {
        // Empty — show all tools.
        appendEntry({ kind: "info", text: toolNames.join("  ") });
        return;
      }

      if (parts.length === 1 && !value.endsWith(" ")) {
        // Completing a tool name.
        const prefix = parts[0];
        const matches = toolNames.filter((n) => n.startsWith(prefix));
        if (matches.length === 1) {
          setValue(matches[0] + " ");
        } else if (matches.length > 1) {
          appendEntry({ kind: "info", text: matches.join("  ") });
        }
      } else {
        // Second word or beyond — offer cached module names.
        if (cachedModuleNames.length) {
          appendEntry({
            kind: "info",
            text: [...cachedModuleNames].sort().join("  "),
          });
        }
      }
    },
    [tools, cachedModuleNames, appendEntry]
  );

  // -------------------------------------------------------------------------
  // Command dispatch
  // -------------------------------------------------------------------------

  const dispatch = useCallback(
    async (raw: string) => {
      const trimmed = raw.trim();
      if (!trimmed) return;

      setHistory((prev) => [...prev, trimmed]);
      appendEntry({ kind: "command", text: trimmed });

      const parts = trimmed.split(/\s+/);
      const toolName = parts[0];
      const rawArgs = parts.slice(1);

      // Built-ins.
      if (toolName === "quit" || toolName === "exit") {
        const client = clientRef.current;
        if (client) await client.close().catch(() => {});
        exit();
        return;
      }

      if (toolName === "help") {
        handleHelp(rawArgs);
        return;
      }

      // MCP tool call.
      const client = clientRef.current;
      if (!client) {
        appendEntry({ kind: "error", text: "MCP server is not connected." });
        return;
      }

      const tool = tools.get(toolName);
      if (!tool) {
        appendEntry({
          kind: "error",
          text: `Unknown tool: ${toolName}. Type 'help' to list tools.`,
        });
        return;
      }

      const schema = tool.inputSchema;
      const paramCount = Object.keys(schema.properties ?? {}).length;
      if (rawArgs.length > paramCount) {
        appendEntry({
          kind: "error",
          text: `${toolName} expects at most ${paramCount} argument(s), got ${rawArgs.length}.`,
        });
        return;
      }

      try {
        const args = buildArgDict(rawArgs, schema);
        const result = await client.callTool(toolName, args);
        appendEntry({ kind: "success", text: formatResult(result) });

        // Cache module names after list_modules.
        if (
          toolName === "list_modules" &&
          result !== null &&
          typeof result === "object" &&
          "data" in (result as object)
        ) {
          const data = (result as { data: unknown }).data;
          if (Array.isArray(data)) {
            setCachedModuleNames(
              data
                .map((m: unknown) =>
                  m !== null && typeof m === "object" && "name" in (m as object)
                    ? String((m as { name: unknown }).name)
                    : null
                )
                .filter((n): n is string => n !== null)
            );
          }
        }
      } catch (err) {
        appendEntry({ kind: "error", text: String(err) });
      }
    },
    [tools, handleHelp, appendEntry, exit]
  );

  // -------------------------------------------------------------------------
  // Layout
  // -------------------------------------------------------------------------

  const rows = stdout?.rows ?? 24;

  return (
    <Box flexDirection="column" height={rows}>
      <OutputLog entries={entries} />
      <Divider />
      <Box flexDirection="row" height={1}>
        <Text color="#ff00ff">{"❯ "}</Text>
        <InputBar
          onSubmit={dispatch}
          onTab={handleTab}
          history={history}
          tools={tools}
        />
      </Box>
      <Divider />
    </Box>
  );
}
