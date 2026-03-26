/**
 * mcp.ts — MCP client wrapper for the egress TUI.
 *
 * Spawns `python3 -m egress.mcp_server` via StdioClientTransport, initializes
 * the session, and exposes a minimal interface used by the App component.
 */

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { parse as yamlParse } from "yaml";

export type JsonSchema = {
  type?: string;
  properties?: Record<string, { type?: string; description?: string }>;
  required?: string[];
};

export type Tool = {
  name: string;
  description?: string;
  inputSchema: JsonSchema;
};

export type ToolMap = Map<string, Tool>;

export interface McpClient {
  tools: ToolMap;
  callTool(name: string, args: Record<string, unknown>): Promise<unknown>;
  close(): Promise<void>;
}

export async function connectMcp(): Promise<McpClient> {
  const transport = new StdioClientTransport({
    command: "bun",
    args: ["run", new URL("./server.ts", import.meta.url).pathname],
    env: { ...process.env } as Record<string, string>,
  });

  const client = new Client(
    { name: "egress-tui", version: "0.1.0" },
    { capabilities: {} }
  );

  try {
    await client.connect(transport);
  } catch (err) {
    await transport.close().catch(() => {});
    throw err;
  }

  const listResult = await client.listTools();
  const tools: ToolMap = new Map(
    listResult.tools.map((t) => [
      t.name,
      {
        name: t.name,
        description: t.description ?? undefined,
        inputSchema: (t.inputSchema as JsonSchema) ?? {},
      },
    ])
  );

  return {
    tools,

    async callTool(
      name: string,
      args: Record<string, unknown>
    ): Promise<unknown> {
      const result = await client.callTool({ name, arguments: args });
      // Concatenate all TextContent blocks into a single string.
      const parts: string[] = [];
      for (const block of result.content as Array<{ type: string; text?: string }>) {
        if (block.type === "text" && block.text != null) {
          parts.push(block.text);
        }
      }
      const raw = parts.join("\n");
      // Try to parse as YAML for structured display; fall back to raw string.
      try {
        return yamlParse(raw);
      } catch {
        return raw;
      }
    },

    async close(): Promise<void> {
      await client.close();
    },
  };
}
