/**
 * OutputLog.tsx — Append-only scrollable log panel.
 *
 * Shows the last N lines that fit in the available terminal height.
 * Each entry has a visual prefix matching the Python Textual color scheme:
 *   command → cyan  ▸
 *   success → green ✓
 *   error   → red   ✗
 *   info    → gray  (no prefix)
 */

import React from "react";
import { Box, Text, useStdout } from "ink";
import type { LogEntry } from "../types.js";

interface Props {
  entries: LogEntry[];
}

function renderEntry(entry: LogEntry, idx: number): React.ReactNode {
  switch (entry.kind) {
    case "command":
      return (
        <Text key={idx} color="#00ffff" bold>
          {"▸ " + entry.text}
        </Text>
      );
    case "success":
      return (
        <Text key={idx}>
          <Text color="#00ff41" bold>{"✓ "}</Text>
          <Text color="#00ff41">{entry.text}</Text>
        </Text>
      );
    case "error":
      return (
        <Text key={idx}>
          <Text color="#ff0066" bold>{"✗ "}</Text>
          <Text color="#ff0066">{entry.text}</Text>
        </Text>
      );
    case "info":
      return (
        <Text key={idx} color="#c0c0d0">
          {entry.text}
        </Text>
      );
  }
}

export function OutputLog({ entries }: Props): React.ReactElement {
  const { stdout } = useStdout();
  // Reserve 3 rows: divider + input row + bottom divider.
  const rows = (stdout?.rows ?? 24) - 3;
  const visible = entries.slice(-rows);

  return (
    <Box flexDirection="column" flexGrow={1} overflow="hidden">
      {visible.map((e, i) => renderEntry(e, i))}
    </Box>
  );
}
