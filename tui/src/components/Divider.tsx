/**
 * Divider.tsx — Horizontal rule using box-drawing characters.
 * Matches the Python Textual Divider: magenta, full-width ─ line.
 */

import React from "react";
import { Text, useStdout } from "ink";

export function Divider(): React.ReactElement {
  const { stdout } = useStdout();
  const cols = (stdout?.columns ?? 80) - 2;
  return (
    <Text color="#ff00ff">{" " + "─".repeat(Math.max(0, cols)) + " "}</Text>
  );
}
