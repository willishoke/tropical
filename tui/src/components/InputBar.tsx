/**
 * InputBar.tsx — Readline-style input with history navigation and tab completion.
 *
 * Key bindings (mirrors Python Textual InputBar):
 *   ↑ / ↓    history navigation
 *   Tab      tool-name completion (1st word) or module-name hints (subsequent)
 *   Return   submit command
 *   Backspace delete last character
 *   Any printable char → append to value
 */

import React, { useState, useCallback } from "react";
import { Text, useInput } from "ink";
import type { ToolMap } from "../mcp.js";

interface Props {
  onSubmit: (value: string) => void;
  onTab: (value: string, setValue: (v: string) => void) => void;
  history: string[];
  tools: ToolMap;
}

export function InputBar({ onSubmit, onTab, history }: Props): React.ReactElement {
  const [value, setValue] = useState("");
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [cursor, setCursor] = useState(0);

  const updateValue = useCallback(
    (v: string) => {
      setValue(v);
      setCursor(v.length);
    },
    []
  );

  useInput((input, key) => {
    if (key.upArrow) {
      if (!history.length) return;
      const newIdx =
        historyIndex === -1
          ? history.length - 1
          : Math.max(0, historyIndex - 1);
      setHistoryIndex(newIdx);
      updateValue(history[newIdx]);
      return;
    }

    if (key.downArrow) {
      if (historyIndex === -1) return;
      if (historyIndex < history.length - 1) {
        const newIdx = historyIndex + 1;
        setHistoryIndex(newIdx);
        updateValue(history[newIdx]);
      } else {
        setHistoryIndex(-1);
        updateValue("");
      }
      return;
    }

    if (key.tab) {
      onTab(value, updateValue);
      return;
    }

    if (key.return) {
      setHistoryIndex(-1);
      const submitted = value;
      updateValue("");
      onSubmit(submitted);
      return;
    }

    if (key.backspace || key.delete) {
      if (cursor > 0) {
        const next = value.slice(0, cursor - 1) + value.slice(cursor);
        setValue(next);
        setCursor(cursor - 1);
      }
      return;
    }

    if (key.leftArrow) {
      setCursor(Math.max(0, cursor - 1));
      return;
    }

    if (key.rightArrow) {
      setCursor(Math.min(value.length, cursor + 1));
      return;
    }

    // Ignore control sequences that aren't printable.
    if (key.ctrl || key.meta || !input) return;

    const next = value.slice(0, cursor) + input + value.slice(cursor);
    setValue(next);
    setCursor(cursor + input.length);
  });

  // Render: characters before cursor, cursor char (underlined or block), chars after.
  const before = value.slice(0, cursor);
  const atCursor = value[cursor] ?? " ";
  const after = value.slice(cursor + 1);

  return (
    <Text color="#00ffff">
      {before}
      <Text color="#00ffff" underline>{atCursor}</Text>
      {after}
    </Text>
  );
}
