/**
 * index.tsx — Entry point for the egress TypeScript/Ink TUI.
 *
 * 1. Queries the terminal background color via OSC 11 (before Ink takes over stdin).
 * 2. Renders the <App> component full-screen.
 */

import React from "react";
import { render } from "ink";
import { App } from "./components/App.js";

// ---------------------------------------------------------------------------
// OSC 11 terminal background detection
// ---------------------------------------------------------------------------

async function queryTerminalBg(fallback = "#1e1e2e"): Promise<string> {
  return new Promise((resolve) => {
    if (!process.stdin.isTTY || !process.stdout.isTTY) {
      resolve(fallback);
      return;
    }

    let buf = "";
    const timeout = setTimeout(() => cleanup(fallback), 200);

    function cleanup(color: string) {
      clearTimeout(timeout);
      try {
        process.stdin.setRawMode(false);
      } catch {}
      process.stdin.pause();
      process.stdin.removeListener("data", onData);
      resolve(color);
    }

    function onData(chunk: Buffer) {
      buf += chunk.toString("binary");
      // OSC response ends with BEL (\x07) or ST (\x1b\\).
      if (buf.includes("\x07") || buf.includes("\x1b\\")) {
        const m = buf.match(
          /rgb:([0-9a-fA-F]+)\/([0-9a-fA-F]+)\/([0-9a-fA-F]+)/
        );
        if (m) {
          const r = parseInt(m[1].slice(0, 2), 16);
          const g = parseInt(m[2].slice(0, 2), 16);
          const b = parseInt(m[3].slice(0, 2), 16);
          cleanup(`#${r.toString(16).padStart(2, "0")}${g.toString(16).padStart(2, "0")}${b.toString(16).padStart(2, "0")}`);
        } else {
          cleanup(fallback);
        }
      }
    }

    try {
      process.stdin.setRawMode(true);
      process.stdin.resume();
      process.stdin.on("data", onData);
      process.stdout.write("\x1b]11;?\x1b\\");
    } catch {
      cleanup(fallback);
    }
  });
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

const bgColor = await queryTerminalBg();
render(<App bgColor={bgColor} />, { exitOnCtrlC: true });
