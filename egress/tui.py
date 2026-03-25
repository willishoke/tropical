"""
egress/tui.py — Textual TUI frontend for the egress MCP server.

Spawns `python -m egress.mcp_server` as a subprocess, connects via the MCP
client library, and provides a two-panel interface:
  top    : append-only output log
  bottom : readline-style command input with history and tab completion
"""

from __future__ import annotations

import asyncio
import io
import re
import select
import sys
import termios
import tty
from pathlib import Path
from typing import Optional

from ruamel.yaml import YAML
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.strip import Strip
from textual.widgets import Input, RichLog, Static
from textual.events import Key

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# ---------------------------------------------------------------------------
# Terminal background detection (OSC 11) — must run before Textual starts
# ---------------------------------------------------------------------------

def _query_terminal_bg(fallback: str = "#1e1e2e", timeout: float = 0.2) -> str:
    """
    Send OSC 11 to the terminal and parse the rgb: response.
    Terminals return 16-bit components (e.g. "3c3c"); we take the high byte.
    Returns a hex color string, or *fallback* on any failure.
    """
    try:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            sys.stdout.write("\033]11;?\033\\")
            sys.stdout.flush()
            buf = ""
            while select.select([sys.stdin], [], [], timeout)[0]:
                ch = sys.stdin.read(1)
                buf += ch
                if ch == "\007" or buf.endswith("\033\\"):
                    break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        m = re.search(r"rgb:([0-9a-fA-F]+)/([0-9a-fA-F]+)/([0-9a-fA-F]+)", buf)
        if m:
            r = int(m.group(1)[:2], 16)
            g = int(m.group(2)[:2], 16)
            b = int(m.group(3)[:2], 16)
            return f"#{r:02x}{g:02x}{b:02x}"
    except Exception:
        pass
    return fallback


# ---------------------------------------------------------------------------
# Demo patch — loaded on startup if the file exists.
# ---------------------------------------------------------------------------

DEMO_PATCH_PATH = Path(__file__).parent / "demo_patch.yaml"


# ---------------------------------------------------------------------------
# YAML formatting helper
# ---------------------------------------------------------------------------

def _to_yaml_str(obj) -> str:
    """Serialize a Python object to a compact YAML string."""
    y = YAML()
    y.default_flow_style = False
    buf = io.StringIO()
    y.dump(obj, buf)
    return buf.getvalue().rstrip()


def _format_result(data) -> str:
    """
    Format a tool result for display.

    Dicts and lists are rendered as YAML; scalars are str().
    """
    if isinstance(data, (dict, list)):
        return _to_yaml_str(data)
    return str(data)


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------

class Divider(Static):
    """A simple horizontal rule drawn with box-drawing characters."""

    DEFAULT_CSS = """
Divider {
    height: 1;
    background: transparent;
    color: #ff00ff;
}
"""

    def render(self):
        inner = max(0, self.size.width - 2)
        return f" {'─' * inner} "


class OutputLog(RichLog):
    """Append-only command/result log with markup support."""

    def log_command(self, command_str: str) -> None:
        self.write(f"[bold #00ffff]▸ {command_str}[/bold #00ffff]")

    def log_success(self, message: str) -> None:
        self.write(f"[bold #00ff41]✓[/bold #00ff41] [#00ff41]{message}[/#00ff41]")

    def log_error(self, message: str) -> None:
        self.write(f"[bold #ff0066]✗[/bold #ff0066] [#ff0066]{message}[/#ff0066]")

    def log_info(self, message: str) -> None:
        self.write(message)


class InputBar(Input):
    """
    Bottom prompt with command history and tab completion.

    History is maintained by the parent App, which this widget references via
    `self.app`.  Key interception happens before Textual's default handling so
    that Up/Down/Tab don't interfere with cursor movement or focus traversal.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._history_index: int = -1  # -1 means "not browsing history"
        self.cursor_blink = False

    def render_line(self, y: int) -> Strip:
        # Extend to full widget width so no cell is left to Textual's
        # background-fill mechanism (which doesn't respect transparency).
        strip = super().render_line(y)
        return strip.extend_cell_length(self.size.width)

    # ------------------------------------------------------------------
    # Key handling
    # ------------------------------------------------------------------

    def on_key(self, event: Key) -> None:
        app: EgressTUI = self.app  # type: ignore[assignment]

        if event.key == "up":
            event.prevent_default()
            event.stop()
            history = app.command_history
            if not history:
                return
            if self._history_index == -1:
                self._history_index = len(history) - 1
            elif self._history_index > 0:
                self._history_index -= 1
            self.value = history[self._history_index]
            self.cursor_position = len(self.value)

        elif event.key == "down":
            event.prevent_default()
            event.stop()
            history = app.command_history
            if self._history_index == -1:
                return
            if self._history_index < len(history) - 1:
                self._history_index += 1
                self.value = history[self._history_index]
            else:
                self._history_index = -1
                self.value = ""
            self.cursor_position = len(self.value)

        elif event.key == "tab":
            event.prevent_default()
            event.stop()
            self._complete()

        elif event.key in ("enter", "return"):
            # Reset history navigation on submit; actual submit is handled by
            # on_input_submitted in the App.
            self._history_index = -1

    def _complete(self) -> None:
        """Attempt tab completion on the current input value."""
        app: EgressTUI = self.app  # type: ignore[assignment]
        text = self.value
        parts = text.strip().split()

        if not parts:
            # Nothing typed; show all tools.
            if app.tools:
                app.output.log_info(
                    "  ".join(sorted(app.tools.keys()))
                )
            return

        if len(parts) == 1 and not text.endswith(" "):
            # Completing a tool name.
            prefix = parts[0]
            matches = [n for n in app.tools if n.startswith(prefix)]
            if len(matches) == 1:
                self.value = matches[0] + " "
                self.cursor_position = len(self.value)
            elif matches:
                app.output.log_info("  ".join(sorted(matches)))
        else:
            # Completing an argument — offer cached module names.
            candidates = sorted(app.cached_module_names)
            if candidates:
                app.output.log_info("  ".join(candidates))


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

class EgressTUI(App):
    CSS = """
App {
    background: transparent;
}
Screen {
    layout: vertical;
    background: transparent;
}
OutputLog {
    height: 1fr;
    background: transparent;
    color: #c0c0d0;
    scrollbar-size-vertical: 0;
}
Horizontal#input-row {
    height: 1;
    background: transparent;
}
#prompt-label {
    width: auto;
    padding: 0 1;
    color: #ff00ff;
    background: transparent;
}
InputBar {
    width: 1fr;
    background: transparent;
    color: #00ffff;
    border: none;
    padding: 0;
}
InputBar:focus {
    border: none;
}
InputBar > .input--cursor {
    background: transparent;
    color: #00ffff;
    text-style: underline;
}
"""

    def __init__(self, bg_color: str = "#1e1e2e", **kwargs):
        super().__init__(**kwargs)
        self._bg_color = bg_color
        self.mcp: Optional[ClientSession] = None
        # tool name → mcp Tool object (has .inputSchema)
        self.tools: dict = {}
        # history of submitted commands (oldest first)
        self.command_history: list[str] = []
        # module names from last successful list_modules call
        self.cached_module_names: list[str] = []
        self._mcp_stop: Optional[asyncio.Event] = None
        self._mcp_done: Optional[asyncio.Event] = None
        self._mcp_ready: Optional[asyncio.Event] = None
        self._mcp_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield OutputLog(id="output", markup=True)
        yield Divider()
        with Horizontal(id="input-row"):
            yield Static("❯", id="prompt-label")
            yield InputBar(id="input")
        yield Divider()

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def output(self) -> OutputLog:
        return self.query_one("#output", OutputLog)

    @property
    def input_bar(self) -> InputBar:
        return self.query_one("#input", InputBar)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def on_mount(self) -> None:
        self._mcp_stop = asyncio.Event()
        self._mcp_done = asyncio.Event()
        self._mcp_ready = asyncio.Event()
        self._mcp_task = asyncio.create_task(self._mcp_lifecycle())
        await self._mcp_ready.wait()
        self.input_bar.focus()

    async def on_unmount(self) -> None:
        if self._mcp_stop:
            self._mcp_stop.set()
        if self._mcp_task and not self._mcp_task.done():
            try:
                await asyncio.wait_for(self._mcp_done.wait(), timeout=3.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._mcp_task.cancel()
                try:
                    await self._mcp_task
                except (asyncio.CancelledError, Exception):
                    pass

    # ------------------------------------------------------------------
    # MCP lifecycle (single task owns the connection from open to close)
    # ------------------------------------------------------------------

    async def _mcp_lifecycle(self) -> None:
        """
        Run the MCP connection from connect to disconnect inside a single task
        so that anyio cancel scopes are always entered and exited in the same
        task context.  Signals _mcp_ready when the session is usable (or has
        failed), and _mcp_done when the connection has been cleanly torn down.
        """
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "egress.mcp_server"],
        )
        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    try:
                        await session.initialize()
                    except Exception as exc:
                        self.output.log_error(f"MCP server failed to start: {exc}")
                        return

                    self.mcp = session

                    # Fetch tool schemas for positional-arg mapping and tab completion.
                    try:
                        tools_response = await self.mcp.list_tools()
                        self.tools = {t.name: t for t in tools_response.tools}
                    except Exception as exc:
                        self.output.log_error(f"list_tools failed: {exc}")

                    # Load demo patch if the file exists.
                    if DEMO_PATCH_PATH.exists() and "load_patch" in self.tools:
                        try:
                            yaml_str = DEMO_PATCH_PATH.read_text()
                            await self._call_tool("load_patch", {"yaml_str": yaml_str})
                        except Exception as exc:
                            self.output.log_error(f"Demo patch load failed: {exc}")

                    self._mcp_ready.set()
                    await self._mcp_stop.wait()
                    self.mcp = None
        except Exception as exc:
            self.output.log_error(f"MCP connection error: {exc}")
        finally:
            self._mcp_ready.set()  # unblock on_mount even if we errored
            self._mcp_done.set()

    # ------------------------------------------------------------------
    # Tool call helper
    # ------------------------------------------------------------------

    async def _call_tool(self, tool_name: str, args: dict):
        """
        Call an MCP tool and return the parsed result content.

        Raises on transport / protocol errors; callers decide how to handle.
        """
        if self.mcp is None:
            raise RuntimeError("MCP session not available")
        result = await self.mcp.call_tool(tool_name, args)
        # result.content is a list of content blocks (usually TextContent).
        # Concatenate all text blocks into a single string.
        parts = []
        for block in result.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        raw = "\n".join(parts)
        # Try to parse as YAML/JSON so we can pretty-print structured results.
        try:
            y = YAML()
            parsed = y.load(io.StringIO(raw))
            return parsed
        except Exception:
            return raw

    # ------------------------------------------------------------------
    # Command dispatch
    # ------------------------------------------------------------------

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        raw = event.value.strip()
        if not raw:
            return

        self.input_bar.value = ""
        self.command_history.append(raw)

        parts = raw.split()
        tool_name = parts[0]
        raw_args = parts[1:]

        self.output.log_command(raw)

        # Built-in commands.
        if tool_name == "quit" or tool_name == "exit":
            await self.action_quit()
            return

        if tool_name == "help":
            self._handle_help(raw_args)
            return

        # MCP tool call.
        await self._dispatch_tool(tool_name, raw_args)

    def _handle_help(self, args: list[str]) -> None:
        if not self.tools:
            self.output.log_error("No tools loaded (MCP server offline?).")
            return

        if not args:
            lines = ["Available tools:"]
            for name in sorted(self.tools):
                desc = getattr(self.tools[name], "description", "") or ""
                lines.append(f"  [bold]{name}[/bold]  {desc}")
            self.output.log_info("\n".join(lines))
        else:
            tool_name = args[0]
            if tool_name not in self.tools:
                self.output.log_error(f"Unknown tool: {tool_name!r}")
                return
            tool = self.tools[tool_name]
            desc = getattr(tool, "description", "") or "(no description)"
            schema = getattr(tool, "inputSchema", {}) or {}
            props = schema.get("properties", {})
            required = schema.get("required", [])
            lines = [
                f"[bold]{tool_name}[/bold]",
                f"  {desc}",
            ]
            if props:
                lines.append("  Arguments:")
                for param_name, prop in props.items():
                    req_marker = "*" if param_name in required else " "
                    ptype = prop.get("type", "any")
                    pdesc = prop.get("description", "")
                    lines.append(
                        f"    {req_marker} {param_name}  [{ptype}]  {pdesc}"
                    )
            self.output.log_info("\n".join(lines))

    async def _dispatch_tool(self, tool_name: str, raw_args: list[str]) -> None:
        if self.mcp is None:
            self.output.log_error("MCP server is not connected.")
            return

        if tool_name not in self.tools:
            self.output.log_error(
                f"Unknown tool: {tool_name!r}. Type 'help' to list tools."
            )
            return

        tool = self.tools[tool_name]
        schema = getattr(tool, "inputSchema", {}) or {}
        props = schema.get("properties", {})

        # Map positional args to named parameters in declaration order.
        param_names = list(props.keys())
        if len(raw_args) > len(param_names):
            self.output.log_error(
                f"{tool_name} expects at most {len(param_names)} argument(s), "
                f"got {len(raw_args)}."
            )
            return

        arg_dict: dict = {}
        for i, raw_val in enumerate(raw_args):
            if i >= len(param_names):
                break
            name = param_names[i]
            prop = props[name]
            arg_dict[name] = _coerce_arg(raw_val, prop.get("type", "string"))

        try:
            result = await self._call_tool(tool_name, arg_dict)
        except Exception as exc:
            self.output.log_error(str(exc))
            return

        self.output.log_success(_format_result(result))


# ---------------------------------------------------------------------------
# Type coercion for positional CLI args
# ---------------------------------------------------------------------------

def _coerce_arg(raw: str, schema_type: str):
    """
    Coerce a raw string argument to the type declared in the tool's JSON schema.
    """
    if schema_type == "integer":
        return int(raw)
    if schema_type == "number":
        return float(raw)
    if schema_type == "boolean":
        return raw.lower() in ("true", "1", "yes")
    # "string" or anything else: pass through as-is.
    return raw


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    bg = _query_terminal_bg()
    app = EgressTUI(bg_color=bg)
    app.run(inline=True)


if __name__ == "__main__":
    main()
