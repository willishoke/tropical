"""
egress/tui.py — Textual TUI frontend for the egress MCP server.

Spawns `python -m egress.mcp_server` as a subprocess, connects via the MCP
client library, and provides a three-panel interface:
  left   : live graph state (auto-refreshes every 500ms)
  right  : append-only output log
  bottom : readline-style command input with history and tab completion
"""

from __future__ import annotations

import asyncio
import io
import sys
from pathlib import Path
from typing import Optional

from ruamel.yaml import YAML
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Input, RichLog, Rule, Static
from textual.events import Key

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

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

class GraphPanel(Static):
    """Left panel — displays live graph state, auto-refreshed every 500ms."""

    DEFAULT_CSS = ""

    def __init__(self, **kwargs):
        super().__init__("", **kwargs)

    def render_state(self, text: str) -> None:
        self.update(text)


class OutputLog(RichLog):
    """Right panel — append-only command/result log with markup support."""

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
Screen {
    layout: vertical;
    background: #0a0a0f;
}
Horizontal {
    height: 1fr;
}
GraphPanel {
    width: 35;
    padding: 1;
    color: #00ff41;
    background: #0a0a0f;
}
Rule.vertical-divider {
    width: 1;
    color: #ff00ff;
}
Rule.horizontal-divider {
    height: 1;
    color: #ff00ff;
    margin: 0;
}
OutputLog {
    width: 1fr;
    background: #0a0a0f;
    color: #c0c0d0;
}
Horizontal#input-row {
    height: 3;
    background: #0d0d1a;
}
#prompt-label {
    width: auto;
    padding: 1 1;
    color: #ff00ff;
}
InputBar {
    width: 1fr;
    background: #0d0d1a;
    color: #00ffff;
    border: tall #1a1a2e;
}
InputBar:focus {
    border: tall #ff00ff;
}
"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mcp: Optional[ClientSession] = None
        # tool name → mcp Tool object (has .inputSchema)
        self.tools: dict = {}
        # history of submitted commands (oldest first)
        self.command_history: list[str] = []
        # module names from last successful list_modules call
        self.cached_module_names: list[str] = []
        self._refresh_in_progress: bool = False
        self._mcp_stop: Optional[asyncio.Event] = None
        self._mcp_done: Optional[asyncio.Event] = None
        self._mcp_ready: Optional[asyncio.Event] = None
        self._mcp_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield GraphPanel(id="graph")
            yield Rule(orientation="vertical", classes="vertical-divider")
            yield OutputLog(id="output", markup=True)
        yield Rule(classes="horizontal-divider")
        with Horizontal(id="input-row"):
            yield Static("▸", id="prompt-label")
            yield InputBar(id="input")

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def graph_panel(self) -> GraphPanel:
        return self.query_one("#graph", GraphPanel)

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
        self.set_interval(0.5, self.refresh_graph)
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
    # Graph panel refresh
    # ------------------------------------------------------------------

    async def refresh_graph(self) -> None:
        """Query MCP for current graph state and update the left panel."""
        if self._refresh_in_progress:
            return
        if self.mcp is None:
            self.graph_panel.render_state("[bold #ff0066]◎ MCP OFFLINE[/bold #ff0066]")
            return
        self._refresh_in_progress = True
        try:
            await self._refresh_graph_inner()
        finally:
            self._refresh_in_progress = False

    async def _refresh_graph_inner(self) -> None:

        lines: list[str] = []

        def _unwrap(result) -> tuple[bool, object]:
            """Extract (ok, data) from a server response envelope."""
            if isinstance(result, dict):
                return result.get("ok", False), result.get("data")
            return False, None

        # Modules
        try:
            ok, modules = _unwrap(await self._call_tool("list_modules", {}))
            lines.append("[bold #ff00ff]▪ MODULES[/bold #ff00ff]")
            if ok and isinstance(modules, list):
                self.cached_module_names = [
                    m.get("name", "") for m in modules if isinstance(m, dict)
                ]
                for m in modules:
                    if isinstance(m, dict):
                        name = m.get("name", "?")
                        mtype = m.get("type_name", "?")
                        n_in = len(m.get("inputs", []))
                        n_out = len(m.get("outputs", []))
                        lines.append(f"  [#00ffff]{name}[/#00ffff]  [dim #ff00ff]{mtype}[/dim #ff00ff]  [dim]in:{n_in}  out:{n_out}[/dim]")
            else:
                lines.append("  [dim](none)[/dim]")
        except Exception:
            lines.append("  [dim #ff0066](unavailable)[/dim #ff0066]")

        lines.append("")

        # Connections
        try:
            ok, conns = _unwrap(await self._call_tool("list_connections", {}))
            lines.append("[bold #ff00ff]▪ CONNECTIONS[/bold #ff00ff]")
            if ok and isinstance(conns, list) and conns:
                for c in conns:
                    if isinstance(c, dict):
                        src = f"{c.get('src', '?')}.{c.get('src_output', '?')}"
                        dst = f"{c.get('dst', '?')}.{c.get('dst_input', '?')}"
                        lines.append(f"  [#00ff41]{src}[/#00ff41] [#ff00ff]→[/#ff00ff] [#00ff41]{dst}[/#00ff41]")
            else:
                lines.append("  [dim](none)[/dim]")
        except Exception:
            lines.append("  [dim #ff0066](unavailable)[/dim #ff0066]")

        lines.append("")

        # Audio status
        try:
            ok, status = _unwrap(await self._call_tool("audio_status", {}))
            if ok and isinstance(status, dict):
                running = status.get("is_running", False)
                dot = "[bold #00ff41]◉[/bold #00ff41]" if running else "[dim #ff0066]◎[/dim #ff0066]"
                state_str = "[bold #00ff41]RUNNING[/bold #00ff41]" if running else "[dim #ff0066]STOPPED[/dim #ff0066]"
                lines.append(f"[bold #ff00ff]▪ AUDIO[/bold #ff00ff]  {dot} {state_str}")
            else:
                lines.append("[bold #ff00ff]▪ AUDIO[/bold #ff00ff]  [dim #ff0066]◎ STOPPED[/dim #ff0066]")
        except Exception:
            lines.append("[bold #ff00ff]▪ AUDIO[/bold #ff00ff]  [dim #ff0066](unavailable)[/dim #ff0066]")

        self.graph_panel.render_state("\n".join(lines))

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
    app = EgressTUI()
    app.run()


if __name__ == "__main__":
    main()
