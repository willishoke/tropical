"""
UDPParamListener — receive float parameter updates over UDP.

Two message formats are supported:

  Text (default):  "<address> <value>\\n"
  OSC binary:      OSC 1.0 datagram with address + ",f" or ",d" type tag

Usage::

    listener = eg.UDPParamListener(port=9000)
    listener.register("freq", freq_param)
    listener.register("gain", gain_param, lo=-60.0, hi=0.0)
    listener.start()
    ...
    listener.stop()

Or as a context manager::

    with eg.UDPParamListener(port=9000) as listener:
        listener.register("freq", freq_param)
        ...
"""

import logging
import socket
import struct
import threading
from typing import Callable, Optional, Tuple

from .param import Param

log = logging.getLogger(__name__)

__all__ = ["UDPParamListener", "parse_text", "parse_osc"]


def parse_text(data: bytes) -> Optional[Tuple[str, float]]:
    """
    Parse a space-delimited text datagram.

    Expected format: b"<address> <value>\\n"

    Returns (address, float_value) or None if the datagram is malformed.
    """
    try:
        text = data.decode("utf-8").strip()
    except UnicodeDecodeError:
        return None
    parts = text.split()
    if len(parts) != 2:
        return None
    try:
        return parts[0], float(parts[1])
    except ValueError:
        return None


def parse_osc(data: bytes) -> Optional[Tuple[str, float]]:
    """
    Parse an OSC 1.0 binary datagram.

    Accepts type tags ",f" (32-bit float) and ",d" (64-bit double).
    Returns (address, float_value) or None if the datagram is malformed
    or uses an unsupported type tag.
    """
    def _read_padded_string(buf: bytes, offset: int) -> Tuple[str, int]:
        """Read a null-terminated, 4-byte-aligned string. Returns (string, new_offset)."""
        end = buf.index(b"\x00", offset)
        s = buf[offset:end].decode("ascii")
        # Align to next 4-byte boundary after the null terminator:
        # (end + 4) & ~3 gives the first byte past the padded string
        aligned = (end + 4) & ~3
        return s, aligned

    try:
        address, pos = _read_padded_string(data, 0)
        type_tag, pos = _read_padded_string(data, pos)
        if type_tag == ",f":
            value = struct.unpack_from(">f", data, pos)[0]
        elif type_tag == ",d":
            value = struct.unpack_from(">d", data, pos)[0]
        else:
            return None
        return address, float(value)
    except Exception as exc:
        log.debug("parse_osc: discarding malformed datagram: %s", exc)
        return None


class UDPParamListener:
    """
    Binds a UDP socket and dispatches incoming messages to registered Params.

    Parameters
    ----------
    port : int
        UDP port to bind on.
    host : str
        Host to bind to. Default "127.0.0.1". Pass "0.0.0.0" to accept
        from any interface.
    parse : callable
        Parse function with signature (bytes) -> (str, float) | None.
        Default is parse_text. Pass parse_osc for OSC binary messages.
    buffer_size : int
        Maximum datagram size in bytes. Default 4096.
    """

    def __init__(
        self,
        port: int,
        host: str = "127.0.0.1",
        parse: Callable[[bytes], Optional[Tuple[str, float]]] = None,
        buffer_size: int = 4096,
    ):
        if parse is None:
            parse = parse_text
        self._port = port
        self._host = host
        self._parse = parse
        self._buffer_size = buffer_size

        # Registry: normalised_address -> (Param, lo, hi, scale_in, scale_out)
        self._registry: dict = {}
        self._lock = threading.Lock()

        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._bound_port: Optional[int] = None

    @property
    def port(self) -> Optional[int]:
        """The port the socket is actually bound to.

        Returns None before start() is called. If the listener was constructed
        with port=0, this reflects the OS-assigned ephemeral port.
        """
        return self._bound_port

    @staticmethod
    def _normalise(address: str) -> str:
        """Strip a single leading slash from address."""
        return address[1:] if address.startswith("/") else address

    def register(
        self,
        address: str,
        param: Param,
        lo: Optional[float] = None,
        hi: Optional[float] = None,
        scale_in: Optional[float] = None,
        scale_out: Optional[float] = None,
    ) -> None:
        """
        Register a Param to receive values from the given address.

        Parameters
        ----------
        address : str
            Address string. A leading slash is stripped transparently,
            so "freq" and "/freq" are equivalent.
        param : Param
            The Param to update on receipt.
        lo : float, optional
            Lower clamp bound applied after scaling.
        hi : float, optional
            Upper clamp bound applied after scaling.
        scale_in : float, optional
            Multiplied into the incoming value before clamping.
        scale_out : float, optional
            Added to the value after scale_in multiplication, before clamping.
        """
        key = self._normalise(address)
        with self._lock:
            self._registry[key] = (param, lo, hi, scale_in, scale_out)

    def unregister(self, address: str) -> None:
        """Remove a registered address. Silent if address was not registered."""
        key = self._normalise(address)
        with self._lock:
            self._registry.pop(key, None)

    def start(self) -> None:
        """
        Bind the socket and start the receiver thread.
        Raises RuntimeError if already running.
        """
        if self._running:
            raise RuntimeError("UDPParamListener is already running.")
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self._host, self._port))
        self._bound_port = self._sock.getsockname()[1]
        self._sock.settimeout(0.1)
        self._running = True
        self._thread = threading.Thread(
            target=self._recv_loop, daemon=True, name=f"egress-udp-{self._bound_port}"
        )
        self._thread.start()

    def stop(self) -> None:
        """
        Signal the receiver thread to stop and join it.
        Closes the socket. Safe to call if not running.
        """
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                log.warning(
                    "UDPParamListener: receiver thread did not stop within 2s "
                    "(port=%s); it may still hold the socket.",
                    self._bound_port,
                )
            self._thread = None
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        self._bound_port = None

    def _recv_loop(self) -> None:
        while self._running:
            try:
                data, _ = self._sock.recvfrom(self._buffer_size)
            except socket.timeout:
                continue
            except OSError:
                break

            result = self._parse(data)
            if result is None:
                continue
            address, raw_value = result

            key = self._normalise(address)
            with self._lock:
                entry = self._registry.get(key)
            if entry is None:
                continue

            param, lo, hi, scale_in, scale_out = entry
            value = raw_value
            if scale_in is not None:
                value *= scale_in
            if scale_out is not None:
                value += scale_out
            if lo is not None and value < lo:
                value = lo
            if hi is not None and value > hi:
                value = hi
            param.value = value

    def __enter__(self) -> "UDPParamListener":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()
