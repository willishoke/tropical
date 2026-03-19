"""
Tests for UDPParamListener, parse_text, and parse_osc.

All tests use loopback sockets with port=0 so the OS assigns a free ephemeral
port for each listener. No DAC, no graph, no audio hardware.
"""

import socket
import struct
import time

import egress as eg


def _send(port: int, data: bytes) -> None:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.sendto(data, ("127.0.0.1", port))
    s.close()


def _poll_eq(param, target: float, tol: float = 1e-6, timeout: float = 1.0) -> bool:
    """Poll param.value until it reaches target within tol, or timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if abs(param.value - target) <= tol:
            return True
        time.sleep(0.0005)
    return False


def _poll_stable(param, expected: float, tol: float = 1e-6,
                 stable_window: float = 0.440, timeout: float = 1.0) -> bool:
    """
    Confirm param.value stays at expected for stable_window seconds.
    Returns True if the value held, False if it diverged or timed out.
    """
    deadline = time.monotonic() + timeout
    stable_until = time.monotonic() + stable_window
    while time.monotonic() < deadline:
        if abs(param.value - expected) > tol:
            return False
        if time.monotonic() >= stable_until:
            return True
        time.sleep(0.0005)
    return True  # window elapsed without divergence


def test_text_basic():
    p = eg.Param(0.0, time_const=0.0)
    with eg.UDPParamListener(port=0) as listener:
        listener.register("freq", p)
        _send(listener.port, b"freq 440.0\n")
        assert _poll_eq(p, 440.0), f"expected 440.0, got {p.value}"


def test_text_slash_stripped_incoming():
    p = eg.Param(0.0, time_const=0.0)
    with eg.UDPParamListener(port=0) as listener:
        listener.register("freq", p)
        _send(listener.port, b"/freq 880.0\n")
        assert _poll_eq(p, 880.0), f"expected 880.0, got {p.value}"


def test_text_slash_stripped_registry():
    p = eg.Param(0.0, time_const=0.0)
    with eg.UDPParamListener(port=0) as listener:
        listener.register("/freq", p)
        _send(listener.port, b"freq 220.0\n")
        assert _poll_eq(p, 220.0), f"expected 220.0, got {p.value}"


def test_text_clamp_lo():
    p = eg.Param(0.0, time_const=0.0)
    with eg.UDPParamListener(port=0) as listener:
        listener.register("gain", p, lo=-60.0, hi=0.0)
        _send(listener.port, b"gain -100.0\n")
        assert _poll_eq(p, -60.0), f"expected -60.0, got {p.value}"


def test_text_clamp_hi():
    p = eg.Param(0.0, time_const=0.0)
    with eg.UDPParamListener(port=0) as listener:
        listener.register("gain", p, lo=-60.0, hi=0.0)
        _send(listener.port, b"gain 10.0\n")
        assert _poll_eq(p, 0.0), f"expected 0.0, got {p.value}"


def test_text_scale():
    p = eg.Param(0.0, time_const=0.0)
    with eg.UDPParamListener(port=0) as listener:
        listener.register("cutoff", p, scale_in=2.0, lo=0.0, hi=20000.0)
        _send(listener.port, b"cutoff 1000.0\n")
        assert _poll_eq(p, 2000.0), f"expected 2000.0, got {p.value}"


def test_text_unknown_address_ignored():
    p = eg.Param(99.0, time_const=0.0)
    with eg.UDPParamListener(port=0) as listener:
        listener.register("freq", p)
        _send(listener.port, b"cutoff 440.0\n")
        assert _poll_stable(p, 99.0), f"expected value to stay 99.0, got {p.value}"


def test_text_malformed_ignored():
    p = eg.Param(99.0, time_const=0.0)
    with eg.UDPParamListener(port=0) as listener:
        listener.register("freq", p)
        _send(listener.port, b"freq not-a-number\n")
        _send(listener.port, b"freq 1.0 2.0 extra\n")
        _send(listener.port, b"\xff\xfe invalid utf-8\n")
        assert _poll_stable(p, 99.0), f"expected value to stay 99.0, got {p.value}"


def test_text_unregister():
    p = eg.Param(0.0, time_const=0.0)
    with eg.UDPParamListener(port=0) as listener:
        listener.register("freq", p)
        _send(listener.port, b"freq 440.0\n")
        assert _poll_eq(p, 440.0), f"first send: expected 440.0, got {p.value}"
        listener.unregister("freq")
        _send(listener.port, b"freq 880.0\n")
        assert _poll_stable(p, 440.0), (
            f"after unregister: expected value to hold at 440.0, got {p.value}"
        )


def _build_osc(address: str, type_tag: str, fmt: str, value) -> bytes:
    """Construct a minimal OSC datagram by hand."""
    def pad(s: bytes) -> bytes:
        s = s + b"\x00"
        remainder = len(s) % 4
        if remainder:
            s = s + b"\x00" * (4 - remainder)
        return s

    return pad(address.encode("ascii")) + pad(type_tag.encode("ascii")) + struct.pack(fmt, value)


def test_osc_f32():
    p = eg.Param(0.0, time_const=0.0)
    datagram = _build_osc("/freq", ",f", ">f", 440.0)
    with eg.UDPParamListener(port=0, parse=eg.parse_osc) as listener:
        listener.register("freq", p)
        _send(listener.port, datagram)
        assert _poll_eq(p, 440.0, tol=1e-3), f"expected 440.0, got {p.value}"


def test_osc_d64():
    p = eg.Param(0.0, time_const=0.0)
    datagram = _build_osc("/cutoff", ",d", ">d", 1200.0)
    with eg.UDPParamListener(port=0, parse=eg.parse_osc) as listener:
        listener.register("cutoff", p)
        _send(listener.port, datagram)
        assert _poll_eq(p, 1200.0, tol=1e-9), f"expected 1200.0, got {p.value}"


def test_osc_unsupported_type_ignored():
    p = eg.Param(99.0, time_const=0.0)

    def pad(s: bytes) -> bytes:
        s = s + b"\x00"
        remainder = len(s) % 4
        if remainder:
            s = s + b"\x00" * (4 - remainder)
        return s

    datagram = pad(b"/freq") + pad(b",s") + pad(b"hello")
    with eg.UDPParamListener(port=0, parse=eg.parse_osc) as listener:
        listener.register("freq", p)
        _send(listener.port, datagram)
        assert _poll_stable(p, 99.0), f"expected value to stay 99.0, got {p.value}"


def test_context_manager_stops_cleanly():
    p = eg.Param(0.0, time_const=0.0)
    with eg.UDPParamListener(port=0) as listener:
        listener.register("freq", p)
        thread = listener._thread
    assert thread is not None
    assert not thread.is_alive()


def test_already_running_raises():
    listener = eg.UDPParamListener(port=0)
    listener.start()
    try:
        raised = False
        try:
            listener.start()
        except RuntimeError:
            raised = True
        assert raised
    finally:
        listener.stop()


def test_port_property_reflects_bound_port():
    listener = eg.UDPParamListener(port=0)
    assert listener.port is None
    listener.start()
    try:
        assert listener.port is not None
        assert isinstance(listener.port, int)
        assert listener.port > 0
    finally:
        listener.stop()
    assert listener.port is None
