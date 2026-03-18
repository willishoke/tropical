"""
Tests for UDPParamListener, parse_text, and parse_osc.

All tests use loopback sockets. No DAC, no graph, no audio hardware.
Ports 19001–19014 are used; each test owns exactly one port.
"""

import socket
import struct
import time

import egress as eg


def _send(port: int, data: bytes) -> None:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.sendto(data, ("127.0.0.1", port))
    s.close()


def test_text_basic():
    p = eg.Param(0.0, time_const=0.0)
    with eg.UDPParamListener(port=19001) as listener:
        listener.register("freq", p)
        _send(19001, b"freq 440.0\n")
        time.sleep(0.005)
    assert abs(p.value - 440.0) < 1e-6


def test_text_slash_stripped_incoming():
    p = eg.Param(0.0, time_const=0.0)
    with eg.UDPParamListener(port=19002) as listener:
        listener.register("freq", p)
        _send(19002, b"/freq 880.0\n")
        time.sleep(0.005)
    assert abs(p.value - 880.0) < 1e-6


def test_text_slash_stripped_registry():
    p = eg.Param(0.0, time_const=0.0)
    with eg.UDPParamListener(port=19003) as listener:
        listener.register("/freq", p)
        _send(19003, b"freq 220.0\n")
        time.sleep(0.005)
    assert abs(p.value - 220.0) < 1e-6


def test_text_clamp_lo():
    p = eg.Param(0.0, time_const=0.0)
    with eg.UDPParamListener(port=19004) as listener:
        listener.register("gain", p, lo=-60.0, hi=0.0)
        _send(19004, b"gain -100.0\n")
        time.sleep(0.005)
    assert abs(p.value - (-60.0)) < 1e-6


def test_text_clamp_hi():
    p = eg.Param(0.0, time_const=0.0)
    with eg.UDPParamListener(port=19005) as listener:
        listener.register("gain", p, lo=-60.0, hi=0.0)
        _send(19005, b"gain 10.0\n")
        time.sleep(0.005)
    assert abs(p.value - 0.0) < 1e-6


def test_text_scale():
    p = eg.Param(0.0, time_const=0.0)
    with eg.UDPParamListener(port=19006) as listener:
        listener.register("cutoff", p, scale_in=2.0, lo=0.0, hi=20000.0)
        _send(19006, b"cutoff 1000.0\n")
        time.sleep(0.005)
    assert abs(p.value - 2000.0) < 1e-6


def test_text_unknown_address_ignored():
    p = eg.Param(99.0, time_const=0.0)
    with eg.UDPParamListener(port=19007) as listener:
        listener.register("freq", p)
        _send(19007, b"cutoff 440.0\n")
        time.sleep(0.005)
    assert abs(p.value - 99.0) < 1e-6


def test_text_malformed_ignored():
    p = eg.Param(99.0, time_const=0.0)
    with eg.UDPParamListener(port=19008) as listener:
        listener.register("freq", p)
        _send(19008, b"freq not-a-number\n")
        _send(19008, b"freq 1.0 2.0 extra\n")
        _send(19008, b"\xff\xfe invalid utf-8\n")
        time.sleep(0.005)
    assert abs(p.value - 99.0) < 1e-6


def test_text_unregister():
    p = eg.Param(0.0, time_const=0.0)
    with eg.UDPParamListener(port=19009) as listener:
        listener.register("freq", p)
        _send(19009, b"freq 440.0\n")
        time.sleep(0.005)
        listener.unregister("freq")
        _send(19009, b"freq 880.0\n")
        time.sleep(0.005)
    assert abs(p.value - 440.0) < 1e-6


def _build_osc(address: str, type_tag: str, fmt: str, value) -> bytes:
    """Construct a minimal OSC datagram by hand."""
    def pad(s: bytes) -> bytes:
        # Null-terminate, then pad to next 4-byte boundary
        s = s + b"\x00"
        remainder = len(s) % 4
        if remainder:
            s = s + b"\x00" * (4 - remainder)
        return s

    return pad(address.encode("ascii")) + pad(type_tag.encode("ascii")) + struct.pack(fmt, value)


def test_osc_f32():
    p = eg.Param(0.0, time_const=0.0)
    datagram = _build_osc("/freq", ",f", ">f", 440.0)
    with eg.UDPParamListener(port=19010, parse=eg.parse_osc) as listener:
        listener.register("freq", p)
        _send(19010, datagram)
        time.sleep(0.005)
    assert abs(p.value - 440.0) < 1e-3


def test_osc_d64():
    p = eg.Param(0.0, time_const=0.0)
    datagram = _build_osc("/cutoff", ",d", ">d", 1200.0)
    with eg.UDPParamListener(port=19011, parse=eg.parse_osc) as listener:
        listener.register("cutoff", p)
        _send(19011, datagram)
        time.sleep(0.005)
    assert abs(p.value - 1200.0) < 1e-9


def test_osc_unsupported_type_ignored():
    p = eg.Param(99.0, time_const=0.0)
    # Build an OSC datagram with a ",s" (string) type tag — unsupported
    def pad(s: bytes) -> bytes:
        s = s + b"\x00"
        remainder = len(s) % 4
        if remainder:
            s = s + b"\x00" * (4 - remainder)
        return s

    datagram = pad(b"/freq") + pad(b",s") + pad(b"hello")
    with eg.UDPParamListener(port=19012, parse=eg.parse_osc) as listener:
        listener.register("freq", p)
        _send(19012, datagram)
        time.sleep(0.005)
    assert abs(p.value - 99.0) < 1e-6


def test_context_manager_stops_cleanly():
    p = eg.Param(0.0, time_const=0.0)
    with eg.UDPParamListener(port=19013) as listener:
        listener.register("freq", p)
        thread = listener._thread
    assert thread is not None
    assert not thread.is_alive()


def test_already_running_raises():
    p = eg.Param(0.0, time_const=0.0)
    listener = eg.UDPParamListener(port=19014)
    listener.register("freq", p)
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
