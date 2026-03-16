"""
DAC — Digital-to-Analog Converter.  Wraps egress_dac_t.
"""

from . import _bindings as _b

__all__ = ["DAC"]


class DAC:
    """
    Real-time audio output via RtAudio.

    Pulls buffers from an egress Graph and sends them to the default output
    device at `sample_rate` Hz across `channels` channels.
    """

    def __init__(self, graph, sample_rate: int = 44100, channels: int = 2):
        """
        Parameters
        ----------
        graph
            An egress.Graph instance.
        sample_rate
            Audio sample rate in Hz (default 44100).
        channels
            Number of output channels (default 2, stereo).
        """
        self._graph = graph
        self._h = _b.check(
            _b.egress_dac_new(graph._h, sample_rate, channels),
            "dac_new",
        )

    def __del__(self):
        if self._h:
            _b.egress_dac_free(self._h)
            self._h = None

    def start(self):
        """Open the audio stream and begin playback."""
        _b.egress_dac_start(self._h)

    def stop(self):
        """Stop playback and close the audio stream."""
        _b.egress_dac_stop(self._h)

    @property
    def is_running(self) -> bool:
        """True if the audio stream is currently active."""
        return bool(_b.egress_dac_is_running(self._h))
