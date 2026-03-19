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

    @property
    def is_reconnecting(self) -> bool:
        """True while a device-disconnect has been detected and reconnection is in progress."""
        return bool(_b.egress_dac_is_reconnecting(self._h))

    def callback_stats(self) -> dict:
        """
        Return a dict of real-time callback diagnostics:
          callback_count  -- total callbacks timed (excludes warmup buffers)
          avg_callback_ms -- mean callback wall-clock time in ms
          max_callback_ms -- worst-case callback wall-clock time in ms
          underrun_count  -- non-zero RtAudioStreamStatus events (driver underruns)
          overrun_count   -- callbacks that exceeded the buffer time budget
        """
        s = _b.EgressDacStats()
        _b.egress_dac_get_stats(self._h, s)
        return {
            "callback_count":  s.callback_count,
            "avg_callback_ms": s.avg_callback_ms,
            "max_callback_ms": s.max_callback_ms,
            "underrun_count":  s.underrun_count,
            "overrun_count":   s.overrun_count,
        }

    def reset_stats(self) -> None:
        """Reset all callback diagnostic counters."""
        _b.egress_dac_reset_stats(self._h)
