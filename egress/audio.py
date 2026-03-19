"""
DAC — Digital-to-Analog Converter.  Wraps egress_dac_t.
"""

import ctypes

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

    @property
    def active_device(self) -> int:
        """Device ID currently open for output (0 if not started)."""
        return int(_b.egress_dac_get_active_device(self._h))

    def switch_device(self, device_id: int) -> bool:
        """
        Switch output to the specified device while running.

        Returns True on success, False if the DAC is not running or the
        device ID is invalid / has no output channels.
        """
        return bool(_b.egress_dac_switch_device(self._h, device_id))

    # ---------- Class-level device enumeration ----------

    @staticmethod
    def list_devices() -> list:
        """
        Return a list of dicts describing all available audio devices.

        Each dict contains:
          id                    -- RtAudio device ID
          name                  -- human-readable device name
          output_channels       -- maximum output channels
          input_channels        -- maximum input channels
          is_default_output     -- True if this is the system default output
          preferred_sample_rate -- driver-preferred sample rate in Hz
          sample_rates          -- list of supported sample rates
        """
        count = _b.egress_audio_device_count()
        if count == 0:
            return []
        id_arr = (ctypes.c_uint * count)()
        _b.egress_audio_get_device_ids(id_arr, count)
        devices = []
        info = _b.EgressDeviceInfo()
        for device_id in id_arr:
            if _b.egress_audio_get_device_info(device_id, ctypes.byref(info)):
                devices.append({
                    "id":                    info.id,
                    "name":                  info.name.decode("utf-8", errors="replace"),
                    "output_channels":       info.output_channels,
                    "input_channels":        info.input_channels,
                    "is_default_output":     bool(info.is_default_output),
                    "preferred_sample_rate": info.preferred_sample_rate,
                    "sample_rates":          list(info.sample_rates[:info.sample_rate_count]),
                })
        return devices

    @staticmethod
    def default_device() -> int:
        """Return the system default output device ID."""
        return int(_b.egress_audio_default_output_device())
