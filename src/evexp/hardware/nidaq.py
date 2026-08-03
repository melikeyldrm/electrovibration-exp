"""NI-DAQmx implementation of DAQDevice.

Not yet used by any session: the real card has not been wired in, and
MockDAQDevice stands in everywhere. It is kept current anyway so that
switching over is a change of constructor rather than a rewrite, and so that
the decisions that are easy to get wrong are already made and written down.

Three of those decisions matter enough to state plainly.

Blocks, not samples. The previous version called Task.read() with no
arguments, which reads one sample. At 1 kHz that survives; at 10 kHz the
per-call overhead alone exceeds the sample interval, the driver's buffer
fills and DAQmx raises error -200279. read_many_sample() into a preallocated
array reads a whole block per call and allocates nothing.

An explicit buffer. cfg_samp_clk_timing() picks a default buffer size when
samps_per_chan is omitted, and the default is not generous. Sizing it in
seconds means a scheduling hiccup in the read loop costs latency instead of
data.

One task for every channel. Sharing a single sample clock across all inputs
makes them simultaneous by construction. Separate tasks per sensor would
need their timing reconciled afterwards, on data that has already been
recorded.
"""

import time
from typing import Optional, Sequence, Tuple

import numpy as np

from evexp.hardware.base import DAQDevice, SensorChunk

# Physical channels for an ATI Nano17's six bridges, wired to ai0..ai5.
DEFAULT_CHANNEL_MAP = {
    "gauge0": "ai0",
    "gauge1": "ai1",
    "gauge2": "ai2",
    "gauge3": "ai3",
    "gauge4": "ai4",
    "gauge5": "ai5",
}


class NiDaqDevice(DAQDevice):
    """Continuous analog input from an NI card via NI-DAQmx.

    channel_map is an ordered mapping from the name a chunk will carry to
    the card's physical channel. Both the names and the order come from
    configuration rather than from this file, because a miswired channel
    that is merely mislabelled produces data that looks fine and is wrong.
    """

    def __init__(self, device_name: str = "Dev1",
                 channel_map: Optional[dict] = None,
                 voltage_range: Tuple[float, float] = (-10.0, 10.0),
                 buffer_seconds: float = 2.0,
                 terminal_config: str = "DIFFERENTIAL"):
        self.device_name = device_name
        self.channel_map = dict(channel_map or DEFAULT_CHANNEL_MAP)
        self.voltage_range = voltage_range
        self.buffer_seconds = buffer_seconds
        self.terminal_config = terminal_config

        self._channels: Tuple[str, ...] = tuple(self.channel_map.keys())
        self._task = None
        self._reader = None
        self._buffer: Optional[np.ndarray] = None
        self._sample_rate_hz: float = 0.0
        self._samples_read = 0
        self._t_start = 0.0

    # --- DAQDevice ---------------------------------------------------------

    @property
    def channels(self) -> Tuple[str, ...]:
        return self._channels

    @property
    def sample_rate_hz(self) -> float:
        return self._sample_rate_hz

    def start(self, sample_rate_hz: float) -> None:
        import nidaqmx
        from nidaqmx.constants import AcquisitionType, TerminalConfiguration
        from nidaqmx.stream_readers import AnalogMultiChannelReader

        if self._task is not None:
            raise RuntimeError("start() called twice; call stop() first")
        if sample_rate_hz <= 0:
            raise ValueError(
                f"sample_rate_hz must be positive, got {sample_rate_hz!r}")

        terminal = getattr(TerminalConfiguration, self.terminal_config)
        low, high = self.voltage_range

        task = nidaqmx.Task()
        try:
            for physical in self.channel_map.values():
                task.ai_channels.add_ai_voltage_chan(
                    f"{self.device_name}/{physical}",
                    terminal_config=terminal,
                    min_val=low, max_val=high,
                )

            samps_per_chan = max(1000, int(sample_rate_hz * self.buffer_seconds))
            task.timing.cfg_samp_clk_timing(
                rate=sample_rate_hz,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=samps_per_chan,
            )
            # The driver derives a buffer from samps_per_chan, but says so
            # only implicitly; setting it outright removes the guesswork.
            task.in_stream.input_buf_size = samps_per_chan

            self._reader = AnalogMultiChannelReader(task.in_stream)
            # Preallocated and reused: allocating inside the read loop is
            # what turns a comfortable margin into a dropped sample.
            self._buffer = np.zeros((len(self._channels), samps_per_chan))

            task.start()
        except Exception:
            task.close()
            raise

        self._task = task
        self._sample_rate_hz = float(sample_rate_hz)
        self._samples_read = 0
        self._t_start = time.perf_counter()

    def read_chunk(self, n_samples: int,
                   timeout_s: float = 10.0) -> SensorChunk:
        if self._task is None or self._reader is None or self._buffer is None:
            raise RuntimeError("start() must be called before reading")
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples!r}")
        if n_samples > self._buffer.shape[1]:
            raise ValueError(
                f"requested {n_samples} samples but the buffer holds "
                f"{self._buffer.shape[1]}; raise buffer_seconds"
            )

        view = self._buffer[:, :n_samples]
        self._reader.read_many_sample(
            view,
            number_of_samples_per_channel=n_samples,
            timeout=timeout_s,
        )

        t0 = self._t_start + self._samples_read / self._sample_rate_hz
        self._samples_read += n_samples
        # Copied out of the reusable buffer: the chunk outlives this call and
        # the next read would otherwise overwrite it underneath its owner.
        return SensorChunk(
            t0=t0,
            sample_rate_hz=self._sample_rate_hz,
            channels=self._channels,
            data=view.copy(),
        )

    def stop(self) -> None:
        if self._task is not None:
            try:
                self._task.stop()
            finally:
                self._task.close()
                self._task = None
        self._reader = None
        self._buffer = None

    # --- diagnostics -------------------------------------------------------

    @property
    def samples_read(self) -> int:
        return self._samples_read

    def expected_samples(self) -> float:
        """Samples the card should have produced by now.

        Compared against samples_read, this is the cheap way to notice that
        the read loop is falling behind - long before DAQmx reports an
        overflow and loses data.
        """
        if self._task is None:
            return 0.0
        return (time.perf_counter() - self._t_start) * self._sample_rate_hz