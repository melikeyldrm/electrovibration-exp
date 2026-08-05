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

NiDaqDevice also implements StimulusOutput, on a separate AO task with its
own sample clock: input and output run independently on the card, and tying
them to one task would only complicate both. Every buffer written to the AO
task passes through safety.check_voltage_limit first - there is no path from
set_amplitude()/stimulus_on() to hardware that skips it.
"""

import time
from typing import Optional, Sequence, Tuple

import numpy as np

from evexp.hardware.base import DAQDevice, SensorChunk, StimulusOutput
from evexp.hardware.safety import check_voltage_limit
from evexp.processing.waveform import cycles_to_duration_s, generate_sine_wave

# Physical channels for an ATI Nano17's six bridges, wired to ai0..ai5.
DEFAULT_CHANNEL_MAP = {
    "gauge0": "ai0",
    "gauge1": "ai1",
    "gauge2": "ai2",
    "gauge3": "ai3",
    "gauge4": "ai4",
    "gauge5": "ai5",
}

# AO defaults for the electrovibration carrier.
#
# 125 Hz, not kHz: Vardar & Kuchenbecker (2021, J. R. Soc. Interface) ran the
# same hardware this project uses - SCT3250 3M touchscreen, PCIe-6321 DAQ,
# Tabor 9200A amplifier - at a 125 Hz input voltage frequency, chosen so the
# resulting electrovibration force (at 2x input, per the V^2 relationship)
# lands at 250 Hz, near the centre of peak mechanoreceptor sensitivity
# (200-300 Hz). Other papers in this literature use comparable values (100,
# 120, 180, 270, 480 Hz) - all tens to a few hundred Hz, never kHz. An
# earlier version of this file had 10 kHz here, which was wrong: confirm
# against Umut's protocol before relying on this default for real data.
#
# 200 kHz sample rate still applies: at 125 Hz that is 1600 samples/cycle,
# comfortably smooth and well under the PCIe-6321's ~900 kS/s AO limit.
DEFAULT_AO_CHANNEL = "ao0"
DEFAULT_STIMULUS_FREQUENCY_HZ = 125.0
DEFAULT_AO_SAMPLE_RATE_HZ = 200_000.0
DEFAULT_AO_CYCLES_PER_BUFFER = 1000


class NiDaqDevice(DAQDevice, StimulusOutput):
    """NI card: analog input for sensors, analog output for the stimulus.

    channel_map is an ordered mapping from the name a chunk will carry to
    the card's physical channel. Both the names and the order come from
    configuration rather than from this file, because a miswired channel
    that is merely mislabelled produces data that looks fine and is wrong.

    AI and AO run as two independent tasks on separate sample clocks; see
    the module docstring for why they are not combined.
    """

    def __init__(self, device_name: str = "Dev1",
                 channel_map: Optional[dict] = None,
                 voltage_range: Tuple[float, float] = (-10.0, 10.0),
                 buffer_seconds: float = 2.0,
                 terminal_config: str = "DIFF",
                 ao_channel: str = DEFAULT_AO_CHANNEL,
                 stimulus_frequency_hz: float = DEFAULT_STIMULUS_FREQUENCY_HZ,
                 ao_sample_rate_hz: float = DEFAULT_AO_SAMPLE_RATE_HZ,
                 ao_voltage_limit_v: float = 2.0,
                 ao_cycles_per_buffer: int = DEFAULT_AO_CYCLES_PER_BUFFER):
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

        # AO / stimulus state.
        self.ao_channel = ao_channel
        self.stimulus_frequency_hz = stimulus_frequency_hz
        self.ao_sample_rate_hz = ao_sample_rate_hz
        self.ao_voltage_limit_v = ao_voltage_limit_v
        self.ao_cycles_per_buffer = ao_cycles_per_buffer
        self._ao_task = None
        self._ao_writer = None
        self._amplitude_v = 0.0
        self._stimulus_active = False

    # --- DAQDevice (analog input) -------------------------------------------

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
            # Not preallocated to samps_per_chan: that number sizes the
            # DAQmx driver's own buffer (in-stream), which is unrelated to
            # the array Python reads into per call. read_chunk() allocates
            # its own read buffer, sized to whatever n_samples it is asked
            # for.
            self._buffer = None

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
        if self._task is None or self._reader is None:
            raise RuntimeError("start() must be called before reading")
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples!r}")

        # Reallocated only when the requested size changes (in practice:
        # once, on the first call - callers ask for the same chunk_samples
        # every time). A slice of a larger array is not guaranteed
        # C-contiguous, which read_many_sample() requires; a right-sized
        # array always is.
        if self._buffer is None or self._buffer.shape[1] != n_samples:
            self._buffer = np.empty((len(self._channels), n_samples))

        self._reader.read_many_sample(
            self._buffer,
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
            data=self._buffer.copy(),
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

    # --- StimulusOutput (analog output) -------------------------------------

    def set_amplitude(self, volts: float) -> None:
        """Set stimulus peak amplitude, in volts.

        Takes effect immediately if already on; otherwise applied the next
        time stimulus_on() opens the AO task.
        """
        self._amplitude_v = volts
        if self._stimulus_active:
            self._rewrite_ao_buffer()

    def stimulus_on(self) -> None:
        """Start continuous sine output at the current amplitude."""
        if self._ao_task is None:
            self._open_ao_task()
        else:
            self._rewrite_ao_buffer()
        self._stimulus_active = True

    def stimulus_off(self) -> None:
        """Stop output and close the AO task."""
        if self._ao_task is not None:
            try:
                self._ao_task.stop()
            finally:
                self._ao_task.close()
                self._ao_task = None
        self._ao_writer = None
        self._stimulus_active = False

    @property
    def is_active(self) -> bool:
        return self._stimulus_active

    def _open_ao_task(self) -> None:
        import nidaqmx
        from nidaqmx.constants import AcquisitionType, RegenerationMode
        from nidaqmx.stream_writers import AnalogSingleChannelWriter

        task = nidaqmx.Task()
        try:
            task.ao_channels.add_ao_voltage_chan(
                f"{self.device_name}/{self.ao_channel}",
                min_val=-self.ao_voltage_limit_v,
                max_val=self.ao_voltage_limit_v,
            )
            buffer_len = self._ao_buffer_len()
            task.timing.cfg_samp_clk_timing(
                rate=self.ao_sample_rate_hz,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=buffer_len,
            )
            # A written buffer keeps replaying until overwritten, so a
            # constant sine needs one write per amplitude change, not a
            # continuous feed from Python.
            task.out_stream.regen_mode = RegenerationMode.ALLOW_REGENERATION

            self._ao_writer = AnalogSingleChannelWriter(task.out_stream)
            self._ao_task = task
            self._write_signal(self._current_signal())
            task.start()
        except Exception:
            task.close()
            self._ao_task = None
            self._ao_writer = None
            raise

    def _rewrite_ao_buffer(self) -> None:
        self._write_signal(self._current_signal())

    def _ao_buffer_len(self) -> int:
        duration_s = cycles_to_duration_s(
            self.stimulus_frequency_hz, self.ao_cycles_per_buffer)
        return int(round(duration_s * self.ao_sample_rate_hz))

    def _current_signal(self) -> np.ndarray:
        duration_s = cycles_to_duration_s(
            self.stimulus_frequency_hz, self.ao_cycles_per_buffer)
        return generate_sine_wave(
            frequency_hz=self.stimulus_frequency_hz,
            amplitude_v=self._amplitude_v,
            duration_s=duration_s,
            sample_rate_hz=self.ao_sample_rate_hz,
        )

    def _write_signal(self, signal: np.ndarray) -> None:
        """Safety check, then write. No other path reaches the AO task."""
        check_voltage_limit(signal, limit_v=self.ao_voltage_limit_v)
        self._ao_writer.write_many_sample(signal)

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