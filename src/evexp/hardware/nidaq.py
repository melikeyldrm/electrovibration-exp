"""NI-DAQmx implementation of DAQDevice + StimulusOutput.

AI (sensors) and AO (stimulus) are two independent tasks, own sample
clocks - not combined, so one doesn't block the other.

Key decisions:
- read_many_sample() into a preallocated buffer, not Task.read() per
  sample - the latter overflows the driver buffer at 10 kHz.
- Explicit samps_per_chan (buffer_seconds) - the DAQmx default is too small.
- One AI task for all channels - shares a sample clock, keeps them in sync.
- Every AO write goes through safety.check_voltage_limit first.
- terminal_config defaults to DEFAULT (differential on the 6321), matching
  the rig's existing acquisition code. With one sensor per card the six
  gauges fit inside the 8-channel differential limit.

One instance drives one card. The rig has three: two reading a force sensor
each, one carrying the stimulus AO and the amplifier's monitor input. The
force cards are joined into a single stream by multi_daq.MultiDaqDevice.
"""

import time
from typing import Optional, Sequence, Tuple

import numpy as np

from evexp.hardware.base import DAQDevice, SensorChunk, StimulusOutput
from evexp.hardware.daq_errors import translate_daq_error
from evexp.hardware.safety import check_voltage_limit
from evexp.processing.signal import cycles_to_duration_s, generate_sine_wave


def gauge_channel_map(prefix: str, first_ai: int = 0) -> dict:
    """Six gauges of one sensor -> ai channels, named by which sensor it is."""
    return {f"{prefix}_gauge{i}": f"ai{first_ai + i}" for i in range(6)}

# One sensor per card, six bridges on ai0..ai5. The channel *names* carry
# which sensor it is (fs1_/fs2_), so the same physical map serves both
# cards - see gauge_channel_map() and multi_daq.py.
DEFAULT_CHANNEL_MAP = gauge_channel_map("fs1")

# AO defaults for the electrovibration carrier.

# 200 kHz sample rate: 1600 samples/cycle, within PCIe-6321's AO limit.
# 5 cycles/buffer: was 1000 (8s, 1.6M samples) -
DEFAULT_AO_CHANNEL = "ao0"
DEFAULT_STIMULUS_FREQUENCY_HZ = 125.0
DEFAULT_AO_SAMPLE_RATE_HZ = 200_000.0
DEFAULT_AO_CYCLES_PER_BUFFER = 5


class NiDaqDevice(DAQDevice, StimulusOutput):
    """NI card: analog input for sensors, analog output for the stimulus.

    channel_map maps a chunk's channel name to the card's physical channel;
    comes from config, not hardcoded here, to keep miswiring visible.
    """

    def __init__(self, device_name: str = "Dev1",
                 channel_map: Optional[dict] = None,
                 voltage_range: Tuple[float, float] = (-10.0, 10.0),
                 buffer_seconds: float = 2.0,
                 terminal_config: str = "DEFAULT",
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
            # Buffer size set explicitly - the driver's default is small.
            task.in_stream.input_buf_size = samps_per_chan

            self._reader = AnalogMultiChannelReader(task.in_stream)
            # Sized in read_chunk() per call, not here - see comment there.
            self._buffer = None

            task.start()
        except nidaqmx.DaqError as e:
            task.close()
            raise translate_daq_error(e, device=self.device_name) from e
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

        # Reallocated only if size changes (normally once). Must be a
        # right-sized array, not a slice - read_many_sample() needs
        # C-contiguous.
        if self._buffer is None or self._buffer.shape[1] != n_samples:
            self._buffer = np.empty((len(self._channels), n_samples))

        try:
            self._reader.read_many_sample(
                self._buffer,
                number_of_samples_per_channel=n_samples,
                timeout=timeout_s,
            )
        except Exception as e:
            # Imported here, not at module top, so this file loads without
            # the driver installed.
            import nidaqmx
            if isinstance(e, nidaqmx.DaqError):
                raise translate_daq_error(e, device=self.device_name) from e
            raise

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
        except nidaqmx.DaqError as e:
            task.close()
            self._ao_task = None
            self._ao_writer = None
            raise translate_daq_error(e, device=self.device_name) from e
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
        try:
            self._ao_writer.write_many_sample(signal)
        except Exception as e:
            import nidaqmx
            if isinstance(e, nidaqmx.DaqError):
                raise translate_daq_error(e, device=self.device_name) from e
            raise

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