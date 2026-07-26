import time
import nidaqmx
from nidaqmx.constants import AcquisitionType

from evexp.hardware.base import DAQDevice, SensorSample


class NiDaqDevice(DAQDevice):
    """NI-DAQmx kartı / NI MAX simularion için DAQDevice implementasyonu."""

    def __init__(self, device_name: str = "Dev1", channels: list[str] | None = None):
        self.device_name = device_name
        self.channels = channels or ["ai0"]
        self._task: nidaqmx.Task | None = None
        self._sample_rate_hz: float = 1000.0

    def start(self, sample_rate_hz: float) -> None:
        self._sample_rate_hz = sample_rate_hz
        self._task = nidaqmx.Task()
        for ch in self.channels:
            self._task.ai_channels.add_ai_voltage_chan(f"{self.device_name}/{ch}")
        self._task.timing.cfg_samp_clk_timing(
            rate=sample_rate_hz,
            sample_mode=AcquisitionType.CONTINUOUS,
        )
        self._task.start()

    def read(self) -> SensorSample:
        if self._task is None:
            raise RuntimeError("start() önce çağrılmalı")
        values = self._task.read()
        if not isinstance(values, list):
            values = [values]
        return SensorSample(
            timestamp=time.time(),
            values={ch: v for ch, v in zip(self.channels, values)},
        )

    def stop(self) -> None:
        if self._task is not None:
            self._task.stop()
            self._task.close()
            self._task = None