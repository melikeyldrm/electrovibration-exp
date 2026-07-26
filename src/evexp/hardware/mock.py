import math
import random
import time

from evexp.hardware.base import DAQDevice, SensorSample


class MockDAQDevice(DAQDevice):
    """Mock DAQ device for testing without physical hardware."""


    def __init__(self, true_threshold: float = 50.0, slope: float = 0.15, noise_std: float = 0.02):
        self.true_threshold = true_threshold
        self.slope = slope
        self.noise_std = noise_std
        self._applied_voltage: float = 0.0
        self._running = False

    def start(self, sample_rate_hz: float) -> None:
        """Initialize the mock device."""
        self._running = True

    def set_applied_voltage(self, voltage: float) -> None:
        """Update the simulated stimulation voltage."""
        self._applied_voltage = voltage

    def read(self) -> SensorSample:
        """Return one simulated sensor sample."""

        if not self._running:
            raise RuntimeError("Device must be started before reading.")
        force_normal = 0.3 + random.gauss(0, self.noise_std)
        force_tangential = 0.05 + random.gauss(0, self.noise_std / 2)
        return SensorSample(
            timestamp=time.time(),
            values={
                "force_normal": force_normal,
                "force_tangential": force_tangential,
                "applied_voltage": self._applied_voltage,
            },
        )

    def detection_probability(self) -> float:
        """Return simulated detection probability."""
        p_detect = 1 / (1 + math.exp(-self.slope * (self._applied_voltage - self.true_threshold)))
        return 0.5 + 0.5 * p_detect

    def stop(self) -> None:
        """Stop the mock device."""
        self._running = False