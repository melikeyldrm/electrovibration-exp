import math
import random
import time

from evexp.hardware.base import DAQDevice, SensorSample, StimulusOutput


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


class MockStimulusOutput(StimulusOutput):
    """Stand-in for the DAQ output -> amplifier -> touchscreen chain.

    Records every on/off transition, so a simulated run can be checked for
    the correct number of activations per trial without any hardware.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._amplitude_v = 0.0
        self._active = False
        self.events: list[tuple[float, str, float]] = []

    def set_amplitude(self, volts: float) -> None:
        """Set target stimulus voltage amplitude."""
        self._amplitude_v = volts

    def stimulus_on(self) -> None:
        self._active = True
        self._record("on")

    def stimulus_off(self) -> None:
        # Calling this while already off is harmless; it keeps the caller simple.
        was_active = self._active
        self._active = False
        if was_active:
            self._record("off")

    @property
    def is_active(self) -> bool:
        """Check if stimulus is currently active. read-only"""
        return self._active

    @property
    def amplitude_v(self) -> float:
        """Get current amplitude in volts. read-only"""
        return self._amplitude_v

    def _record(self, kind: str) -> None:
        """Record event timestamp, action, and voltage amplitude."""
        self.events.append((time.time(), kind, self._amplitude_v))
        if self.verbose:
            print(f"[stimulus] {kind:<3s} {self._amplitude_v:6.2f} V")