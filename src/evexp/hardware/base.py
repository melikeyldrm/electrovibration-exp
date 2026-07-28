from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SensorSample:
    timestamp: float
    values: dict  


class DAQDevice(ABC):

    @abstractmethod
    def start(self, sample_rate_hz: float) -> None: ...

    @abstractmethod
    def read(self) -> SensorSample: ...

    @abstractmethod
    def stop(self) -> None: ...

    def __enter__(self):
        self.start(sample_rate_hz=1000)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()

class StimulusOutput(ABC):
    """Abstract interface for the electrovibration stimulus output path.

    Physically this is: DAQ analog output -> high-voltage amplifier -> touchscreen.
    This is deliberately separate from DAQDevice: that interface describes
    acquisition (reading sensors), whereas this one describes actuation
    (emitting a signal). Keeping them apart means a real amplifier can be
    swapped in without touching the acquisition code, and vice versa.
    """

    @abstractmethod
    def set_amplitude(self, volts: float) -> None:
        """Set the peak amplitude of the stimulus waveform, in volts."""

    @abstractmethod
    def stimulus_on(self) -> None:
        """Start emitting the stimulus waveform."""

    @abstractmethod
    def stimulus_off(self) -> None:
        """Stop emitting the stimulus waveform."""

    @property
    @abstractmethod
    def is_active(self) -> bool:
        """True while the stimulus waveform is being emitted."""