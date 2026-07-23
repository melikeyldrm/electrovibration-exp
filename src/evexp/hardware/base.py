from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SensorSample:
    timestamp: float
    values: dict  


class DAQDevice(ABC):
    """Herhangi bir çok-kanallı veri toplama cihazı için soyut arayüz."""

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