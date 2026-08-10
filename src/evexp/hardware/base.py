"""Hardware interfaces: acquisition in, actuation out.

Acquisition is block-oriented. A sensor sampled at 10 kHz produces a sample
every 100 us, and Python cannot service a call that often - the per-call
overhead alone exceeds the sample interval, the driver's buffer fills, and
the acquisition fails with an overflow. Reading in blocks of a few hundred
samples moves the loop rate down to something Python is comfortable with
while the card's own DMA keeps the timing exact.

The single-sample read() is kept as a convenience for code that only wants a
current value, but it is a thin wrapper over the block read and is not the
path a real session should take.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

# A block covering roughly 50 ms. Short enough that the displayed values and
# the trial boundaries stay responsive, long enough that the read loop runs
# at a rate Python can sustain: 20 calls a second rather than 10,000.
CHUNK_FRACTION_OF_SECOND = 20
MIN_CHUNK_SAMPLES = 64
MAX_CHUNK_SAMPLES = 4096


def default_chunk_samples(sample_rate_hz: float) -> int:
    """Block size to read at a given sample rate."""
    target = int(sample_rate_hz // CHUNK_FRACTION_OF_SECOND)
    return max(MIN_CHUNK_SAMPLES, min(MAX_CHUNK_SAMPLES, target))


@dataclass
class SensorSample:
    """One instant, one value per channel."""
    timestamp: float
    values: dict


@dataclass(frozen=True)
class SensorChunk:
    """A block of consecutive samples across every channel.

    Stored channels-first, matching what the NI-DAQmx readers produce, so
    that no transpose or copy is needed on the acquisition path.

    t0 is a host-clock timestamp for the first sample and is only as good as
    the host clock. Sample *spacing* comes from the card's own clock and is
    far more precise than t0 is; anything needing exact intervals should
    derive them from sample_rate_hz rather than from repeated timestamps.
    """

    t0: float
    sample_rate_hz: float
    channels: Tuple[str, ...]
    data: np.ndarray            # shape (n_channels, n_samples)

    def __post_init__(self) -> None:
        if self.data.ndim != 2:
            raise ValueError(
                f"SensorChunk.data must be 2-D (channels, samples), "
                f"got shape {self.data.shape}"
            )
        if self.data.shape[0] != len(self.channels):
            raise ValueError(
                f"SensorChunk has {len(self.channels)} channel names but "
                f"{self.data.shape[0]} rows of data"
            )
        if self.sample_rate_hz <= 0:
            raise ValueError(
                f"SensorChunk.sample_rate_hz must be positive, "
                f"got {self.sample_rate_hz!r}"
            )

    @property
    def n_channels(self) -> int:
        return self.data.shape[0]

    @property
    def n_samples(self) -> int:
        return self.data.shape[1]

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.sample_rate_hz

    @property
    def t_end(self) -> float:
        return self.t0 + self.duration_s

    def times(self) -> np.ndarray:
        """Host-clock timestamp for each sample, spaced by the card clock."""
        return self.t0 + np.arange(self.n_samples) / self.sample_rate_hz

    def channel(self, name: str) -> np.ndarray:
        try:
            return self.data[self.channels.index(name)]
        except ValueError:
            raise KeyError(
                f"no channel {name!r}; available: {', '.join(self.channels)}"
            ) from None

    def latest(self) -> SensorSample:
        """The final sample of the block, as a per-channel mapping. """
        # to use at gui 
        return SensorSample(
            timestamp=self.t_end,
            values={name: float(self.data[i, -1])
                    for i, name in enumerate(self.channels)},
        )

    def mean(self) -> Dict[str, float]:
        return {name: float(self.data[i].mean())
                for i, name in enumerate(self.channels)}


class DAQDevice(ABC):
    """Continuous multi-channel analog acquisition."""

    @property
    @abstractmethod
    def channels(self) -> Tuple[str, ...]:
        """Channel names, in the order they appear in a chunk's rows."""

    @property
    @abstractmethod
    def sample_rate_hz(self) -> float:
        """Configured sample rate. Meaningful only after start()."""

    @abstractmethod
    def start(self, sample_rate_hz: float) -> None:
        """Begin continuous acquisition."""

    @abstractmethod
    def read_chunk(self, n_samples: int,
                   timeout_s: float = 10.0) -> SensorChunk:
        """Return the next n_samples on every channel.

        Blocks until the samples are available, so it must be called from a
        worker thread rather than from the Qt event loop.
        """

    @abstractmethod
    def stop(self) -> None:
        """End acquisition and release the device."""

    # --- convenience -------------------------------------------------------

    def read(self) -> SensorSample:
        """One sample per channel.

        Provided for scripts and probes. A session should read blocks: at
        realistic sample rates this call cannot keep up with the device.
        """
        return self.read_chunk(1).latest()

    def default_chunk_samples(self) -> int:
        return default_chunk_samples(self.sample_rate_hz)

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