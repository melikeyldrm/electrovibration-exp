"""Several DAQ cards presented as one acquisition stream.

The rig reads its two force sensors on separate cards, and the amplifier's
monitor input on a third. SensorAcquisition, the ring buffer and everything
above them expect a single DAQDevice with one channel list, so the cards are
joined here rather than threaded through the whole stack.

Each card runs on its own sample clock. They are configured at the same rate
and started back to back, but crystals differ by a hundred ppm or so, so the
cards drift slowly relative to one another - about ten milliseconds over half
an hour. That is immaterial for normal force, which is what drives the
staircase, and it matches how the rig's existing acquisition code works. It
would matter for comparing the two sensors sample by sample at the stimulus
frequency; if that is ever needed, the fix is an RTSI or PFI link and a
source= argument on the follower cards' timing, not a change here.
"""

from typing import Optional, Sequence, Tuple

import numpy as np

from evexp.hardware.base import DAQDevice, SensorChunk


class MultiDaqDevice(DAQDevice):
    """Reads several cards per chunk and stacks them into one block.

    Channel names must be unique across cards - they are what every layer
    above addresses the data by, so a collision would silently shadow one
    card's channel with another's.
    """

    def __init__(self, devices: Sequence[DAQDevice]):
        if not devices:
            raise ValueError("MultiDaqDevice needs at least one device")
        self._devices = tuple(devices)

        names: Tuple[str, ...] = ()
        for device in self._devices:
            names += tuple(device.channels)
        if len(set(names)) != len(names):
            duplicates = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(
                f"channel names must be unique across cards; repeated: "
                f"{', '.join(duplicates)}")
        self._channels = names
        self._sample_rate_hz = 0.0

    @property
    def channels(self) -> Tuple[str, ...]:
        return self._channels

    @property
    def sample_rate_hz(self) -> float:
        return self._sample_rate_hz

    @property
    def devices(self) -> Tuple[DAQDevice, ...]:
        return self._devices

    def start(self, sample_rate_hz: float) -> None:
        started = []
        try:
            for device in self._devices:
                device.start(sample_rate_hz)
                started.append(device)
        except Exception:
            # A half-started set would hold its cards open with no way to
            # release them, so unwind before letting the error out.
            for device in reversed(started):
                try:
                    device.stop()
                except Exception:                       # noqa: BLE001
                    pass
            raise
        self._sample_rate_hz = float(sample_rate_hz)

    def read_chunk(self, n_samples: int,
                   timeout_s: float = 10.0) -> SensorChunk:
        chunks = [device.read_chunk(n_samples, timeout_s)
                  for device in self._devices]
        # Each card returns its own count; a short read on one would
        # misalign the stack, so refuse rather than pad.
        lengths = {chunk.n_samples for chunk in chunks}
        if len(lengths) != 1:
            raise RuntimeError(
                f"cards returned different sample counts: {sorted(lengths)}")
        # t0 comes from the first card. The others are within a read of it,
        # and their exact offsets are not knowable without a shared trigger.
        return SensorChunk(
            t0=chunks[0].t0,
            sample_rate_hz=chunks[0].sample_rate_hz,
            channels=self._channels,
            data=np.vstack([chunk.data for chunk in chunks]),
        )

    def stop(self) -> None:
        """Stop every card, then report the first failure if any."""
        first_error: Optional[Exception] = None
        for device in self._devices:
            try:
                device.stop()
            except Exception as exc:                    # noqa: BLE001
                first_error = first_error or exc
        if first_error is not None:
            raise first_error