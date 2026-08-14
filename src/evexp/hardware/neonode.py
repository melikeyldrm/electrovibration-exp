"""Neonode NNAMC1580PCEV IR position sensor - PositionSource implementation.
 What IS in place and testable right now:

    - the connection lifecycle (connect/disconnect, error taxonomy)
    - the background-thread + lock pattern that turns a stream of device
      reports into PositionSource.read() snapshots, mirroring
      SensorAcquisition's worker-thread/snapshot split in acquisition.py
    - the raw-device-units -> millimetres conversion

The only two things that need filling in once the sensor enumerates are
marked TODO below: the VID/PID (or whatever transport.list_devices() needs
to find it) and _decode_report()'s actual byte layout. Everything else -
SessionController wiring, the thread, the mm conversion, error handling -
does not change when those are filled in.
"""

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

from evexp.hardware.position import PositionSample, PositionSource


class NeonodeConnectionError(RuntimeError):
    """The sensor could not be opened or stopped responding."""


@dataclass(frozen=True)
class NeonodeCalibration:
    """Maps raw device units to millimetres along the travel axis.

    Kept separate from ScreenCalibration (hardware/screen.py) even though
    the shape is similar - one is a property of the touch sensor's own
    coordinate system, the other of the monitor panel, and they are
    calibrated independently, by different people, at different times.
    """
    mm_per_unit_x: float
    mm_per_unit_y: float = 1.0
    origin_x: float = 0.0   # raw device units at physical x = 0
    origin_y: float = 0.0

    def to_mm(self, x_raw: float, y_raw: float) -> Tuple[float, float]:
        x_mm = (x_raw - self.origin_x) * self.mm_per_unit_x
        y_mm = (y_raw - self.origin_y) * self.mm_per_unit_y
        return x_mm, y_mm


class NeonodeTransport(ABC):
    """The USB/HID layer, isolated so it can be swapped for a fake in tests.

    Everything above this class (the thread, the parsing, the mm
    conversion) can be developed and tested without hardware by handing
    NeonodePositionSource a fake transport that returns canned reports.
    """

    @abstractmethod
    def open(self) -> None:
        """Enumerate and open the device. Raise NeonodeConnectionError if
        it can't be found or claimed."""

    @abstractmethod
    def close(self) -> None:
        """Release the device. Must be safe to call even if open() failed
        or was never called."""

    @abstractmethod
    def read_report(self, timeout_s: float) -> Optional[bytes]:
        """Block up to timeout_s for one report; return None on timeout.

        Raise NeonodeConnectionError if the device stops responding (as
        opposed to simply having nothing new to say)."""


class HidNeonodeTransport(NeonodeTransport):
    """Real transport, over hidapi. NOT YET WORKING.

    TODO once the sensor enumerates (currently it does not, on any machine
    tried):
      - confirm VID/PID with `python -m hid` or Neonode's own tooling and
        fill them in below (placeholders now - do not trust these numbers)
      - confirm whether this is a standard HID device or needs Neonode's
        vendor SDK instead of raw hidapi
    """

    # TODO: sensor takınca doldur:
    VENDOR_ID = 0x0000
    PRODUCT_ID = 0x0000

    def __init__(self, vendor_id: int = VENDOR_ID, product_id: int = PRODUCT_ID):
        self._vendor_id = vendor_id
        self._product_id = product_id
        self._device = None

    def open(self) -> None:
        try:
            import hid  # hidapi - not yet confirmed as the right library
        except ImportError as e:
            raise NeonodeConnectionError(
                "hidapi not installed (pip install hidapi)") from e

        try:
            device = hid.device()
            device.open(self._vendor_id, self._product_id)
        except OSError as e:
            raise NeonodeConnectionError(
                f"could not open Neonode (vid={self._vendor_id:#06x}, "
                f"pid={self._product_id:#06x}) - enumeration failing as of "
                f"the last check; see hardware notes"
            ) from e
        self._device = device

    def close(self) -> None:
        if self._device is not None:
            self._device.close()
            self._device = None

    def read_report(self, timeout_s: float) -> Optional[bytes]:
        if self._device is None:
            raise NeonodeConnectionError("read_report() called before open()")
        # TODO: confirm hidapi's timeout units (ms) and report size once
        # a real device is available to test against.
        raise NotImplementedError(
            "Neonode wire protocol not yet observed - USB enumeration has "
            "not succeeded on any machine tried. Fill this in once the "
            "device opens and a report can be captured."
        )


def _decode_report(report: bytes) -> Optional[Tuple[float, float]]:
    """Raw HID report bytes -> (x_raw, y_raw) in device units.

    Returns None if the report indicates no contact (protocol-specific -
    exact encoding unknown until a report can be captured from real
    hardware). Kept as a free function, not a method, so it can be unit
    tested against captured byte sequences without constructing a whole
    NeonodePositionSource.
    """
    raise NotImplementedError(
        "byte layout not yet known - capture a report from real hardware "
        "first (e.g. via Neonode's own demo tooling), then fill this in "
        "and add tests/test_neonode.py cases against the captured bytes"
    )


class NeonodePositionSource(PositionSource):
    """Background thread turns a stream of device reports into read() snapshots.

    Mirrors SensorAcquisition's split (hardware/acquisition.py): a worker
    thread blocks on the transport and updates a snapshot under a lock; read()
    only ever takes the lock briefly and never touches the transport, so it
    stays safe to call from both a UI timer and the acquisition thread.
    """

    def __init__(self, transport: NeonodeTransport,
                 calibration: NeonodeCalibration,
                 stale_after_s: float = 0.5):
        self._transport = transport
        self._calibration = calibration
        self._stale_after_s = stale_after_s
        self._lock = threading.Lock()
        self._latest: Optional[PositionSample] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_requested = threading.Event()

    def connect(self) -> None:
        """Open the transport and start the background read thread.

        Raises NeonodeConnectionError if the device can't be opened.
        """
        self._transport.open()
        self._stop_requested.clear()
        self._thread = threading.Thread(
            target=self._read_loop, name="neonode-read", daemon=True)
        self._thread.start()

    def disconnect(self) -> None:
        self._stop_requested.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._transport.close()
        with self._lock:
            self._latest = None

    def __enter__(self) -> "NeonodePositionSource":
        self.connect()
        return self

    def __exit__(self, *exc_info) -> None:
        self.disconnect()

    def _read_loop(self) -> None:
        while not self._stop_requested.is_set():
            try:
                report = self._transport.read_report(timeout_s=0.1)
            except NeonodeConnectionError:
                # Device dropped out mid-session. Clear the last known
                # position rather than keep serving a stale one - "no
                # finger detected" is the honest state here, same as
                # ManualPositionSource.clear().
                with self._lock:
                    self._latest = None
                break
            if report is None:
                continue

            decoded = _decode_report(report)
            if decoded is None:
                with self._lock:
                    self._latest = None
                continue

            x_raw, y_raw = decoded
            x_mm, y_mm = self._calibration.to_mm(x_raw, y_raw)
            with self._lock:
                self._latest = PositionSample(
                    t=time.perf_counter(), x_mm=x_mm, y_mm=y_mm)

    def read(self) -> Optional[PositionSample]:
        with self._lock:
            sample = self._latest
        if sample is None:
            return None
        if time.perf_counter() - sample.t > self._stale_after_s:
            return None
        return sample