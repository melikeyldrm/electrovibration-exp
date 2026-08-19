"""Neonode NNAMC1580PCEV IR position sensor - PositionSource implementation.

The sensor exposes two HID (Human Interface Device - the standard USB
protocol used by keyboards, mice and touch panels) interfaces. The
digitizer one carries standard touch reports but Windows reserves it as a
system pointing device and will not let a program read it. The vendor one
(usage page 0xFF00) carries the zForce protocol and is what this uses.

That interface is a feature-report pipe rather than a stream: the host
writes to Feature Report 1 and reads from Feature Report 2, and the sensor
sends nothing until it has been put in detection mode and enabled. Reports
are 257 bytes - a shorter buffer makes every read fail outright rather than
return a short result, which is worth knowing because the failure looks
like a broken device.

The Windows HID API is called through ctypes. hidapi writes to this device
correctly but its feature-report reads always fail here, while the same
HidD_GetFeature call through ctypes returns the sensor's response. hidapi
is still used to enumerate, which works fine.

Coordinates arrive in units of 0.1 mm from the sensor's own origin, so the
calibration is a fixed scale factor rather than something to measure.
"""

import ctypes
import threading
import time
from abc import ABC, abstractmethod
from ctypes import wintypes
from dataclasses import dataclass
from typing import List, Optional, Tuple

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

    The defaults reflect the protocol: the sensor reports in units of
    0.1 mm from its own origin. Set origin_x to align the sensor's zero
    with the start of the cue track, and negate mm_per_unit_x if the
    sensor is mounted against the direction of travel.
    """
    mm_per_unit_x: float = 0.1
    mm_per_unit_y: float = 0.1
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


VENDOR_ID = 0x1536
PRODUCT_ID = 0x0101
VENDOR_USAGE_PAGE = 0xFF00

WRITE_REPORT_ID = 1
READ_REPORT_ID = 2
FEATURE_REPORT_LENGTH = 257

# From the zForce AIR protocol. Detection mode makes the sensor produce
# touch notifications; enable starts them flowing.
SET_DETECTION_MODE = bytes.fromhex("EE1240020200670C8001FF810100820100830100")
ENABLE = bytes.fromhex("EE09400202006503810100")
DISABLE = bytes.fromhex("EE08400202006502 8000".replace(" ", ""))

NOTIFICATION_FRAME = 0xF0
TOUCH_TAG = 0x42

EVENT_DOWN, EVENT_MOVE, EVENT_UP = 0, 1, 2

_GENERIC_READ = 0x80000000
_GENERIC_WRITE = 0x40000000
_FILE_SHARE_RW = 3
_OPEN_EXISTING = 3
_INVALID_HANDLE = ctypes.c_void_p(-1).value


class HidNeonodeTransport(NeonodeTransport):
    """The sensor's vendor HID interface, over the Windows HID API."""

    def __init__(self, vendor_id: int = VENDOR_ID,
                 product_id: int = PRODUCT_ID):
        self._vendor_id = vendor_id
        self._product_id = product_id
        self._handle = None
        self._hid_dll = None
        self._kernel32 = None
        self._last_payload = b""

    def _find_path(self) -> str:
        try:
            import hid
        except ImportError as e:
            raise NeonodeConnectionError(
                "hidapi not installed (pip install hidapi)") from e
        for d in hid.enumerate(self._vendor_id, self._product_id):
            if d.get("usage_page") == VENDOR_USAGE_PAGE:
                path = d["path"]
                return path.decode() if isinstance(path, bytes) else path
        return ""

    def open(self) -> None:
        if not hasattr(ctypes, "WinDLL"):
            raise NeonodeConnectionError(
                "the Neonode transport uses the Windows HID API and only "
                "runs on Windows")
        path = self._find_path()
        if not path:
            raise NeonodeConnectionError(
                f"no Neonode vendor interface (vid={self._vendor_id:#06x}, "
                f"pid={self._product_id:#06x}) - check the sensor is "
                f"plugged in and Neonode Workbench is closed")

        self._hid_dll = ctypes.WinDLL("hid")
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # Declared explicitly: the default restype is a 32-bit int, which
        # truncates a 64-bit handle into failures that look like a fault
        # in the device rather than in this call.
        self._kernel32.CreateFileW.restype = ctypes.c_void_p

        handle = self._kernel32.CreateFileW(
            ctypes.c_wchar_p(path),
            wintypes.DWORD(_GENERIC_READ | _GENERIC_WRITE),
            wintypes.DWORD(_FILE_SHARE_RW), None,
            wintypes.DWORD(_OPEN_EXISTING), wintypes.DWORD(0), None)
        if handle == _INVALID_HANDLE:
            raise NeonodeConnectionError(
                f"could not open the Neonode - is Workbench holding it? "
                f"(win32 error {ctypes.get_last_error()})")
        self._handle = handle

        # Order matters: detection mode first, then enable.
        self._send(SET_DETECTION_MODE)
        time.sleep(0.15)
        self._receive()
        self._send(ENABLE)
        time.sleep(0.15)
        self._receive()

    def close(self) -> None:
        if self._handle is None:
            return
        try:
            self._send(DISABLE)
        except Exception:                              # noqa: BLE001
            pass
        self._kernel32.CloseHandle(wintypes.HANDLE(self._handle))
        self._handle = None

    def _send(self, message: bytes) -> bool:
        buffer = ctypes.create_string_buffer(FEATURE_REPORT_LENGTH)
        payload = bytes([WRITE_REPORT_ID, len(message)]) + message
        buffer[:len(payload)] = payload
        return bool(self._hid_dll.HidD_SetFeature(
            wintypes.HANDLE(self._handle), buffer,
            wintypes.ULONG(FEATURE_REPORT_LENGTH)))

    def _receive(self) -> bytes:
        buffer = ctypes.create_string_buffer(FEATURE_REPORT_LENGTH)
        buffer[0] = bytes([READ_REPORT_ID])
        if not self._hid_dll.HidD_GetFeature(
                wintypes.HANDLE(self._handle), buffer,
                wintypes.ULONG(FEATURE_REPORT_LENGTH)):
            return b""
        raw = buffer.raw
        return bytes(raw[2:2 + raw[1]]) if raw[1] else b""

    def read_report(self, timeout_s: float) -> Optional[bytes]:
        """Poll until a message that is not a repeat of the last one.

        The sensor holds its last message in the buffer, so the same one
        reads back until a new one lands. Each notification carries a
        timestamp, so comparing whole payloads drops repeats without also
        dropping a finger that genuinely has not moved.
        """
        if self._handle is None:
            raise NeonodeConnectionError("read_report() called before open()")
        deadline = time.perf_counter() + timeout_s
        while True:
            payload = self._receive()
            if payload and payload != self._last_payload:
                self._last_payload = payload
                return payload
            if time.perf_counter() >= deadline:
                return None
            time.sleep(0.002)


def _decode_report(report: bytes) -> Optional[Tuple[float, float]]:
    """One notification -> (x, y) in device units, or None for no contact.

    Only the first touch is used: the experiment tracks one finger, and
    the sensor reports at most two. An up event is no contact.
    """
    touches = _parse_touches(report)
    if not touches:
        return None
    touch_id, event, x_raw, y_raw = touches[0]
    if event == EVENT_UP:
        return None
    return float(x_raw), float(y_raw)


def _parse_touches(report: bytes) -> List[Tuple[int, int, int, int]]:
    """Every touch in a notification, as (id, event, x_raw, y_raw).

    Each touch is a TLV (tag-length-value: a small self-describing chunk of
    binary data - a tag byte says what it is, a length byte says how many
    bytes follow, then the actual value): tag 0x42, a length, then id,
    event, x, y and sizes. Scanning for the tag rather than indexing from a
    fixed offset keeps this working whether the frame carries one touch or
    three, and with or without the trailing timestamp - both of which vary
    with the number of fingers and the firmware version.
    """
    if not report or report[0] != NOTIFICATION_FRAME:
        return []
    touches = []
    i = 0
    while i < len(report) - 1:
        if report[i] != TOUCH_TAG:
            i += 1
            continue
        length = report[i + 1]
        body = report[i + 2:i + 2 + length]
        if len(body) < 6:
            break
        touches.append((
            body[0],
            body[1],
            int.from_bytes(body[2:4], "big", signed=True),
            int.from_bytes(body[4:6], "big", signed=True),
        ))
        i += 2 + length
    return touches


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