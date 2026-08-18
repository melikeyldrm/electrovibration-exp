"""Bring the Neonode up in raw mode and print touch notifications.

The vendor interface is a feature-report pipe, not a stream: the host
writes to Feature Report 1 and reads from Feature Report 2, and nothing
arrives until the sensor is put in detection mode and enabled.

    feature report   257 bytes: [report id][byte count][ASN.1 message]

Called through ctypes rather than the hid package: hidapi writes to this
device fine but its feature-report reads always fail here, while the same
HidD_GetFeature call through ctypes returns the sensor's response. Windows
only, which the rig is.

Coordinates arrive in units of 0.1 mm, so no ruler calibration is needed.
Close Neonode Workbench before running; it holds the device open.
"""

import ctypes
import sys
import time
from ctypes import wintypes

import hid

VENDOR_ID = 0x1536
PRODUCT_ID = 0x0101
VENDOR_USAGE_PAGE = 0xFF00

WRITE_REPORT_ID = 1
READ_REPORT_ID = 2
FEATURE_LENGTH = 257

SET_DETECTION_MODE = bytes.fromhex("EE1240020200670C8001FF810100820100830100")
ENABLE = bytes.fromhex("EE09400202006503810100")

TOUCH_TAG = 0x42
NOTIFICATION_FRAME = 0xF0
RESPONSE_FRAME = 0xEF

GENERIC_READ = 0x80000000
GENERIC_WRITE = 0x40000000
FILE_SHARE_RW = 3
OPEN_EXISTING = 3
INVALID_HANDLE = ctypes.c_void_p(-1).value

hid_dll = ctypes.WinDLL("hid")
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
# Declared explicitly: the default restype is a 32-bit int, which truncates
# a 64-bit handle and produces failures that look like device faults.
kernel32.CreateFileW.restype = ctypes.c_void_p


def open_handle():
    for d in hid.enumerate(VENDOR_ID, PRODUCT_ID):
        if d.get("usage_page") != VENDOR_USAGE_PAGE:
            continue
        path = d["path"]
        path = path.decode() if isinstance(path, bytes) else path
        handle = kernel32.CreateFileW(
            ctypes.c_wchar_p(path), wintypes.DWORD(GENERIC_READ | GENERIC_WRITE),
            wintypes.DWORD(FILE_SHARE_RW), None,
            wintypes.DWORD(OPEN_EXISTING), wintypes.DWORD(0), None)
        return None if handle == INVALID_HANDLE else handle
    return None


def send(handle, message: bytes) -> bool:
    buffer = ctypes.create_string_buffer(FEATURE_LENGTH)
    payload = bytes([WRITE_REPORT_ID, len(message)]) + message
    buffer[:len(payload)] = payload
    return bool(hid_dll.HidD_SetFeature(
        wintypes.HANDLE(handle), buffer, wintypes.ULONG(FEATURE_LENGTH)))


def receive(handle) -> bytes:
    """The ASN.1 message in feature report 2, or empty if there is none.

    The read fails outright when the sensor has nothing queued, which is
    ordinary rather than an error worth reporting.
    """
    buffer = ctypes.create_string_buffer(FEATURE_LENGTH)
    buffer[0] = bytes([READ_REPORT_ID])
    if not hid_dll.HidD_GetFeature(
            wintypes.HANDLE(handle), buffer, wintypes.ULONG(FEATURE_LENGTH)):
        return b""
    raw = buffer.raw
    return bytes(raw[2:2 + raw[1]]) if raw[1] else b""


def parse_touches(payload: bytes):
    """Every touch element in a notification frame.

    Each is a TLV: tag 0x42, a length, then id, event, x, y, sizes.
    Scanning for the tag rather than a fixed offset keeps this working
    whether the frame carries one touch or three, with or without the
    trailing timestamp.
    """
    touches = []
    i = 0
    while i < len(payload) - 1:
        if payload[i] != TOUCH_TAG:
            i += 1
            continue
        length = payload[i + 1]
        body = payload[i + 2:i + 2 + length]
        if len(body) < 6:
            break
        touches.append({
            "id": body[0],
            "event": {0: "down", 1: "move", 2: "up"}.get(body[1], str(body[1])),
            "x_mm": int.from_bytes(body[2:4], "big", signed=True) / 10.0,
            "y_mm": int.from_bytes(body[4:6], "big", signed=True) / 10.0,
        })
        i += 2 + length
    return touches


def setup_step(handle, label: str, message: bytes) -> None:
    print(f"{label}...")
    send(handle, message)
    time.sleep(0.15)
    print(f"  {receive(handle).hex(' ') or '(no response)'}")


def main() -> None:
    handle = open_handle()
    if handle is None:
        print("Could not open the vendor interface - sensor plugged in, "
              "Workbench closed?")
        return
    show_raw = "--raw" in sys.argv

    try:
        setup_step(handle, "Setting detection mode", SET_DETECTION_MODE)
        setup_step(handle, "Enabling touch notifications", ENABLE)

        print("\nSlide a finger across the sensor. Ctrl+C to stop.\n")
        # Polled rather than driven by the input report: the buffer holds
        # the last message, so the same one is read repeatedly until a new
        # one lands. The trailing timestamp differs between notifications,
        # so comparing whole payloads drops repeats without dropping a
        # finger that genuinely stayed still.
        previous = None
        while True:
            payload = receive(handle)
            if not payload or payload == previous:
                time.sleep(0.002)
                continue
            previous = payload
            if show_raw:
                print(payload.hex(" "))
            if payload[0] == NOTIFICATION_FRAME:
                for t in parse_touches(payload):
                    print(f"  id={t['id']} {t['event']:<4s} "
                          f"x={t['x_mm']:7.1f} mm  y={t['y_mm']:7.1f} mm")
            elif payload[0] == RESPONSE_FRAME:
                print(f"  response: {payload.hex(' ')}")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        kernel32.CloseHandle(wintypes.HANDLE(handle))


if __name__ == "__main__":
    main()