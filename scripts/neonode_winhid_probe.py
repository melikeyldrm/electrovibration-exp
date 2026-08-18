"""Talk to the Neonode through the Windows HID API directly.

hidapi writes to this device successfully but every read fails, which
points at the wrapper rather than the sensor. This goes straight to
HidD_GetFeature/HidD_SetFeature via ctypes, and first asks the driver what
the real report lengths are - hidapi never exposes those, so until now the
buffer sizes have been guesswork.

Read-only apart from the same detection-mode message Workbench sends.
Close Neonode Workbench before running.
"""

import ctypes
from ctypes import wintypes

import hid

VENDOR_ID = 0x1536
PRODUCT_ID = 0x0101
VENDOR_USAGE_PAGE = 0xFF00

SET_DETECTION_MODE = bytes.fromhex("EE1240020200670C8001FF810100820100830100")

GENERIC_READ = 0x80000000
GENERIC_WRITE = 0x40000000
FILE_SHARE_READ = 0x00000001
FILE_SHARE_WRITE = 0x00000002
OPEN_EXISTING = 3
INVALID_HANDLE = ctypes.c_void_p(-1).value

hid_dll = ctypes.WinDLL("hid")
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)


class HIDP_CAPS(ctypes.Structure):
    _fields_ = [
        ("Usage", ctypes.c_ushort),
        ("UsagePage", ctypes.c_ushort),
        ("InputReportByteLength", ctypes.c_ushort),
        ("OutputReportByteLength", ctypes.c_ushort),
        ("FeatureReportByteLength", ctypes.c_ushort),
        ("Reserved", ctypes.c_ushort * 17),
        ("NumberLinkCollectionNodes", ctypes.c_ushort),
        ("NumberInputButtonCaps", ctypes.c_ushort),
        ("NumberInputValueCaps", ctypes.c_ushort),
        ("NumberInputDataIndices", ctypes.c_ushort),
        ("NumberOutputButtonCaps", ctypes.c_ushort),
        ("NumberOutputValueCaps", ctypes.c_ushort),
        ("NumberOutputDataIndices", ctypes.c_ushort),
        ("NumberFeatureButtonCaps", ctypes.c_ushort),
        ("NumberFeatureValueCaps", ctypes.c_ushort),
        ("NumberFeatureDataIndices", ctypes.c_ushort),
    ]


def vendor_path() -> str:
    for d in hid.enumerate(VENDOR_ID, PRODUCT_ID):
        if d.get("usage_page") == VENDOR_USAGE_PAGE:
            path = d["path"]
            return path.decode() if isinstance(path, bytes) else path
    return ""


def open_handle(path: str, access: int):
    handle = kernel32.CreateFileW(
        ctypes.c_wchar_p(path), wintypes.DWORD(access),
        wintypes.DWORD(FILE_SHARE_READ | FILE_SHARE_WRITE),
        None, wintypes.DWORD(OPEN_EXISTING), wintypes.DWORD(0), None)
    if handle == INVALID_HANDLE:
        return None, ctypes.get_last_error()
    return handle, 0


def caps_for(handle):
    preparsed = ctypes.c_void_p()
    if not hid_dll.HidD_GetPreparsedData(
            wintypes.HANDLE(handle), ctypes.byref(preparsed)):
        return None
    caps = HIDP_CAPS()
    status = hid_dll.HidP_GetCaps(preparsed, ctypes.byref(caps))
    hid_dll.HidD_FreePreparsedData(preparsed)
    return caps if status == 0x00110000 else caps


def get_feature(handle, report_id: int, length: int):
    buffer = ctypes.create_string_buffer(length)
    buffer[0] = bytes([report_id])
    ok = hid_dll.HidD_GetFeature(
        wintypes.HANDLE(handle), buffer, wintypes.ULONG(length))
    if not ok:
        return None, ctypes.get_last_error()
    return buffer.raw, 0


def set_feature(handle, report_id: int, message: bytes, length: int):
    buffer = ctypes.create_string_buffer(length)
    payload = bytes([report_id, len(message)]) + message
    buffer[:len(payload)] = payload
    ok = hid_dll.HidD_SetFeature(
        wintypes.HANDLE(handle), buffer, wintypes.ULONG(length))
    return bool(ok), (0 if ok else ctypes.get_last_error())


def main() -> None:
    path = vendor_path()
    if not path:
        print("Vendor interface not found.")
        return
    print(f"path: {path}\n")

    # Read/write first; if the driver refuses that, read-only still allows
    # HidD_GetFeature and would explain hidapi's asymmetric behaviour.
    for label, access in (("read+write", GENERIC_READ | GENERIC_WRITE),
                          ("read only", GENERIC_READ),
                          ("no access flags", 0)):
        handle, error = open_handle(path, access)
        if handle is None:
            print(f"open {label}: FAILED (win32 error {error})")
            continue
        print(f"open {label}: ok")

        caps = caps_for(handle)
        if caps is not None:
            print(f"  usage_page={caps.UsagePage:#06x} usage={caps.Usage}")
            print(f"  input report length   = {caps.InputReportByteLength}")
            print(f"  output report length  = {caps.OutputReportByteLength}")
            print(f"  feature report length = {caps.FeatureReportByteLength}")
            length = caps.FeatureReportByteLength or 66
        else:
            print("  HidD_GetPreparsedData failed")
            length = 66

        for report_id in (1, 2):
            data, error = get_feature(handle, report_id, length)
            if data is None:
                print(f"  get feature {report_id}: failed (win32 {error})")
            else:
                print(f"  get feature {report_id}: {data.hex(' ')}")

        if access & GENERIC_WRITE:
            ok, error = set_feature(handle, 1, SET_DETECTION_MODE, length)
            print(f"  set feature 1: {'ok' if ok else f'failed (win32 {error})'}")
            data, error = get_feature(handle, 2, length)
            if data is None:
                print(f"  get feature 2 after write: failed (win32 {error})")
            else:
                print(f"  get feature 2 after write: {data.hex(' ')}")

        kernel32.CloseHandle(wintypes.HANDLE(handle))
        print()


if __name__ == "__main__":
    main()