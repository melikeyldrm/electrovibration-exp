"""Capture raw HID reports from the Neonode sensor.

The sensor exposes two HID interfaces and they carry different things:

    usage_page 13 (digitizer)  standard HID touch reports - fixed layout
    usage_page 65280 (0xFF00)  vendor channel - the zForce message protocol

The digitizer interface is easier to decode but Windows reserves touch
collections for itself, so opening succeeds and the first read fails with
"read error". When that happens this falls back to the vendor channel
automatically. Pass a usage_page to force one:

    python scripts/capture_neonode_report.py
    python scripts/capture_neonode_report.py 65280

Close Neonode Workbench first - it holds the device open. Then slide a
finger slowly across the sensor and watch which bytes track the motion.
Ctrl+C to stop.
"""

import sys
import time

import hid

VENDOR_ID = 0x1536
PRODUCT_ID = 0x0101

DIGITIZER_USAGE_PAGE = 13
VENDOR_USAGE_PAGE = 0xFF00      # 65280

READ_TIMEOUT_MS = 1000


def find_interface(usage_page: int):
    """The device path for one of the sensor's interfaces, or None.

    Selected by path rather than by VID/PID: open(vid, pid) takes whichever
    interface it finds first, which is not the one we want half the time.
    """
    for d in hid.enumerate(VENDOR_ID, PRODUCT_ID):
        if d.get("usage_page") == usage_page:
            return d["path"]
    return None


def capture(usage_page: int, allow_fallback: bool) -> bool:
    """Listen on one interface. False if it could not be read at all."""
    path = find_interface(usage_page)
    if path is None:
        available = sorted({d.get("usage_page")
                            for d in hid.enumerate(VENDOR_ID, PRODUCT_ID)})
        print(f"No interface with usage_page {usage_page}. Available: {available}")
        return False

    device = hid.device()
    try:
        device.open_path(path)
    except OSError as e:
        print(f"Failed to open usage_page {usage_page}: {e}")
        print("Close Neonode Workbench and try again.")
        return False

    print(f"Connected: {device.get_manufacturer_string()} / "
          f"{device.get_product_string()} (usage_page {usage_page})")
    print("Slide a finger across the sensor. Ctrl+C to stop.\n")

    previous = None
    first_read = True
    try:
        while True:
            try:
                report = device.read(64, timeout_ms=READ_TIMEOUT_MS)
            except OSError as e:
                # Windows reserves system pointing devices: the open
                # succeeds, the read does not. Only meaningful on the first
                # read - later failures mean the device went away.
                if first_read and allow_fallback:
                    print(f"  read refused ({e}) - Windows is holding this "
                          "interface")
                    return False
                raise
            first_read = False
            if not report:
                continue
            hex_bytes = " ".join(f"{b:02x}" for b in report)
            # Marking what moved since the last report makes the coordinate
            # bytes stand out from the framing, which is the whole point.
            if previous is not None and len(previous) == len(report):
                changed = [i for i, (a, b) in enumerate(zip(previous, report))
                           if a != b]
                print(f"{hex_bytes}   changed: {changed}")
            else:
                print(hex_bytes)
            previous = report
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        device.close()
    return True


def main() -> None:
    if len(sys.argv) > 1:
        capture(int(sys.argv[1]), allow_fallback=False)
        return
    if not capture(DIGITIZER_USAGE_PAGE, allow_fallback=True):
        print(f"Falling back to the vendor channel "
              f"(usage_page {VENDOR_USAGE_PAGE}).\n")
        capture(VENDOR_USAGE_PAGE, allow_fallback=False)


if __name__ == "__main__":
    main()