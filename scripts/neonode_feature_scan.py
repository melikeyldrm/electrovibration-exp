"""Work out how the Neonode's feature-report pipe wants to be talked to.

Writing to the vendor interface succeeds but reading fails, and there are
two plausible reasons: Windows rejects a feature-report buffer whose length
does not match the descriptor exactly, or the sensor errors when asked for
data it does not yet have. This tries both, plus every report id in the
descriptor, and reports what actually worked.

Close Neonode Workbench before running.
"""

import time

import hid

VENDOR_ID = 0x1536
PRODUCT_ID = 0x0101
VENDOR_USAGE_PAGE = 0xFF00

SET_DETECTION_MODE = bytes.fromhex("EE1240020200670C8001FF810100820100830100")

# Every report id the HID descriptor declares for this collection.
REPORT_IDS = (1, 2, 4, 128, 130)


def open_vendor_interface():
    for d in hid.enumerate(VENDOR_ID, PRODUCT_ID):
        if d.get("usage_page") == VENDOR_USAGE_PAGE:
            print(f"interface: {d}")
            device = hid.device()
            device.open_path(d["path"])
            return device
    return None


def scan_lengths(device, report_id: int):
    """Which buffer lengths Windows accepts for this report."""
    working = []
    for length in range(2, 100):
        try:
            raw = device.get_feature_report(report_id, length)
        except OSError:
            continue
        if raw:
            working.append((length, bytes(raw)))
    return working


def main() -> None:
    device = open_vendor_interface()
    if device is None:
        print("Vendor interface not found.")
        return

    try:
        print("\n--- reading each report id at every buffer length ---")
        for report_id in REPORT_IDS:
            hits = scan_lengths(device, report_id)
            if not hits:
                print(f"  report {report_id}: no length accepted")
                continue
            lengths = [n for n, _ in hits]
            print(f"  report {report_id}: accepted lengths {lengths}")
            print(f"      first result: {hits[0][1].hex(' ')}")

        print("\n--- write, then wait for the input report ---")
        report = bytearray([1, len(SET_DETECTION_MODE)]) + SET_DETECTION_MODE
        report += bytes(66 - len(report))
        written = device.send_feature_report(bytes(report))
        print(f"  send_feature_report returned {written}")

        # The guide says an input report signals that a feature report is
        # ready to be read. If one arrives, the pipe works and the earlier
        # failures were just reads with nothing to fetch.
        device.set_nonblocking(0)
        for attempt in range(3):
            data = device.read(64, timeout_ms=500)
            print(f"  input report attempt {attempt + 1}: "
                  f"{bytes(data).hex(' ') if data else '(nothing)'}")
            if data:
                break

        print("\n--- reading report 2 again after the write ---")
        for report_id in (1, 2):
            hits = scan_lengths(device, report_id)
            if hits:
                print(f"  report {report_id}: {hits[0][1].hex(' ')}")
            else:
                print(f"  report {report_id}: still unreadable")
    finally:
        device.close()


if __name__ == "__main__":
    main()