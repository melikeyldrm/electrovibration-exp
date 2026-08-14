"""Capture raw HID reports from the Neonode sensor.

Move your finger across the sensor and watch which bytes change.
Use this to figure out the report layout for _decode_report().

Steps:
    1) Fill in VENDOR_ID and PRODUCT_ID below.
    2) Close Neonode Workbench first (it may be holding the device).
    3) Run: python capture_neonode_report.py
    4) Slide your finger slowly across the sensor.
    5) Ctrl+C to stop, copy the output.
"""

import hid

# TODO: fill in real values from Workbench / Device Manager
VENDOR_ID = 0x0000
PRODUCT_ID = 0x0000

REPORT_SIZE = 64  # typical HID report size, adjust if needed


def main() -> None:
    device = hid.device()
    try:
        device.open(VENDOR_ID, PRODUCT_ID)
    except OSError as e:
        print(f"Failed to open (vid={VENDOR_ID:#06x}, pid={PRODUCT_ID:#06x}): {e}")
        print("Check Workbench is closed and VID/PID are correct.")
        return

    device.set_nonblocking(0)
    print(f"Connected: {device.get_manufacturer_string()} / {device.get_product_string()}")
    print("Move your finger across the sensor. Ctrl+C to stop.\n")

    try:
        while True:
            report = device.read(REPORT_SIZE)
            if report:
                print(report)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        device.close()


if __name__ == "__main__":
    main()