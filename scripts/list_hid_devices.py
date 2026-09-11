"""List every HID device Windows exposes, so the Neonode can be identified.

Run it twice - once with the sensor unplugged, once plugged in - and
compare. Close Neonode Workbench first; it may hold the device open.
"""

import hid


def main() -> None:
    devices = hid.enumerate()
    if not devices:
        print("No HID devices found at all - is the hid package installed "
              "with its backend DLL?")
        return

    print(f"{len(devices)} HID interface(s):\n")
    for d in sorted(devices, key=lambda x: (x["vendor_id"], x["product_id"])):
        vid, pid = d["vendor_id"], d["product_id"]
        maker = (d.get("manufacturer_string") or "").strip()
        product = (d.get("product_string") or "").strip()
        label = " / ".join(p for p in (maker, product) if p) or "(no strings)"
        print(f"  VID={vid:#06x} PID={pid:#06x}  {label}")
        print(f"      usage_page={d.get('usage_page')} "
              f"usage={d.get('usage')} "
              f"interface={d.get('interface_number')}")

    print("\nLook for anything mentioning Neonode or zForce. If nothing "
          "matches, plug/unplug and diff the two runs.")


if __name__ == "__main__":
    main()