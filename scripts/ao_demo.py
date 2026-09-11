"""AO (stimulus output) demo against a real or NI MAX simulated card.

Run with e.g.:
    python scripts/ao_demo.py --device-name Dev1
"""

import argparse
import time

from evexp.hardware.nidaq import NiDaqDevice
from evexp.hardware.safety import SafetyLimitExceeded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-name", default="Dev1")
    parser.add_argument("--ao-channel", default="ao0")
    parser.add_argument("--frequency", type=float, default=125.0, help="Hz")
    parser.add_argument("--voltage-limit", type=float, default=2.0, help="V")
    args = parser.parse_args()

    daq = NiDaqDevice(
        device_name=args.device_name,
        ao_channel=args.ao_channel,
        stimulus_frequency_hz=args.frequency,
        ao_voltage_limit_v=args.voltage_limit,
    )

    print(f"AO channel: {args.device_name}/{args.ao_channel}, "
          f"{args.frequency:g} Hz carrier, ±{args.voltage_limit} V limit")

    # --- normal path -------------------------------------------------------
    print("\n[1] set_amplitude(0.5) then stimulus_on()")
    daq.set_amplitude(0.5)
    daq.stimulus_on()
    print(f"    is_active = {daq.is_active}")
    time.sleep(2.0)

    print("[2] set_amplitude(1.2) while active (should rewrite the buffer)")
    daq.set_amplitude(1.2)
    time.sleep(2.0)

    print("[3] stimulus_off()")
    daq.stimulus_off()
    print(f"    is_active = {daq.is_active}")

    # --- safety path ---------------------------------------------------
    print(f"\n[4] set_amplitude({args.voltage_limit + 3.0}) then stimulus_on() "
          f"-> expect SafetyLimitExceeded, no write reaching the card")
    daq.set_amplitude(args.voltage_limit + 3.0)
    try:
        daq.stimulus_on()
        print("    UNEXPECTED: no exception raised - safety check did not fire")
    except SafetyLimitExceeded as exc:
        print(f"    OK, blocked as expected: {exc}")
    finally:
        print(f"    is_active = {daq.is_active}")

    print("\nDone.")


if __name__ == "__main__":
    main()