"""Run acquisition alone for a long stretch and log lag periodically.

No trial loop, no UI - just the same DAQ setup run_gui.py uses, left running
so a real (or NI MAX simulated) session's read-loop lag can be watched over
time without babysitting a full staircase session.

Run with e.g.:
    python scripts/watch_lag.py --minutes 30
    python scripts/watch_lag.py --mock --minutes 2       # no hardware needed
"""

import argparse
import time

from evexp.hardware.acquisition import SensorAcquisition
from evexp.hardware.multi_daq import MultiDaqDevice
from evexp.hardware.nidaq import NiDaqDevice, gauge_channel_map


def build_real_device(args) -> MultiDaqDevice:
    return MultiDaqDevice([
        NiDaqDevice(device_name=args.daq_fs1,
                    channel_map=gauge_channel_map("fs1"),
                    terminal_config="DIFF"),
        NiDaqDevice(device_name=args.daq_fs2,
                    channel_map=gauge_channel_map("fs2"),
                    terminal_config="DIFF"),
        NiDaqDevice(device_name=args.daq_stim,
                    channel_map={"current": args.monitor_channel},
                    terminal_config="RSE"),
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true",
                         help="Use MockDAQDevice instead of real/NI MAX cards.")
    parser.add_argument("--daq-fs1", default="Dev1")
    parser.add_argument("--daq-fs2", default="Dev3")
    parser.add_argument("--daq-stim", default="Dev2")
    parser.add_argument("--monitor-channel", default="ai0")
    parser.add_argument("--sample-rate", type=float, default=10000.0, help="Hz")
    parser.add_argument("--minutes", type=float, default=30.0)
    parser.add_argument("--interval-s", type=float, default=60.0,
                         help="How often to print stats.")
    args = parser.parse_args()

    if args.mock:
        from evexp.hardware.mock import MockDAQDevice
        device = MockDAQDevice(realtime=True)
        print("Device: MockDAQDevice - no hardware")
    else:
        device = build_real_device(args)
        print(f"Device: {args.daq_fs1} (FS1) + {args.daq_fs2} (FS2) + "
              f"{args.daq_stim} (monitor on {args.monitor_channel})")

    acquisition = SensorAcquisition(device)
    acquisition.start(sample_rate_hz=args.sample_rate)
    print(f"Started at {args.sample_rate:g} Hz. Watching for "
          f"{args.minutes:g} min, logging every {args.interval_s:g} s. "
          f"Ctrl+C to stop early.\n")

    deadline = time.perf_counter() + args.minutes * 60.0
    try:
        while time.perf_counter() < deadline:
            time.sleep(args.interval_s)
            elapsed_min = (time.perf_counter() - deadline + args.minutes * 60.0) / 60.0
            print(f"[{elapsed_min:5.1f} min] {acquisition.stats().describe()}")
    except KeyboardInterrupt:
        print("\nStopped early.")
    finally:
        print(f"\nFinal: {acquisition.stats().describe()}")
        acquisition.stop()


if __name__ == "__main__":
    main()
