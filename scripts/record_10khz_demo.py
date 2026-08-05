"""Proof-of-concept: record N seconds at 10 kHz and verify the file.

Uses MockDAQDevice by default (runs anywhere, no hardware). Pass --real to
use NiDaqDevice against a real or NI MAX simulated card instead - same
script, same output format either way, since both implement DAQDevice.

This does not go through SensorAcquisition/the ring buffer; it calls
read_chunk() directly in a simple loop, to keep the demo minimal and easy
to read end to end.
"""

import argparse
import time
from pathlib import Path

import h5py
import numpy as np


def record(device, duration_s: float, sample_rate_hz: float,
           chunk_samples: int, output_path: Path) -> None:
    device.start(sample_rate_hz=sample_rate_hz)
    try:
        n_chunks = int(round(duration_s * sample_rate_hz / chunk_samples))
        blocks = []
        t0 = None
        t_read_start = time.perf_counter()
        for i in range(n_chunks):
            chunk = device.read_chunk(chunk_samples)
            if t0 is None:
                t0 = chunk.t0
            blocks.append(chunk.data)
        elapsed = time.perf_counter() - t_read_start
    finally:
        device.stop()

    data = np.hstack(blocks)  # (n_channels, total_samples)
    n_channels, n_samples = data.shape

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.attrs["sample_rate_hz"] = sample_rate_hz
        f.attrs["t0"] = t0
        f.attrs["channels"] = list(device.channels)
        f.create_dataset("raw", data=data)

    print(f"Recorded {n_samples} samples x {n_channels} channels "
          f"({n_samples / sample_rate_hz:.3f} s of data) in {elapsed:.3f} s wall time")
    print(f"Saved to {output_path}")

    # Read back and sanity-check, rather than trusting the write blindly.
    with h5py.File(output_path, "r") as f:
        reloaded = f["raw"][:]
        channels = list(f.attrs["channels"])
    assert reloaded.shape == data.shape, "shape mismatch on read-back"
    print(f"\nRead-back check OK. Per-channel stats:")
    for i, name in enumerate(channels):
        ch = reloaded[i]
        print(f"  {name:>8s}: mean={ch.mean():+.4f} V  std={ch.std():.4f} V  "
              f"min={ch.min():+.4f} V  max={ch.max():+.4f} V")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", action="store_true",
                         help="use NiDaqDevice instead of MockDAQDevice")
    parser.add_argument("--device-name", default="Dev1")
    parser.add_argument("--duration", type=float, default=2.0, help="seconds")
    parser.add_argument("--sample-rate", type=float, default=10000.0, help="Hz")
    parser.add_argument("--chunk-samples", type=int, default=500)
    parser.add_argument("--output", default="data/raw_10khz_demo.h5")
    args = parser.parse_args()

    if args.real:
        from evexp.hardware.nidaq import NiDaqDevice
        device = NiDaqDevice(device_name=args.device_name)
        print(f"Using NiDaqDevice({args.device_name!r}) - real or NI MAX simulated card")
    else:
        from evexp.hardware.mock import MockDAQDevice
        device = MockDAQDevice(realtime=True)
        print("Using MockDAQDevice (no hardware)")

    record(device, args.duration, args.sample_rate,
           args.chunk_samples, Path(args.output))


if __name__ == "__main__":
    main()