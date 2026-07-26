import time
from evexp.hardware.nidaq import NiDaqDevice

with NiDaqDevice(device_name="Dev1", channels=["ai0"]) as daq:
    for i in range(5):
        sample = daq.read()
        print(f"t={sample.timestamp:.3f}  values={sample.values}")
        time.sleep(0.2)