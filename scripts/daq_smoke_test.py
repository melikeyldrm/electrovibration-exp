import nidaqmx
from nidaqmx.system import System

system = System.local()
print("devices found:")
for device in system.devices:
    print(f"  {device.name}  —  {device.product_type}  —  simulated: {device.is_simulated}")

with nidaqmx.Task() as task:
    task.ai_channels.add_ai_voltage_chan("Dev1/ai0")
    sample = task.read()
    print(f"\nDev1/ai0 okundu: {sample:.4f} V")