"""Raw gauge volts -> normal force.

Nano17 reports 6 gauge voltages; the 6x6 matrix, bias, and mounting
rotation convert that to Fx/Fy/Fz/Tx/Ty/Tz. Real matrix comes from ATI's
.cal file. Until then, is_placeholder marks the numbers as not real forces.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

# Borrowed from another Nano17 ( Setup_FS1.5.py), not this sensor's
# real .cal. Swap for the real matrix when it arrives.
GAIN_FS1 = [
    [-0.00196, -0.06523, -0.07955, -1.66690, -0.03715, 1.57517],
    [0.07846, 1.91376, -0.04191, -0.99642, 0.02254, -0.85845],
    [1.86516, -0.02924, 1.96260, 0.05072, 1.78680, -0.00166],
    [1.04821, 11.73104, 10.65601, -5.85953, -9.94313, -5.24938],
    [-11.77058, 0.62323, 7.15594, 10.39180, 5.67526, -9.73148],
    [0.30751, 7.34154, 0.35701, 7.75355, -0.14530, 6.72782],
]


@dataclass(frozen=True)
class ForceCalibration:
    """matrix: (6,6) rows Fx..Tz, cols gauges. bias: (6,). mounting: (3,3)."""

    matrix: np.ndarray
    bias: np.ndarray = field(default_factory=lambda: np.zeros(6))
    mounting: np.ndarray = field(default_factory=lambda: np.eye(3))
    force_units: str = "N"
    torque_units: str = "N-mm"
    serial: str = ""
    is_placeholder: bool = False

    def __post_init__(self) -> None:
        if self.matrix.shape != (6, 6):
            raise ValueError(f"matrix must be 6x6, got {self.matrix.shape}")
        if self.bias.shape != (6,):
            raise ValueError(f"bias must have 6 entries, got {self.bias.shape}")
        if self.mounting.shape != (3, 3):
            raise ValueError(f"mounting must be 3x3, got {self.mounting.shape}")

    @classmethod
    def placeholder(cls) -> "ForceCalibration":
        return cls(matrix=np.eye(6), is_placeholder=True)

    def with_bias(self, bias: Sequence[float]) -> "ForceCalibration":
        return ForceCalibration(
            matrix=self.matrix, bias=np.asarray(bias, dtype=float),
            mounting=self.mounting, force_units=self.force_units,
            torque_units=self.torque_units, serial=self.serial,
            is_placeholder=self.is_placeholder,
        )

    def with_mounting(self, rotation: np.ndarray) -> "ForceCalibration":
        return ForceCalibration(
            matrix=self.matrix, bias=self.bias,
            mounting=np.asarray(rotation, dtype=float),
            force_units=self.force_units, torque_units=self.torque_units,
            serial=self.serial, is_placeholder=self.is_placeholder,
        )

    def wrench(self, gauge_volts: np.ndarray) -> np.ndarray:
        """(6,) or (6,n) gauge volts -> forces+torques, same shape."""
        raw = np.asarray(gauge_volts, dtype=float)
        if raw.shape[0] != 6:
            raise ValueError(f"expected 6 gauge channels, got {raw.shape[0]}")
        if raw.ndim == 1:
            return self.matrix @ (raw - self.bias)
        return self.matrix @ (raw - self.bias[:, None])

    def forces_in_screen_frame(self, gauge_volts: np.ndarray) -> np.ndarray:
        """Fx/Fy/Fz rotated into the screen frame (+z out of the glass)."""
        return self.mounting @ self.wrench(gauge_volts)[:3]

    def normal_force(self, gauge_volts: np.ndarray) -> np.ndarray:
        return self.forces_in_screen_frame(gauge_volts)[2]

    def tangential_force(self, gauge_volts: np.ndarray) -> np.ndarray:
        forces = self.forces_in_screen_frame(gauge_volts)
        return np.hypot(forces[0], forces[1])

    def describe(self) -> str:
        label = self.serial or "unknown sensor"
        if self.is_placeholder:
            return f"{label}: PLACEHOLDER calibration - not real forces"
        biased = "biased" if np.any(self.bias) else "unbiased"
        rotated = "rotated" if not np.allclose(self.mounting, np.eye(3)) \
            else "unrotated"
        return f"{label}: {self.force_units}/{self.torque_units}, {biased}, {rotated}"


class ForceSource(ABC):
    """Provides the participant's current normal force."""

    @abstractmethod
    def read_normal_force(self) -> Optional[float]:
        """Latest normal force in N, or None if no contact. Must not block."""


def measure_bias(acquisition, n_samples: int = 100,
                  poll_interval_s: float = 0.001,
                  gauge_names: Sequence[str] = (
                      "gauge0", "gauge1", "gauge2", "gauge3", "gauge4", "gauge5")
                  ) -> np.ndarray:
    """Average n_samples of gauge_names; call once at session start, no touch."""
    samples = []
    while len(samples) < n_samples:
        latest = acquisition.latest()
        if latest is not None:
            samples.append([latest[name] for name in gauge_names])
        time.sleep(poll_interval_s)
    return np.mean(samples, axis=0)


class AcquisitionForceSource(ForceSource):
    """Live normal force from SensorAcquisition.latest(), calibrated."""

    def __init__(self, acquisition, calibration: "ForceCalibration",
                 gauge_names: Sequence[str] = (
                     "gauge0", "gauge1", "gauge2", "gauge3", "gauge4", "gauge5")):
        self._acquisition = acquisition
        self._calibration = calibration
        self._gauge_names = tuple(gauge_names)

    def read_normal_force(self) -> Optional[float]:
        latest = self._acquisition.latest()
        if latest is None:
            return None
        try:
            gauge_volts = np.array([latest[name] for name in self._gauge_names])
        except KeyError:
            return None
        return float(self._calibration.normal_force(gauge_volts))