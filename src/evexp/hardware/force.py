"""Raw gauge volts -> normal force.

The rig carries two ATI Nano17s under the same plate. Each reports 6 gauge
voltages and has its own 6x6 matrix and its own bias; the plate's total
force is the sum of the two, so both are calibrated separately and only
then added. Real matrices come from ATI's .cal files - until they arrive,
is_placeholder marks the numbers as not real forces.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

# Channel names for the two sensors, in gauge order. The acquisition device
# publishes all twelve on one clock; these names pick each sensor's six out.
FS1_GAUGE_CHANNELS: Tuple[str, ...] = tuple(f"fs1_gauge{i}" for i in range(6))
FS2_GAUGE_CHANNELS: Tuple[str, ...] = tuple(f"fs2_gauge{i}" for i in range(6))
ALL_GAUGE_CHANNELS: Tuple[str, ...] = FS1_GAUGE_CHANNELS + FS2_GAUGE_CHANNELS

# From Setup_FS1.5.py - the rig's two sensors, but not read from their .cal
# files. Swap for the real matrices when they arrive.
GAIN_FS1 = [
    [-0.00196, -0.06523, -0.07955, -1.66690, -0.03715, 1.57517],
    [0.07846, 1.91376, -0.04191, -0.99642, 0.02254, -0.85845],
    [1.86516, -0.02924, 1.96260, 0.05072, 1.78680, -0.00166],
    [1.04821, 11.73104, 10.65601, -5.85953, -9.94313, -5.24938],
    [-11.77058, 0.62323, 7.15594, 10.39180, 5.67526, -9.73148],
    [0.30751, 7.34154, 0.35701, 7.75355, -0.14530, 6.72782],
]
GAIN_FS2 = [
    [-0.00195, 0.01080, 0.06581, -1.65824, -0.09366, 1.64348],
    [-0.17825, 1.86454, 0.06494, -0.92514, 0.00966, -0.97718],
    [1.89526, -0.01688, 1.83664, -0.00175, 1.85168, -0.00273],
    [-1.14584, 11.34313, 10.51476, -5.61246, -10.06706, -5.95481],
    [-12.55627, 0.04926, 5.41259, 10.20378, 6.61359, -10.16940],
    [-0.76650, 7.03372, -0.09259, 7.02651, -0.52256, 7.07061],
]

# Setup_FS1.5.py negates Fy after summing the two sensors. Expressed here as
# a mounting rotation rather than a sign buried in the summing code, so the
# frame convention lives in one place. Replace with the measured mounting
# rotation once the sensor orientation relative to the screen is known.
FLIP_Y = np.diag([1.0, -1.0, 1.0]) # validate !!!


@dataclass(frozen=True)
class ForceCalibration:
    """One sensor. matrix: (6,6) rows Fx..Tz, cols gauges. bias: (6,). mounting: (3,3)."""

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


@dataclass(frozen=True)
class DualForceCalibration:
    """Both sensors. Gauge volts arrive stacked (12,) or (12,n): rows 0-5
    are FS1, rows 6-11 FS2, matching ALL_GAUGE_CHANNELS.

    Exposes the same forces/normal_force/tangential_force interface as a
    single ForceCalibration, so the writers and force sources do not need to
    know how many sensors are under the plate.
    """

    fs1: ForceCalibration
    fs2: ForceCalibration

    @classmethod
    def from_gain_matrices(cls, gain_fs1=GAIN_FS1, gain_fs2=GAIN_FS2,
                           mounting: np.ndarray = FLIP_Y,
                           is_placeholder: bool = True) -> "DualForceCalibration":
        def one(gain, serial):
            return ForceCalibration(
                matrix=np.asarray(gain, dtype=float),
                mounting=np.asarray(mounting, dtype=float),
                serial=serial, is_placeholder=is_placeholder,
            )
        # Builds two ForceCalibration objects from the gain matrices and 
        # wraps them in one DualForceCalibration
        return cls(fs1=one(gain_fs1, "FS1"), fs2=one(gain_fs2, "FS2"))

    @classmethod
    def placeholder(cls) -> "DualForceCalibration":
        return cls(fs1=ForceCalibration.placeholder(),
                   fs2=ForceCalibration.placeholder())

    @property
    def is_placeholder(self) -> bool:
        return self.fs1.is_placeholder or self.fs2.is_placeholder

    def with_bias(self, bias_fs1: Sequence[float],
                  bias_fs2: Sequence[float]) -> "DualForceCalibration":
        return DualForceCalibration(fs1=self.fs1.with_bias(bias_fs1),
                                    fs2=self.fs2.with_bias(bias_fs2))

    @staticmethod
    def split(gauge_volts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raw = np.asarray(gauge_volts, dtype=float)
        if raw.shape[0] != 12:
            raise ValueError(
                f"expected 12 gauge channels (6 per sensor), got {raw.shape[0]}")
        return raw[:6], raw[6:]

    def forces_per_sensor(self, gauge_volts: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray]:
        """Each sensor's Fx/Fy/Fz in the screen frame, kept separate."""
        volts1, volts2 = self.split(gauge_volts)
        return (self.fs1.forces_in_screen_frame(volts1),
                self.fs2.forces_in_screen_frame(volts2))

    def forces_in_screen_frame(self, gauge_volts: np.ndarray) -> np.ndarray:
        """Total Fx/Fy/Fz on the plate: the two sensors carry it in parallel."""
        forces1, forces2 = self.forces_per_sensor(gauge_volts)
        return forces1 + forces2

    def normal_force(self, gauge_volts: np.ndarray) -> np.ndarray:
        return self.forces_in_screen_frame(gauge_volts)[2]

    def tangential_force(self, gauge_volts: np.ndarray) -> np.ndarray:
        forces = self.forces_in_screen_frame(gauge_volts)
        return np.hypot(forces[0], forces[1])

    def describe(self) -> str:
        return f"two sensors - FS1 {self.fs1.describe()}; FS2 {self.fs2.describe()}"


class ForceSource(ABC):
    """Provides the participant's current normal force."""

    @abstractmethod
    def read_normal_force(self) -> Optional[float]:
        """Latest normal force in N, or None if no contact. Must not block."""


def measure_bias(acquisition, n_samples: int = 100,
                  poll_interval_s: float = 0.001,
                  gauge_names: Sequence[str] = ALL_GAUGE_CHANNELS
                  ) -> np.ndarray:
    """Average n_samples of gauge_names; call once at session start, no touch."""
    samples = []
    while len(samples) < n_samples:
        latest = acquisition.latest()
        if latest is not None:
            samples.append([latest[name] for name in gauge_names])
        time.sleep(poll_interval_s)
    return np.mean(samples, axis=0)


def measure_dual_bias(acquisition, n_samples: int = 100,
                      poll_interval_s: float = 0.001
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Both sensors' bias from one pass, so they share the same rest period."""
    volts = measure_bias(acquisition, n_samples, poll_interval_s,
                         ALL_GAUGE_CHANNELS)
    return volts[:6], volts[6:]


class AcquisitionForceSource(ForceSource):
    """Live normal force from SensorAcquisition.latest(), calibrated."""

    def __init__(self, acquisition, calibration,
                 gauge_names: Sequence[str] = ALL_GAUGE_CHANNELS):
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