"""Turning raw strain-gauge volts into a normal force.

The ATI Nano17 does not report forces. It reports six bridge voltages, and
the conversion to three forces and three torques is a 6x6 matrix supplied by
ATI with that specific sensor, in a .cal file keyed to its serial number.
Using another sensor's matrix produces numbers that look entirely reasonable
and are wrong, which is why the matrix is loaded from file rather than typed
into the source.

Two further corrections sit on top of it:

  bias      The gauges have an offset that drifts with temperature, so the
            voltages at rest are not zero. Measured at the start of every
            session with nothing touching the sensor, and subtracted.

  mounting  The matrix yields forces in the sensor's own frame. What the
            experiment needs is the component normal to the touchscreen and
            the component along the direction of sliding. Unless the sensor
            happens to be mounted perfectly square to the screen, that is a
            rotation - and it has to be right, because the friction
            coefficient is a ratio of two of these components and inherits
            the error twice over.

Until the .cal file and the mounting geometry are known, identity stands in
for both, and the resulting "forces" are raw volts wearing a newton label.
ForceCalibration.is_placeholder marks that state so a session can refuse to
record forces it cannot actually interpret.
"""

import math
import random
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Tuple
from xml.etree import ElementTree

import numpy as np

AXES = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")


@dataclass(frozen=True)
class ForceCalibration:
    """Converts six gauge voltages into forces and torques.

    matrix    (6, 6) - rows are Fx, Fy, Fz, Tx, Ty, Tz; columns are gauges.
    bias      (6,)   - gauge voltages with no load, subtracted before the
                       matrix is applied.
    mounting  (3, 3) - rotation from the sensor frame into the screen frame,
                       where +z is normal to the glass and +x is the
                       direction of sliding.
    """

    matrix: np.ndarray
    bias: np.ndarray = field(default_factory=lambda: np.zeros(6))
    mounting: np.ndarray = field(default_factory=lambda: np.eye(3))
    force_units: str = "N"
    torque_units: str = "N-mm"
    serial: str = ""
    is_placeholder: bool = False

    def __post_init__(self) -> None:
        if self.matrix.shape != (6, 6):
            raise ValueError(
                f"calibration matrix must be 6x6, got {self.matrix.shape}")
        if self.bias.shape != (6,):
            raise ValueError(
                f"bias must have 6 entries, got {self.bias.shape}")
        if self.mounting.shape != (3, 3):
            raise ValueError(
                f"mounting rotation must be 3x3, got {self.mounting.shape}")

    # --- construction ------------------------------------------------------

    @classmethod
    def placeholder(cls) -> "ForceCalibration":
        """Identity everywhere: volts pass through unchanged.

        Lets the whole acquisition path be built and tested before the .cal
        file exists, while flagging that the numbers are not yet forces.
        """
        return cls(matrix=np.eye(6), is_placeholder=True)

    @classmethod
    def from_ati_cal_file(cls, path, serial: str = "") -> "ForceCalibration":
        """Load ATI's .cal file for one sensor.

        The file is XML; each of the six axes carries a row of six gauge
        coefficients and a scale factor. Parsed strictly - a file that does
        not have exactly six axes of six values each is more likely to be
        the wrong file than a file this code should be lenient about.
        """
        root = ElementTree.parse(Path(path)).getroot()
        calibration = root.find(".//Calibration")
        if calibration is None:
            raise ValueError(f"{path}: no <Calibration> element found")

        rows = []
        names = []
        for axis in calibration.findall("Axis"):
            name = axis.get("Name", "")
            values = [float(v) for v in (axis.get("values") or "").split()]
            if len(values) != 6:
                raise ValueError(
                    f"{path}: axis {name!r} has {len(values)} coefficients, "
                    "expected 6"
                )
            scale = float(axis.get("scale", 1.0))
            rows.append(np.asarray(values) / scale)
            names.append(name)

        if len(rows) != 6:
            raise ValueError(
                f"{path}: found {len(rows)} axes, expected 6 ({', '.join(AXES)})"
            )

        return cls(
            matrix=np.vstack(rows),
            force_units=calibration.get("ForceUnits", "N"),
            torque_units=calibration.get("TorqueUnits", "N-mm"),
            serial=serial or calibration.get("Serial", ""),
        )

    def with_bias(self, bias: Sequence[float]) -> "ForceCalibration":
        """A copy with the measured no-load offsets applied."""
        return ForceCalibration(
            matrix=self.matrix,
            bias=np.asarray(bias, dtype=float),
            mounting=self.mounting,
            force_units=self.force_units,
            torque_units=self.torque_units,
            serial=self.serial,
            is_placeholder=self.is_placeholder,
        )

    def with_mounting(self, rotation: np.ndarray) -> "ForceCalibration":
        """A copy with the sensor-to-screen rotation applied."""
        return ForceCalibration(
            matrix=self.matrix,
            bias=self.bias,
            mounting=np.asarray(rotation, dtype=float),
            force_units=self.force_units,
            torque_units=self.torque_units,
            serial=self.serial,
            is_placeholder=self.is_placeholder,
        )

    # --- application -------------------------------------------------------

    def wrench(self, gauge_volts: np.ndarray) -> np.ndarray:
        """Forces and torques from gauge voltages.

        Accepts a single reading of shape (6,) or a block of shape (6, n),
        and returns the same shape. Vectorised because the block form is
        what the acquisition thread actually has.
        """
        raw = np.asarray(gauge_volts, dtype=float)
        if raw.shape[0] != 6:
            raise ValueError(
                f"expected 6 gauge channels, got {raw.shape[0]}")
        if raw.ndim == 1:
            return self.matrix @ (raw - self.bias)
        return self.matrix @ (raw - self.bias[:, None])

    def forces_in_screen_frame(self, gauge_volts: np.ndarray) -> np.ndarray:
        """Force vector rotated into the screen frame: +z out of the glass."""
        wrench = self.wrench(gauge_volts)
        return self.mounting @ wrench[:3]

    def normal_force(self, gauge_volts: np.ndarray) -> np.ndarray:
        """Component pressing into the screen, positive for a push."""
        return self.forces_in_screen_frame(gauge_volts)[2]

    def tangential_force(self, gauge_volts: np.ndarray) -> np.ndarray:
        """Magnitude of the in-plane force, i.e. friction."""
        forces = self.forces_in_screen_frame(gauge_volts)
        return np.hypot(forces[0], forces[1])

    def describe(self) -> str:
        label = self.serial or "unknown sensor"
        if self.is_placeholder:
            return (f"{label}: PLACEHOLDER calibration - values are raw volts, "
                    "not forces")
        biased = "biased" if np.any(self.bias) else "unbiased"
        rotated = "rotated" if not np.allclose(self.mounting, np.eye(3)) \
            else "unrotated"
        return (f"{label}: {self.force_units}/{self.torque_units}, "
                f"{biased}, {rotated}")


class ForceSource(ABC):
    """Provides the normal force the participant is currently applying."""

    @abstractmethod
    def read_normal_force(self) -> Optional[float]:
        """Latest normal force in newtons, or None if nothing is in contact.

        Must not block: the UI calls this from a timer tick.
        """


class SimulatedForceSource(ForceSource):
    """A plausible force signal with no hardware attached.

    Drifts slowly around the target with a little noise, which is enough to
    exercise the feedback colouring and the per-trial force statistics. It
    does not pretend to model a fingertip; it exists so that the code paths
    downstream of the sensor can be built and tested before the sensor is.
    """

    def __init__(self, target_n: float = 1.0, drift_n: float = 0.35,
                 drift_period_s: float = 9.0, noise_n: float = 0.02,
                 seed: Optional[int] = None):
        self.target_n = target_n
        self.drift_n = drift_n
        self.drift_period_s = drift_period_s
        self.noise_n = noise_n
        self._rng = random.Random(seed)
        self._t0 = time.perf_counter()

    def read_normal_force(self) -> Optional[float]:
        elapsed = time.perf_counter() - self._t0
        drift = self.drift_n * math.sin(2 * math.pi * elapsed / self.drift_period_s)
        return self.target_n + drift + self._rng.gauss(0.0, self.noise_n)


class ManualForceSource(ForceSource):
    """Force driven by whatever the UI pushes in.

    Wired to the pointer's vertical position during development, so the
    force feedback can be steered deliberately rather than waited on: to see
    the "pressing too hard" state, press too hard.

    Thread-safe, because the value is written from the UI and may be read
    from the acquisition thread.
    """

    def __init__(self, initial_n: Optional[float] = None):
        self._lock = threading.Lock()
        self._value = initial_n

    def push(self, force_n: float) -> None:
        with self._lock:
            self._value = float(force_n)

    def clear(self) -> None:
        """No contact: subsequent reads report nothing rather than zero."""
        with self._lock:
            self._value = None

    def read_normal_force(self) -> Optional[float]:
        with self._lock:
            return self._value