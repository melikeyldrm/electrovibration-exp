from pathlib import Path
from typing import List

import h5py
import numpy as np

from evexp.psychophysics.trial import TrialResult


class HDF5TrialWriter:
    """Writes a completed staircase run to HDF5, following the variable
    categories identified in the literature review (electrical, mechanical,
    finger mechanics, motion, psychophysical, metadata).

    Each category is stored as its own dataset under a per-session group,
    so future sensor streams (force, position, impedance) can be added
    without changing the schema for existing categories.
    """

    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        results: List[TrialResult],
        session_meta: dict,
        threshold_estimate: float,
    ) -> None:
        if not results:
            return

        trial_index = np.array([r.trial_index for r in results])
        applied_voltage = np.array([r.applied_voltage for r in results])
        stimulus_interval = np.array([r.stimulus_interval for r in results])
        response_interval = np.array([r.response_interval for r in results])
        correct = np.array([r.correct for r in results])
        timestamp = np.array([r.timestamp for r in results])

        with h5py.File(self.output_path, "w") as f:
            session = f.create_group("session")
            for key, value in session_meta.items():
                session.attrs[key] = value
            session.attrs["threshold_estimate_v"] = threshold_estimate

            electrical = f.create_group("electrical")
            electrical.create_dataset("applied_voltage", data=applied_voltage)

            psychophysical = f.create_group("psychophysical")
            psychophysical.create_dataset("trial_index", data=trial_index)
            psychophysical.create_dataset("stimulus_interval", data=stimulus_interval)
            psychophysical.create_dataset("response_interval", data=response_interval)
            psychophysical.create_dataset("correct", data=correct)

            metadata = f.create_group("metadata")
            metadata.create_dataset("timestamp", data=timestamp)