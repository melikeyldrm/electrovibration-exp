import csv
from pathlib import Path
from typing import List

from evexp.psychophysics.trial import TrialResult


class TrialLogger:
    """Writes trial results to a CSV file, one row per trial.

    CSV is used here for simplicity during development; the project will
    move to HDF5 once multi-sensor (force, position, voltage) streams
    need to be logged together (see data/hdf5_writer.py, planned for Day 3).
    """

    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, results: List[TrialResult]) -> None:
        if not results:
            return
        fieldnames = list(results[0].__dataclass_fields__.keys())
        with self.output_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(r.__dict__)