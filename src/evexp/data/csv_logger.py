"""Per-trial CSV logging.

Written incrementally (one row appended per trial) rather than all at once,
so progress survives if the session is interrupted.
"""

import csv
from dataclasses import asdict, fields
from pathlib import Path
from typing import List

from evexp.psychophysics.trial import TrialResult


class CSVTrialLogger:
    """Appends one row per completed trial to a CSV file.

    The file is reopened for each write so that a partially completed session
    is still readable if the experiment is interrupted - which matters when a
    participant needs to stop part way through.
    """

    FIELDNAMES = [f.name for f in fields(TrialResult)]

    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDNAMES).writeheader()

    def log(self, result: TrialResult) -> None:
        with self.output_path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDNAMES).writerow(asdict(result))

    def log_all(self, results: List[TrialResult]) -> None:
        for result in results:
            self.log(result)

class MinimalCSVTrialLogger:
    """Bare-bones CSV logger: trial number, voltage and correctness only.

    For a quick look at how a session is progressing without the full set
    of TrialResult fields that CSVTrialLogger writes.
    """

    FIELDNAMES = ["trial_no", "applied_voltage", "correct"]

    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDNAMES).writeheader()

    def log(self, result: TrialResult) -> None:
        if result.training:
            return  # training trials are not part of the numbered sequence
        row = {
            "trial_no": result.trial_index,
            "applied_voltage": result.applied_voltage,
            "correct": result.correct,
        }
        with self.output_path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDNAMES).writerow(row)

    def log_all(self, results: List[TrialResult]) -> None:
        for result in results:
            self.log(result)