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

    The file is reopened for each write so a partially completed session
    stays readable if the experiment is interrupted.
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