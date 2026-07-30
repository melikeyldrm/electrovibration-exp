import csv

from evexp.data.csv_logger import MinimalCSVTrialLogger
from evexp.psychophysics.trial import TrialResult


def test_minimal_logger_writes_only_three_columns(tmp_path):
    path = tmp_path / "out.csv"
    logger = MinimalCSVTrialLogger(str(path))

    logger.log(TrialResult(
        trial_index=0, applied_voltage=1.5, stimulus_interval=1,
        response_interval=1, correct=True, reversal=False,
        timestamp=0.0, response_time_s=1.0, training=False,
    ))
    logger.log(TrialResult(
        trial_index=0, applied_voltage=2.0, stimulus_interval=1,
        response_interval=2, correct=False, reversal=False,
        timestamp=0.0, response_time_s=1.0, training=True,  # should be skipped
    ))

    with path.open() as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1  # training row skipped
    assert set(rows[0].keys()) == {"trial_no", "applied_voltage", "correct"}
    assert rows[0]["trial_no"] == "0"
    assert rows[0]["applied_voltage"] == "1.5"
    assert rows[0]["correct"] == "True"