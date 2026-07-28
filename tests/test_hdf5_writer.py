import h5py

from evexp.data.hdf5_writer import HDF5TrialWriter
from evexp.hardware.mock import MockStimulusOutput
from evexp.psychophysics.staircase import StaircaseConfig, StaircaseController
from evexp.psychophysics.trial import SimulatedRunner, Trial2IFC


def test_hdf5_writer_creates_expected_groups(tmp_path):
    cfg = StaircaseConfig(start_value=90, step_sizes=[8, 4, 2, 1],
                          n_reversals_to_stop=4, rule="1up2down",
                          min_value=0, max_value=150)
    staircase = StaircaseController(cfg)
    trial = Trial2IFC(MockStimulusOutput(), staircase)
    results = SimulatedRunner(trial, true_threshold=50.0).run_until_done(max_trials=100)

    output_path = tmp_path / "test_run.h5"
    HDF5TrialWriter(str(output_path)).write(
        results, {"experiment_id": "test"}, staircase.threshold_estimate
    )

    with h5py.File(output_path, "r") as f:
        assert "electrical" in f
        assert "psychophysical" in f
        assert "metadata" in f
        assert len(f["electrical"]["applied_voltage"]) == len(results)