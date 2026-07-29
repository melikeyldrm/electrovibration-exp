import random

from evexp.hardware.mock import MockStimulusOutput
from evexp.psychophysics.staircase import StaircaseConfig, StaircaseController
from evexp.psychophysics.trial import SimulatedRunner, Trial2IFC, TrialState


def make_trial():
    cfg = StaircaseConfig(start_value=90, step_sizes=[8, 4, 2, 1],
                          n_reversals_to_stop=8, rule="1up2down",
                          min_value=0, max_value=150)
    return Trial2IFC(MockStimulusOutput(), StaircaseController(cfg),
                     rng=random.Random(0))


def test_state_sequence_follows_2ifc_structure():
    trial = make_trial()
    assert trial.state is TrialState.READY
    trial.start_trial()
    assert trial.state is TrialState.INTERVAL_1
    trial.advance()
    assert trial.state is TrialState.GAP
    trial.advance()
    assert trial.state is TrialState.INTERVAL_2
    assert trial.advance() is None
    assert trial.state is TrialState.AWAITING_RESPONSE


def test_stimulus_is_active_in_exactly_one_interval():
    trial = make_trial()
    trial.start_trial()
    active_in_1 = trial.stimulus.is_active
    trial.advance()
    trial.advance()
    active_in_2 = trial.stimulus.is_active
    trial.advance()
    assert active_in_1 != active_in_2
    assert not trial.stimulus.is_active


def test_response_is_scored_against_the_stimulus_interval():
    trial = make_trial()
    trial.start_trial()
    trial.advance(); trial.advance(); trial.advance()
    assert trial.submit_response(trial.stimulus_interval).correct

    trial.start_trial()
    trial.advance(); trial.advance(); trial.advance()
    assert not trial.submit_response(3 - trial.stimulus_interval).correct


def test_simulated_session_converges_near_true_threshold():
    trial = make_trial()
    runner = SimulatedRunner(trial, true_threshold=50.0, rng=random.Random(0))
    results = runner.run_until_done()
    assert len(results) > 10
    assert abs(trial.staircase.threshold_estimate - 50.0) < 15

def test_training_trial_does_not_update_staircase():
    trial = make_trial()
    voltage_before = trial.staircase.value
    reversals_before = len(trial.staircase.reversals)

    trial.start_trial(training=True)
    trial.advance(); trial.advance(); trial.advance()
    result = trial.submit_response(trial.stimulus_interval)

    assert result.training is True
    assert result.reversal is False
    assert trial.staircase.value == voltage_before
    assert len(trial.staircase.reversals) == reversals_before
    assert trial.trial_index == 0  # index artmamali


def test_training_trial_returns_to_ready_state():
    trial = make_trial()
    trial.start_trial(training=True)
    trial.advance(); trial.advance(); trial.advance()
    trial.submit_response(1)
    assert trial.state is TrialState.READY


def test_voltages_over_trials_excludes_training():
    trial = make_trial()

    # one training trial
    trial.start_trial(training=True)
    trial.advance(); trial.advance(); trial.advance()
    trial.submit_response(trial.stimulus_interval)

    # one real trial
    trial.start_trial(training=False)
    trial.advance(); trial.advance(); trial.advance()
    trial.submit_response(trial.stimulus_interval)

    voltages = trial.voltages_over_trials()
    assert len(voltages) == 1
    assert voltages[0] == 90.0  # staircase start value


def test_submit_response_raises_in_wrong_state():
    trial = make_trial()
    import pytest as pt
    with pt.raises(RuntimeError):
        trial.submit_response(1)  # READY state'de cagrilmamali


def test_invalid_response_raises():
    trial = make_trial()
    trial.start_trial()
    trial.advance(); trial.advance(); trial.advance()
    import pytest as pt
    with pt.raises(ValueError):
        trial.submit_response(3)  # sadece 1 veya 2 gecerli