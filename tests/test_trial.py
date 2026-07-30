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


def advance_to_awaiting_response(trial):
    """Drive the trial through every timed phase (PRE_INTERVAL_WAIT ->
    INTERVAL_1 -> GAP -> PRE_INTERVAL_WAIT -> INTERVAL_2) until it reaches
    the untimed AWAITING_RESPONSE phase. Robust to the exact phase count,
    so future state-machine changes don't silently break every test."""
    while trial.advance() is not None:
        pass


def test_state_sequence_follows_2ifc_structure():
    trial = make_trial()
    assert trial.state is TrialState.READY
    trial.start_trial()
    assert trial.state is TrialState.PRE_INTERVAL_WAIT
    trial.advance()
    assert trial.state is TrialState.INTERVAL_1
    trial.advance()
    assert trial.state is TrialState.GAP
    trial.advance()
    assert trial.state is TrialState.PRE_INTERVAL_WAIT
    trial.advance()
    assert trial.state is TrialState.INTERVAL_2
    assert trial.advance() is None
    assert trial.state is TrialState.AWAITING_RESPONSE


def test_stimulus_is_active_in_exactly_one_interval():
    trial = make_trial()
    trial.start_trial()
    trial.advance()  # PRE_INTERVAL_WAIT -> INTERVAL_1
    active_in_1 = trial.stimulus.is_active
    trial.advance()  # INTERVAL_1 -> GAP
    trial.advance()  # GAP -> PRE_INTERVAL_WAIT
    trial.advance()  # PRE_INTERVAL_WAIT -> INTERVAL_2
    active_in_2 = trial.stimulus.is_active
    trial.advance()  # INTERVAL_2 -> AWAITING_RESPONSE
    assert active_in_1 != active_in_2
    assert not trial.stimulus.is_active


def test_response_is_scored_against_the_stimulus_interval():
    trial = make_trial()
    trial.start_trial()
    advance_to_awaiting_response(trial)
    assert trial.submit_response(trial.stimulus_interval).correct

    trial.start_trial()
    advance_to_awaiting_response(trial)
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
    advance_to_awaiting_response(trial)
    result = trial.submit_response(trial.stimulus_interval)

    assert result.training is True
    assert result.reversal is False
    assert trial.staircase.value == voltage_before
    assert len(trial.staircase.reversals) == reversals_before
    assert trial.trial_index == 0  # index artmamali


def test_training_trial_returns_to_ready_state():
    trial = make_trial()
    trial.start_trial(training=True)
    advance_to_awaiting_response(trial)
    trial.submit_response(1)
    assert trial.state is TrialState.READY


def test_voltages_over_trials_excludes_training():
    trial = make_trial()

    # one training trial
    trial.start_trial(training=True)
    advance_to_awaiting_response(trial)
    trial.submit_response(trial.stimulus_interval)

    # one real trial
    trial.start_trial(training=False)
    advance_to_awaiting_response(trial)
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
    advance_to_awaiting_response(trial)
    import pytest as pt
    with pt.raises(ValueError):
        trial.submit_response(3)  # sadece 1 veya 2 gecerli


def test_pre_interval_wait_appears_before_each_interval():
    """PRE_INTERVAL_WAIT must occur exactly twice: before interval 1 and before interval 2."""
    staircase = StaircaseController(StaircaseConfig(
        start_value=1.0, step_sizes=[5.0, 1.0], n_reversals_to_stop=99,
        rule="3down1up", min_value=0.01, max_value=2.0, domain="db",
        consecutive_required=False,
    ))
    trial = Trial2IFC(MockStimulusOutput(), staircase, rng=random.Random(0))

    states_visited = []
    trial.start_trial(training=True)
    states_visited.append(trial.state)
    while trial.state is not TrialState.AWAITING_RESPONSE:
        trial.advance()
        states_visited.append(trial.state)

    wait_count = states_visited.count(TrialState.PRE_INTERVAL_WAIT)
    assert wait_count == 2, f"expected PRE_INTERVAL_WAIT twice, got {wait_count}"
    assert states_visited == [
        TrialState.PRE_INTERVAL_WAIT,
        TrialState.INTERVAL_1,
        TrialState.GAP,
        TrialState.PRE_INTERVAL_WAIT,
        TrialState.INTERVAL_2,
        TrialState.AWAITING_RESPONSE,
    ]

def make_production_trial():
    """Uses the actual protocol config (dB, 3-down/1-up, 2V ceiling) —
    catches integration issues that the linear-config tests above can't."""
    cfg = StaircaseConfig(
        start_value=2.0, step_sizes=[5.0, 1.0], n_reversals_to_stop=5,
        rule="3down1up", min_value=0.01, max_value=2.0,
        domain="db", consecutive_required=False,
    )
    return Trial2IFC(MockStimulusOutput(), StaircaseController(cfg),
                      rng=random.Random(0), training_voltage=2.0)


def test_full_session_runs_with_production_db_config():
    """Sanity check: the real protocol parameters actually produce a
    completed, sane session end-to-end (not just the toy linear config)."""
    trial = make_production_trial()
    runner = SimulatedRunner(trial, true_threshold=1.2, slope=15.0, rng=random.Random(1))
    results = runner.run_until_done()
    assert trial.staircase.finished
    assert all(0.01 <= r.applied_voltage <= 2.0 for r in results)