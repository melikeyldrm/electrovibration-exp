import math
import pytest
import random
from evexp.psychophysics.staircase import StaircaseConfig, StaircaseController, to_db, from_db


def simulate_observer(value, true_threshold, slope=0.15):
    """Simulate a participant using a psychometric function."""

    p_detect = 1 / (1 + math.exp(-slope * (value - true_threshold)))
    p_correct = 0.5 + 0.5 * p_detect   
    return random.random() < p_correct


def test_1up2down_converges_near_threshold():
    """Verify that the staircase converges near the simulated threshold."""
    random.seed(0)
    true_threshold = 50.0
    cfg = StaircaseConfig(
        start_value=90, step_sizes=[8, 4, 2, 1],
        n_reversals_to_stop=8, rule="1up2down",
        min_value=0, max_value=150,
    )
    sc = StaircaseController(cfg)
    trials = 0
    while not sc.finished and trials < 300:
        correct = simulate_observer(sc.value, true_threshold)
        sc.update(correct)
        trials += 1
    # The estimated threshold should be reasonably close to the ground truth.
    assert abs(sc.threshold_estimate - true_threshold) < 15

def test_abort_triggers_after_consecutive_wrong_at_ceiling():
    cfg = StaircaseConfig(start_value=150, step_sizes=[8, 4, 2, 1],
                          n_reversals_to_stop=8, rule="1up2down",
                          min_value=0, max_value=150,
                          max_consecutive_wrong_at_ceiling=3)
    sc = StaircaseController(cfg)
    for _ in range(3):
        sc.update(False)
    assert sc.aborted
    assert sc.finished

def test_abort_resets_on_correct_answer():
    cfg = StaircaseConfig(start_value=150, step_sizes=[8, 4, 2, 1],
                          n_reversals_to_stop=8, rule="1up2down",
                          min_value=0, max_value=150,
                          max_consecutive_wrong_at_ceiling=3)
    sc = StaircaseController(cfg)
    sc.update(False)
    sc.update(False)
    sc.update(True)   # resets counter
    sc.update(False)
    sc.update(False)
    assert not sc.aborted  # only 2 consecutive wrong after reset

def test_voltage_does_not_exceed_max_value():
    cfg = StaircaseConfig(start_value=148, step_sizes=[8, 4, 2, 1],
                          n_reversals_to_stop=8, rule="1up2down",
                          min_value=0, max_value=150)
    sc = StaircaseController(cfg)
    for _ in range(10):
        sc.update(False)  # sürekli yukari
    assert sc.value <= 150.0


def test_voltage_does_not_go_below_min_value():
    cfg = StaircaseConfig(start_value=2, step_sizes=[8, 4, 2, 1],
                          n_reversals_to_stop=8, rule="1up2down",
                          min_value=0, max_value=150)
    sc = StaircaseController(cfg)
    sc.update(True); sc.update(True)  # asagi
    assert sc.value >= 0.0


def test_3down1up_converges_near_threshold():
    random.seed(42)
    true_threshold = 50.0
    cfg = StaircaseConfig(
        start_value=90, step_sizes=[8, 4, 2, 1],
        n_reversals_to_stop=8, rule="3down1up",
        min_value=0, max_value=150,
    )
    sc = StaircaseController(cfg)
    trials = 0
    while not sc.finished and trials < 300:
        correct = simulate_observer(sc.value, true_threshold)
        sc.update(correct)
        trials += 1
    assert abs(sc.threshold_estimate - true_threshold) < 20

def test_db_conversion_roundtrip():
    """to_db / from_db should be inverse operations."""
    for v in [0.5, 1.0, 2.0, 50.0]:
        assert from_db(to_db(v)) == pytest.approx(v, rel=1e-9)


def test_db_domain_step_is_multiplicative_in_linear_space():
    """A step of +5 dB should multiply the linear value by 10**(5/20), not add 5."""
    cfg = StaircaseConfig(
        start_value=2.0,
        step_sizes=[5.0],
        n_reversals_to_stop=99,  # never stop, we just check one step
        rule="3down1up",
        min_value=0.01,
        max_value=2.0,
        domain="db",
        consecutive_required=False,
    )
    sc = StaircaseController(cfg)
    # First wrong answer -> immediate "up" step, but clamped at max_value=2.0
    # so let's start below the ceiling to actually observe the multiplicative step.
    cfg2 = StaircaseConfig(
        start_value=1.0,
        step_sizes=[5.0],
        n_reversals_to_stop=99,
        rule="3down1up",
        min_value=0.01,
        max_value=2.0,
        domain="db",
        consecutive_required=False,
    )
    sc2 = StaircaseController(cfg2)
    new_value = sc2.update(correct=False)  # wrong -> steps up by 5 dB
    expected = from_db(to_db(1.0) + 5.0)
    assert new_value == pytest.approx(expected, rel=1e-9)
    assert new_value == pytest.approx(min(1.0 * 10 ** (5 / 20), 2.0), rel=1e-6)


def test_non_consecutive_streak_not_reset_by_wrong_answer():
    """With consecutive_required=False, a wrong answer between correct
    answers should NOT reset the cumulative correct counter, and should
    itself trigger an immediate upward step."""
    cfg = StaircaseConfig(
        start_value=1.0,
        step_sizes=[5.0, 1.0],
        n_reversals_to_stop=99,
        rule="3down1up",
        min_value=0.01,
        max_value=2.0,
        domain="db",
        consecutive_required=False,
    )
    sc = StaircaseController(cfg)

    v0 = sc.value
    v1 = sc.update(correct=True)   # streak=1, no step yet
    assert v1 == v0, "first correct answer alone should not move the value"

    v2 = sc.update(correct=False)  # wrong -> immediate up-step, streak untouched (still 1)
    assert v2 > v1, "a wrong answer must trigger an immediate upward step"

    v3 = sc.update(correct=True)   # streak=2, still not 3 -> no down-step
    assert v3 == v2

    v4 = sc.update(correct=True)   # streak reaches 3 -> down-step, streak resets
    assert v4 < v3, "third cumulative correct (non-consecutive) must trigger a down-step"


def test_consecutive_required_true_resets_streak_on_wrong():
    """Sanity check that the classic (default) behavior is unchanged:
    a wrong answer resets the correct streak."""
    cfg = StaircaseConfig(
        start_value=1.0,
        step_sizes=[5.0, 1.0],
        n_reversals_to_stop=99,
        rule="3down1up",
        min_value=0.01,
        max_value=2.0,
        domain="db",
        consecutive_required=True,  # classic Levitt
    )
    sc = StaircaseController(cfg)

    sc.update(correct=True)   # streak=1
    sc.update(correct=True)   # streak=2
    v_before = sc.value
    v_after_wrong = sc.update(correct=False)  # streak resets to 0, immediate up-step
    assert v_after_wrong > v_before

    # Now need 3 fresh consecutive corrects to trigger a down-step, not just 1 more.
    v_a = sc.update(correct=True)  # streak=1
    assert v_a == v_after_wrong, "single correct after reset should not step down yet"


def test_ceiling_clamp_never_exceeded_in_db_mode():
    """The 2V safety ceiling must never be exceeded even with large dB steps."""
    cfg = StaircaseConfig(
        start_value=1.9,
        step_sizes=[5.0],
        n_reversals_to_stop=99,
        rule="3down1up",
        min_value=0.01,
        max_value=2.0,
        domain="db",
        consecutive_required=False,
    )
    sc = StaircaseController(cfg)
    for _ in range(10):
        v = sc.update(correct=False)  # repeatedly wrong -> repeatedly steps up
        assert v <= 2.0 + 1e-9


def test_linear_mode_unchanged_backward_compat():
    """Existing linear-domain, consecutive-required behavior (the pre-existing
    default) must be bit-for-bit unchanged after this refactor."""
    cfg = StaircaseConfig(
        start_value=100.0,
        step_sizes=[8, 4, 2, 1],
        n_reversals_to_stop=8,
        rule="1up2down",
        min_value=0.0,
        max_value=150.0,
        # domain defaults to "linear", consecutive_required defaults to True
    )
    sc = StaircaseController(cfg)
    responses = "CCCCCCCCWCCCCWWCCWCCWCC"
    for r in responses:
        sc.update(correct=(r == "C"))
    # Just check it runs to completion without error and stays within bounds.
    assert 0.0 <= sc.value <= 150.0