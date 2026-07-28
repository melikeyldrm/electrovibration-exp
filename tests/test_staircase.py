import math
import random
from evexp.psychophysics.staircase import StaircaseController, StaircaseConfig


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