import math
import random
from evexp.psychophysics.staircase import StaircaseController, StaircaseConfig


def simulate_observer(value, true_threshold, slope=0.15):
    p_detect = 1 / (1 + math.exp(-slope * (value - true_threshold)))
    p_correct = 0.5 + 0.5 * p_detect   # 2AFC guess tabanı %50
    return random.random() < p_correct


def test_1up2down_converges_near_threshold():
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
    assert abs(sc.threshold_estimate - true_threshold) < 15