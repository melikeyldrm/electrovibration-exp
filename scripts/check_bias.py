"""Runs the staircase many times with different random seeds to check
whether the threshold estimate is systematically biased."""

import random
from pathlib import Path

import yaml

from evexp.hardware.mock import MockStimulusOutput
from evexp.psychophysics.staircase import StaircaseConfig, StaircaseController
from evexp.psychophysics.trial import SimulatedRunner, Trial2AFC, TrialTiming

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "experiment.yaml"

cfg = yaml.safe_load(CONFIG_PATH.read_text())
sim_cfg = cfg["simulation"]

biases = []
for seed in range(20):
    staircase = StaircaseController(StaircaseConfig(**cfg["staircase"]))
    trial = Trial2AFC(
        stimulus=MockStimulusOutput(),
        staircase=staircase,
        timing=TrialTiming(**cfg["timing"]),
        rng=random.Random(seed),
    )
    runner = SimulatedRunner(
        trial,
        true_threshold=sim_cfg["true_threshold_v"],
        slope=sim_cfg["slope"],
        rng=random.Random(seed + 1000),
    )
    runner.run_until_done(max_trials=sim_cfg["max_trials"])
    bias = staircase.threshold_estimate - sim_cfg["true_threshold_v"]
    biases.append(bias)
    print(f"seed={seed:2d}  estimate={staircase.threshold_estimate:.3f} V  bias={bias:+.3f} V")

mean_bias = sum(biases) / len(biases)
print(f"\nMean bias over {len(biases)} runs: {mean_bias:+.3f} V")
print(f"True threshold: {sim_cfg['true_threshold_v']} V")


print("\n--- Detailed debug for seed=3 ---")
staircase = StaircaseController(StaircaseConfig(**cfg["staircase"]))
trial = Trial2AFC(
    stimulus=MockStimulusOutput(),
    staircase=staircase,
    timing=TrialTiming(**cfg["timing"]),
    rng=random.Random(3),
)
runner = SimulatedRunner(
    trial,
    true_threshold=sim_cfg["true_threshold_v"],
    slope=sim_cfg["slope"],
    rng=random.Random(3 + 1000),
)
runner.run_until_done(max_trials=sim_cfg["max_trials"])

print(f"Total reversals: {len(staircase.reversals)}")
print(f"Reversal values: {[round(r, 3) for r in staircase.reversals]}")
print(f"Finished: {staircase.finished}, Aborted: {staircase.aborted}")
print(f"Trial count: {trial.trial_index}")
print(f"Threshold estimate: {staircase.threshold_estimate:.3f}")