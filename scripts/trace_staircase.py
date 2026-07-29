"""Print the expected staircase trajectory for a noiseless observer.

Produces a reference trace to check a manual GUI session against. The observer
is deterministic - always correct at or above the threshold, always wrong below -
so this exercises the staircase bookkeeping (step schedule, reversal detection,
threshold estimation) rather than any model of human performance.
"""

from pathlib import Path

import yaml

from evexp.psychophysics.staircase import StaircaseConfig, StaircaseController

TRUE_THRESHOLD_V = 60.0
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "experiment.yaml"


def main():
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    staircase = StaircaseController(StaircaseConfig(**cfg["staircase"]))

    answers = []
    print(f"{'trial':>5} {'shown V':>8} {'ans':>4} {'step':>5} {'rev':>4} "
          f"{'next V':>7} {'nrev':>5}")
    trial = 0
    while not staircase.finished and trial < 200:
        shown = staircase.value
        step = staircase.step_size
        correct = shown >= TRUE_THRESHOLD_V
        before = len(staircase.reversals)
        staircase.update(correct)
        reversal = len(staircase.reversals) > before
        answers.append("C" if correct else "W")
        print(f"{trial:>5} {shown:>8.1f} {answers[-1]:>4} {step:>5.1f} "
              f"{'yes' if reversal else '':>4} {staircase.value:>7.1f} "
              f"{len(staircase.reversals):>5}")
        trial += 1

    print(f"\nsequence:  {''.join(answers)}")
    print(f"reversals: {staircase.reversals}")
    print(f"threshold: {staircase.threshold_estimate:.2f} V "
          f"(true {TRUE_THRESHOLD_V} V)")


if __name__ == "__main__":
    main()