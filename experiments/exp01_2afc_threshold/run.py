"""Headless simulated run of the exp01 2AFC staircase.

Verifies that the trial state machine, staircase, and loggers work end to end
before any GUI or hardware is involved.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import yaml

from evexp.data.csv_logger import CSVTrialLogger
from evexp.hardware.mock import MockStimulusOutput
from evexp.psychophysics.staircase import StaircaseConfig, StaircaseController
from evexp.psychophysics.trial import SimulatedRunner, Trial2AFC, TrialTiming

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "experiment.yaml"


def main():
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    session_cfg, sim_cfg = cfg["session"], cfg["simulation"]
    out_dir = Path(session_cfg["output_dir"])

    timing_cfg = cfg["timing"]
    staircase = StaircaseController(StaircaseConfig(**cfg["staircase"]))
    trial = Trial2AFC(
        stimulus=MockStimulusOutput(),
        staircase=staircase,
        # Pick out only the fields TrialTiming takes: the timing block also
        # carries speed_options_mm_s (for the GUI setup dialog), which is not
        # a TrialTiming argument. Mirrors run_gui.py's explicit construction.
        timing=TrialTiming(
            pre_interval_wait_s=timing_cfg["pre_interval_wait_s"],
            gap_s_override=timing_cfg.get("gap_s_override"),
            cursor_speed_mm_s=timing_cfg["cursor_speed_mm_s"],
            cursor_travel_mm=timing_cfg["cursor_travel_mm"],
        ),
    )
    runner = SimulatedRunner(
        trial,
        true_threshold=sim_cfg["true_threshold_v"],
        slope=sim_cfg["slope"],
    )

    results = runner.run_until_done(max_trials=sim_cfg["max_trials"])

    print(f"Total trials:       {len(results)}")
    print(f"Reversals:          {len(staircase.reversals)}")
    print(f"Threshold estimate: {staircase.threshold_estimate:.1f} V "
          f"(true: {sim_cfg['true_threshold_v']} V)")

    stem = session_cfg["experiment_id"]
    CSVTrialLogger(str(out_dir / f"{stem}.csv")).log_all(results)

    plt.figure(figsize=(8, 4))
    plt.plot(trial.voltages_over_trials(), marker="o", markersize=3, linewidth=1)
    plt.axhline(sim_cfg["true_threshold_v"], color="red", linestyle="--",
                label=f"true threshold ({sim_cfg['true_threshold_v']} V)")
    plt.axhline(staircase.threshold_estimate, color="green", linestyle="--",
                label=f"estimate ({staircase.threshold_estimate:.1f} V)")
    plt.xlabel("Trial")
    plt.ylabel("Applied voltage (V)")
    plt.title(f"Staircase convergence ({cfg['staircase']['rule']}, simulated observer)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{stem}_convergence.png", dpi=150)
    print(f"Convergence plot saved to {out_dir / f'{stem}_convergence.png'}")


if __name__ == "__main__":
    main()