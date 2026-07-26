import matplotlib.pyplot as plt

from evexp.hardware.mock import MockDAQDevice
from evexp.psychophysics.staircase import StaircaseController, StaircaseConfig
from evexp.psychophysics.trial import TrialController
from evexp.acquisition.sync_logger import TrialLogger
from evexp.data.hdf5_writer import HDF5TrialWriter


def main():
    device = MockDAQDevice(true_threshold=50.0)
    staircase_cfg = StaircaseConfig(
        start_value=90,
        step_sizes=[8, 4, 2, 1],
        n_reversals_to_stop=8,
        rule="1up2down",
        min_value=0,
        max_value=150,
    )
    staircase = StaircaseController(staircase_cfg)
    controller = TrialController(device, staircase)

    results = controller.run_until_done()

    print(f"Total trials: {len(results)}")
    print(f"Threshold estimate: {staircase.threshold_estimate:.1f} V "
          f"(true threshold: {device.true_threshold} V)")

    # CSV log (human-readable, quick inspection)
    TrialLogger("data/exp01_pilot_run.csv").write(results)

    # HDF5 log (structured, matches the Data to Record schema)
    session_meta = {
        "experiment_id": "exp01_waveform_pilot",
        "participant_id": "sim_participant_01",
        "staircase_rule": staircase_cfg.rule,
    }
    HDF5TrialWriter("data/exp01_pilot_run.h5").write(
        results, session_meta, staircase.threshold_estimate
    )

    # Convergence plot
    voltages = controller.voltages_over_trials()
    plt.figure(figsize=(8, 4))
    plt.plot(voltages, marker="o", markersize=3, linewidth=1)
    plt.axhline(device.true_threshold, color="red", linestyle="--",
                label=f"true threshold ({device.true_threshold} V)")
    plt.axhline(staircase.threshold_estimate, color="green", linestyle="--",
                label=f"estimated threshold ({staircase.threshold_estimate:.1f} V)")
    plt.xlabel("Trial")
    plt.ylabel("Applied voltage (V)")
    plt.title("Staircase convergence (1-up/2-down, simulated observer)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/exp01_convergence_plot.png", dpi=150)
    print("Convergence plot saved to data/exp01_convergence_plot.png")


if __name__ == "__main__":
    main()