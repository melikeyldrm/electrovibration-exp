from evexp.acquisition.sync_logger import TrialLogger
from evexp.hardware.mock import MockDAQDevice
from evexp.psychophysics.staircase import StaircaseController, StaircaseConfig
from evexp.psychophysics.trial import TrialController



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
    print("\nLast 5 trials:")
    for r in results[-5:]:
        print(f"  trial={r.trial_index:3d}  voltage={r.applied_voltage:6.2f}  "
              f"correct={r.correct}")
        
    logger = TrialLogger("data/exp01_pilot_run.csv")
    logger.write(results)
    print(f"\nResults written to data/exp01_pilot_run.csv")


if __name__ == "__main__":
    main()