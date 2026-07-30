"""Interactive 2IFC threshold session with a real participant."""

import shutil
import sys
from datetime import datetime
from pathlib import Path

import yaml
from PyQt5.QtWidgets import QApplication, QInputDialog

from evexp.data.csv_logger import CSVTrialLogger
from evexp.hardware.mock import MockStimulusOutput
from evexp.hardware.position import ManualPositionSource
from evexp.psychophysics.staircase import StaircaseConfig, StaircaseController
from evexp.psychophysics.trial import Trial2IFC, TrialTiming
from evexp.ui.experimenter_window import ExperimenterWindow
from evexp.ui.participant_window import ParticipantWindow
from evexp.ui.session_controller import SessionController

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "experiment.yaml"


def main():
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    session_cfg = cfg["session"]
    timing_cfg = cfg["timing"]

    app = QApplication(sys.argv)

    participant_id, ok = QInputDialog.getText(
        None, "Session setup", "Participant ID:"
    )
    if not ok or not participant_id.strip():
        print("No participant ID entered, exiting.")
        sys.exit(0)
    participant_id = participant_id.strip()

    staircase = StaircaseController(StaircaseConfig(**cfg["staircase"]))
    trial = Trial2IFC(
        stimulus=MockStimulusOutput(verbose=False),
        staircase=staircase,
        timing=TrialTiming(**timing_cfg),
        training_voltage=cfg.get("training", {}).get("voltage", 2.0),
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(session_cfg["output_dir"]) / (
        f"{session_cfg['experiment_id']}_{participant_id}_{stamp}.csv"
    )

    logger = CSVTrialLogger(str(output_path))
    print(f"Logging trials to {output_path}")

    snapshot_path = output_path.with_suffix(".yaml")
    shutil.copy(CONFIG_PATH, snapshot_path)
    print(f"Config snapshot saved to {snapshot_path}")

    reveal = cfg.get("debug", {}).get("reveal_stimulus", False)
    if reveal:
        print("WARNING: reveal_stimulus is enabled - data from this session "
              "is not a valid threshold measurement.")

    travel_mm = timing_cfg["cursor_travel_mm"]

    # Finger position. Today this is driven by the mouse over the cue track;
    # swapping in a NeonodePositionSource here is the only change needed once
    # the touch sensor works - the UI and speed readout are unaffected.
    position_source = ManualPositionSource(travel_mm=travel_mm)
    print("Position source: mouse (move the pointer along the cue track)")

    participant = ParticipantWindow(
        position_source=position_source,
        travel_mm=travel_mm,
    )
    experimenter = ExperimenterWindow(
        session_cfg["experiment_id"],
        participant_id,
        target_speed_mm_s=timing_cfg["cursor_speed_mm_s"],
    )

    controller = SessionController(
        trial, participant, experimenter, logger,
        reveal_stimulus=reveal,
        n_training=cfg.get("training", {}).get("n_trials", 0),
        position_source=position_source,
    )

    experimenter.show()
    participant.show()
    participant.activateWindow()
    participant.setFocus()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()