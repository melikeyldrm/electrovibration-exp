"""Interactive 2IFC threshold session with a real participant.

Uses the same Trial2IFC state machine and staircase as the headless run in
run.py; only the response source differs - keyboard input instead of a
simulated observer.
"""

import sys
from datetime import datetime
from pathlib import Path

import yaml
from PyQt5.QtWidgets import QApplication

from evexp.data.csv_logger import CSVTrialLogger
from evexp.hardware.mock import MockStimulusOutput
from evexp.psychophysics.staircase import StaircaseConfig, StaircaseController
from evexp.psychophysics.trial import Trial2IFC, TrialTiming
from evexp.ui.experimenter_window import ExperimenterWindow
from evexp.ui.participant_window import ParticipantWindow
from evexp.ui.session_controller import SessionController

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "experiment.yaml"


def main():
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    session_cfg = cfg["session"]

    app = QApplication(sys.argv)

    staircase = StaircaseController(StaircaseConfig(**cfg["staircase"]))
    trial = Trial2IFC(
        # Swap this for the real amplifier implementation once hardware is
        # connected; nothing else in this file needs to change.
        stimulus=MockStimulusOutput(verbose=False),
        staircase=staircase,
        timing=TrialTiming(**cfg["timing"]),
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(session_cfg["output_dir"]) / (
        f"{session_cfg['experiment_id']}_{session_cfg['participant_id']}_{stamp}.csv"
    )
    logger = CSVTrialLogger(str(output_path))
    print(f"Logging trials to {output_path}")

    participant = ParticipantWindow()
    experimenter = ExperimenterWindow(
        session_cfg["experiment_id"], session_cfg["participant_id"]
    )
    reveal = cfg.get("debug", {}).get("reveal_stimulus", False)
    if reveal:
        print("WARNING: reveal_stimulus is enabled - data from this session "
              "is not a valid threshold measurement.")
    controller = SessionController(trial, participant, experimenter, logger,
                                   reveal_stimulus=reveal)
    experimenter.show()
    participant.show()
    participant.activateWindow()
    participant.setFocus()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()