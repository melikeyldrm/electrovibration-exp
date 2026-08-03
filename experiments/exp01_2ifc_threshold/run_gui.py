"""Interactive 2IFC threshold session with a real participant."""

import sys
from datetime import datetime
from pathlib import Path

import yaml
from PyQt5.QtWidgets import QApplication

from evexp.data.csv_logger import CSVTrialLogger
from evexp.hardware.acquisition import SensorAcquisition
from evexp.hardware.force import ManualForceSource, SimulatedForceSource
from evexp.hardware.mock import MockDAQDevice, MockStimulusOutput
from evexp.processing.force_feedback import ForceBands
from evexp.hardware.position import ManualPositionSource
from evexp.hardware.screen import ScreenCalibration
from evexp.psychophysics.staircase import StaircaseConfig, StaircaseController
from evexp.psychophysics.trial import Trial2IFC, TrialTiming
from evexp.ui.experimenter_window import ExperimenterWindow
from evexp.ui.participant_window import ParticipantWindow
from evexp.ui.session_controller import SessionController
from evexp.ui.setup_dialog import ask_for_setup

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "experiment.yaml"


def write_config_snapshot(cfg: dict, path: Path) -> None:
    """Record the configuration this session actually ran with.

    Deliberately a dump of the in-memory config rather than a copy of the
    file: choices made at runtime - the speed condition in particular - are
    not in the file, and a snapshot that disagrees with the session it
    documents is worse than no snapshot, because it will be believed.

    The cost is that the comments in experiment.yaml are not carried over.
    The snapshot exists for reproducibility, not for explanation, and the
    commented file stays in version control.
    """
    header = (
        "# Effective configuration for this session, including choices made\n"
        "# in the setup dialog. Generated automatically - do not edit.\n"
        f"# Written {datetime.now().isoformat(timespec='seconds')}\n"
    )
    path.write_text(header + yaml.safe_dump(cfg, sort_keys=False))


def main():
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    session_cfg = cfg["session"]
    timing_cfg = cfg["timing"]
    display_cfg = cfg["display"]

    app = QApplication(sys.argv)

    # Physical calibration of the participant display. Built before anything
    # is drawn, because the cue track geometry and the pacing speed both
    # depend on it, and a wrong calibration silently corrupts every recorded
    # speed rather than failing visibly.
    calibration = ScreenCalibration.from_config(display_cfg)
    print(f"Display calibration: {calibration.describe()}")
    for warning in calibration.warnings():
        print(f"WARNING: {warning}")

    setup = ask_for_setup(
        speed_options=timing_cfg["speed_options_mm_s"],
        default_speed_mm_s=timing_cfg["cursor_speed_mm_s"],
        experiment_id=session_cfg["experiment_id"],
    )
    if setup is None:
        print("Setup cancelled, exiting.")
        sys.exit(0)

    participant_id = setup.participant_id
    cue_speed_mm_s = setup.target_speed_mm_s
    # Fold the runtime choice back into the config so that everything
    # downstream - trial timing, the console readout, the snapshot - reads
    # the same single value.
    timing_cfg["cursor_speed_mm_s"] = cue_speed_mm_s
    session_cfg["participant_id"] = participant_id
    print(f"Participant {participant_id}, sliding speed {cue_speed_mm_s:g} mm/s")

    staircase = StaircaseController(StaircaseConfig(**cfg["staircase"]))
    trial = Trial2IFC(
        stimulus=MockStimulusOutput(verbose=False),
        staircase=staircase,
        timing=TrialTiming(
            pre_interval_wait_s=timing_cfg["pre_interval_wait_s"],
            interval_s=timing_cfg["interval_s"],
            gap_s=timing_cfg["gap_s"],
            cursor_speed_mm_s=timing_cfg["cursor_speed_mm_s"],
            cursor_travel_mm=timing_cfg["cursor_travel_mm"],
        ),
        training_voltage=cfg.get("training", {}).get("voltage", 2.0),
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(session_cfg["output_dir"]) / (
        f"{session_cfg['experiment_id']}_{participant_id}_{stamp}.csv"
    )

    logger = CSVTrialLogger(str(output_path))
    print(f"Logging trials to {output_path}")

    snapshot_path = output_path.with_suffix(".yaml")
    write_config_snapshot(cfg, snapshot_path)
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

    # Force feedback. No sensor yet - the "mouse_y" option lets the applied
    # force be driven deliberately during development (drag the mouse
    # vertically over the track) instead of only ever showing the on-target
    # colour. Swapping in a Nano17-backed source later is the only change
    # needed; the UI and the colour mapping are unaffected.
    force_cfg = cfg["force"]
    force_bands = ForceBands(
        target_n=force_cfg["target_n"],
        full_scale_n=force_cfg["full_scale_n"],
    )
    force_source_kind = force_cfg.get("source", "simulated")
    if force_source_kind == "mouse_y":
        force_source = ManualForceSource(initial_n=force_bands.target_n)
    elif force_source_kind == "simulated":
        force_source = SimulatedForceSource(target_n=force_bands.target_n)
    else:
        raise ValueError(
            f"unknown force.source {force_source_kind!r} in config; "
            "expected 'mouse_y' or 'simulated' (nano17 not wired in yet)"
        )
    print(f"Force source: {force_source_kind}")

    # Continuous sensor acquisition, on its own thread. Nothing displays it
    # yet; it runs from here so that the threading, the shutdown path and the
    # timing diagnostics are exercised in every session rather than only once
    # the real card arrives.
    acq_cfg = cfg["acquisition"]
    acquisition = SensorAcquisition(
        device=MockDAQDevice(realtime=True),
        position_source=position_source,
        ring_seconds=acq_cfg.get("ring_seconds", 30.0),
        chunk_samples=acq_cfg.get("chunk_samples") or None,
    )
    acquisition.start(sample_rate_hz=acq_cfg["sample_rate_hz"])
    print(f"Acquisition: {len(acquisition.channels)} channels at "
          f"{acquisition.sample_rate_hz:g} Hz (simulated)")

    # Stopping from aboutToQuit rather than after app.exec_() so that the
    # worker is joined on every exit path, including the window being closed
    # and the Escape confirmation on the console.
    def shutdown() -> None:
        acquisition.stop()
        print(acquisition.stats().describe())

    app.aboutToQuit.connect(shutdown)

    participant = ParticipantWindow(
        calibration=calibration,
        position_source=position_source,
        force_source=force_source,
        force_bands=force_bands,
        force_smoothing_samples=force_cfg.get("smoothing_samples", 5),
        travel_mm=travel_mm,
        cue_speed_mm_s=cue_speed_mm_s,
    )
    experimenter = ExperimenterWindow(
        session_cfg["experiment_id"],
        participant_id,
        target_speed_mm_s=cue_speed_mm_s,
    )

    controller = SessionController(
        trial, participant, experimenter, logger,
        reveal_stimulus=reveal,
        n_training=cfg.get("training", {}).get("n_trials", 0),
        position_source=position_source,
    )

    experimenter.show()
    if display_cfg.get("fullscreen", False):
        participant.showFullScreen()
    else:
        participant.show()
    participant.activateWindow()
    participant.setFocus()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()