"""Interactive 2IFC threshold session with a real participant."""

import argparse
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml
from PyQt5.QtWidgets import QApplication

from evexp.data.csv_logger import CSVTrialLogger
from evexp.data.raw_csv_writer import RawTrialWriter
from evexp.hardware.acquisition import SensorAcquisition
from evexp.hardware.force import (AcquisitionForceSource, DualForceCalibration,
                                  measure_dual_bias)
from evexp.hardware.dev_sources import ManualForceSource, ManualPositionSource, SimulatedForceSource
from evexp.hardware.mock import MockDAQDevice, MockStimulusOutput
from evexp.hardware.multi_daq import MultiDaqDevice
from evexp.hardware.neonode import (HidNeonodeTransport, NeonodeCalibration,
                                    NeonodeConnectionError,
                                    NeonodePositionSource)
from evexp.hardware.nidaq import NiDaqDevice, gauge_channel_map
from evexp.processing.force_feedback import ForceBands
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real-daq", action="store_true",
        help="Use NiDaqDevice (real or NI MAX simulated card) instead of "
             "MockDAQDevice/MockStimulusOutput for both acquisition and "
             "the stimulus. Default: mock, no hardware needed."
    )
    parser.add_argument(
        "--daq-fs1", default="Dev1",
        help="NI-DAQmx device reading force sensor 1. Only with --real-daq."
    )
    parser.add_argument(
        "--daq-fs2", default="Dev2",
        help="NI-DAQmx device reading force sensor 2. Only with --real-daq."
    )
    parser.add_argument(
        "--daq-stim", default="Dev3",
        help="NI-DAQmx device carrying the stimulus AO and the amplifier "
             "monitor input. Only with --real-daq."
    )
    parser.add_argument(
        "--monitor-channel", default="ai0",
        help="AI channel on the stimulus card reading the amplifier's "
             "monitor output."
    )
    parser.add_argument(
        "--neonode", action="store_true",
        help="Track finger position with the Neonode IR sensor instead of "
             "the mouse. Falls back to the mouse if the sensor cannot be "
             "opened."
    )
    parser.add_argument(
        "--list-screens", action="store_true",
        help="Print the index and geometry of every connected monitor, "
             "then exit. Use this to find the right "
             "display.participant_screen_index before a real session."
    )
    return parser.parse_args()


def list_screens() -> None:
    app = QApplication(sys.argv)
    for i, screen in enumerate(app.screens()):
        geo = screen.geometry()
        primary = " (primary)" if screen is app.primaryScreen() else ""
        print(f"[{i}] {screen.name()}{primary}: "
              f"{geo.width()}x{geo.height()} at ({geo.x()}, {geo.y()})")


def move_to_screen(window, app: QApplication, screen_index: Optional[int]) -> None:
    """Move window onto the requested monitor before showing it fullscreen.

    Qt's showFullScreen() fills whichever screen the window is currently on
    - normally wherever it was created, i.e. the primary monitor - so on a
    two-monitor rig the participant window has to be moved explicitly or it
    opens on the experimenter's screen instead of the touchscreen.
    """
    if screen_index is None:
        return
    screens = app.screens()
    if not 0 <= screen_index < len(screens):
        print(f"WARNING: participant_screen_index {screen_index} is out of "
              f"range (0-{len(screens) - 1} available); using the default "
              "screen instead. Run with --list-screens to check indices.")
        return
    window.move(screens[screen_index].geometry().topLeft())


def main():
    args = parse_args()
    if args.list_screens:
        list_screens()
        sys.exit(0)

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

    # Three cards: one per force sensor, one for the stimulus. The two
    # force cards are joined into a single channel list so that acquisition
    # and everything above it still sees one device; the stimulus card is
    # kept separate because it is doing a different job (AO out, monitor
    # in), and its AI channel rides along in the same joined stream.
    #
    # Each card keeps its own sample clock - see hardware/multi_daq.py for
    # why that is acceptable here and what would change it.
    if args.real_daq:
        # ao_voltage_limit_v drives both the safety.check_voltage_limit()
        # check and the AO channel's own hardware range (see nidaq.py) - tied
        # to the staircase's max_value so there is one ceiling to edit, not
        # two numbers that happen to agree until someone changes only one.
        stimulus_card = NiDaqDevice(
            device_name=args.daq_stim,
            channel_map={"current": args.monitor_channel},
            ao_voltage_limit_v=cfg["staircase"]["max_value"],
        )
        acquisition_device = MultiDaqDevice([
            NiDaqDevice(device_name=args.daq_fs1,
                        channel_map=gauge_channel_map("fs1")),
            NiDaqDevice(device_name=args.daq_fs2,
                        channel_map=gauge_channel_map("fs2")),
            stimulus_card,
        ])
        stimulus_output = stimulus_card
        print(f"DAQ: {args.daq_fs1} (FS1) + {args.daq_fs2} (FS2) + "
              f"{args.daq_stim} (stimulus, monitor on {args.monitor_channel})")
    else:
        acquisition_device = MockDAQDevice(realtime=True)
        stimulus_output = MockStimulusOutput(verbose=False)
        print("DAQ: MockDAQDevice/MockStimulusOutput - no hardware")

    staircase = StaircaseController(StaircaseConfig(**cfg["staircase"]))
    trial = Trial2IFC(
        stimulus=stimulus_output,
        staircase=staircase,
        timing=TrialTiming(
            pre_interval_wait_s=timing_cfg["pre_interval_wait_s"],
            gap_s=timing_cfg["gap_s"],
            cursor_speed_mm_s=timing_cfg["cursor_speed_mm_s"],
            cursor_travel_mm=timing_cfg["cursor_travel_mm"],
        ),
        training_voltage=cfg.get("training", {}).get("voltage", 2.0),
    )
    print(f"Interval duration: {trial.timing.interval_s:.3f} s "
          f"({timing_cfg['cursor_travel_mm']:g} mm / {cue_speed_mm_s:g} mm/s, one-way)")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Everything for this participant lives in one folder; filenames are
    # built from participant id, speed condition and trial number, with one
    # session stamp shared by every file from the same run.
    participant_dir = Path(session_cfg["output_dir"]) / f"p{participant_id}"
    participant_dir.mkdir(parents=True, exist_ok=True)
    output_path = participant_dir / (
        f"p{participant_id}_s{cue_speed_mm_s:g}_staircase_{stamp}.csv"
    )

    logger = CSVTrialLogger(str(output_path))
    print(f"Logging trials to {output_path}")

    raw_writer = RawTrialWriter(
        output_dir=session_cfg["output_dir"],
        participant_id=participant_id,
        speed_mm_s=cue_speed_mm_s,
        stamp=stamp,
    )
    print(f"Raw per-trial CSVs -> {raw_writer.participant_dir}")

    # Two Nano17s under the plate. Matrices are the ones from
    # Setup_FS1.5.py, not read from these sensors' own .cal files, so the
    # numbers stay marked placeholder (see hardware/force.py).
    force_calibration = DualForceCalibration.from_gain_matrices()

    snapshot_path = output_path.with_suffix(".yaml")
    write_config_snapshot(cfg, snapshot_path)
    print(f"Config snapshot saved to {snapshot_path}")

    reveal = cfg.get("debug", {}).get("reveal_stimulus", False)
    if reveal:
        print("WARNING: reveal_stimulus is enabled - data from this session "
              "is not a valid threshold measurement.")

    travel_mm = timing_cfg["cursor_travel_mm"]

    # Finger position. The Neonode reports in its own coordinates, so
    # origin_x has to line the sensor's zero up with the start of the cue
    # track, and the scale sign has to match the direction of travel -
    # neither is knowable until the sensor is mounted on the rig.
    if args.neonode:
       
        position_source = NeonodePositionSource(
            transport=HidNeonodeTransport(),
            calibration=NeonodeCalibration(),
        )
        try:
            position_source.connect()
        except NeonodeConnectionError as exc:
            sys.exit(f"Neonode unavailable ({exc}); aborting - rerun without "
                      "--neonode to develop on the mouse instead.")
        print("Position source: Neonode IR sensor")
    else:
        position_source = ManualPositionSource(travel_mm=travel_mm)
        print("Position source: mouse (move the pointer along the cue track)")

    # Continuous sensor acquisition, on its own thread. Built before the
    # force source below, not after: the "nano17" force option reads
    # acquisition.latest(), so acquisition has to exist first.
    acq_cfg = cfg["acquisition"]
    acquisition = SensorAcquisition(
        device=acquisition_device,
        position_source=position_source,
        ring_seconds=acq_cfg.get("ring_seconds", 30.0),
        chunk_samples=acq_cfg.get("chunk_samples") or None,
    )
    acquisition.start(sample_rate_hz=acq_cfg["sample_rate_hz"])
    print(f"Acquisition: {len(acquisition.channels)} channels at "
          f"{acquisition.sample_rate_hz:g} Hz "
          f"({'real/NI MAX' if args.real_daq else 'simulated'})")

    # Bias: once per session, right after acquisition starts, nothing
    # touching either sensor. Both are averaged over the same rest period.
    print("Measuring force bias (nothing touching the sensors)...")
    bias_fs1, bias_fs2 = measure_dual_bias(acquisition, n_samples=100)
    force_calibration = force_calibration.with_bias(bias_fs1, bias_fs2)
    print(f"Force calibration: {force_calibration.describe()}")

    # Force feedback. "nano17" reads the live normal force off the DAQ's
    # most recent sample via acquisition.latest() (see
    # hardware.force.AcquisitionForceSource) - the actual 10 kHz sampling
    # happens in the acquisition thread above, this just polls the freshest
    # value at whatever rate the UI timer asks. "mouse_y" and "simulated"
    # remain for developing without any DAQ channels wired up at all.
    force_cfg = cfg["force"]
    force_bands = ForceBands(
        target_n=force_cfg["target_n"],
        full_scale_n=force_cfg["full_scale_n"],
        in_band_half_width_n=force_cfg.get("in_band_half_width_n"),
    )
    force_source_kind = force_cfg.get("source", "simulated")
    if force_source_kind == "mouse_y":
        force_source = ManualForceSource(initial_n=force_bands.target_n)
    elif force_source_kind == "simulated":
        force_source = SimulatedForceSource(target_n=force_bands.target_n)
    elif force_source_kind == "nano17":
        force_source = AcquisitionForceSource(acquisition, force_calibration)
    else:
        raise ValueError(
            f"unknown force.source {force_source_kind!r} in config; "
            "expected 'mouse_y', 'simulated', or 'nano17'"
        )
    print(f"Force source: {force_source_kind}")

    # Stopping from aboutToQuit rather than after app.exec_() so that the
    # worker is joined on every exit path, including the window being closed
    # and the Escape confirmation on the console.
    def shutdown() -> None:
        acquisition.stop()
        if isinstance(position_source, NeonodePositionSource):
            position_source.disconnect()
        print(acquisition.stats().describe())
        # AO task is separate from acquisition's AI task and is not closed
        # by acquisition.stop() - close it explicitly if it was left running.
        if args.real_daq and stimulus_output.is_active:
            stimulus_output.stimulus_off()

    app.aboutToQuit.connect(shutdown)

    # Without this, Ctrl+C is only noticed whenever Python next happens to
    # run - typically inside _poll_sensors, mid-callback - and the
    # exception propagates out of Qt's C++ event loop instead of through
    # app.quit(). That skips aboutToQuit, so shutdown() never runs: the DAQ
    # cards stay open and the trial in progress is never written. Routing
    # SIGINT through app.quit() makes Ctrl+C behave like closing the window.
    signal.signal(signal.SIGINT, lambda *_: app.quit())

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
        force_source=force_source,
        force_bands=force_bands,
        cursor_speed_mm_s=cue_speed_mm_s,
        acquisition=acquisition,
        raw_writer=raw_writer,
        force_calibration=force_calibration,
    )

    experimenter.show()
    move_to_screen(participant, app, display_cfg.get("participant_screen_index"))
    if display_cfg.get("fullscreen", False):
        participant.showFullScreen()
    else:
        participant.show()
    participant.activateWindow()
    participant.setFocus()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()