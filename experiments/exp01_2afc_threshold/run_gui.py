"""Interactive 2AFC threshold session with a real participant."""

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
from evexp.hardware.nidaq import NiDaqDevice
from evexp.processing.force_feedback import ForceBands
from evexp.hardware.screen import ScreenCalibration
from evexp.psychophysics.staircase import StaircaseConfig, StaircaseController
from evexp.psychophysics.trial import Trial2AFC, TrialTiming
from evexp.ui.experimenter_window import ExperimenterWindow
from evexp.ui.participant_window import ParticipantWindow
from evexp.ui.session_controller import SessionController
from evexp.ui.setup_dialog import ask_for_setup

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "experiment.yaml"


def _gauge_channel_map_from_config(prefix: str, ai_channels: list) -> dict:
    """config/experiment.yaml's daq.force_sensor_N.ai_channels -> channel_map,
    named by which sensor it is (see gauge_channel_map() in hardware/nidaq.py
    for the equivalent used by non-config-driven callers)."""
    return {f"{prefix}_gauge{i}": ch for i, ch in enumerate(ai_channels)}


def write_config_snapshot(cfg: dict, path: Path) -> None:
    """Record the configuration this session actually ran with, including
    the runtime choices made in the setup dialog."""
    header = (
        "# Effective configuration for this session, including choices made\n"
        "# in the setup dialog. Generated automatically - do not edit.\n"
        f"# Written {datetime.now().isoformat(timespec='seconds')}\n"
    )
    path.write_text(header + yaml.safe_dump(cfg, sort_keys=False))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
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
    """Move window onto the requested monitor before showing it fullscreen."""
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
    force_cfg = cfg["force"]

    app = QApplication(sys.argv)

    calibration = ScreenCalibration.from_config(display_cfg)
    print(f"Display calibration: {calibration.describe()}")
    for warning in calibration.warnings():
        print(f"WARNING: {warning}")

    setup = ask_for_setup(
        speed_options=timing_cfg["speed_options_mm_s"],
        default_speed_mm_s=timing_cfg["cursor_speed_mm_s"],
        force_options=force_cfg["force_levels_n"],
        default_force_n=force_cfg["target_n"],
        experiment_id=session_cfg["experiment_id"],
    )
    if setup is None:
        print("Setup cancelled, exiting.")
        sys.exit(0)

    participant_id = setup.participant_id
    cue_speed_mm_s = setup.target_speed_mm_s
    target_force_n = setup.target_force_n

    timing_cfg["cursor_speed_mm_s"] = cue_speed_mm_s
    force_cfg["target_n"] = target_force_n
    session_cfg["participant_id"] = participant_id
    print(f"Participant {participant_id}, sliding speed {cue_speed_mm_s:g} mm/s, "
          f"target force {target_force_n:g} N")

    daq_cfg = cfg.get("daq", {})
    if daq_cfg.get("enabled", False):
        stim_cfg = daq_cfg["stimulus_card"]
        fs1_cfg = daq_cfg["force_sensor_1"]
        fs2_cfg = daq_cfg["force_sensor_2"]

        # ao_voltage_limit_v ties the AO hardware range to the staircase ceiling.
        stimulus_card = NiDaqDevice(
            device_name=stim_cfg["device"],
            ao_channel=stim_cfg["voltage_set_channel"],
            channel_map={"current": stim_cfg["current_read_channel"]},
            terminal_config=stim_cfg["terminal_config"],
            ao_voltage_limit_v=cfg["staircase"]["max_value"],
        )
        acquisition_device = MultiDaqDevice([
            NiDaqDevice(device_name=fs1_cfg["device"],
                        channel_map=_gauge_channel_map_from_config("fs1", fs1_cfg["ai_channels"]),
                        terminal_config=fs1_cfg["terminal_config"]),
            NiDaqDevice(device_name=fs2_cfg["device"],
                        channel_map=_gauge_channel_map_from_config("fs2", fs2_cfg["ai_channels"]),
                        terminal_config=fs2_cfg["terminal_config"]),
            stimulus_card,
        ])
        stimulus_output = stimulus_card
        print(f"DAQ: {fs1_cfg['device']} (FS1) + {fs2_cfg['device']} (FS2) + "
              f"{stim_cfg['device']} (stimulus: voltage set on "
              f"{stim_cfg['voltage_set_channel']}, current read on "
              f"{stim_cfg['current_read_channel']})")
    else:
        acquisition_device = MockDAQDevice(realtime=True)
        stimulus_output = MockStimulusOutput(verbose=False)
        print("DAQ: MockDAQDevice/MockStimulusOutput - no hardware")

    staircase = StaircaseController(StaircaseConfig(**cfg["staircase"]))
    trial = Trial2AFC(
        stimulus=stimulus_output,
        staircase=staircase,
        timing=TrialTiming(
            pre_interval_wait_s=timing_cfg["pre_interval_wait_s"],
            gap_s_override=timing_cfg.get("gap_s_override"),
            cursor_speed_mm_s=timing_cfg["cursor_speed_mm_s"],
            cursor_travel_mm=timing_cfg["cursor_travel_mm"],
        ),
        training_voltage=cfg.get("training", {}).get("voltage", 2.0),
    )
    print(f"Interval duration: {trial.timing.interval_s:.3f} s "
          f"({timing_cfg['cursor_travel_mm']:g} mm / {cue_speed_mm_s:g} mm/s, one-way)")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # One folder per participant; filenames share a session stamp per run.
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

    force_calibration = DualForceCalibration.from_gain_matrices()

    snapshot_path = output_path.with_suffix(".yaml")
    write_config_snapshot(cfg, snapshot_path)
    print(f"Config snapshot saved to {snapshot_path}")

    reveal = cfg.get("debug", {}).get("reveal_stimulus", False)
    if reveal:
        print("WARNING: reveal_stimulus is enabled - data from this session "
              "is not a valid threshold measurement.")

    travel_mm = timing_cfg["cursor_travel_mm"]

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
          f"({'real/NI MAX' if daq_cfg.get('enabled', False) else 'simulated'})")

    print("Measuring force bias (nothing touching the sensors)...")
    bias_fs1, bias_fs2 = measure_dual_bias(acquisition, n_samples=100)
    force_calibration = force_calibration.with_bias(bias_fs1, bias_fs2)
    print(f"Force calibration: {force_calibration.describe()}")

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

    def shutdown() -> None:
        acquisition.stop()
        if isinstance(position_source, NeonodePositionSource):
            position_source.disconnect()
        print(acquisition.stats().describe())
        print(f"Invalid trials discarded: {controller.n_invalid}")
        # AO task is not closed by acquisition.stop(); close it explicitly.
        if daq_cfg.get("enabled", False) and stimulus_output.is_active:
            stimulus_output.stimulus_off()

    app.aboutToQuit.connect(shutdown)
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
        force_tolerance_pct=cfg.get("validity", {}).get("force_tolerance_pct"),
        speed_tolerance_pct=cfg.get("validity", {}).get("speed_tolerance_pct"),
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