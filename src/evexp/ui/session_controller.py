"""Glue between the Trial2IFC state machine and the Qt event loop.

Every phase transition is driven by a one-shot QTimer rather than by waiting,
so the interface stays responsive throughout. This is also why Trial2IFC itself
contains no Qt code: swapping this controller for the headless SimulatedRunner
requires no change to the trial logic. The same separation holds for force: the
force source and the per-trial statistics live here, not in Trial2IFC, so the
trial state machine stays hardware-free.

A separate repeating timer polls the position and force sources and pushes the
derived finger speed to the experimenter console. Polling is used rather than a
callback so that the display refreshes at a fixed, predictable rate regardless
of how fast the underlying source produces samples - the mouse fires events
only when it moves, whereas a real sensor streams at ~100 Hz.

Raw per-trial recording (RawSessionWriter) is optional and wired in here
rather than in Trial2IFC for the same reason force is: it needs the
acquisition ring buffer and a force calibration, neither of which the trial
state machine has any business knowing about. It is cut and written as soon
as a trial's response comes in, not deferred, because SensorAcquisition's
ring buffer holds only ~30 s - a trial cut out late is gone for good.
"""

import time
from typing import Optional

from PyQt5.QtCore import QObject, QTimer

from evexp.data.csv_logger import CSVTrialLogger
from evexp.data.raw_hdf5_writer import RawSessionWriter
from evexp.hardware.acquisition import SensorAcquisition
from evexp.hardware.force import ForceCalibration, ForceSource
from evexp.hardware.position import PositionSource
from evexp.processing.force_feedback import ForceBands, ForceTrialAccumulator
from evexp.processing.speed import SpeedEstimator
from evexp.psychophysics.trial import Trial2IFC, TrialState
from evexp.ui.experimenter_window import ExperimenterWindow
from evexp.ui.participant_window import ParticipantWindow

SENSOR_POLL_INTERVAL_MS = 50   # 20 Hz: fast enough to be live, slow enough to read

# States during which the participant is actually stroking the screen, as
# opposed to waiting or resting between intervals. Force is only meaningful
# to summarise while this is happening - averaging in the gap, where the
# finger is lifted, would pull the mean toward "no contact" for reasons that
# have nothing to do with how well the participant pressed.
_STROKING_STATES = (TrialState.INTERVAL_1, TrialState.INTERVAL_2)


class SessionController(QObject):
    def __init__(
        self,
        trial: Trial2IFC,
        participant: ParticipantWindow,
        experimenter: ExperimenterWindow,
        logger: CSVTrialLogger,
        reveal_stimulus: bool = False,
        n_training: int = 0,
        position_source: Optional[PositionSource] = None,
        force_source: Optional[ForceSource] = None,
        force_bands: Optional[ForceBands] = None,
        cursor_speed_mm_s: Optional[float] = None,
        acquisition: Optional[SensorAcquisition] = None,
        raw_writer: Optional[RawSessionWriter] = None,
        force_calibration: Optional[ForceCalibration] = None,
    ):
        super().__init__()
        self.trial = trial
        self.participant = participant
        self.experimenter = experimenter
        self.logger = logger
        self.reveal_stimulus = reveal_stimulus
        self._n_training_total = n_training
        self._n_training = 0
        self._training_done = (n_training == 0)
        self._cursor_speed_mm_s = cursor_speed_mm_s

        # Wire participant inputs to the two entry points of the trial loop.
        # SPACE -> _begin_trial, 1/2 keypress -> _on_response.
        # _advance is never called by the participant; only QTimer fires it.
        self.participant.startRequested.connect(self._begin_trial)
        self.participant.responseGiven.connect(self._on_response)

        # Live finger-speed readout on the experimenter console.
        self._position_source = position_source
        self._speed_estimator = SpeedEstimator()

        # Per-trial force summary. Only built if a force source was actually
        # supplied - a session run without one (or without the sensor wired
        # in yet) logs None for these columns rather than fabricating zeros.
        self._force_source = force_source
        self._force_accumulator = (
            ForceTrialAccumulator(force_bands)
            if force_source is not None and force_bands is not None else None
        )
        self._collecting_force = False

        # Raw per-trial signal recording. All three are required together;
        # a session missing any one of them (acquisition not started, no
        # writer configured, or no calibration - even a placeholder one)
        # simply does not record raw signals, rather than half-recording.
        self._acquisition = acquisition
        self._raw_writer = raw_writer
        self._force_calibration = force_calibration
        self._raw_recording_enabled = (
            acquisition is not None and raw_writer is not None
            and force_calibration is not None
        )
        self._trial_start_perf = 0.0
        # Both intervals' boundaries are tracked, not just one: which one
        # carries the stimulus is randomised per trial (Trial2IFC picks it
        # in start_trial()), so both have to be available when the trial
        # ends and _write_raw_trial() can finally look up which was which.
        self._interval1_start_perf = 0.0
        self._interval1_end_perf = 0.0
        self._interval2_start_perf = 0.0
        self._interval2_end_perf = 0.0

        self._sensor_timer = QTimer(self)
        self._sensor_timer.setInterval(SENSOR_POLL_INTERVAL_MS)
        self._sensor_timer.timeout.connect(self._poll_sensors)
        if position_source is not None or force_source is not None:
            self._sensor_timer.start()

        self._refresh_status()

    def _begin_trial(self) -> None:
        training = not self._training_done
        if not training:
            self.experimenter.clear_message()
        if self._force_accumulator is not None:
            self._force_accumulator.reset()
        self._collecting_force = False
        # Marks the start of the raw-signal window for this trial. Recorded
        # here (not lazily on first use) so it covers the whole trial,
        # NOT the raw-recording window boundary any more - see _advance()
        # for where that is actually marked. Kept for anything that still
        # wants "when did this trial begin" in the wider sense.
        self._trial_start_perf = time.perf_counter()
        duration_s = self.trial.start_trial(training=training)
        label = "Training" if training else f"trial {self.trial.trial_index }"
        self._log_console(f"\n{label}  {self.trial.applied_voltage:6.3f} V")
        # start_trial() now enters PRE_INTERVAL_WAIT, not INTERVAL_1.
        self.participant.show_pre_interval_wait(1, duration_s)
        self._refresh_status()
        self._arm(duration_s)

    def _advance(self) -> None:
        # Called only by QTimer, never directly. Each call moves the state
        # machine one step:
        #   PRE_INTERVAL_WAIT -> INTERVAL_1 -> GAP
        #   -> PRE_INTERVAL_WAIT -> INTERVAL_2 -> AWAITING_RESPONSE
        # Returns None at AWAITING_RESPONSE because that phase has no fixed duration.
        duration_s = self.trial.advance()
        state = self.trial.state
        self._collecting_force = state in _STROKING_STATES

        if state is TrialState.INTERVAL_1:
            self.participant.show_interval(1, duration_s)
            self._announce_interval(1)
            self._interval1_start_perf = time.perf_counter()
        elif state is TrialState.GAP:
            self.participant.show_gap()
            # Reached right as interval_1 ends (interval_1 is always the one
            # right before the gap - interval_2 is followed by
            # AWAITING_RESPONSE instead, never by GAP).
            self._interval1_end_perf = time.perf_counter()
        elif state is TrialState.PRE_INTERVAL_WAIT:
            # Reached only on the way to interval 2; the wait before interval 1
            # is entered by _begin_trial above.
            self.participant.show_pre_interval_wait(2, duration_s)
        elif state is TrialState.INTERVAL_2:
            self.participant.show_interval(2, duration_s)
            self._announce_interval(2)
            self._interval2_start_perf = time.perf_counter()
        elif state is TrialState.AWAITING_RESPONSE:
            self.participant.show_response_prompt()
            self._log_console("  -> press 1 or 2")
            self._interval2_end_perf = time.perf_counter()

        self._refresh_status()
        if duration_s is not None:
            self._arm(duration_s)

    def _on_response(self, response_interval: int) -> None:
        self._collecting_force = False
        result = self.trial.submit_response(response_interval)

        # Cut and write the raw signal window first, before anything else
        # in this method - the ring buffer holds only ~30 s, so this is the
        # one step in _on_response where delay actually costs data.
        # Training trials are skipped: they have no trial_index in the
        # staircase's numbering and are not the data being collected.
        if not result.training and self._raw_recording_enabled:
            self._write_raw_trial(result)

        # Filled in here, not by Trial2IFC: these come from hardware the
        # trial state machine has no knowledge of. TrialResult is a plain,
        # mutable dataclass for exactly this reason - the CSV logger reads
        # its field list dynamically, so adding data here needs no change to
        # the logger or to Trial2IFC.
        if self._force_accumulator is not None:
            stats = self._force_accumulator.stats()
            result.mean_normal_force_n = stats.mean_n
            result.std_normal_force_n = stats.std_n
            result.force_in_band_fraction = stats.in_band_fraction
        result.cursor_speed_mm_s = self._cursor_speed_mm_s

        # Log every trial immediately so a crash mid-session still yields
        # partial data. Training rows are included with training=True flag.
        self.logger.log(result)

        verdict = "correct" if result.correct else "WRONG"
        reversal = "  [reversal]" if result.reversal else ""
        force_note = ""
        if result.mean_normal_force_n is not None:
            force_note = (f"   force {result.mean_normal_force_n:.2f} N "
                          f"({result.force_in_band_fraction * 100:.0f}% in band)")
        self._log_console(
            f"  answered {response_interval} -> {verdict}{reversal}   "
            f"next: {self.trial.staircase.value:.3f} V{force_note}"
        )

        if result.training:
            self._n_training += 1
            if self._n_training >= self._n_training_total:
                self._training_done = True
                self.participant.show_ready_post_training()
                self.experimenter.announce_training_done()
            else:
                self.participant.show_ready()
            self._refresh_status()
            return

        self.experimenter.add_result(result)

        if self.trial.staircase.aborted:
            self._sensor_timer.stop()
            self.participant.show_aborted()
            self.experimenter.announce_abort()
        elif self.trial.finished:
            self._sensor_timer.stop()
            self._save_convergence_plot()
            self.participant.show_finished()
            self.experimenter.announce_completion(self.trial.staircase)
        else:
            self.participant.show_ready()

        self._refresh_status()

    def _poll_sensors(self) -> None:
        """Sample position and force at a fixed rate.

        Position feeds the live speed readout on the experimenter console.
        Force feeds both that same console (not yet wired to show it) and,
        while a stroke is actually in progress, the per-trial accumulator
        that ends up in the CSV.
        """
        if self._position_source is not None:
            self._speed_estimator.add(self._position_source.read())
            speed = self._speed_estimator.speed_mm_s(now=time.perf_counter())
            self.experimenter.update_speed(speed)

        if self._force_source is not None:
            force = self._force_source.read_normal_force()
            if self._force_accumulator is not None and self._collecting_force:
                self._force_accumulator.add(force)

    def _write_raw_trial(self, result) -> None:
        """Cut only the stimulus-carrying interval from the ring buffer.

        Which interval that is (1 or 2) is randomised per trial by
        Trial2IFC, hence the lookup here rather than a fixed choice.

        Best-effort: a failure here must not stop the trial loop or lose the
        CSV row that follows, so errors are logged (when reveal_stimulus is
        on) and swallowed rather than raised. An empty window - the ring
        buffer already overwrote it, or acquisition was never running - is
        skipped the same way, since RawSessionWriter has nothing useful to
        do with zero samples.
        """
        try:
            if result.stimulus_interval == 1:
                t_start, t_end = self._interval1_start_perf, self._interval1_end_perf
            else:
                t_start, t_end = self._interval2_start_perf, self._interval2_end_perf

            block = self._acquisition.window(t_start, t_end)
            if block.shape[1] == 0:
                self._log_console(
                    "  [raw] skipped: empty window (ring buffer overrun or "
                    "acquisition not running)"
                )
                return

            channels = self._acquisition.channels
            gauge_indices = [channels.index(f"gauge{i}") for i in range(6)]
            gauge_chunk = block[gauge_indices, :]

            positions = [
                p for p in self._acquisition.recent_positions()
                if t_start <= p.t <= t_end
            ]

            self._raw_writer.append_trial(
                trial_index=result.trial_index,
                gauge_chunk=gauge_chunk,
                calibration=self._force_calibration,
                sample_rate_hz=self._acquisition.sample_rate_hz,
                t0=t_start,
                commanded_voltage=result.applied_voltage,
                positions=positions,
            )
            self._log_console(
                f"  [raw] wrote trial {result.trial_index} "
                f"(stimulus interval {result.stimulus_interval}): "
                f"{gauge_chunk.shape[1]} samples, {len(positions)} positions"
            )
        except Exception as exc:                       # noqa: BLE001
            self._log_console(f"  [raw] FAILED to write trial: {exc}")

    def _announce_interval(self, number: int) -> None:
        # Only prints when reveal_stimulus is True (debug/development mode).
        # Never enable this during real data collection.
        marker = "  <<< STIMULUS" if self.trial.stimulus.is_active else ""
        self._log_console(f"  interval {number}{marker}")

    def _log_console(self, message: str) -> None:
        if self.reveal_stimulus:
            print(message, flush=True)

    def _arm(self, duration_s: float) -> None:
        # QTimer.singleShot does not block the event loop, so the GUI stays
        # responsive during intervals and the gap.
        QTimer.singleShot(int(duration_s * 1000), self._advance)

    def _refresh_status(self) -> None:
        self.experimenter.update_status(self.trial.state.name, self.trial.staircase)

    def _save_convergence_plot(self) -> None:
        try:
            import matplotlib.pyplot as plt
            from pathlib import Path

            voltages = self.trial.voltages_over_trials()
            threshold = self.trial.staircase.threshold_estimate

            plt.figure(figsize=(8, 4))
            plt.plot(voltages, marker="o", markersize=3, linewidth=1)
            plt.axhline(threshold, color="green", linestyle="--",
                        label=f"threshold estimate ({threshold:.3f} V)")
            plt.xlabel("Trial")
            plt.ylabel("Applied voltage (V)")
            plt.title(f"Staircase convergence ({self.trial.staircase.cfg.rule})")
            plt.legend()
            plt.tight_layout()

            csv_path = Path(self.logger.output_path)
            plot_path = csv_path.with_suffix(".png")
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"Convergence plot saved to {plot_path}")
        except Exception as e:
            print(f"Could not save convergence plot: {e}")