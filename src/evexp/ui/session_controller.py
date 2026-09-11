"""Glue between the Trial2AFC state machine and the Qt event loop.

Phase transitions are driven by a one-shot QTimer, not blocking waits, so
Trial2AFC itself stays Qt-free and hardware-free.
"""

import time
from typing import Optional

from PyQt5.QtCore import QObject, QTimer

from evexp.data.csv_logger import CSVTrialLogger

from evexp.data.raw_csv_writer import RawTrialWriter

from evexp.hardware.acquisition import SensorAcquisition
from evexp.hardware.force import ForceCalibration, ForceSource
from evexp.hardware.position import PositionSource
from evexp.processing.force_feedback import ForceBands, ForceTrialAccumulator
from evexp.processing.signal import SpeedEstimator, SpeedTrialAccumulator
from evexp.psychophysics.trial import Trial2AFC, TrialState
from evexp.ui.experimenter_window import ExperimenterWindow
from evexp.ui.participant_window import ParticipantWindow
from evexp.ui.recording_controller import RecordingController

SENSOR_POLL_INTERVAL_MS = 50   # 20 Hz

# Maps the states where the participant is actually stroking the screen to
# their interval number; not present means neither interval is running.
_INTERVAL_NUMBER_BY_STATE = {TrialState.INTERVAL_1: 1, TrialState.INTERVAL_2: 2}


class SessionController(QObject):
    """Runs a full session: drives Trial2AFC, updates both windows, and logs
    each trial's result."""

    def __init__(
        self,
        trial: Trial2AFC,
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
        raw_writer: Optional[RawTrialWriter] = None,
        force_calibration: Optional[ForceCalibration] = None,
        force_tolerance_pct: Optional[float] = None,
        speed_tolerance_pct: Optional[float] = None,
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
        # None disables the corresponding validity check.
        self._force_tolerance_pct = force_tolerance_pct
        self._speed_tolerance_pct = speed_tolerance_pct
        self._n_invalid = 0
        self._current_is_training = False

        self.participant.startRequested.connect(self._begin_trial)
        self.participant.responseGiven.connect(self._on_response)

        # Live finger-speed readout on the experimenter console.
        self._position_source = position_source
        self._speed_estimator = SpeedEstimator()

        # Per-trial summaries, for the CSV columns; None if no source configured.
        self._force_source = force_source
        self._force_accumulator = (
            ForceTrialAccumulator(force_bands)
            if force_source is not None and force_bands is not None else None
        )
        self._force_bands = force_bands
        self._speed_accumulator = (
            SpeedTrialAccumulator() if position_source is not None else None
        )
        # Separate per-interval accumulators for the validity check, so an
        # over-target interval and an under-target one can't average out.
        self._force_accumulators_by_interval = (
            {1: ForceTrialAccumulator(force_bands), 2: ForceTrialAccumulator(force_bands)}
            if force_source is not None and force_bands is not None else None
        )
        self._speed_accumulators_by_interval = (
            {1: SpeedTrialAccumulator(), 2: SpeedTrialAccumulator()}
            if position_source is not None else None
        )
        self._collecting_interval: Optional[int] = None

        self._recording = RecordingController(
            acquisition, raw_writer, force_calibration, force_bands,
            log_fn=self._log_console)
        self._trial_start_perf = 0.0

        self._sensor_timer = QTimer(self)
        self._sensor_timer.setInterval(SENSOR_POLL_INTERVAL_MS)
        self._sensor_timer.timeout.connect(self._poll_sensors)
        if position_source is not None or force_source is not None:
            self._sensor_timer.start()

        self._refresh_status()

    @property
    def n_invalid(self) -> int:
        """Trials discarded this session for being out of force/speed tolerance."""
        return self._n_invalid

    def _begin_trial(self) -> None:
        training = not self._training_done
        self._current_is_training = training
        if not training:
            self.experimenter.clear_message()
        if self._force_accumulator is not None:
            self._force_accumulator.reset()
        if self._speed_accumulator is not None:
            self._speed_accumulator.reset()
        if self._force_accumulators_by_interval is not None:
            for acc in self._force_accumulators_by_interval.values():
                acc.reset()
        if self._speed_accumulators_by_interval is not None:
            for acc in self._speed_accumulators_by_interval.values():
                acc.reset()
        self._collecting_interval = None
        self._trial_start_perf = time.perf_counter()
        duration_s = self.trial.start_trial(training=training)
        label = "Training" if training else f"trial {self.trial.trial_index }"
        self._log_console(f"\n{label}  {self.trial.applied_voltage:6.3f} V")
        self.participant.show_pre_interval_wait(1, duration_s)
        self._refresh_status()
        self._arm(duration_s)

    def _advance(self) -> None:
        """Called only by QTimer. Steps the state machine one phase."""
        duration_s = self.trial.advance()
        state = self.trial.state
        self._collecting_interval = _INTERVAL_NUMBER_BY_STATE.get(state)

        if state is TrialState.INTERVAL_1:
            self.participant.show_interval(1, duration_s)
            self._announce_interval(1)
            self._recording.mark_interval1_start()
        elif state is TrialState.GAP:
            self.participant.show_gap()
            self._recording.mark_interval1_end()
        elif state is TrialState.PRE_INTERVAL_WAIT:
            self.participant.show_pre_interval_wait(2, duration_s)
        elif state is TrialState.INTERVAL_2:
            self.participant.show_interval(2, duration_s)
            self._announce_interval(2)
            self._recording.mark_interval2_start()
        elif state is TrialState.AWAITING_RESPONSE:
            self.participant.show_response_prompt()
            self._log_console("  -> press 1 or 2")
            self._recording.mark_interval2_end()
            # Delay so the tail of the interval has landed in the ring buffer.
            QTimer.singleShot(250, self._recording.capture_interval2)

        self._refresh_status()
        if duration_s is not None:
            self._arm(duration_s)

    def _on_response(self, response_interval: int) -> None:
        self._collecting_interval = None

        # Measured before scoring: validity gates whether the staircase
        # updates. Trial-wide accumulators feed the CSV; per-interval ones
        # feed the validity check.
        force_stats = (self._force_accumulator.stats()
                      if self._force_accumulator is not None else None)
        speed_stats = (self._speed_accumulator.stats()
                      if self._speed_accumulator is not None else None)
        force_stats_by_interval = (
            {i: acc.stats() for i, acc in self._force_accumulators_by_interval.items()}
            if self._force_accumulators_by_interval is not None else {})
        speed_stats_by_interval = (
            {i: acc.stats() for i, acc in self._speed_accumulators_by_interval.items()}
            if self._speed_accumulators_by_interval is not None else {})
        problems = self._validity_problems(force_stats_by_interval, speed_stats_by_interval)
        # Training is never discarded; the check still runs to warn the
        # experimenter so they can correct the participant.
        result = self.trial.submit_response(
            response_interval, valid=(self._current_is_training or not problems))

        if force_stats is not None:
            result.mean_normal_force_n = force_stats.mean_n
            result.std_normal_force_n = force_stats.std_n
            result.force_in_band_fraction = force_stats.in_band_fraction
        if speed_stats is not None:
            result.mean_speed_mm_s = speed_stats.mean_mm_s
        result.cursor_speed_mm_s = self._cursor_speed_mm_s

        if problems and self._current_is_training:
            reason = "; ".join(problems)
            self._log_console(f"  off target (training, not repeated) — {reason}")
            self.experimenter.announce_training_off_target(reason)
        elif not result.valid:
            self._n_invalid += 1
            reason = "; ".join(problems)
            self._log_console(f"  DISCARDED — {reason}")
            self.experimenter.announce_invalid_trial(reason)
            self.participant.show_ready()
            self._refresh_status()
            return

        # write_trial() overwrites the force/speed stats above with exact
        # values from the ring-buffer window, when raw recording is enabled.
        if not result.training:
            self._recording.write_trial(result)

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
        """Sample position and force at a fixed rate."""
        if self._position_source is not None:
            sample = self._position_source.read()
            self._speed_estimator.add(sample)
            speed = self._speed_estimator.speed_mm_s(now=time.perf_counter())
            self.experimenter.update_speed(speed)
            if self._collecting_interval is not None:
                if self._speed_accumulator is not None:
                    self._speed_accumulator.add(sample)
                if self._speed_accumulators_by_interval is not None:
                    self._speed_accumulators_by_interval[self._collecting_interval].add(sample)

        if self._force_source is not None:
            force = self._force_source.read_normal_force()
            if self._collecting_interval is not None:
                if self._force_accumulator is not None:
                    self._force_accumulator.add(force)
                if self._force_accumulators_by_interval is not None:
                    self._force_accumulators_by_interval[self._collecting_interval].add(force)

    def _validity_problems(self, force_stats_by_interval: dict,
                           speed_stats_by_interval: dict) -> list:
        """Which measured channels fell outside tolerance of target, if any.

        Checked per interval, not on a trial-wide average. Empty list means
        valid. A check is skipped only if its tolerance or source isn't
        configured; zero contact/movement during a configured interval is
        itself a problem, not something to skip.
        """
        problems = []
        for interval in (1, 2):
            if self._force_tolerance_pct is not None and self._force_bands is not None \
                    and interval in force_stats_by_interval:
                force_stats = force_stats_by_interval[interval]
                target = self._force_bands.target_n
                if force_stats.mean_n is None:
                    problems.append(f"interval {interval} force: no contact detected")
                elif target:
                    deviation_pct = abs(force_stats.mean_n - target) / abs(target) * 100.0
                    if deviation_pct > self._force_tolerance_pct:
                        direction = "too light" if force_stats.mean_n < target else "too heavy"
                        problems.append(
                            f"interval {interval} force {direction} "
                            f"(%{deviation_pct:.0f})")

            if self._speed_tolerance_pct is not None and self._cursor_speed_mm_s \
                    and interval in speed_stats_by_interval:
                speed_stats = speed_stats_by_interval[interval]
                if speed_stats.mean_mm_s is None:
                    problems.append(f"interval {interval} speed: no movement detected")
                else:
                    target = self._cursor_speed_mm_s
                    deviation_pct = abs(speed_stats.mean_mm_s - target) / abs(target) * 100.0
                    if deviation_pct > self._speed_tolerance_pct:
                        direction = "too slow" if speed_stats.mean_mm_s < target else "too fast"
                        problems.append(
                            f"interval {interval} speed {direction} "
                            f"(%{deviation_pct:.0f})")
        return problems

    def _announce_interval(self, number: int) -> None:
        # Only prints when reveal_stimulus is True (debug/development mode).
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