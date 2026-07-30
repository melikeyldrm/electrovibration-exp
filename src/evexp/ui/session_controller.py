"""Glue between the Trial2IFC state machine and the Qt event loop.

Every phase transition is driven by a one-shot QTimer rather than by waiting,
so the interface stays responsive throughout. This is also why Trial2IFC itself
contains no Qt code: swapping this controller for the headless SimulatedRunner
requires no change to the trial logic.

A separate repeating timer polls the position source and pushes the derived
finger speed to the experimenter console. Polling is used rather than a
callback so that the display refreshes at a fixed, predictable rate regardless
of how fast the underlying source produces samples - the mouse fires events
only when it moves, whereas a real sensor streams at ~100 Hz.
"""

import time
from typing import Optional

from PyQt5.QtCore import QObject, QTimer

from evexp.data.csv_logger import CSVTrialLogger
from evexp.hardware.position import PositionSource
from evexp.processing.speed import SpeedEstimator
from evexp.psychophysics.trial import Trial2IFC, TrialState
from evexp.ui.experimenter_window import ExperimenterWindow
from evexp.ui.participant_window import ParticipantWindow

SPEED_POLL_INTERVAL_MS = 50   # 20 Hz: fast enough to be live, slow enough to read


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

        # Wire participant inputs to the two entry points of the trial loop.
        # SPACE -> _begin_trial, 1/2 keypress -> _on_response.
        # _advance is never called by the participant; only QTimer fires it.
        self.participant.startRequested.connect(self._begin_trial)
        self.participant.responseGiven.connect(self._on_response)

        # Live finger-speed readout on the experimenter console.
        self._position_source = position_source
        self._speed_estimator = SpeedEstimator()
        self._speed_timer = QTimer(self)
        self._speed_timer.setInterval(SPEED_POLL_INTERVAL_MS)
        self._speed_timer.timeout.connect(self._poll_speed)
        if position_source is not None:
            self._speed_timer.start()

        self._refresh_status()

    def _begin_trial(self) -> None:
        training = not self._training_done
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

        if state is TrialState.INTERVAL_1:
            self.participant.show_interval(1, duration_s)
            self._announce_interval(1)
        elif state is TrialState.GAP:
            self.participant.show_gap()
        elif state is TrialState.PRE_INTERVAL_WAIT:
            # Reached only on the way to interval 2; the wait before interval 1
            # is entered by _begin_trial above.
            self.participant.show_pre_interval_wait(2, duration_s)
        elif state is TrialState.INTERVAL_2:
            self.participant.show_interval(2, duration_s)
            self._announce_interval(2)
        elif state is TrialState.AWAITING_RESPONSE:
            self.participant.show_response_prompt()
            self._log_console("  -> press 1 or 2")

        self._refresh_status()
        if duration_s is not None:
            self._arm(duration_s)

    def _on_response(self, response_interval: int) -> None:
        result = self.trial.submit_response(response_interval)
        # Log every trial immediately so a crash mid-session still yields
        # partial data. Training rows are included with training=True flag.
        self.logger.log(result)

        verdict = "correct" if result.correct else "WRONG"
        reversal = "  [reversal]" if result.reversal else ""
        self._log_console(
            f"  answered {response_interval} -> {verdict}{reversal}   "
            f"next: {self.trial.staircase.value:.3f} V"
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
            self._speed_timer.stop()
            self.participant.show_aborted()
            self.experimenter.announce_abort()
        elif self.trial.finished:
            self._speed_timer.stop()
            self._save_convergence_plot()
            self.participant.show_finished()
            self.experimenter.announce_completion(self.trial.staircase)
        else:
            self.participant.show_ready()
            self._refresh_status()

    def _poll_speed(self) -> None:
        """Sample the position source and update the experimenter's readout."""
        if self._position_source is None:
            return
        self._speed_estimator.add(self._position_source.read())
        speed = self._speed_estimator.speed_mm_s(now=time.perf_counter())
        self.experimenter.update_speed(speed)

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