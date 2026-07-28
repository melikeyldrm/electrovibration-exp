"""Glue between the Trial2IFC state machine and the Qt event loop.

Every phase transition is driven by a one-shot QTimer rather than by waiting,
so the interface stays responsive throughout. This is also why Trial2IFC itself
contains no Qt code: swapping this controller for the headless SimulatedRunner
requires no change to the trial logic.
"""

from PyQt5.QtCore import QObject, QTimer

from evexp.data.csv_logger import CSVTrialLogger
from evexp.psychophysics.trial import Trial2IFC, TrialState
from evexp.ui.experimenter_window import ExperimenterWindow
from evexp.ui.participant_window import ParticipantWindow


class SessionController(QObject):
    def __init__(
        self,
        trial: Trial2IFC,
        participant: ParticipantWindow,
        experimenter: ExperimenterWindow,
        logger: CSVTrialLogger,
        reveal_stimulus: bool = False,
    ):
        super().__init__()
        self.trial = trial
        self.participant = participant
        self.experimenter = experimenter
        self.logger = logger
        self.reveal_stimulus = reveal_stimulus

        self.participant.startRequested.connect(self._begin_trial)
        self.participant.responseGiven.connect(self._on_response)
        self._refresh_status()

    def _begin_trial(self) -> None:
        duration_s = self.trial.start_trial()
        self.participant.show_interval(1, duration_s)
        self._log_console(
            f"\ntrial {self.trial.trial_index:<3d} {self.trial.applied_voltage:6.1f} V"
        )
        self._announce_interval(1)
        self._refresh_status()
        self._arm(duration_s)

    def _advance(self) -> None:
        duration_s = self.trial.advance()
        state = self.trial.state

        if state is TrialState.GAP:
            self.participant.show_gap()
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
        self.logger.log(result)
        self.experimenter.add_result(result)

        verdict = "correct" if result.correct else "WRONG"
        reversal = "  [reversal]" if result.reversal else ""
        self._log_console(
            f"  answered {response_interval} -> {verdict}{reversal}   "
            f"next: {self.trial.staircase.value:.1f} V"
        )

        if self.trial.finished:
            self.participant.show_finished()
            self.experimenter.announce_completion(self.trial.staircase)
        else:
            self.participant.show_ready()
            self._refresh_status()

    def _announce_interval(self, number: int) -> None:
        marker = "  <<< STIMULUS" if self.trial.stimulus.is_active else ""
        self._log_console(f"  interval {number}{marker}")

    def _log_console(self, message: str) -> None:
        if self.reveal_stimulus:
            print(message, flush=True)

    def _arm(self, duration_s: float) -> None:
        QTimer.singleShot(int(duration_s * 1000), self._advance)

    def _refresh_status(self) -> None:
        self.experimenter.update_status(self.trial.state.name, self.trial.staircase)