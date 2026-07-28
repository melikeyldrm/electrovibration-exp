"""Experimenter-facing monitor window.

Shows everything the participant must not see: applied voltage, which interval
carried the stimulus, whether each response was correct, and the live state of
the staircase. Intended for a second screen.
"""

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (QHBoxLayout, QHeaderView, QLabel, QTableWidget,
                             QTableWidgetItem, QVBoxLayout, QWidget)

from evexp.psychophysics.staircase import StaircaseController
from evexp.psychophysics.trial import TrialResult


class ExperimenterWindow(QWidget):
    COLUMNS = ["Trial", "Voltage (V)", "Stim interval", "Response",
               "Correct", "Reversal", "RT (s)"]

    def __init__(self, experiment_id: str, participant_id: str):
        super().__init__()
        self.setWindowTitle("Experimenter console")
        self.resize(860, 620)

        header = QLabel(f"{experiment_id}   |   participant: {participant_id}")
        header.setFont(QFont("Segoe UI", 12, QFont.Medium))

        self._status = QLabel("")
        self._status.setFont(QFont("Consolas", 11))

        self._table = QTableWidget(0, len(self.COLUMNS))
        self._table.setHorizontalHeaderLabels(self.COLUMNS)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)

        top = QHBoxLayout()
        top.addWidget(header)
        top.addStretch()

        layout = QVBoxLayout()
        layout.addLayout(top)
        layout.addWidget(self._status)
        layout.addWidget(self._table)
        self.setLayout(layout)

    def update_status(self, state_name: str, staircase: StaircaseController) -> None:
        threshold = staircase.threshold_estimate
        threshold_text = "-" if threshold != threshold else f"{threshold:.1f} V"
        self._status.setText(
            f"state: {state_name:<18s} "
            f"next voltage: {staircase.value:6.1f} V   "
            f"step: {staircase.step_size:4.1f} V   "
            f"reversals: {len(staircase.reversals)}/{staircase.cfg.n_reversals_to_stop}   "
            f"threshold: {threshold_text}"
        )

    def add_result(self, result: TrialResult) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)
        values = [
            str(result.trial_index),
            f"{result.applied_voltage:.1f}",
            str(result.stimulus_interval),
            str(result.response_interval),
            "yes" if result.correct else "no",
            "yes" if result.reversal else "",
            f"{result.response_time_s:.2f}",
        ]
        for column, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setTextAlignment(Qt.AlignCenter)
            if not result.correct:
                item.setForeground(QColor("#c0392b"))
            elif result.reversal:
                item.setForeground(QColor("#1d9e75"))
            self._table.setItem(row, column, item)
        self._table.scrollToBottom()

    def announce_completion(self, staircase: StaircaseController) -> None:
        self._status.setText(
            f"SESSION COMPLETE - threshold estimate: "
            f"{staircase.threshold_estimate:.1f} V "
            f"(mean of last {min(6, len(staircase.reversals))} reversals)"
        )

    def announce_abort(self) -> None:
        self._status.setText(
            "SESSION ABORTED - stimulus not detected at ceiling voltage. "
            "No threshold estimate recorded."
        )

    def announce_training_done(self) -> None:
        self._status.setText(
            "Training complete - real session starting on next SPACE press"
        )