"""Experimenter-facing monitor window.

Shows everything the participant must not see: applied voltage, which interval
carried the stimulus, whether each response was correct, the live state of the
staircase, and the participant's live finger speed. Intended for a second
screen.

Finger speed is shown here rather than on the participant's screen on purpose:
a changing number in front of the participant would compete with the
perceptual task. The experimenter can watch it and correct the participant
verbally if they drift far from the target pace.
"""

from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (QApplication, QHBoxLayout, QHeaderView, QLabel,
                             QMessageBox, QTableWidget, QTableWidgetItem,
                             QVBoxLayout, QWidget)

from evexp.psychophysics.staircase import StaircaseController
from evexp.psychophysics.trial import TrialResult


class ExperimenterWindow(QWidget):
    COLUMNS = ["Trial", "Voltage (V)", "Stim interval", "Response",
               "Correct", "Reversal", "RT (s)"]

    def __init__(self, experiment_id: str, participant_id: str,
                 target_speed_mm_s: float = 50.0):
        super().__init__()
        self.setWindowTitle("Experimenter console")
        self.resize(860, 620)
        self._target_speed = target_speed_mm_s

        header = QLabel(f"{experiment_id}   |   participant: {participant_id}")
        header.setFont(QFont("Segoe UI", 12, QFont.Medium))

        self._status = QLabel("")
        self._status.setFont(QFont("Consolas", 11))

        self._speed = QLabel("finger speed:      -")
        self._speed.setFont(QFont("Consolas", 11))

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
        layout.addWidget(self._speed)
        layout.addWidget(self._table)
        self.setLayout(layout)

    def update_status(self, state_name: str, staircase: StaircaseController) -> None:
        threshold = staircase.threshold_estimate
        threshold_text = "-" if threshold != threshold else f"{threshold:.3f} V"
        unit = "dB" if staircase.cfg.domain == "db" else "V"
        self._status.setText(
            f"state: {state_name:<20s} "
            f"next voltage: {staircase.value:6.3f} V   "
            f"step: {staircase.step_size:4.1f} {unit}   "
            f"reversals: {len(staircase.reversals)}/{staircase.total_reversals_needed}   "
            f"threshold: {threshold_text}"
        )

    def update_speed(self, speed_mm_s: Optional[float]) -> None:
        """Refresh the live finger-speed readout.

        Pass None when no finger is detected. The deviation from target is
        shown alongside the raw value so the experimenter can judge at a
        glance whether the participant needs correcting.
        """
        if speed_mm_s is None:
            self._speed.setText("finger speed:      -   (no finger detected)")
            self._speed.setStyleSheet("color: #8a8a8a;")
            return

        deviation = speed_mm_s - self._target_speed
        self._speed.setText(
            f"finger speed: {speed_mm_s:6.1f} mm/s   "
            f"(target {self._target_speed:.0f}, {deviation:+.1f})"
        )
        # Colour is a coarse at-a-glance cue only; the number is what matters.
        if abs(deviation) <= 10.0:
            self._speed.setStyleSheet("color: #1d9e75;")
        elif abs(deviation) <= 25.0:
            self._speed.setStyleSheet("color: #d9a95c;")
        else:
            self._speed.setStyleSheet("color: #c0392b;")

    def add_result(self, result: TrialResult) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)
        values = [
            str(result.trial_index),
            f"{result.applied_voltage:.3f}",
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
        n_averaged = min(staircase.cfg.n_reversals_to_stop,
                         len(staircase.reversals))
        self._status.setText(
            f"SESSION COMPLETE - threshold estimate: "
            f"{staircase.threshold_estimate:.3f} V "
            f"(mean of last {n_averaged} reversals)"
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
    def keyPressEvent(self, event) -> None:
        """Escape closes the session, with confirmation.

        This lives here rather than on the participant window so that a
        participant leaning on the keyboard cannot end a session mid-staircase.
        """
        if event.key() != Qt.Key_Escape:
            super().keyPressEvent(event)
            return
        reply = QMessageBox.question(
            self,
            "End session",
            "End the session now?\n\n"
            "Trials completed so far are already saved, but the staircase "
            "will not reach its stopping criterion, so no valid threshold "
            "estimate will be produced.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            QApplication.quit()