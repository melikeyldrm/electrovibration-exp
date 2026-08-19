"""Experimenter-facing monitor window.

Shows everything the participant must not see: applied voltage, which
interval carried the stimulus, correctness, live staircase state, and the
participant's live finger speed.

Laid out as a console: status cards along the top, staircase trace and
trial log in a resizable splitter below - scannable at a glance, since the
experimenter is watching the participant, not the screen.
"""

import math
import time
from typing import List, Optional, Tuple

from PyQt5.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt5.QtGui import QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import (QApplication, QFrame, QHBoxLayout, QHeaderView,
                             QLabel, QMessageBox, QSplitter, QTableWidget,
                             QTableWidgetItem, QVBoxLayout, QWidget)

from evexp.psychophysics.staircase import StaircaseController
from evexp.psychophysics.trial import TrialResult
from evexp.ui import theme


class StatusCard(QFrame):
    """One labelled value in the status row."""

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.NoFrame)
        self.setStyleSheet(
            f"background-color: {theme.CONSOLE_PANEL};"
            f"border: 1px solid {theme.CONSOLE_BORDER};"
            f"border-radius: 4px;"
        )

        self._label = QLabel(label.upper())
        self._label.setFont(QFont(theme.CONSOLE_FONT, theme.CONSOLE_SIZE_LABEL,
                                  QFont.DemiBold))
        self._label.setStyleSheet(
            f"color: {theme.CONSOLE_LABEL}; border: none; letter-spacing: 1px;")

        self._value = QLabel("-")
        self._value.setFont(QFont(theme.CONSOLE_MONO, theme.CONSOLE_SIZE_VALUE))
        self._value.setStyleSheet(f"color: {theme.CONSOLE_TEXT}; border: none;")

        layout = QVBoxLayout()
        layout.setContentsMargins(12, 8, 12, 9)
        layout.setSpacing(2)
        layout.addWidget(self._label)
        layout.addWidget(self._value)
        self.setLayout(layout)

    def set_value(self, text: str, colour: Optional[str] = None) -> None:
        self._value.setText(text)
        self._value.setStyleSheet(
            f"color: {colour or theme.CONSOLE_TEXT}; border: none;")


class ConvergencePlot(QWidget):
    """Live staircase trace, updated after every trial.

    Drawn on a log voltage axis: the staircase steps in dB, so on a linear
    axis every step shrinks as voltage falls and the trace collapses into a
    smear right where it matters most, near the threshold. On a log axis
    each dB step is the same height, so reversals and convergence are
    visible at a glance.
    """

    PADDING_LEFT = 62
    PADDING_RIGHT = 16
    PADDING_TOP = 14
    PADDING_BOTTOM = 26
    MIN_SPAN_RATIO = 2.0   # never zoom in tighter than a factor of two

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(190)
        self._points: List[Tuple[float, bool]] = []   # (voltage, is_reversal)
        self._threshold: Optional[float] = None

    def add_point(self, voltage: float, is_reversal: bool) -> None:
        self._points.append((voltage, is_reversal))
        self.update()

    def set_threshold(self, threshold: Optional[float]) -> None:
        # NaN until enough reversals exist; treat it as "no estimate yet".
        if threshold is None or threshold != threshold:
            self._threshold = None
        else:
            self._threshold = threshold
        self.update()

    def clear(self) -> None:
        self._points.clear()
        self._threshold = None
        self.update()

    def _bounds(self) -> Tuple[float, float]:
        values = [v for v, _ in self._points if v > 0]
        if self._threshold is not None:
            values.append(self._threshold)
        if not values:
            return 0.5, 2.0
        lo, hi = min(values), max(values)
        lo *= 10 ** (-2.0 / 20.0)   # 2 dB of headroom either side
        hi *= 10 ** (2.0 / 20.0)
        if hi / lo < self.MIN_SPAN_RATIO:
            mid = math.sqrt(lo * hi)
            half = math.sqrt(self.MIN_SPAN_RATIO)
            lo, hi = mid / half, mid * half
        return lo, hi

    def _y_for(self, voltage: float, lo: float, hi: float) -> float:
        top = self.PADDING_TOP
        bottom = self.height() - self.PADDING_BOTTOM
        if voltage <= 0:
            return bottom
        frac = (math.log10(voltage) - math.log10(lo)) / \
               (math.log10(hi) - math.log10(lo))
        frac = min(1.0, max(0.0, frac))
        return bottom - frac * (bottom - top)

    def _x_for(self, index: int, n: int) -> float:
        left = self.PADDING_LEFT
        right = self.width() - self.PADDING_RIGHT
        if n <= 1:
            return left
        return left + (index / (n - 1)) * (right - left)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        # Panel fill plus a hairline border, so the plot reads as a card like
        # the status row above it rather than floating on the window.
        card = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.setPen(QPen(QColor(theme.CONSOLE_BORDER), 1))
        painter.setBrush(QColor(theme.CONSOLE_PANEL))
        painter.drawRoundedRect(card, 4, 4)

        lo, hi = self._bounds()
        left = self.PADDING_LEFT
        right = self.width() - self.PADDING_RIGHT
        top = self.PADDING_TOP
        bottom = self.height() - self.PADDING_BOTTOM

        # Gridlines, labelled in volts even though the spacing is logarithmic:
        # the experimenter thinks in the volts on the DAQ, not in dB.
        painter.setFont(QFont(theme.CONSOLE_MONO, 8))
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = bottom - frac * (bottom - top)
            painter.setPen(QPen(QColor(theme.CONSOLE_GRID), 1))
            painter.drawLine(int(left), int(y), int(right), int(y))
            value = 10 ** (math.log10(lo) + frac * (math.log10(hi) - math.log10(lo)))
            painter.setPen(QPen(QColor(theme.CONSOLE_LABEL), 1))
            painter.drawText(QRectF(0, y - 8, left - 8, 16),
                             Qt.AlignRight | Qt.AlignVCenter, f"{value:.3f}")

        if not self._points:
            painter.setPen(QPen(QColor(theme.CONSOLE_LABEL), 1))
            painter.setFont(QFont(theme.CONSOLE_FONT, theme.CONSOLE_SIZE_BODY))
            painter.drawText(self.rect(), Qt.AlignCenter,
                             "Staircase trace appears after the first trial")
            return

        n = len(self._points)

        # Threshold estimate
        if self._threshold is not None:
            y = self._y_for(self._threshold, lo, hi)
            pen = QPen(QColor(theme.CONSOLE_THRESHOLD), 1, Qt.DashLine)
            painter.setPen(pen)
            painter.drawLine(int(left), int(y), int(right), int(y))

        # Trace
        painter.setPen(QPen(QColor(theme.CONSOLE_TRACE), 2))
        previous: Optional[QPointF] = None
        for i, (voltage, _) in enumerate(self._points):
            point = QPointF(self._x_for(i, n), self._y_for(voltage, lo, hi))
            if previous is not None:
                painter.drawLine(previous, point)
            previous = point

        # Markers; reversals emphasised because they are what the threshold
        # is actually computed from.
        for i, (voltage, is_reversal) in enumerate(self._points):
            point = QPointF(self._x_for(i, n), self._y_for(voltage, lo, hi))
            painter.setPen(Qt.NoPen)
            if is_reversal:
                painter.setBrush(QColor(theme.CONSOLE_OK))
                painter.drawEllipse(point, 4.5, 4.5)
            else:
                painter.setBrush(QColor(theme.CONSOLE_TRACE))
                painter.drawEllipse(point, 2.5, 2.5)

        painter.setPen(QPen(QColor(theme.CONSOLE_LABEL), 1))
        painter.setFont(QFont(theme.CONSOLE_MONO, 8))
        painter.drawText(QRectF(left, bottom + 4, right - left, 18),
                         Qt.AlignLeft, "trial 0")
        painter.drawText(QRectF(left, bottom + 4, right - left, 18),
                         Qt.AlignRight, f"trial {n - 1}")


class ExperimenterWindow(QWidget):
    """The console window: status cards, live staircase plot, and a trial
    log table, updated as the session runs (see update_status, add_result,
    update_speed)."""

    COLUMNS = ["Trial", "Voltage (V)", "Stim", "Resp",
               "Correct", "Reversal", "RT (s)"]

    def __init__(self, experiment_id: str, participant_id: str,
                 target_speed_mm_s: float = 50.0):
        super().__init__()
        self.setWindowTitle("Experimenter console")
        self.resize(1080, 860)
        self.setStyleSheet(f"background-color: {theme.CONSOLE_BG};")
        self._target_speed = target_speed_mm_s
        self._session_start = time.time()
        self._n_trials = 0

        # --- header --------------------------------------------------------
        title = QLabel(experiment_id)
        title.setFont(QFont(theme.CONSOLE_FONT, 15, QFont.DemiBold))
        title.setStyleSheet(f"color: {theme.CONSOLE_TEXT};")

        subtitle = QLabel(f"participant {participant_id}")
        subtitle.setFont(QFont(theme.CONSOLE_FONT, theme.CONSOLE_SIZE_BODY))
        subtitle.setStyleSheet(f"color: {theme.CONSOLE_LABEL};")

        header = QHBoxLayout()
        header.setSpacing(12)
        header.addWidget(title)
        header.addWidget(subtitle)
        header.addStretch()

        # --- status cards --------------------------------------------------
        self._card_state = StatusCard("phase")
        self._card_voltage = StatusCard("next voltage")
        self._card_step = StatusCard("step")
        self._card_reversals = StatusCard("reversals")
        self._card_threshold = StatusCard("threshold")
        self._card_trials = StatusCard("trials")
        self._card_elapsed = StatusCard("elapsed")

        cards = QHBoxLayout()
        cards.setSpacing(8)
        for card in (self._card_state, self._card_voltage, self._card_step,
                     self._card_reversals, self._card_threshold,
                     self._card_trials, self._card_elapsed):
            cards.addWidget(card)

        # --- speed readout -------------------------------------------------
        self._speed = QLabel("finger speed    -")
        self._speed.setFont(QFont(theme.CONSOLE_MONO, 13))
        self._speed.setStyleSheet(
            f"color: {theme.CONSOLE_LABEL};"
            f"background-color: {theme.CONSOLE_PANEL};"
            f"border: 1px solid {theme.CONSOLE_BORDER};"
            f"border-radius: 4px; padding: 8px 12px;"
        )

        # --- message line ----------------------------------------------------
        self._message = QLabel("")
        self._message.setFont(QFont(theme.CONSOLE_FONT,
                                    theme.CONSOLE_SIZE_BODY, QFont.DemiBold))
        self._message.setStyleSheet(f"color: {theme.CONSOLE_ACCENT};")
        self._message.setVisible(False)

        # --- plot + table ----------------------------------------------------
        self._plot = ConvergencePlot()

        self._table = QTableWidget(0, len(self.COLUMNS))
        self._table.setHorizontalHeaderLabels(self.COLUMNS)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setShowGrid(False)
        self._table.setAlternatingRowColors(True)
        self._table.setFont(QFont(theme.CONSOLE_MONO, 11))
        self._table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {theme.CONSOLE_PANEL};
                alternate-background-color: {theme.CONSOLE_PANEL_ALT};
                color: {theme.CONSOLE_TEXT};
                border: 1px solid {theme.CONSOLE_BORDER};
                border-radius: 4px;
                gridline-color: {theme.CONSOLE_BORDER};
            }}
            QHeaderView::section {{
                background-color: {theme.CONSOLE_BG};
                color: {theme.CONSOLE_LABEL};
                border: none;
                border-bottom: 1px solid {theme.CONSOLE_BORDER};
                padding: 7px;
                font-weight: 600;
            }}
            QTableWidget::item {{ padding: 5px; }}
        """)

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self._plot)
        splitter.addWidget(self._table)
        splitter.setSizes([230, 430])
        splitter.setHandleWidth(6)
        splitter.setStyleSheet(
            f"QSplitter::handle {{ background-color: {theme.CONSOLE_BG}; }}")

        layout = QVBoxLayout()
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(10)
        layout.addLayout(header)
        layout.addLayout(cards)
        layout.addWidget(self._speed)
        layout.addWidget(self._message)
        layout.addWidget(splitter, stretch=1)
        self.setLayout(layout)

        self._clock = QTimer(self)
        self._clock.setInterval(1000)
        self._clock.timeout.connect(self._tick_clock)
        self._clock.start()
        self._tick_clock()

    # --- live updates ------------------------------------------------------

    def _tick_clock(self) -> None:
        seconds = int(time.time() - self._session_start)
        self._card_elapsed.set_value(f"{seconds // 60:02d}:{seconds % 60:02d}")

    def update_status(self, state_name: str,
                      staircase: StaircaseController) -> None:
        self._card_state.set_value(state_name.replace("_", " ").lower())
        self._card_voltage.set_value(f"{staircase.value:.3f} V")

        unit = "dB" if staircase.cfg.domain == "db" else "V"
        self._card_step.set_value(f"{staircase.step_size:.0f} {unit}")

        done = len(staircase.reversals)
        needed = staircase.total_reversals_needed
        self._card_reversals.set_value(
            f"{done}/{needed}",
            theme.CONSOLE_OK if done >= needed else None,
        )

        threshold = staircase.threshold_estimate
        if threshold != threshold:   # NaN: no reversals yet
            self._card_threshold.set_value("-")
        else:
            self._card_threshold.set_value(f"{threshold:.3f} V")
        self._plot.set_threshold(threshold)

        self._card_trials.set_value(str(self._n_trials))

    def update_speed(self, speed_mm_s: Optional[float]) -> None:
        """Refresh the live finger-speed readout.

        Pass None when no finger is detected. Deviation from target is
        shown next to the raw value so the experimenter can judge at a
        glance whether the participant needs correcting.
        """
        base = (f"color: %s; background-color: {theme.CONSOLE_PANEL};"
                f"border: 1px solid {theme.CONSOLE_BORDER};"
                f"border-radius: 4px; padding: 8px 12px;")

        if speed_mm_s is None:
            self._speed.setText("finger speed        -     no finger detected")
            self._speed.setStyleSheet(base % theme.CONSOLE_LABEL)
            return

        deviation = speed_mm_s - self._target_speed
        self._speed.setText(
            f"finger speed  {speed_mm_s:6.1f} mm/s     "
            f"target {self._target_speed:.0f}     {deviation:+6.1f}"
        )
        if abs(deviation) <= 10.0:
            colour = theme.CONSOLE_OK
        elif abs(deviation) <= 25.0:
            colour = theme.CONSOLE_WARN
        else:
            colour = theme.CONSOLE_BAD
        self._speed.setStyleSheet(base % colour)

    def add_result(self, result: TrialResult) -> None:
        self._n_trials += 1
        self._plot.add_point(result.applied_voltage, result.reversal)

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
                item.setForeground(QColor(theme.CONSOLE_BAD))
            elif result.reversal:
                item.setForeground(QColor(theme.CONSOLE_OK))
            else:
                item.setForeground(QColor(theme.CONSOLE_TEXT))
            self._table.setItem(row, column, item)
        self._table.scrollToBottom()

    # --- session end -------------------------------------------------------

    def _announce(self, text: str, colour: str) -> None:
        self._message.setText(text)
        self._message.setStyleSheet(f"color: {colour};")
        self._message.setVisible(True)

    def announce_completion(self, staircase: StaircaseController) -> None:
        self._clock.stop()
        n_averaged = min(staircase.cfg.n_reversals_to_stop,
                         len(staircase.reversals))
        self._plot.set_threshold(staircase.threshold_estimate)
        self._announce(
            f"Session complete — threshold {staircase.threshold_estimate:.3f} V "
            f"(mean of last {n_averaged} reversals)",
            theme.CONSOLE_OK,
        )

    def announce_abort(self) -> None:
        self._clock.stop()
        self._announce(
            "Session aborted — stimulus not detected at the ceiling voltage. "
            "No threshold estimate recorded.",
            theme.CONSOLE_BAD,
        )

    def announce_training_done(self) -> None:
        self._announce(
            "Practice complete — recorded session starts on the next SPACE press",
            theme.CONSOLE_ACCENT,
        )

    # --- input -------------------------------------------------------------

    def keyPressEvent(self, event) -> None:
        """Escape ends the session, with confirmation.

        Lives here rather than on the participant window so a participant
        leaning on the keyboard can't truncate a session mid-staircase.
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

    def clear_message(self) -> None:
        """Hide the announcement line."""
        self._message.setVisible(False)