"""Fullscreen window shown to the participant.

Deliberately minimal: the participant must not see the applied voltage, which
interval carried the stimulus, or whether their previous answer was correct.
All of that lives in the experimenter window on a separate screen.
"""

from PyQt5.QtCore import QElapsedTimer, QPointF, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget


class CueTrack(QWidget):
    """Open-loop pacing cue: a marker sweeping left to right at a fixed rate.

    The participant follows the marker with their fingertip. Actual finger
    speed is NOT measured in this phase - no position sensor is connected yet -
    so this is guidance only, not a closed-loop constraint. Once the motorised
    stage and force sensor are added, real speed can be logged and compared
    against this cue.
    """

    TRACK_MARGIN_PX = 80
    FRAME_INTERVAL_MS = 16   # ~60 fps

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(180)
        self._progress = 0.0
        self._running = False
        self._duration_ms = 2000
        self._elapsed = QElapsedTimer()
        self._timer = QTimer(self)
        self._timer.setInterval(self.FRAME_INTERVAL_MS)
        self._timer.timeout.connect(self._on_frame)

    def start(self, duration_s: float) -> None:
        self._duration_ms = max(1, int(duration_s * 1000))
        self._progress = 0.0
        self._running = True
        self._elapsed.restart()
        self._timer.start()
        self.update()

    def clear(self) -> None:
        self._running = False
        self._timer.stop()
        self._progress = 0.0
        self.update()

    def _on_frame(self) -> None:
        fraction = self._elapsed.elapsed() / self._duration_ms
        if fraction >= 1.0:
            self._progress = 1.0
            self._running = False
            self._timer.stop()
        else:
            self._progress = fraction
        self.update()

    def paintEvent(self, event) -> None:
        if not self._running:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        y = self.height() // 2
        x0 = self.TRACK_MARGIN_PX
        x1 = self.width() - self.TRACK_MARGIN_PX
        painter.setPen(QPen(QColor("#3c3c3c"), 6, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(x0, y, x1, y)
        cx = x0 + self._progress * (x1 - x0)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#4da3ff"))
        painter.drawEllipse(QPointF(cx, y), 22, 22)


class ParticipantWindow(QWidget):
    """Participant-facing display and keyboard input.

    Key handling is gated by an input mode rather than accepted at all times,
    so a participant pressing 1 during an interval cannot desynchronise the
    trial state machine.
    """

    startRequested = pyqtSignal()
    responseGiven = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Participant")
        self.setStyleSheet("background-color: #101010;")
        self.resize(1000, 620)

        self._message = QLabel("", self)
        self._message.setAlignment(Qt.AlignCenter)
        self._message.setFont(QFont("Segoe UI", 30, QFont.Medium))
        self._message.setStyleSheet("color: #f0f0f0;")

        self._hint = QLabel("", self)
        self._hint.setAlignment(Qt.AlignCenter)
        self._hint.setFont(QFont("Segoe UI", 16))
        self._hint.setStyleSheet("color: #8a8a8a;")

        self._cue = CueTrack(self)

        layout = QVBoxLayout()
        layout.setContentsMargins(40, 60, 40, 60)
        layout.addWidget(self._message)
        layout.addWidget(self._cue, stretch=1)
        layout.addWidget(self._hint)
        self.setLayout(layout)

        self._input_mode = None   # None | "start" | "response"
        self.show_ready()

    def show_ready(self) -> None:
        self._input_mode = "start"
        self._cue.clear()
        self._message.setText("Ready")
        self._hint.setText("Press SPACE to begin the next trial")

    def show_interval(self, number: int, duration_s: float) -> None:
        self._input_mode = None
        self._message.setText(f"Interval {number}")
        self._hint.setText("Follow the marker with your fingertip")
        self._cue.start(duration_s)

    def show_gap(self) -> None:
        self._input_mode = None
        self._cue.clear()
        self._message.setText("")
        self._hint.setText("Lift your finger and return to the start")

    def show_response_prompt(self) -> None:
        self._input_mode = "response"
        self._cue.clear()
        self._message.setText("Which interval?")
        self._hint.setText("Press 1 or 2")

    def show_finished(self) -> None:
        self._input_mode = None
        self._cue.clear()
        self._message.setText("Session complete")
        self._hint.setText("Thank you - you may lift your finger")

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()
            return
        if self._input_mode == "start" and key == Qt.Key_Space:
            self._input_mode = None
            self.startRequested.emit()
        elif self._input_mode == "response" and key in (Qt.Key_1, Qt.Key_2):
            self._input_mode = None
            self.responseGiven.emit(1 if key == Qt.Key_1 else 2)