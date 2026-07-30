"""Fullscreen window shown to the participant.

Deliberately minimal: the participant must not see the applied voltage, which
interval carried the stimulus, whether their previous answer was correct, or
their numeric finger speed. All of that lives in the experimenter window on a
separate screen - a changing number on the participant's screen would pull
attention away from the perceptual task.
"""

from typing import Optional

from PyQt5.QtCore import QElapsedTimer, QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget

from evexp.hardware.position import PositionSource
from evexp.ui import theme


class CueTrack(QWidget):
    """Pacing cue plus the participant's own tracked position.

    Two markers share one track: a circle sweeping left to right at a fixed
    rate (the pace the participant is asked to match), and a square showing
    where the participant's finger actually is. The gap between them *is* the
    speed feedback - no numeric readout is needed, and the two markers differ
    in both shape and hue so they stay distinguishable without relying on
    colour discrimination.

    The square is fed by a PositionSource. Today that source is the mouse;
    when the Neonode touch sensor is wired up nothing here changes.
    """

    TRACK_MARGIN_PX = 80
    FRAME_INTERVAL_MS = 16   # ~60 fps

    def __init__(self, position_source: Optional[PositionSource] = None,
                 travel_mm: float = 100.0, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(180)
        self.setMouseTracking(True)
        self._position_source = position_source
        self._travel_mm = travel_mm
        self._progress = 0.0
        self._running = False
        self._track_visible = False
        self._duration_ms = 2000
        self._elapsed = QElapsedTimer()
        self._timer = QTimer(self)
        self._timer.setInterval(self.FRAME_INTERVAL_MS)
        self._timer.timeout.connect(self._on_frame)

    # --- track geometry ----------------------------------------------------

    def _track_bounds(self) -> tuple[float, float, float]:
        x0 = float(self.TRACK_MARGIN_PX)
        x1 = float(self.width() - self.TRACK_MARGIN_PX)
        y = self.height() / 2.0
        return x0, x1, y

    def _mm_to_px(self, x_mm: float) -> float:
        x0, x1, _ = self._track_bounds()
        fraction = x_mm / self._travel_mm if self._travel_mm else 0.0
        return x0 + min(1.0, max(0.0, fraction)) * (x1 - x0)

    def _px_to_mm(self, x_px: float) -> float:
        x0, x1, _ = self._track_bounds()
        span = x1 - x0
        if span <= 0:
            return 0.0
        return (x_px - x0) / span * self._travel_mm

    # --- phase control -----------------------------------------------------

    def show_track_only(self) -> None:
        """Show the track and the participant's marker, but no moving cue.

        Used during the pre-interval wait so the participant can place their
        finger at the start position before the interval begins.
        """
        self._running = False
        self._timer.stop()
        self._progress = 0.0
        self._track_visible = True
        self._timer.start()   # keep repainting so the marker follows the finger
        self.update()

    def start(self, duration_s: float) -> None:
        self._duration_ms = max(1, int(duration_s * 1000))
        self._progress = 0.0
        self._running = True
        self._track_visible = True
        self._elapsed.restart()
        self._timer.start()
        self.update()

    def clear(self) -> None:
        self._running = False
        self._track_visible = False
        self._timer.stop()
        self._progress = 0.0
        self.update()

    def _on_frame(self) -> None:
        if self._running:
            fraction = self._elapsed.elapsed() / self._duration_ms
            if fraction >= 1.0:
                self._progress = 1.0
                self._running = False
            else:
                self._progress = fraction
        self.update()

    # --- position input ----------------------------------------------------

    def mouseMoveEvent(self, event) -> None:
        """Feed pointer position into the position source, if it accepts it.

        Only ManualPositionSource has push(); a real sensor source produces
        its own samples and ignores the mouse entirely.
        """
        push = getattr(self._position_source, "push", None)
        if push is not None and self._track_visible:
            push(self._px_to_mm(event.pos().x()))
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:
        clear = getattr(self._position_source, "clear", None)
        if clear is not None:
            clear()
        super().leaveEvent(event)

    # --- painting ----------------------------------------------------------

    def paintEvent(self, event) -> None:
        if not self._track_visible:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        x0, x1, y = self._track_bounds()

        # Travel path
        painter.setPen(QPen(QColor(theme.TRACK), theme.TRACK_HEIGHT_PX,
                            Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(int(x0), int(y), int(x1), int(y))

        # Participant's finger position (square)
        if self._position_source is not None:
            sample = self._position_source.read()
            if sample is not None:
                mx = self._mm_to_px(sample.x_mm)
                half = theme.MARKER_SIZE_PX / 2.0
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(theme.PARTICIPANT_MARKER))
                painter.drawRect(QRectF(mx - half, y - half,
                                        theme.MARKER_SIZE_PX,
                                        theme.MARKER_SIZE_PX))

        # Pacing cue (circle) - only while an interval is running
        if self._running:
            cx = x0 + self._progress * (x1 - x0)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(theme.CURSOR))
            painter.drawEllipse(QPointF(cx, y),
                                theme.CURSOR_RADIUS_PX,
                                theme.CURSOR_RADIUS_PX)


class ParticipantWindow(QWidget):
    """Participant-facing display and keyboard input.

    Key handling is gated by an input mode rather than accepted at all times,
    so a participant pressing 1 during an interval cannot desynchronise the
    trial state machine.
    """

    startRequested = pyqtSignal()
    responseGiven = pyqtSignal(int)

    def __init__(self, position_source: Optional[PositionSource] = None,
                 travel_mm: float = 100.0):
        super().__init__()
        self.setWindowTitle("Participant")
        self.setStyleSheet(f"background-color: {theme.BACKGROUND};")
        self.resize(1000, 620)

        self._message = QLabel("", self)
        self._message.setAlignment(Qt.AlignCenter)
        self._message.setFont(QFont(theme.FONT_FAMILY,
                                    theme.FONT_SIZE_LARGE, QFont.Medium))
        self._message.setStyleSheet(f"color: {theme.TEXT_PRIMARY};")

        self._hint = QLabel("", self)
        self._hint.setAlignment(Qt.AlignCenter)
        self._hint.setFont(QFont(theme.FONT_FAMILY, theme.FONT_SIZE_BODY))
        self._hint.setStyleSheet(f"color: {theme.TEXT_SECONDARY};")

        self._cue = CueTrack(position_source, travel_mm, self)

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

    def show_pre_interval_wait(self, number: int) -> None:
        """Wait screen shown before *each* interval, per protocol.

        The track and the participant's marker are visible so they can settle
        their finger on the start position; the pacing cue has not started yet.
        """
        self._input_mode = None
        self._message.setText(f"Get ready — interval {number}")
        self._hint.setText("Place your finger at the start of the track")
        self._cue.show_track_only()

    def show_interval(self, number: int, duration_s: float) -> None:
        self._input_mode = None
        self._message.setText(f"Interval {number}")
        self._hint.setText("Follow the circle with your fingertip")
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
        elif self._input_mode == "response":
            self._hint.setText("Please press only 1 or 2.")

    def show_aborted(self) -> None:
        self._input_mode = None
        self._cue.clear()
        self._message.setText("Session stopped")
        self._hint.setText(
            "The stimulus could not be detected at maximum voltage."
        )

    def show_ready_post_training(self) -> None:
        self._input_mode = "start"
        self._cue.clear()
        self._message.setText("Ready")
        self._hint.setText(
            "Training complete — press SPACE to begin the real session"
        )