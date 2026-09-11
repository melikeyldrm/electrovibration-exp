"""Fullscreen window shown to the participant.

Shows no applied voltage, stimulus interval, correctness, or numeric
speed - those live in the experimenter window. Applied force is shown
indirectly, as the fill colour/border of the participant's own marker.
All geometry is specified in millimetres and converted to pixels through
a ScreenCalibration at draw time.
"""

import math
import time
from typing import Optional

from PyQt5.QtCore import QElapsedTimer, QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget

from evexp.hardware.force import ForceSource
from evexp.hardware.position import PositionSource
from evexp.hardware.screen import ScreenCalibration
from evexp.processing.force_feedback import (ForceBands, ForceSmoother,
                                              force_border_px, force_color)
from evexp.ui import theme


class CueTrack(QWidget):
    """Pacing cue plus the participant's own tracked position.

    A circle paces at fixed physical speed; a square shows where the
    finger actually is, with fill/border carrying force feedback: blue
    (too light) through green (on target) to red (too hard). A single
    one-way sweep from the start marker to the far end
    (TrialTiming.interval_s = travel_mm / cue_speed_mm_s); no return leg.
    """

    FRAME_INTERVAL_MS = 16    # ~60 fps
    MIN_MARGIN_PX = 40        # breathing room at each end of the track

    def __init__(self, calibration: ScreenCalibration,
                 position_source: Optional[PositionSource] = None,
                 force_source: Optional[ForceSource] = None,
                 force_bands: Optional[ForceBands] = None,
                 force_smoothing_samples: int = 5,
                 travel_mm: float = 100.0,
                 cue_speed_mm_s: float = 50.0,
                 parent=None):
        super().__init__(parent)
        self.setMinimumHeight(200)
        self.setMouseTracking(True)
        self._cal = calibration
        self._position_source = position_source
        self._force_source = force_source
        self._force_bands = force_bands or ForceBands(target_n=1.0, full_scale_n=0.6)
        self._force_smoother = ForceSmoother(window_samples=force_smoothing_samples)
        self._travel_mm = travel_mm
        self._cue_speed_mm_s = cue_speed_mm_s

        self._cue_mm = 0.0
        self._running = False
        self._track_visible = False
        self._duration_ms = 2000
        self._elapsed = QElapsedTimer()
        self._timer = QTimer(self)
        self._timer.setInterval(self.FRAME_INTERVAL_MS)
        self._timer.timeout.connect(self._on_frame)
        self._warned_too_narrow = False

    # --- track geometry ----------------------------------------------------

    def set_cue_speed(self, speed_mm_s: float) -> None:
        """Set the physical pace the cue asks for, in mm/s."""
        self._cue_speed_mm_s = speed_mm_s

    def _track_length_px(self) -> float:
        """Length of the track on screen, honouring the calibration.

        Falls back to the available width if the window is too narrow, and
        warns once since that breaks the calibration.
        """
        wanted = self._cal.mm_to_px_x(self._travel_mm)
        available = self.width() - 2 * self.MIN_MARGIN_PX
        if available <= 0:
            return 1.0
        if wanted > available:
            if not self._warned_too_narrow:
                self._warned_too_narrow = True
                print(
                    f"WARNING: window is too narrow for a {self._travel_mm:.0f} mm "
                    f"track ({wanted:.0f} px needed, {available:.0f} px available). "
                    "The track has been shrunk to fit, so displayed distances and "
                    "speeds are NOT physically calibrated. Run fullscreen on the "
                    "calibrated display."
                )
            return float(available)
        return wanted

    def _effective_mm_per_px(self) -> float:
        """mm per pixel actually in force along the track."""
        return self._travel_mm / self._track_length_px()

    def _track_bounds(self) -> tuple[float, float, float]:
        length = self._track_length_px()
        x0 = (self.width() - length) / 2.0
        y = self.height() / 2.0
        return x0, x0 + length, y

    def _mm_to_px(self, x_mm: float) -> float:
        x0, _, _ = self._track_bounds()
        clamped = min(self._travel_mm, max(0.0, x_mm))
        return x0 + clamped / self._effective_mm_per_px()

    def _px_to_mm(self, x_px: float) -> float:
        x0, _, _ = self._track_bounds()
        x_mm = (x_px - x0) * self._effective_mm_per_px()
        return min(self._travel_mm, max(0.0, x_mm))

    # --- phase control -----------------------------------------------------

    def show_track_only(self) -> None:
        """Show the track and both markers, but no moving cue."""
        self._running = False
        self._timer.stop()
        self._cue_mm = 0.0
        self._track_visible = True
        self._timer.start()   # keep repainting so the marker follows the finger
        self.update()

    def start(self, duration_s: float) -> None:
        self._duration_ms = max(1, int(duration_s * 1000))
        self._cue_mm = 0.0
        self._running = True
        self._track_visible = True
        self._elapsed.restart()
        self._timer.start()
        self.update()

    def clear(self) -> None:
        self._running = False
        self._track_visible = False
        self._timer.stop()
        self._cue_mm = 0.0
        self._force_smoother.reset()
        self.update()

    def _cue_position_mm(self, elapsed_s: float) -> float:
        """Cue position for a constant-speed one-way sweep, clamped to the track length."""
        if self._cue_speed_mm_s <= 0 or self._travel_mm <= 0:
            return 0.0
        return min(self._travel_mm, elapsed_s * self._cue_speed_mm_s)

    def _on_frame(self) -> None:
        if self._running:
            elapsed_ms = self._elapsed.elapsed()
            if elapsed_ms >= self._duration_ms:
                self._running = False
            else:
                self._cue_mm = self._cue_position_mm(elapsed_ms / 1000.0)
        self.update()

    # --- position input ----------------------------------------------------

    def mouseMoveEvent(self, event) -> None:
        """Feed pointer position into the position and (if manual) force sources."""
        if self._track_visible:
            push_position = getattr(self._position_source, "push", None)
            if push_position is not None:
                push_position(self._px_to_mm(event.pos().x()))

            push_force = getattr(self._force_source, "push", None)
            if push_force is not None:
                push_force(self._force_from_pointer_y(event.pos().y()))
        super().mouseMoveEvent(event)

    def _force_from_pointer_y(self, y_px: float) -> float:
        """Map vertical pointer position to a force for manual driving."""
        span = max(1.0, float(self.height()))
        fraction = 1.0 - min(1.0, max(0.0, y_px / span))
        max_force = self._force_bands.target_n + 2.0 * self._force_bands.full_scale_n
        return fraction * max_force

    def leaveEvent(self, event) -> None:
        clear = getattr(self._position_source, "clear", None)
        if clear is not None:
            clear()
        force_clear = getattr(self._force_source, "clear", None)
        if force_clear is not None:
            force_clear()
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

        # Pacing cue (circle), drawn before the participant's marker so the
        # marker stays visible when the two overlap.
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(theme.CURSOR))
        cue_x_mm = self._cue_mm if self._running else 0.0
        painter.drawEllipse(QPointF(self._mm_to_px(cue_x_mm), y),
                            theme.CURSOR_RADIUS_PX,
                            theme.CURSOR_RADIUS_PX)

        # Participant's finger position (square), on top; fill/border carry force feedback.
        if self._position_source is not None:
            sample = self._position_source.read()
            if sample is not None:
                raw_force = (self._force_source.read_normal_force()
                            if self._force_source is not None else None)
                smoothed = self._force_smoother.add(raw_force)

                fill = force_color(smoothed, self._force_bands)
                border_px = force_border_px(smoothed, self._force_bands)

                mx = self._mm_to_px(sample.x_mm)
                half = theme.MARKER_SIZE_PX / 2.0
                rect = QRectF(mx - half, y - half,
                              theme.MARKER_SIZE_PX, theme.MARKER_SIZE_PX)

                if border_px > 0:
                    painter.setPen(QPen(QColor(fill).darker(130), border_px))
                else:
                    painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(fill))
                painter.drawRect(rect)


class ParticipantWindow(QWidget):
    """Participant-facing display and keyboard input.

    Key handling is gated by an input mode so a stray keypress during an
    interval can't desynchronise the trial state machine.
    """

    COUNTDOWN_TICK_MS = 100

    startRequested = pyqtSignal()
    responseGiven = pyqtSignal(int)

    def __init__(self, calibration: ScreenCalibration,
                 position_source: Optional[PositionSource] = None,
                 force_source: Optional[ForceSource] = None,
                 force_bands: Optional[ForceBands] = None,
                 force_smoothing_samples: int = 5,
                 travel_mm: float = 100.0,
                 cue_speed_mm_s: float = 50.0):
        super().__init__()
        self.setWindowTitle("Participant")
        self.setStyleSheet(f"background-color: {theme.BACKGROUND};")
        self.resize(1100, 700)
        self._calibration = calibration

        self._message = QLabel("", self)
        self._message.setAlignment(Qt.AlignCenter)
        self._message.setFixedHeight(theme.MESSAGE_AREA_HEIGHT_PX)
        self._message.setFont(QFont(theme.FONT_FAMILY,
                                    theme.FONT_SIZE_LARGE, QFont.Medium))
        self._message.setStyleSheet(f"color: {theme.TEXT_PRIMARY};")

        # Shows the pre-interval countdown, then the interval number.
        self._numeral = QLabel("", self)
        self._numeral.setAlignment(Qt.AlignCenter)
        self._numeral.setFixedHeight(theme.NUMERAL_AREA_HEIGHT_PX)
        self._numeral.setFont(QFont(theme.FONT_FAMILY,
                                    theme.FONT_SIZE_NUMERAL, QFont.Light))
        self._numeral.setStyleSheet(f"color: {theme.TEXT_EMPHASIS};")

        self._hint = QLabel("", self)
        self._hint.setAlignment(Qt.AlignCenter)
        self._hint.setFixedHeight(theme.HINT_AREA_HEIGHT_PX)
        self._hint.setFont(QFont(theme.FONT_FAMILY, theme.FONT_SIZE_BODY))
        self._hint.setStyleSheet(f"color: {theme.TEXT_SECONDARY};")

        self._cue = CueTrack(calibration, position_source, force_source,
                             force_bands, force_smoothing_samples,
                             travel_mm, cue_speed_mm_s, self)

        layout = QVBoxLayout()
        layout.setContentsMargins(60, 50, 60, 50)
        layout.setSpacing(24)
        layout.addWidget(self._message)
        layout.addWidget(self._numeral)
        layout.addWidget(self._cue, stretch=1)
        layout.addWidget(self._hint)
        self.setLayout(layout)

        self._countdown_end = 0.0
        self._countdown_timer = QTimer(self)
        self._countdown_timer.setInterval(self.COUNTDOWN_TICK_MS)
        self._countdown_timer.timeout.connect(self._tick_countdown)

        self._input_mode = None   # None | "start" | "response"
        self.show_ready()

    # --- countdown ---------------------------------------------------------

    def _start_countdown(self, duration_s: float) -> None:
        self._countdown_end = time.perf_counter() + duration_s
        self._tick_countdown()
        self._countdown_timer.start()

    def _stop_countdown(self) -> None:
        self._countdown_timer.stop()

    def _tick_countdown(self) -> None:
        remaining = self._countdown_end - time.perf_counter()
        if remaining <= 0:
            self._countdown_timer.stop()
            self._numeral.setText("")
            return
        self._numeral.setText(str(math.ceil(remaining)))

    # --- phase screens -----------------------------------------------------

    def show_ready(self) -> None:
        self._input_mode = "start"
        self._stop_countdown()
        self._cue.clear()
        self._numeral.setText("")
        self._message.setText("Ready when you are")
        self._hint.setText("Press SPACE to start the next trial")

    def show_pre_interval_wait(self, number: int, duration_s: float) -> None:
        """Wait screen shown before each interval, with a countdown."""
        self._input_mode = None
        self._message.setText(f"Interval {number} starting")
        self._hint.setText("Place your fingertip on the start marker")
        self._cue.show_track_only()
        self._start_countdown(duration_s)

    def show_interval(self, number: int, duration_s: float) -> None:
        self._input_mode = None
        self._stop_countdown()
        self._message.setText(f"Interval {number}")
        self._numeral.setText("")
        self._hint.setText("Follow the circle with your fingertip")
        self._cue.start(duration_s)

    def show_gap(self) -> None:
        self._input_mode = None
        self._stop_countdown()
        self._cue.clear()
        self._numeral.setText("")
        self._message.setText("")
        self._hint.setText("Lift your finger and return to the start marker")

    def show_response_prompt(self) -> None:
        self._input_mode = "response"
        self._stop_countdown()
        self._cue.clear()
        self._numeral.setText("")
        self._message.setText("Which interval contained the stimulus?")
        self._hint.setText("Press 1 or 2 — take as long as you need")

    def show_finished(self) -> None:
        self._input_mode = None
        self._stop_countdown()
        self._cue.clear()
        self._numeral.setText("")
        self._message.setText("Session complete")
        self._hint.setText("Thank you — you may lift your finger")

    def show_aborted(self) -> None:
        self._input_mode = None
        self._stop_countdown()
        self._cue.clear()
        self._numeral.setText("")
        self._message.setText("Session stopped")
        self._hint.setText("Please wait for the experimenter")

    def show_ready_post_training(self) -> None:
        self._input_mode = "start"
        self._stop_countdown()
        self._cue.clear()
        self._numeral.setText("")
        self._message.setText("Practice complete")
        self._hint.setText("Press SPACE to begin the recorded session")

    # --- input -------------------------------------------------------------

    def keyPressEvent(self, event) -> None:
        # No Escape handling: closing the session is the experimenter's call.
        key = event.key()
        if self._input_mode == "start" and key == Qt.Key_Space:
            self._input_mode = None
            self.startRequested.emit()
        elif self._input_mode == "response" and key in (Qt.Key_1, Qt.Key_2):
            self._input_mode = None
            self.responseGiven.emit(1 if key == Qt.Key_1 else 2)
        elif self._input_mode == "response":
            self._hint.setText("Please press only 1 or 2")