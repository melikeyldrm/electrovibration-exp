"""Fullscreen window shown to the participant.

Deliberately minimal: the participant must not see the applied voltage, which
interval carried the stimulus, whether their previous answer was correct, or
their numeric finger speed. All of that lives in the experimenter window on a
separate screen - a changing number in front of the participant would compete
with the perceptual task.

The one number that *is* shown to the participant, indirectly, is applied
force - as the fill colour and border weight of their own position marker.
That is feedback about the participant's own present action, not a result to
be interpreted, so it does not compete with the perceptual task the way a
numeric readout would.

There is no keyboard shortcut to close this window. Ending a session is the
experimenter's decision, made from the console; a stray Escape keypress by the
participant must not truncate a session mid-staircase.

All geometry here is specified in millimetres and converted to pixels through
a ScreenCalibration at draw time. Sliding speed is an experimental variable,
so the cue track has to be the same physical length - and the pacing cue the
same physical speed - regardless of the window it happens to be drawn in.
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

    Two markers share one track: a circle moving at a fixed physical speed
    (the pace the participant is asked to match), and a square showing where
    the participant's finger actually is. The gap between them *is* the speed
    feedback - no numeric readout is needed, and the two markers differ in
    both shape and hue so they stay distinguishable without relying on colour
    discrimination.

    The square's own fill and border additionally carry force feedback: its
    colour moves from blue (too light) through green (on target) to red (too
    hard), and its border thickens with distance from target regardless of
    colour, so the feedback still works without colour discrimination.

    The track is a single one-way sweep: the cue starts at the start marker
    and travels at cue_speed_mm_s until it reaches the far end, at which
    point the interval ends (TrialTiming.interval_s is exactly travel_mm /
    cue_speed_mm_s - see psychophysics/trial.py). There is no return leg and
    no reversal; a session that used to see one out-and-back stroke per
    interval now sees one one-way stroke.

    Both the square and the force source are fed externally. Today position
    comes from the mouse and force from a simulated or manually-driven
    source; when the Neonode sensor and the Nano17 are wired up, nothing
    here changes.
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

        Falls back to the available width if the window is too narrow to hold
        the full physical travel. That fallback breaks the calibration, so it
        warns once rather than failing silently: a session run in a too-small
        window would otherwise record speeds that are quietly wrong.
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
        """mm per pixel actually in force along the track.

        Equal to the calibration whenever the track fits, which is the only
        case that produces valid data; using the drawn length keeps the
        display self-consistent in the degraded case too.
        """
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
        """Show the track, start marker and the participant's own marker, but
        no moving cue.

        Used during the pre-interval wait so the participant can settle their
        fingertip on the start position before the interval begins.
        """
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
        """Cue position for a constant-speed one-way sweep.

        Linear in elapsed time, clamped to the track length. The clamp is a
        safety net for frame-timing jitter near the end of the interval
        (interval_s is set to exactly travel_mm / cue_speed_mm_s upstream,
        so the cue should reach the far end just as the interval ends, not
        before) - it is not meant to hide a real mismatch between the two.
        """
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
        """Feed pointer position into the position and (if manual) force sources.

        Only ManualPositionSource has push(); a real sensor source produces
        its own samples and ignores the mouse entirely. The same applies to
        force: ManualForceSource is a development stand-in, driven here by
        the pointer's vertical position (up = more force) so the feedback
        colours can be exercised deliberately instead of only ever showing
        the on-target colour.
        """
        if self._track_visible:
            push_position = getattr(self._position_source, "push", None)
            if push_position is not None:
                push_position(self._px_to_mm(event.pos().x()))

            push_force = getattr(self._force_source, "push", None)
            if push_force is not None:
                push_force(self._force_from_pointer_y(event.pos().y()))
        super().mouseMoveEvent(event)

    def _force_from_pointer_y(self, y_px: float) -> float:
        """Map vertical pointer position to a force for manual driving.

        Top of the widget is the highest force the scale shows meaningfully
        (target plus twice the full-scale distance, i.e. solidly past
        saturated red); bottom is zero. Purely a development convenience -
        it has no physical meaning once a real sensor is in use.
        """
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

        # Start marker: a vertical notch, so "place your finger at the start"
        # refers to a specific visible point rather than a vague region. Every
        # stroke then begins from the same physical spot.
        half_notch = theme.START_MARKER_HEIGHT_PX / 2.0
        painter.setPen(QPen(QColor(theme.START_MARKER), 3,
                            Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(int(x0), int(y - half_notch),
                         int(x0), int(y + half_notch))

        # Pacing cue (circle) - only while an interval is running. Drawn
        # before the participant's marker so that the marker stays visible
        # when the two overlap, which is exactly when the participant is on
        # pace and most needs to be able to tell.
        if self._running:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(theme.CURSOR))
            painter.drawEllipse(QPointF(self._mm_to_px(self._cue_mm), y),
                                theme.CURSOR_RADIUS_PX,
                                theme.CURSOR_RADIUS_PX)

        # Participant's finger position (square), on top. Fill and border
        # carry force feedback; position comes from the position source
        # independently of it, so a working position source with no force
        # source yet still draws a marker, just an uncoloured one.
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

    Key handling is gated by an input mode rather than accepted at all times,
    so a participant pressing 1 during an interval cannot desynchronise the
    trial state machine.

    The layout uses fixed-height text areas so that the track stays at a
    constant screen position through every phase - the participant is
    touching the screen, so the track drifting as text changes length would
    be worse than cosmetic.
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

        # Shows the pre-interval countdown, then the interval number. One
        # widget for both so the digit never moves between the two.
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
        # Ceiling, so the last visible digit is 1 rather than 0: the interval
        # begins the moment the count would reach zero.
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
        """Wait screen shown before *each* interval, per protocol.

        The track and the participant's marker are visible so they can settle
        their fingertip on the start marker; a countdown tells them exactly
        when the interval begins, so the first stroke isn't rushed.
        """
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