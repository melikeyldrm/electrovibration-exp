"""Session setup: participant identity and the sliding speed condition.

Shown once, before any window opens. Both values it collects end up in the
filename, the config snapshot and the trial log, so they are treated as data
rather than as convenience: the speed in particular is an experimental
condition, and a session whose recorded speed does not match the speed the
participant was actually paced at is worse than no session at all.

Speed is offered as a small set of buttons rather than a free-text field.
The conditions are fixed by the protocol, so anything outside them is a
typo, and a typo here is invisible until analysis.
"""

from dataclasses import dataclass
from typing import List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QButtonGroup, QDialog, QDialogButtonBox,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QVBoxLayout)

from evexp.ui import theme

# Characters that would be awkward or unsafe in a filename. The participant
# ID becomes part of the output path, so it is restricted at the point of
# entry rather than mangled silently later.
_FORBIDDEN = set('\\/:*?"<>|')


@dataclass(frozen=True)
class SessionSetup:
    """What the experimenter chose before the session started."""
    participant_id: str
    target_speed_mm_s: float


class SetupDialog(QDialog):
    """Modal dialog collecting the participant ID and speed condition."""

    def __init__(self, speed_options: List[float],
                 default_speed_mm_s: float,
                 experiment_id: str = "",
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Session setup")
        self.setModal(True)
        self.setMinimumWidth(430)
        self.setStyleSheet(f"background-color: {theme.CONSOLE_BG};")

        self._speed_options = list(speed_options)
        self._selected_speed = self._initial_speed(default_speed_mm_s)

        layout = QVBoxLayout()
        layout.setContentsMargins(28, 24, 28, 22)
        layout.setSpacing(14)

        if experiment_id:
            title = QLabel(experiment_id)
            title.setFont(QFont(theme.CONSOLE_FONT, 14, QFont.DemiBold))
            title.setStyleSheet(f"color: {theme.CONSOLE_TEXT};")
            layout.addWidget(title)

        layout.addWidget(self._section_label("Participant ID"))

        self._id_field = QLineEdit()
        self._id_field.setFont(QFont(theme.CONSOLE_MONO, 13))
        self._id_field.setStyleSheet(
            f"background-color: {theme.CONSOLE_PANEL};"
            f"color: {theme.CONSOLE_TEXT};"
            f"border: 1px solid {theme.CONSOLE_BORDER};"
            f"border-radius: 4px; padding: 8px 10px;"
        )
        self._id_field.textChanged.connect(self._revalidate)
        layout.addWidget(self._id_field)

        layout.addSpacing(6)
        layout.addWidget(self._section_label("Sliding speed"))

        self._speed_group = QButtonGroup(self)
        self._speed_group.setExclusive(True)
        speed_row = QHBoxLayout()
        speed_row.setSpacing(8)
        for index, speed in enumerate(self._speed_options):
            button = QPushButton(f"{speed:g} mm/s")
            button.setCheckable(True)
            button.setCursor(Qt.PointingHandCursor)
            button.setFont(QFont(theme.CONSOLE_FONT, 12, QFont.DemiBold))
            button.setMinimumHeight(44)
            button.setStyleSheet(self._speed_button_style())
            button.setChecked(speed == self._selected_speed)
            self._speed_group.addButton(button, index)
            speed_row.addWidget(button)
        self._speed_group.idClicked.connect(self._on_speed_clicked)
        layout.addLayout(speed_row)

        self._warning = QLabel("")
        self._warning.setWordWrap(True)
        self._warning.setFont(QFont(theme.CONSOLE_FONT, theme.CONSOLE_SIZE_BODY))
        self._warning.setStyleSheet(f"color: {theme.CONSOLE_BAD};")
        layout.addWidget(self._warning)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self._buttons.button(QDialogButtonBox.Ok).setText("Start session")
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        self.setLayout(layout)
        self._revalidate()
        self._id_field.setFocus()

    # --- construction helpers ----------------------------------------------

    def _initial_speed(self, default_speed_mm_s: float) -> float:
        if not self._speed_options:
            raise ValueError("speed_options must contain at least one speed")
        if default_speed_mm_s in self._speed_options:
            return default_speed_mm_s
        # A default outside the offered set means the config disagrees with
        # itself; pick something valid rather than starting with no selection.
        return self._speed_options[0]

    def _section_label(self, text: str) -> QLabel:
        label = QLabel(text.upper())
        label.setFont(QFont(theme.CONSOLE_FONT, theme.CONSOLE_SIZE_LABEL,
                            QFont.DemiBold))
        label.setStyleSheet(
            f"color: {theme.CONSOLE_LABEL}; letter-spacing: 1px;")
        return label

    def _speed_button_style(self) -> str:
        return f"""
            QPushButton {{
                background-color: {theme.CONSOLE_PANEL};
                color: {theme.CONSOLE_TEXT};
                border: 1px solid {theme.CONSOLE_BORDER};
                border-radius: 4px;
            }}
            QPushButton:hover {{
                border-color: {theme.CONSOLE_ACCENT};
            }}
            QPushButton:checked {{
                background-color: {theme.CONSOLE_ACCENT};
                color: #ffffff;
                border-color: {theme.CONSOLE_ACCENT};
            }}
        """

    # --- validation --------------------------------------------------------

    def _on_speed_clicked(self, index: int) -> None:
        self._selected_speed = self._speed_options[index]

    def _validation_error(self) -> Optional[str]:
        text = self._id_field.text().strip()
        if not text:
            return None   # nothing typed yet: no error, but not valid either
        if any(char in _FORBIDDEN for char in text):
            return "Participant ID cannot contain \\ / : * ? \" < > |"
        return None

    def _revalidate(self) -> None:
        error = self._validation_error()
        self._warning.setText(error or "")
        has_id = bool(self._id_field.text().strip())
        self._buttons.button(QDialogButtonBox.Ok).setEnabled(
            has_id and error is None)

    # --- result ------------------------------------------------------------

    def setup(self) -> SessionSetup:
        """The chosen values. Only meaningful after the dialog is accepted."""
        return SessionSetup(
            participant_id=self._id_field.text().strip(),
            target_speed_mm_s=float(self._selected_speed),
        )


def ask_for_setup(speed_options: List[float], default_speed_mm_s: float,
                  experiment_id: str = "") -> Optional[SessionSetup]:
    """Run the dialog. Returns None if the experimenter cancelled."""
    dialog = SetupDialog(speed_options, default_speed_mm_s, experiment_id)
    if dialog.exec_() != QDialog.Accepted:
        return None
    return dialog.setup()