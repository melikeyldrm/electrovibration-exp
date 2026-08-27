"""Tests for the session setup dialog.

The dialog collects three things that become data: the participant ID,
which ends up in the output filename, and the sliding speed and target
force, which are experimental conditions. All are checked here for the
failure modes that would not be noticed until analysis - an ID that cannot
be written to disk, or a condition that was never actually chosen.
"""

import os

import pytest

# A widget cannot be constructed without a QApplication, and CI has no
# display, so Qt is asked for its headless backend before it is imported.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SPEEDS = [20.0, 50.0, 80.0]
FORCES = [0.5, 1.0, 1.5]


@pytest.fixture(scope="module")
def qapp():
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture
def dialog(qapp):
    from evexp.ui.setup_dialog import SetupDialog
    return SetupDialog(SPEEDS, default_speed_mm_s=50.0,
                       force_options=FORCES, default_force_n=1.0,
                       experiment_id="exp01_2ifc_threshold")


def ok_button(dialog):
    from PyQt5.QtWidgets import QDialogButtonBox
    return dialog._buttons.button(QDialogButtonBox.Ok)


# --- speed selection -------------------------------------------------------

def test_offers_every_configured_speed(dialog):
    labels = {b.text() for b in dialog._speed_group.buttons()}
    assert labels == {"20 mm/s", "50 mm/s", "80 mm/s"}


def test_default_speed_starts_selected(dialog):
    checked = [b for b in dialog._speed_group.buttons() if b.isChecked()]
    assert len(checked) == 1
    assert checked[0].text() == "50 mm/s"
    assert dialog.setup().target_speed_mm_s == 50.0


def test_selecting_a_speed_changes_the_result(dialog):
    dialog._id_field.setText("p07")
    dialog._speed_group.button(2).click()
    assert dialog.setup().target_speed_mm_s == 80.0


def test_speed_selection_is_exclusive(dialog):
    dialog._speed_group.button(0).click()
    checked = [b for b in dialog._speed_group.buttons() if b.isChecked()]
    assert len(checked) == 1


def test_default_outside_the_options_falls_back(qapp):
    """A config that disagrees with itself must not leave nothing selected."""
    from evexp.ui.setup_dialog import SetupDialog

    dialog = SetupDialog(SPEEDS, default_speed_mm_s=37.0,
                         force_options=FORCES, default_force_n=1.0)
    checked = [b for b in dialog._speed_group.buttons() if b.isChecked()]
    assert len(checked) == 1
    assert dialog.setup().target_speed_mm_s == 20.0


def test_empty_speed_options_is_rejected(qapp):
    from evexp.ui.setup_dialog import SetupDialog

    with pytest.raises(ValueError):
        SetupDialog([], default_speed_mm_s=50.0,
                    force_options=FORCES, default_force_n=1.0)


# --- force selection ---------------------------------------------------------

def test_offers_every_configured_force(dialog):
    labels = {b.text() for b in dialog._force_group.buttons()}
    assert labels == {"0.5 N", "1 N", "1.5 N"}


def test_default_force_starts_selected(dialog):
    checked = [b for b in dialog._force_group.buttons() if b.isChecked()]
    assert len(checked) == 1
    assert checked[0].text() == "1 N"
    assert dialog.setup().target_force_n == 1.0


def test_selecting_a_force_changes_the_result(dialog):
    dialog._id_field.setText("p07")
    dialog._force_group.button(2).click()
    assert dialog.setup().target_force_n == 1.5


def test_force_selection_is_exclusive(dialog):
    dialog._force_group.button(0).click()
    checked = [b for b in dialog._force_group.buttons() if b.isChecked()]
    assert len(checked) == 1


def test_default_force_outside_the_options_falls_back(qapp):
    from evexp.ui.setup_dialog import SetupDialog

    dialog = SetupDialog(SPEEDS, default_speed_mm_s=50.0,
                         force_options=FORCES, default_force_n=0.9)
    checked = [b for b in dialog._force_group.buttons() if b.isChecked()]
    assert len(checked) == 1
    assert dialog.setup().target_force_n == 0.5


def test_empty_force_options_is_rejected(qapp):
    from evexp.ui.setup_dialog import SetupDialog

    with pytest.raises(ValueError):
        SetupDialog(SPEEDS, default_speed_mm_s=50.0,
                    force_options=[], default_force_n=1.0)


# --- participant ID --------------------------------------------------------

def test_cannot_start_without_an_id(dialog):
    assert not ok_button(dialog).isEnabled()


def test_typing_an_id_enables_start(dialog):
    dialog._id_field.setText("p07")
    assert ok_button(dialog).isEnabled()


def test_whitespace_only_id_is_not_enough(dialog):
    dialog._id_field.setText("   ")
    assert not ok_button(dialog).isEnabled()


def test_id_is_stripped(dialog):
    dialog._id_field.setText("  p07  ")
    assert dialog.setup().participant_id == "p07"


@pytest.mark.parametrize("bad_id", ["p/07", "p:07", "p*07", "p?07", "p|07"])
def test_path_unsafe_ids_are_blocked(dialog, bad_id):
    """The ID becomes part of a filename, so it is restricted on entry."""
    dialog._id_field.setText(bad_id)
    assert not ok_button(dialog).isEnabled()
    assert dialog._warning.text() != ""


def test_warning_clears_once_the_id_is_fixed(dialog):
    dialog._id_field.setText("p/07")
    assert dialog._warning.text() != ""
    dialog._id_field.setText("p07")
    assert dialog._warning.text() == ""
    assert ok_button(dialog).isEnabled()


def test_ids_with_dashes_and_underscores_are_allowed(dialog):
    dialog._id_field.setText("pilot_02-b")
    assert ok_button(dialog).isEnabled()
    assert dialog.setup().participant_id == "pilot_02-b"